#!/usr/bin/env python
"""5-fold ensemble inference for OrdFused-CXR on the held-out test set.

Averages P(cavity)=σ(CORN logit_0) across all fold checkpoints, tunes tau on
the *training* out-of-fold predictions (never on test), scores + writes a
submission. Mirrors predict.py so OrdFused and the CNN baselines are scored
identically.

Usage:
    python predict_ordfused.py --encoder tf_efficientnetv2_s
    python predict_ordfused.py --encoder tf_efficientnetv2_s --no-tabular
    # ensemble several trained OrdFused variants into one submission:
    python predict_ordfused.py --ensemble ordfused_tf_efficientnetv2_s ordfused_densenet201 --name ordfused
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ID_COL, challenge_score, make_folds, set_seed, sweep_tau, sweep_tau_accuracy
from ordfused import ENCODER_FAMILIES, OrdFusedDataset, OrdFusedModel, corn_cavity_prob

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def resolve_preprocess(encoder_name, img_size_hint):
    """(img_size, mean, std) for this checkpoint's encoder — the registry for
    foundation models (which don't use 224/ImageNet stats), else the
    checkpoint's own saved img_size with plain ImageNet stats (matches every
    CNN checkpoint trained so far)."""
    fam = ENCODER_FAMILIES.get(encoder_name)
    if fam is not None:
        return fam["img_size"], fam["mean"], fam["std"]
    return img_size_hint, _DEFAULT_MEAN, _DEFAULT_STD


class CFG:
    def __init__(self, args, split, img_size=None, mean=None, std=None):
        self.csv_path = os.path.join(args.data_dir, f"{split}.csv")
        self.image_dir = os.path.join(args.data_dir, split, "CXR")
        self.image_ext = ".dcm"
        self.img_size = img_size if img_size is not None else args.img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        self.norm_mean = mean if mean is not None else _DEFAULT_MEAN
        self.norm_std = std if std is not None else _DEFAULT_STD
        self.n_folds = args.n_folds
        self.batch_size = args.batch_size


def load_fold_models(model_dir, device, lora_r=16, lora_alpha=16, lora_dropout=0.05):
    """Returns (fold, encoder_name, img_size_hint, model) tuples — img_size_hint
    is the checkpoint's own saved img_size (used as a fallback for plain-timm
    encoders not in ENCODER_FAMILIES).

    lora_r/alpha/dropout must match whatever the checkpoint was actually
    trained with -- PEFT bakes the LoRA rank into the adapter matrices'
    shapes, so a mismatch here fails load_state_dict with a shape error, not
    a silent wrong answer. Default (16/16/0.05) matches every checkpoint
    trained so far; override if train_ordfused.py's --lora-r was ever
    overridden from its own default for the run being loaded."""
    ckpts = sorted(glob.glob(os.path.join(model_dir, "fold*_best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no fold*_best.pt under {model_dir}")
    models = []
    for p in ckpts:
        sd = torch.load(p, map_location=device, weights_only=False)
        m = OrdFusedModel(sd["encoder"], use_tabular=sd["use_tabular"], pretrained=False,
                           lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout).to(device)
        m.load_state_dict(sd["model"])
        m.eval()
        models.append((sd["fold"], sd["encoder"], sd.get("img_size", 224), m))
        print(f"loaded {p} (fold {sd['fold']}, best {sd['best']})")
    return models


def group_by_preprocess(models):
    """(fold, encoder, img_size_hint, model) list -> {(size,mean,std): [(fold, model), ...]}.

    Needed because foundation-model checkpoints (CheXFound @ 512, RAD-DINO @
    518) can't share a dataloader/preprocessing pass with the 224px CNN
    checkpoints in the same ensemble."""
    groups = {}
    for fold, enc, img_size_hint, m in models:
        key = resolve_preprocess(enc, img_size_hint)
        groups.setdefault(key, []).append((fold, m))
    return groups


@torch.no_grad()
def ensemble_probs(models, dl, device, tta=True, zero_tabular=False):
    """Mean P(cavity) across the given (fold, model) pairs, and — when
    tta=True — across {original, horizontally-flipped} for each model (the
    one augmentation already used in training, so it's a safe, cheap addition).

    zero_tabular=True zeroes the tabular vector before every forward pass —
    a diagnostic for checkpoints trained with use_tabular=True, simulating
    metadata being unavailable at test time (NOT the same as a --no-tabular
    checkpoint, whose head was actually trained without the tabular branch)."""
    out = []
    for x, tab, _yb, _yo in dl:
        x, tab = x.to(device), tab.to(device)
        if zero_tabular:
            tab = torch.zeros_like(tab)
        variants = [x, x.flip(-1)] if tta else [x]
        ps = [corn_cavity_prob(m(xv, tab)) for xv in variants for _f, m in models]
        out.append(torch.stack(ps, 0).mean(0).cpu().numpy())
    return np.concatenate(out)


def weighted_group_probs(df_subset, groups, args, split, device, tta, zero_tabular=False):
    """Probability array (aligned to df_subset row order), averaged across
    preprocessing groups with weight = number of models in each group — i.e.
    equivalent to a flat per-model average across every model regardless of
    which preprocessing group it needed."""
    total = np.zeros(len(df_subset))
    total_models = 0
    for (size, mean, std), group_models in groups.items():
        cfg = CFG(args, split, img_size=size, mean=mean, std=std)
        dl = DataLoader(OrdFusedDataset(df_subset, cfg, False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
        probs = ensemble_probs(group_models, dl, device, tta=tta, zero_tabular=zero_tabular)
        n = len(group_models)
        total += probs * n
        total_models += n
    return total / max(total_models, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="tf_efficientnetv2_s")
    ap.add_argument("--no-tabular", action="store_true")
    ap.add_argument("--ensemble", nargs="*", default=None,
                    help="run-dir names under --out-dir to ensemble together "
                         "(overrides --encoder). E.g. ordfused_densenet201 ordfused_tf_efficientnetv2_s")
    ap.add_argument("--name", default=None, help="submission name (default derived from run dir)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224,
                    help="fallback resolution for plain-timm checkpoints missing a "
                         "saved img_size; foundation-model checkpoints always use "
                         "their own registry resolution regardless of this flag")
    ap.add_argument("--tta", action="store_true", default=True,
                    help="average predictions over original + horizontally-flipped "
                         "image at inference (on by default)")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--zero-tabular-at-test", action="store_true",
                    help="diagnostic: zero the tabular vector for every forward pass "
                         "on a checkpoint trained with use_tabular=True, to estimate "
                         "the score if real challenge test images arrive without "
                         "metadata. This is a train/inference mismatch, NOT the same "
                         "as a proper --no-tabular ablation checkpoint -- treat the "
                         "result as a lower bound on how bad the gap could be.")
    ap.add_argument("--tau-objective", default="accuracy", choices=["accuracy", "challenge"],
                    help="tau selection target: 'accuracy' (default) maximizes plain "
                         "presence-detection accuracy -- the right choice now that real "
                         "Dice comes from nnUNet separately (gate_segmentation.py); "
                         "'challenge' uses the old 0.7*acc+0.3*presence_dice proxy")
    ap.add_argument("--lora-r", type=int, default=16,
                    help="must match the --lora-r the checkpoint being loaded was trained "
                         "with (see train_ordfused.py) -- PEFT bakes rank into adapter shapes")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-oof", default=None,
                    help="optional path to write (our_id, oof_prob) for every training-set "
                         "row to CSV, feeding ensemble_submissions.py. Only meaningful "
                         "combined with --no-tabular (or an --ensemble of --no-tabular run "
                         "dirs) -- a checkpoint trained with tabular fusion needs metadata "
                         "at inference, which the real challenge test set will not provide, "
                         "so its OOF/test probs are not valid ensemble inputs.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--submissions-dir", default=os.path.join(HERE, "..", "submissions"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.submissions_dir, exist_ok=True)

    if args.ensemble:
        run_dirs = args.ensemble
        name = args.name or "ordfused_ensemble"
    else:
        tag = args.encoder if not args.no_tabular else f"{args.encoder}_notab"
        run_dirs = [f"ordfused_{tag}"]
        name = args.name or f"ordfused_{tag}"
        if args.zero_tabular_at_test:
            name += "_zerotabtest"

    # (fold, model) pairs across every requested run dir
    all_models = []
    for rd in run_dirs:
        all_models += load_fold_models(os.path.join(args.out_dir, rd), device,
                                        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                        lora_dropout=args.lora_dropout)

    # --- tau on training OOF: each fold's checkpoint predicts only its own
    # held-out slice, so the threshold never sees the test set. When ensembling
    # multiple run dirs, average across all models sharing that held-out fold
    # (grouped by preprocessing, since foundation-model checkpoints need a
    # different resolution/normalization than the CNN checkpoints). ---
    # make_folds only needs img_size/n_folds off a CFG-shaped object — any
    # preprocessing values work here since only the "fold" column is used.
    fold_cfg = CFG(args, "train")
    df_tr = pd.read_csv(fold_cfg.csv_path)
    df_tr = make_folds(df_tr, fold_cfg, group_col=None, seed=args.seed)
    oof = np.zeros(len(df_tr))
    for f in range(args.n_folds):
        fold_models = [(fo, enc, sz, m) for (fo, enc, sz, m) in all_models if fo == f]
        va = df_tr[df_tr.fold == f]
        if len(va) == 0 or not fold_models:
            continue
        groups = group_by_preprocess(fold_models)
        oof[va.index.to_numpy()] = weighted_group_probs(
            va, groups, args, "train", device, args.tta,
            zero_tabular=args.zero_tabular_at_test)
    oof_y = (df_tr.cavity != "none").astype(int).to_numpy()
    sweep_fn = sweep_tau_accuracy if args.tau_objective == "accuracy" else sweep_tau
    tau, oof_score = sweep_fn(oof_y, oof)
    print(f"[{name}] OOF tau*={tau:.3f} {args.tau_objective}={oof_score:.4f}")

    if args.dump_oof:
        pd.DataFrame({ID_COL: df_tr[ID_COL].to_numpy(), "oof_prob": oof}).to_csv(
            args.dump_oof, index=False)
        print(f"wrote OOF probs to {args.dump_oof}")

    # --- test: full ensemble (all folds/backbones) on every test image ---
    df_te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    test_groups = group_by_preprocess(all_models)
    test_prob = weighted_group_probs(df_te, test_groups, args, "test", device, args.tta,
                                      zero_tabular=args.zero_tabular_at_test)

    out = df_te[[ID_COL]].copy()
    out["prob_cavity"] = test_prob
    out["pred_cavity"] = (test_prob >= tau).astype(int)
    if "cavity" in df_te.columns:
        y_true = (df_te.cavity != "none").astype(int).to_numpy()
        s, acc, dice = challenge_score(y_true, test_prob, tau)
        print(f"[{name}] TEST score={s:.4f} acc={acc:.4f} dice={dice:.4f} (tau from OOF)")
        out["cavity_true"] = df_te["cavity"].to_numpy()

    out_path = os.path.join(args.submissions_dir, f"submission_{name}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
