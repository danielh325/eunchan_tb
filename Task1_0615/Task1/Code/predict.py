#!/usr/bin/env python
"""5-fold ensemble inference for one trained model on the held-out test set.

Averages sigmoid probabilities across all fold checkpoints in
--out-dir/<model>/fold*_best.pt, re-sweeps tau on the *training* OOF
predictions (never on test), then scores + writes predictions for test.csv.

Usage:
    python predict.py --model resnet50d
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
from common import (
    CavityDataset,
    CNNDetector,
    ID_COL,
    challenge_score,
    make_folds,
    set_seed,
    sweep_tau,
    sweep_tau_accuracy,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))


class CFG:
    def __init__(self, args, split):
        self.csv_path = os.path.join(args.data_dir, f"{split}.csv")
        self.image_dir = os.path.join(args.data_dir, split, "CXR")
        self.mask_dir = os.path.join(args.data_dir, split, "CXR_label")
        self.image_ext = ".dcm"
        self.mask_ext = ".nii.gz"
        self.img_size = args.img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)
        self.n_folds = args.n_folds
        self.batch_size = args.batch_size


def load_fold_models(model_dir, model_name, device):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "fold*_best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no fold*_best.pt checkpoints under {model_dir}")
    models = []
    for p in ckpts:
        sd = torch.load(p, map_location=device, weights_only=False)
        m = CNNDetector(model_name, pretrained=False).to(device)
        m.load_state_dict(sd["model"])
        m.eval()
        models.append(m)
        print(f"loaded {p} (fold {sd['fold']}, best {sd['best']})")
    return models


@torch.no_grad()
def ensemble_probs(models, dl, device, tta=True):
    """Mean sigmoid probability across the given fold models, and -- when
    tta=True -- across {original, horizontally-flipped} for each model (the
    one augmentation already used in training, so it's a safe, cheap addition;
    mirrors predict_ordfused.py's ensemble_probs)."""
    all_probs = []
    for x, _yb, _yo in dl:
        x = x.to(device)
        variants = [x, x.flip(-1)] if tta else [x]
        ps = [torch.sigmoid(m(xv)) for xv in variants for m in models]
        p = torch.stack(ps, dim=0).mean(0)
        all_probs.append(p.cpu().numpy())
    return np.concatenate(all_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true", default=True,
                    help="average predictions over original + horizontally-flipped "
                         "image at inference (on by default)")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--submissions-dir", default=os.path.join(HERE, "..", "submissions"))
    ap.add_argument("--out-tag", default=None,
                    help="override the submission filename's model tag (defaults to --model) "
                         "-- use this when running a preprocessing variant (e.g. lung-crop) of "
                         "an existing --model so its submission doesn't overwrite the original")
    ap.add_argument("--tau-objective", default="accuracy", choices=["accuracy", "challenge"],
                    help="tau selection target: 'accuracy' (default) maximizes plain "
                         "presence-detection accuracy -- the right choice now that real "
                         "Dice comes from nnUNet separately (gate_segmentation.py); "
                         "'challenge' uses the old 0.7*acc+0.3*presence_dice proxy")
    ap.add_argument("--dump-oof", default=None,
                    help="optional path to write (our_id, oof_prob) for every training-set "
                         "row to CSV, alongside the normal submission output -- feeds "
                         "ensemble_submissions.py, which combines multiple models' OOF+test "
                         "probabilities into one weighted ensemble with a freshly-swept tau.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(args.submissions_dir, exist_ok=True)

    models = load_fold_models(model_dir, args.model, device)

    # --- Re-derive tau on training OOF (out-of-fold) predictions: each fold's
    # own checkpoint predicts on its own held-out validation slice, so this is
    # a fair, test-set-independent threshold ---
    train_cfg = CFG(args, "train")
    df_tr = pd.read_csv(train_cfg.csv_path)
    df_tr = make_folds(df_tr, train_cfg, group_col=None, seed=args.seed)
    oof_prob = np.zeros(len(df_tr))
    for f, m in enumerate(models):
        va = df_tr[df_tr.fold == f]
        if len(va) == 0:
            continue
        dl_va = DataLoader(CavityDataset(va, train_cfg, False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        p = ensemble_probs([m], dl_va, device, tta=args.tta)
        oof_prob[va.index.to_numpy()] = p
    oof_y = (df_tr.cavity != "none").astype(int).to_numpy()
    sweep_fn = sweep_tau_accuracy if args.tau_objective == "accuracy" else sweep_tau
    tau, oof_score = sweep_fn(oof_y, oof_prob)
    print(f"[{args.model}] OOF tau*={tau:.3f} {args.tau_objective}={oof_score:.4f}")

    if args.dump_oof:
        pd.DataFrame({ID_COL: df_tr[ID_COL].to_numpy(), "oof_prob": oof_prob}).to_csv(
            args.dump_oof, index=False)
        print(f"wrote OOF probs to {args.dump_oof}")

    # --- Ensemble inference on the held-out test split ---
    test_cfg = CFG(args, "test")
    df_te = pd.read_csv(test_cfg.csv_path)
    dl_te = DataLoader(CavityDataset(df_te, test_cfg, False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    test_prob = ensemble_probs(models, dl_te, device, tta=args.tta)

    out = df_te[[ID_COL]].copy()
    out["prob_cavity"] = test_prob
    out["pred_cavity"] = (test_prob >= tau).astype(int)

    if "cavity" in df_te.columns:
        y_true = (df_te.cavity != "none").astype(int).to_numpy()
        s, acc, dice = challenge_score(y_true, test_prob, tau)
        print(f"[{args.model}] TEST score={s:.4f} acc={acc:.4f} dice={dice:.4f} (tau from OOF)")
        out["cavity_true"] = df_te["cavity"].to_numpy()

    out_path = os.path.join(args.submissions_dir, f"submission_{args.out_tag or args.model}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
