#!/usr/bin/env python
"""5-fold ensemble inference for HybridFused-CXR on the held-out test set.

Mirrors predict_ordfused.py: averages P(cavity)=sigma(CORN logit_0) across all
fold checkpoints (+ horizontal-flip TTA), tunes tau on training OOF only,
writes a submission CSV with the same prob_cavity/pred_cavity schema so it
plugs straight into ensemble_submissions.py alongside the CNN and OrdFused
single-stream members.

Usage:
    python predict_hybrid_fused.py --cnn-encoder densenet201 --vit-encoder eva_x_base --no-tabular
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
from hybrid_fused import HybridFusedDataset, HybridFusedModel
from ordfused import ENCODER_FAMILIES, corn_cavity_prob
from train_hybrid_fused import BranchCFG, _CNN_MEAN, _CNN_STD

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))


def load_fold_models(model_dir, device):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "fold*_best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no fold*_best.pt under {model_dir}")
    models = []
    for p in ckpts:
        sd = torch.load(p, map_location=device, weights_only=False)
        m = HybridFusedModel(sd["cnn_encoder"], sd["vit_encoder"],
                              use_tabular=sd["use_tabular"], pretrained=False).to(device)
        m.load_state_dict(sd["model"])
        m.eval()
        models.append((sd["fold"], sd["cnn_encoder"], sd["vit_encoder"],
                        sd.get("cnn_img_size", 224), m))
        print(f"loaded {p} (fold {sd['fold']}, best {sd['best']})")
    return models


@torch.no_grad()
def ensemble_probs(models, dl, device, tta=True, zero_tabular=False):
    out = []
    for x_cnn, x_vit, tab, _yb, _yo in dl:
        x_cnn, x_vit, tab = x_cnn.to(device), x_vit.to(device), tab.to(device)
        if zero_tabular:
            tab = torch.zeros_like(tab)
        variants = [(x_cnn, x_vit), (x_cnn.flip(-1), x_vit.flip(-1))] if tta else [(x_cnn, x_vit)]
        ps = [corn_cavity_prob(m(xc, xv, tab)) for xc, xv in variants for _f, m in models]
        out.append(torch.stack(ps, 0).mean(0).cpu().numpy())
    return np.concatenate(out)


def build_dl(df_subset, cnn_encoder, vit_encoder, cnn_img_size, args, split):
    image_dir = os.path.join(args.data_dir, split, "CXR")
    cnn_cfg = BranchCFG(image_dir, ".dcm", cnn_img_size, _CNN_MEAN, _CNN_STD)
    vit_family = ENCODER_FAMILIES[vit_encoder]
    vit_cfg = BranchCFG(image_dir, ".dcm", vit_family["img_size"], vit_family["mean"], vit_family["std"])
    ds = HybridFusedDataset(df_subset, cnn_cfg, vit_cfg, train=False)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-encoder", default="densenet201")
    ap.add_argument("--vit-encoder", default="eva_x_base")
    ap.add_argument("--no-tabular", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--zero-tabular-at-test", action="store_true")
    ap.add_argument("--tau-objective", default="accuracy", choices=["accuracy", "challenge"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-oof", default=None,
                    help="write (our_id, oof_prob) for the train set to CSV -- only valid "
                         "as an ensemble input when combined with --no-tabular, same caveat "
                         "as predict_ordfused.py")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--submissions-dir", default=os.path.join(HERE, "..", "submissions"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.submissions_dir, exist_ok=True)

    tag = f"{args.cnn_encoder}_{args.vit_encoder}"
    if args.no_tabular:
        tag += "_notab"
    name = args.name or f"hybrid_{tag}"
    if args.zero_tabular_at_test:
        name += "_zerotabtest"

    models = load_fold_models(os.path.join(args.out_dir, f"hybrid_{tag}"), device)
    cnn_img_size = models[0][3]

    fold_csv = os.path.join(args.data_dir, "train.csv")
    df_tr = pd.read_csv(fold_csv)

    class _FoldCFG:
        img_size = cnn_img_size
        n_folds = args.n_folds

    df_tr = make_folds(df_tr, _FoldCFG(), group_col=None, seed=args.seed)
    oof = np.zeros(len(df_tr))
    for f in range(args.n_folds):
        fold_models = [(fo, ce, ve, sz, m) for (fo, ce, ve, sz, m) in models if fo == f]
        va = df_tr[df_tr.fold == f]
        if len(va) == 0 or not fold_models:
            continue
        dl = build_dl(va, args.cnn_encoder, args.vit_encoder, cnn_img_size, args, "train")
        oof[va.index.to_numpy()] = ensemble_probs(
            [(fo, m) for fo, _ce, _ve, _sz, m in fold_models], dl, device, tta=args.tta,
            zero_tabular=args.zero_tabular_at_test)
    oof_y = (df_tr.cavity != "none").astype(int).to_numpy()
    sweep_fn = sweep_tau_accuracy if args.tau_objective == "accuracy" else sweep_tau
    tau, oof_score = sweep_fn(oof_y, oof)
    print(f"[{name}] OOF tau*={tau:.3f} {args.tau_objective}={oof_score:.4f}")

    if args.dump_oof:
        pd.DataFrame({ID_COL: df_tr[ID_COL].to_numpy(), "oof_prob": oof}).to_csv(
            args.dump_oof, index=False)
        print(f"wrote OOF probs to {args.dump_oof}")

    df_te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    dl_te = build_dl(df_te, args.cnn_encoder, args.vit_encoder, cnn_img_size, args, "test")
    test_prob = ensemble_probs([(fo, m) for fo, _ce, _ve, _sz, m in models], dl_te, device,
                                tta=args.tta, zero_tabular=args.zero_tabular_at_test)

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
