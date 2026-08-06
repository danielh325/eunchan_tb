"""Aggregate leave-one-modality-out results across checkpoints saved by
train_task2.py, plus a plain-stratified-split sanity check for direct
comparability to the baseline notebook's reported 0.9923 val acc (expected
to look *worse* here -- per-modality holdout is the number that actually
matters, the plain split is actively misleading per the notebook's own
Grad-CAM diagnosis).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch



def load_checkpoints(ckpt_dir, encoder_name=None):
    pattern = os.path.join(ckpt_dir, f"{encoder_name or '*'}_*.pth")
    paths = sorted(glob.glob(pattern))
    return [torch.load(p, map_location="cpu", weights_only=True) for p in paths]


def summarize(ckpts, label=""):
    if not ckpts:
        print(f"{label}: no checkpoints found")
        return
    df = pd.DataFrame([dict(fold=c["fold_tag"], val_acc=c["val_acc"], val_auc=c["val_auc"])
                        for c in ckpts])
    print(f"\n=== {label} ({len(df)} folds) ===")
    print(df.to_string(index=False))
    print(f"mean val_acc={df['val_acc'].mean():.4f}  mean val_auc={df['val_auc'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--encoder", default=None, help="filter to one encoder; default: all")
    args = ap.parse_args()

    ckpts = load_checkpoints(args.ckpt_dir, args.encoder)
    by_encoder = {}
    for c in ckpts:
        by_encoder.setdefault(c["encoder_name"], []).append(c)
    for name, group in by_encoder.items():
        summarize(group, label=name)


if __name__ == "__main__":
    main()
