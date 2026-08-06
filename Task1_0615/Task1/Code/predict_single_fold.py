#!/usr/bin/env python
"""Run inference on test.csv using a single fold checkpoint (not the full
5-fold ensemble) -- for quickly checking one fold's number while the rest of
a run is still training. Uses that fold's own saved tau (tuned on its own
held-out validation slice during training), not a fresh OOF sweep across all
folds -- the standard predict_ordfused.py's tau-sweep needs every fold's
checkpoint to be meaningful, which doesn't apply here.

Usage:
    python predict_single_fold.py --ckpt runs/ordfused_eva_x_base/fold0_best.pt \
        --name eva_x_base_fold0
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ID_COL, challenge_score, set_seed
from ordfused import ENCODER_FAMILIES, OrdFusedDataset, OrdFusedModel, corn_cavity_prob

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))
_DEFAULT_MEAN, _DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class CFG:
    def __init__(self, args, img_size, mean, std):
        self.csv_path = os.path.join(args.data_dir, "test.csv")
        self.image_dir = os.path.join(args.data_dir, "test", "CXR")
        self.image_ext = ".dcm"
        self.img_size = img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        self.norm_mean = mean
        self.norm_std = std
        self.batch_size = args.batch_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="path to a single fold*_best.pt")
    ap.add_argument("--name", required=True, help="submission name")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--submissions-dir", default=os.path.join(HERE, "..", "submissions"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.submissions_dir, exist_ok=True)

    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    tau = sd["best"]["tau"]
    print(f"loaded {args.ckpt} (fold {sd['fold']}, best {sd['best']})")
    print(f"using this fold's own saved tau={tau:.3f} (from its held-out validation slice during training)")

    fam = ENCODER_FAMILIES.get(sd["encoder"])
    img_size = fam["img_size"] if fam else sd.get("img_size", 224)
    mean = fam["mean"] if fam else _DEFAULT_MEAN
    std = fam["std"] if fam else _DEFAULT_STD

    model = OrdFusedModel(sd["encoder"], use_tabular=sd["use_tabular"], pretrained=False).to(device)
    model.load_state_dict(sd["model"])
    model.eval()

    cfg = CFG(args, img_size, mean, std)
    df_te = pd.read_csv(cfg.csv_path)
    dl_te = DataLoader(OrdFusedDataset(df_te, cfg, False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    probs = []
    with torch.no_grad():
        for x, tab, _yb, _yo in dl_te:
            x, tab = x.to(device), tab.to(device)
            probs.append(corn_cavity_prob(model(x, tab)).cpu().numpy())
    import numpy as np
    test_prob = np.concatenate(probs)

    out = df_te[[ID_COL]].copy()
    out["prob_cavity"] = test_prob
    out["pred_cavity"] = (test_prob >= tau).astype(int)
    if "cavity" in df_te.columns:
        y_true = (df_te.cavity != "none").astype(int).to_numpy()
        s, acc, dice = challenge_score(y_true, test_prob, tau)
        print(f"[{args.name}] TEST score={s:.4f} acc={acc:.4f} dice={dice:.4f} (tau from this fold's own training)")
        out["cavity_true"] = df_te["cavity"].to_numpy()

    out_path = os.path.join(args.submissions_dir, f"submission_{args.name}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
