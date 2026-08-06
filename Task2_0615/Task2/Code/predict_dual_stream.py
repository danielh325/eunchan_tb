"""Inference + ensemble for GlobalLocalTBClassifier checkpoints (see
train_dual_stream.py). Kept as a separate, self-contained script rather than
folded into predict_task2.py's ensemble loop: dual-stream checkpoints need
two images per sample (potentially two different img_sizes/backbones), which
doesn't fit predict_task2.py's single-image-per-checkpoint grouping without
risking a subtle cross-wiring bug -- same reasoning Task1 kept OrdFused and
the plain-CNN baselines on separate, individually-verified predict paths.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ID_COL, LABEL_COL, LABEL_MAP, detection_accuracy, safe_auc, safe_f1, sweep_tau_accuracy
from dataset import GlobalLocalCfg, GlobalLocalTBDataset
from model import GlobalLocalTBClassifier


def load_checkpoints(ckpt_dir, pattern="dual_*.pth"):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    return [torch.load(p, map_location="cpu", weights_only=True) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--global-image-dir", required=True, help="dir of RAW uncropped images")
    ap.add_argument("--local-image-dir", required=True, help="dir of preprocess.py's ch0 (lung-cropped) images")
    ap.add_argument("--bone-suppress", action="store_true",
                     help="MUST match whatever --bone-suppress (or its absence) was used at "
                          "training time (train_dual_stream.py) -- same train/serve-parity "
                          "requirement as predict_task2.py's flag of the same name")
    ap.add_argument("--out", default="submission_dual_stream.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--score", action="store_true",
                     help="print accuracy/AUC/F1 against real TB/Normal labels in --csv, if present")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = load_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"no dual_*.pth checkpoints found under {args.ckpt_dir}")

    df = pd.read_csv(args.csv)
    has_real_labels = LABEL_COL in df.columns
    ids = df[ID_COL].tolist()
    pred_df = df if has_real_labels else df.assign(**{"TB/Normal": "Normal"})
    if "Modality_DICOM" not in pred_df.columns:
        pred_df = pred_df.assign(Modality_DICOM="unknown")

    prob_sum = np.zeros(len(df), dtype=np.float64)
    n_models = 0

    # every checkpoint pairs a (global_encoder, local_encoder) combo -- group
    # by that combo so checkpoints sharing the same pair (e.g. multiple
    # leave-one-modality-out folds) reuse one dataset/dataloader pass.
    by_pair = {}
    for c in ckpts:
        by_pair.setdefault((c["global_encoder"], c["local_encoder"]), []).append(c)

    for (global_enc, local_enc), group in by_pair.items():
        cfg = GlobalLocalCfg(global_enc, local_enc,
                              global_image_dir=args.global_image_dir, local_image_dir=args.local_image_dir,
                              use_bone_suppression=args.bone_suppress)
        ds = GlobalLocalTBDataset(pred_df, cfg, train=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

        for ckpt in group:
            model = GlobalLocalTBClassifier(global_enc, local_enc, pretrained=False).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            probs = []
            with torch.no_grad():
                for xg, xl, _y, _dom in loader:
                    xg = xg.to(device, non_blocking=True)
                    xl = xl.to(device, non_blocking=True)
                    probs.append(torch.sigmoid(model(xg, xl)).cpu().numpy())
            probs = np.concatenate(probs)
            prob_sum += probs
            n_models += 1
            print(f"scored dual({global_enc}+{local_enc}) / {ckpt['fold_tag']} "
                  f"(tau={ckpt['tau']:.2f}, val_acc={ckpt['val_acc']:.4f})")

    prob = prob_sum / n_models
    pred = (prob >= 0.5).astype(int)
    out = pd.DataFrame({ID_COL: ids, "prob_tb": prob, "pred_TB/Normal": np.where(pred == 1, "TB", "Normal")})
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({n_models} models ensembled)")

    if args.score:
        if not has_real_labels:
            print(f"--score requested but {args.csv} has no '{LABEL_COL}' column -- skipping")
        else:
            y_true = df[LABEL_COL].astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy()
            acc_half = detection_accuracy(y_true, pred)
            auc = safe_auc(y_true, prob)
            f1_half = safe_f1(y_true, pred)
            tau_star, acc_tau = sweep_tau_accuracy(y_true, prob)
            print(f"\n=== SCORE against real labels in {args.csv} ({len(y_true)} samples) ===")
            print(f"accuracy@0.5={acc_half:.4f}  AUC={auc:.4f}  F1@0.5={f1_half:.4f}")
            print(f"best-tau={tau_star:.2f}  accuracy@best-tau={acc_tau:.4f}  "
                  f"(informational only, tau chosen on this same set)")


if __name__ == "__main__":
    main()
