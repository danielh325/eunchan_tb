#!/usr/bin/env python
"""Score the ACTUAL submission pipeline (predict_task1_submission.py's output)
against real ground truth on the 111-case held-out test set. Unlike the
various ad-hoc diagnostic sweeps used during development, this exercises the
literal code path that ships in the Docker image -- run this against a
submission.csv + masks/ dir produced by either:
  (a) running the built container directly:
      docker run --rm --device nvidia.com/gpu=all \\
        -v Task1_0615/Task1/Data/test/CXR:/input:ro -v /tmp/out:/output \\
        lisa-task1-submission:v1
  (b) or predict_task1_submission.py directly (faster iteration, same code):
      python predict_task1_submission.py --image-dir ../Data/test/CXR \\
        --out-dir /tmp/out

Usage:
    python evaluate_task1.py --pred-dir /tmp/out
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))
_DEFAULT_NNUNET_LABELS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data",
    "nnUNet_raw", "Dataset001_Task1", "labelsTr"))


def dice_score_np(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return np.nan
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * np.logical_and(pred, gt).sum() / (ps + gs)


def load_gt_mask(cid, labels_dir, size=512):
    """transpose=True: labelsTr stores rows/columns swapped relative to the
    raw image -- see train_segmentation.py's load_mask() docstring for the
    empirical confirmation (5x test Dice recovery). Never skip this for
    labelsTr-sourced masks."""
    path = os.path.join(labels_dir, f"{cid}.nii.gz")
    if not os.path.exists(path):
        return np.zeros((size, size), dtype=np.uint8)
    m = sitk.GetArrayFromImage(sitk.ReadImage(path))
    m = np.squeeze(m).T
    m = (m > 0).astype(np.uint8)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="dir containing submission.csv + masks/ "
                    "from predict_task1_submission.py / the built container")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--nnunet-labels-dir", default=_DEFAULT_NNUNET_LABELS)
    ap.add_argument("--mask-eval-size", type=int, default=512)
    args = ap.parse_args()

    sub = pd.read_csv(os.path.join(args.pred_dir, "submission.csv"))
    sub["our_id"] = sub["our_id"].astype(str)

    df_te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    df_te["our_id"] = df_te["our_id"].astype(str)
    y_true = dict(zip(df_te["our_id"], (df_te["cavity"] != "none").astype(int)))

    merged = sub[sub["our_id"].isin(y_true)].copy()
    if len(merged) != len(df_te):
        print(f"!! WARNING: predicted {len(merged)}/{len(df_te)} test cases -- "
              f"missing cases are silently excluded from these numbers, check --pred-dir")
    merged["y_true"] = merged["our_id"].map(y_true)

    acc = accuracy_score(merged["y_true"], merged["pred_cavity"])
    auc = roc_auc_score(merged["y_true"], merged["prob_cavity"])
    f1 = f1_score(merged["y_true"], merged["pred_cavity"])
    print(f"[classification] n={len(merged)} accuracy={acc:.4f} AUC={auc:.4f} F1={f1:.4f}")

    mask_dir = os.path.join(args.pred_dir, "masks")
    dices = []
    for _id in df_te["our_id"]:
        gt = load_gt_mask(_id, args.nnunet_labels_dir, args.mask_eval_size)
        mp = os.path.join(mask_dir, f"{_id}.png")
        if os.path.exists(mp):
            pred_mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            if pred_mask.shape != gt.shape:
                pred_mask = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask = np.zeros_like(gt)  # no mask file -- classifier called this case negative
        dices.append(dice_score_np(pred_mask, gt))
    dice = float(np.nanmean(dices))
    print(f"[segmentation] n={len(dices)} Dice={dice:.4f}")

    composite = 0.7 * acc + 0.3 * dice
    print(f"[composite] 0.7*accuracy + 0.3*Dice = {composite:.4f}")


if __name__ == "__main__":
    main()