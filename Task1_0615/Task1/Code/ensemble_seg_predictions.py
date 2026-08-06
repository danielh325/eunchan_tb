#!/usr/bin/env python
"""Ensemble RAD-DINO's, CheXFound's, and EVA-X's segmentation decoders
(train_segmentation.py) by averaging their sigmoid probability maps -- same
late-fusion principle used throughout this project (predict_ordfused.py,
ensemble_submissions.py).

Why this might beat any one alone: the three backbones have different
pretraining (RAD-DINO's DINOv2 self-supervised corpus, CheXFound's ~987K CXR
corpus, EVA-X's own MIM pretraining) and different resolutions/architectures
(518/512/224, DINOv2 vs. EVA), so they plausibly make different errors --
the same bet that worked for the classification ensemble.

Resolution handling: each model's probability map is at its OWN native size
(from ENCODER_FAMILIES img_size -- 518/512/224). All are resized to a common
--eval-size before averaging. Ground truth is reloaded fresh from source
masks at --eval-size for scoring, rather than reusing any model's own
resized copy (avoids subtly comparing against different resample lineages
of the same mask).

No-leakage discipline: the ensemble threshold is swept on the DEV split's
saved probabilities (dev_probs.pt, written by train_segmentation.py) and only
then applied to test (test_probs.pt) -- never swept on test itself, same
rule as every other tau/threshold decision in this project.

Usage:
    python train_segmentation.py --encoder rad_dino          # writes runs_seg/rad_dino/{test,dev}_probs.pt
    python train_segmentation.py --encoder chexfound_vitl16  # writes runs_seg/chexfound_vitl16/{test,dev}_probs.pt
    python train_segmentation.py --encoder eva_x_base        # writes runs_seg/eva_x_base/{test,dev}_probs.pt
    python ensemble_seg_predictions.py   # defaults to all three
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ID_COL, make_folds
from train_segmentation import DEFAULT_DATA_DIR, _DEFAULT_NNUNET_LABELS, CFG, dice_score_np

HERE = os.path.dirname(os.path.abspath(__file__))


def _parse_weights(spec, tags):
    if not spec:
        return {t: 1.0 for t in tags}
    weights = {}
    for item in spec.split(","):
        tag, w = item.split("=")
        weights[tag] = float(w)
    missing = set(tags) - set(weights)
    if missing:
        raise ValueError(f"--weights is missing entries for: {sorted(missing)}")
    return weights


def _load_mask(path, size, transpose=False):
    """transpose=True for masks sourced from nnU-Net's labelsTr (TEST ground
    truth) -- see the matching note in train_segmentation.py's load_mask();
    confirmed empirically that plain transpose recovers TEST dice from
    collapsed (~0.04) to a real number, and only for that mask source."""
    m = sitk.GetArrayFromImage(sitk.ReadImage(path))
    m = np.squeeze(m)
    if transpose:
        m = m.T
    m = (m > 0).astype(np.uint8)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def combine_probs(per_run_probs, weights, eval_size):
    """per_run_probs: {tag: {id: (H,W) np array at that run's own resolution}}
    -> {id: (eval_size,eval_size) np array}, weighted mean after resizing
    every run's map to a common resolution."""
    ids = sorted(set.intersection(*[set(p) for p in per_run_probs.values()]))
    total_w = sum(weights.values())
    combined = {}
    for _id in ids:
        acc = None
        for tag, w in weights.items():
            arr = per_run_probs[tag][_id]
            arr = cv2.resize(arr, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR)
            acc = arr * w if acc is None else acc + arr * w
        combined[_id] = acc / total_w
    return combined


def score(combined, gt, threshold):
    dices = [dice_score_np((combined[i] >= threshold).astype(np.uint8), gt[i])
             for i in combined]
    return float(np.nanmean(dices))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["rad_dino", "chexfound_vitl16", "eva_x_base"])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs_seg"))
    ap.add_argument("--eval-size", type=int, default=512,
                    help="common resolution both models' probability maps are resized "
                         "to before averaging/scoring (independent of either model's "
                         "own training resolution)")
    ap.add_argument("--weights", default=None, help="tag=weight,tag=weight,... (default: uniform)")
    ap.add_argument("--threshold-grid", type=int, default=17)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--nnunet-labels-dir", default=_DEFAULT_NNUNET_LABELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    weights = _parse_weights(args.weights, args.runs)

    # --- DEV: sweep the ensemble threshold here, never on test ---
    dev_probs_by_run = {}
    for tag in args.runs:
        p = os.path.join(args.out_dir, tag, "dev_probs.pt")
        dev_probs_by_run[tag] = torch.load(p, map_location="cpu", weights_only=False)["probs"]
    combined_dev = combine_probs(dev_probs_by_run, weights, args.eval_size)

    cfg = CFG(argparse.Namespace(encoder=args.runs[0], n_folds=args.n_folds))  # img_size unused below
    df_tr_all = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_tr_all = make_folds(df_tr_all, cfg, group_col=None, seed=args.seed)
    dev_df = df_tr_all[df_tr_all.fold == 0]
    dev_mask_dir = os.path.join(args.data_dir, "train", "CXR_label")
    gt_dev = {}
    for _id in combined_dev:
        row_ids = dev_df[dev_df[ID_COL].astype(str) == str(_id)]
        assert len(row_ids) == 1, f"dev id {_id} not found (or duplicated) in fold-0 dev split"
        mp = os.path.join(dev_mask_dir, f"{_id}.nii.gz")
        gt_dev[_id] = _load_mask(mp, args.eval_size) if os.path.exists(mp) \
            else np.zeros((args.eval_size, args.eval_size), dtype=np.uint8)

    grid = np.linspace(0.1, 0.9, args.threshold_grid)
    best_t, best_dev_dice = 0.5, -1.0
    for t in grid:
        d = score(combined_dev, gt_dev, t)
        if d > best_dev_dice:
            best_dev_dice, best_t = d, float(t)
    print(f"[ensemble_seg] members={args.runs} weights={weights} eval_size={args.eval_size}")
    print(f"[ensemble_seg] dev threshold sweep: t*={best_t:.2f} dev_dice={best_dev_dice:.4f} "
          f"(n={len(combined_dev)})")

    # --- TEST: apply the dev-swept threshold, report once ---
    test_probs_by_run = {}
    for tag in args.runs:
        p = os.path.join(args.out_dir, tag, "test_probs.pt")
        sd = torch.load(p, map_location="cpu", weights_only=False)
        test_probs_by_run[tag] = sd["probs"]
        print(f"[ensemble_seg]   {tag} standalone TEST dice={sd['test_dice']:.4f} "
              f"(own threshold={sd['threshold']:.2f}, own resolution)")
    combined_test = combine_probs(test_probs_by_run, weights, args.eval_size)

    gt_test = {}
    for _id in combined_test:
        mp = os.path.join(args.nnunet_labels_dir, f"{_id}.nii.gz")
        gt_test[_id] = _load_mask(mp, args.eval_size, transpose=True)

    test_dice = score(combined_test, gt_test, best_t)
    print(f"[ensemble_seg] TEST dice={test_dice:.4f} (n={len(combined_test)}, "
          f"threshold={best_t:.2f} from dev, real held-out, same 111 cases as nnU-Net)")


if __name__ == "__main__":
    main()
