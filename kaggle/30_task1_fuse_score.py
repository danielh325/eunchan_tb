#!/usr/bin/env python
"""STEP 3 -- CPU only (run on Kaggle or on the Mac). Metric-aware fusion.

WHY THIS IS A SEPARATE STEP, AND WHY IT MATTERS MORE THAN IT LOOKS
------------------------------------------------------------------
Re-reading Code/Task1_evaluation_code.ipynb line by line, the organizers'
metric has a property the current pipeline does not optimise for:

    dice_score(pred, gt):
        if pred.sum()==0 and gt.sum()==0:  return np.nan   # EXCLUDED from mean
        if exactly one is empty:           return 0.0      # COUNTED in mean
    mean_dice = np.nanmean(...)

So the Dice mean is taken over {cases where pred OR gt is non-empty}, not over
all cases and not over positives. Three consequences:

  1. A false-positive mask on a cavity-negative case is punished TWICE: it
     costs 0.7/N of accuracy and it adds a 0.0 to the Dice numerator while
     growing the denominator by one. Suppressing one FP is worth roughly
     0.7/N + 0.3*meandice/K, not just the accuracy term.
  2. A true negative is free -- it is excluded from Dice entirely. Emitting an
     empty mask on a confidently-negative case is strictly better than emitting
     a small speculative one.
  3. Therefore "tune the segmentation threshold for best Dice" and "tune the
     classifier threshold for best accuracy" are NOT separable. They share the
     same emptiness decision. This script optimises them jointly, on OOF only.

The paper's finding that classifier-gating "slightly reduced Dice"
(0.229 -> 0.222) was measured against a Dice definition that did not model the
NaN-exclusion, so it understates gating's value. Re-measure it here.

Everything is scored at NATIVE resolution (predictions upsampled back through
the lung-crop box and the 1024 pack frame), which is where the organizers score.

Usage:
    python 30_task1_fuse_score.py \
        --seg segprob_resnet34.npz segprob_timm-efficientnet-b0.npz segprob_se_resnext50_32x4d.npz \
        --cls clsprob_convnext_small.npz clsprob_densenet201.npz \
        --write-masks out_masks/
"""
import argparse
import itertools
import json
import os

import cv2
import numpy as np
import pandas as pd

PACK = os.environ.get("MMTB_PACK", "/kaggle/input/mmtb-2026-pack")
WORK = os.environ.get("MMTB_WORK", "/kaggle/working")
SIZE = 256


def official_score(preds, gts):
    """Exactly Code/Task1_evaluation_code.ipynb, on already-binarised arrays."""
    acc, dice = [], []
    for p, g in zip(preds, gts):
        ps, gs = int(p.sum()), int(g.sum())
        acc.append(float((ps > 0) == (gs > 0)))
        if ps == 0 and gs == 0:
            dice.append(np.nan)
        elif ps == 0 or gs == 0:
            dice.append(0.0)
        else:
            dice.append(2.0 * np.logical_and(p, g).sum() / (ps + gs))
    a = float(np.mean(acc))
    d = float(np.nanmean(dice))
    return 0.7 * a + 0.3 * d, a, d


def keep_components(mask, min_area):
    if min_area <= 0 or mask.sum() == 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def to_native(prob_crop, box, native_hw):
    """crop-frame 256x256 prob -> 1024 pack frame -> native HxW."""
    y0, y1, x0, x1 = box
    canvas = np.zeros((1024, 1024), np.float32)
    canvas[y0:y1, x0:x1] = cv2.resize(prob_crop.astype(np.float32),
                                      (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    h, w = native_hw
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)


def load_gt_native(_id, native_hw):
    m = cv2.imread(f"{PACK}/task1/mask/{_id}.png", cv2.IMREAD_GRAYSCALE)
    h, w = native_hw
    return (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)


def build(split_ids, probs, boxes, meta, t_seg, min_area, cls_p, tau, gate):
    """Returns binarised native-resolution predictions under one config."""
    out = []
    mi = meta.set_index("our_id")
    for k, _id in enumerate(split_ids):
        hw = (int(mi.loc[_id, "native_h"]), int(mi.loc[_id, "native_w"]))
        if gate and cls_p is not None and cls_p[k] < tau:
            out.append(np.zeros(hw, np.uint8))
            continue
        p = to_native(probs[k], boxes[str(_id)], hw)
        m = keep_components((p >= t_seg).astype(np.uint8),
                            int(min_area * hw[0] * hw[1] / (1024 * 1024)))
        if gate and cls_p is not None and cls_p[k] >= tau and m.sum() == 0:
            # recovery: the classifier is confident, so lower the bar rather
            # than inherit the segmentor's miss (an empty mask on a GT-positive
            # case scores 0.0 and is COUNTED, so a bad guess is free upside).
            m = keep_components((p >= t_seg * 0.5).astype(np.uint8), 0)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="+", required=True)
    ap.add_argument("--cls", nargs="*", default=[])
    ap.add_argument("--write-masks", default="")
    a = ap.parse_args()

    meta = pd.read_csv(f"{PACK}/task1/meta.csv")
    boxes = json.load(open(f"{WORK}/lungbox.json"))
    segs = [np.load(os.path.join(WORK, f) if not os.path.isabs(f) else f) for f in a.seg]
    tr_ids, te_ids = segs[0]["train_ids"], segs[0]["test_ids"]
    oof_p = np.mean([s["oof"].astype(np.float32) for s in segs], 0)
    test_p = np.mean([s["test"].astype(np.float32) for s in segs], 0)

    cls_oof = cls_test = None
    if a.cls:
        cs = [np.load(os.path.join(WORK, f) if not os.path.isabs(f) else f) for f in a.cls]
        cls_oof = np.mean([c["oof"] for c in cs], 0)
        cls_test = np.mean([c["test"] for c in cs], 0)

    gt_tr = [load_gt_native(i, (int(r.native_h), int(r.native_w)))
             for i, r in zip(tr_ids, meta.set_index("our_id").loc[tr_ids].itertuples())]
    gt_te = [load_gt_native(i, (int(r.native_h), int(r.native_w)))
             for i, r in zip(te_ids, meta.set_index("our_id").loc[te_ids].itertuples())]

    # ---- joint sweep on OOF (444 train cases) only -----------------------
    best = None
    t_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    a_grid = [0, 64, 256, 1024]
    tau_grid = [0.0] if cls_oof is None else [0.3, 0.4, 0.5, 0.6, 0.7]
    gate_grid = [False] if cls_oof is None else [False, True]
    print(f"{'t_seg':>6}{'minA':>7}{'tau':>6}{'gate':>6}  {'score':>7}{'acc':>7}{'dice':>7}")
    for t, mn, tau, g in itertools.product(t_grid, a_grid, tau_grid, gate_grid):
        preds = build(tr_ids, oof_p, boxes, meta, t, mn, cls_oof, tau, g)
        s, acc, d = official_score(preds, gt_tr)
        print(f"{t:6.2f}{mn:7d}{tau:6.2f}{str(g):>6}  {s:7.4f}{acc:7.4f}{d:7.4f}", flush=True)
        if best is None or s > best[0]:
            best = (s, t, mn, tau, g)
    _, t, mn, tau, g = best
    print(f"\nOOF-selected config: t_seg={t} min_area={mn} tau={tau} gate={g}")

    # ---- apply once to test ---------------------------------------------
    preds = build(te_ids, test_p, boxes, meta, t, mn, cls_test, tau, g)
    s, acc, d = official_score(preds, gt_te)
    print(f"\n=== TEST (n={len(te_ids)}, official metric) ===")
    print(f"  accuracy    {acc:.4f}")
    print(f"  mean dice   {d:.4f}   (nanmean, organizers' NaN rule)")
    print(f"  FINAL SCORE {s:.4f}   = 0.7*acc + 0.3*dice")
    print(f"  reference: current pipeline = 0.7*0.8468 + 0.3*0.229 = 0.6614")

    if a.write_masks:
        import SimpleITK as sitk
        os.makedirs(a.write_masks, exist_ok=True)
        for _id, m in zip(te_ids, preds):
            sitk.WriteImage(sitk.GetImageFromArray(m.astype(np.uint8)),
                            os.path.join(a.write_masks, f"{_id}.nii.gz"))
        print(f"\nwrote {len(preds)} masks -> {a.write_masks}")


if __name__ == "__main__":
    main()
