#!/usr/bin/env python
"""Score a trained YOLOv8-seg cavity model with real Dice, directly comparable
to nnU-Net's summary.json and train_segmentation.py's TEST dice line.

Sweeps YOLO's detection confidence threshold on VAL only (never on test) --
same never-tune-on-test discipline as sweep_threshold() in
train_segmentation.py and sweep_tau_accuracy() in common.py -- then reports
the real held-out TEST dice at that threshold.

Usage:
    python eval_yolo_seg.py --weights runs/segment/cavity_yolov8m/weights/best.pt
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import SimpleITK as sitk
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_segmentation import dice_score_np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(HERE, "yolo_seg_data")
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))
_DEFAULT_NNUNET_LABELS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data",
    "nnUNet_raw", "Dataset001_Task1", "labelsTr"))


def load_binary_mask(path, img_size, transpose=False):
    """transpose defaults to False and should stay that way, including for
    labelsTr-sourced masks (TEST split) -- see train_segmentation.py's
    load_mask() docstring: an 8-transform sweep found identity (no transpose)
    is correct, transpose was one of the worst options (dice ~0.05 vs.
    0.27-0.29 for identity)."""
    m = sitk.GetArrayFromImage(sitk.ReadImage(path))
    m = np.squeeze(m)
    if transpose:
        m = m.T
    m = (m > 0).astype(np.uint8)
    return cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)


def predict_union_masks(model, image_paths, img_size, conf, device):
    """image path (by our_id) -> unioned binary prediction mask at img_size --
    every kept instance mask (YOLO's own conf filtering applied at this
    threshold) OR'd together into one cavity mask, for Dice comparison
    against the single-class ground truth."""
    results = model.predict(source=image_paths, imgsz=img_size, conf=conf,
                            device=device, verbose=False, retina_masks=True)
    out = {}
    for r in results:
        _id = os.path.splitext(os.path.basename(r.path))[0]
        canvas = np.zeros((img_size, img_size), dtype=np.uint8)
        if r.masks is not None:
            for inst_mask in r.masks.data.cpu().numpy():
                m = cv2.resize(inst_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                canvas = np.maximum(canvas, (m > 0.5).astype(np.uint8))
        out[_id] = canvas
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="output dir from prepare_yolo_seg_data.py")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--nnunet-labels-dir", default=_DEFAULT_NNUNET_LABELS)
    ap.add_argument("--conf-grid-min", type=float, default=0.05)
    ap.add_argument("--conf-grid-n", type=int, default=13)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.weights)
    grid = np.linspace(args.conf_grid_min, 0.7, args.conf_grid_n)
    lowest_conf = float(grid.min())

    def load_split(split_name, mask_dir, mask_transpose=False):
        img_paths = sorted(glob.glob(os.path.join(args.data_root, "images", split_name, "*.png")))
        gt = {os.path.splitext(os.path.basename(p))[0]:
              load_binary_mask(os.path.join(mask_dir, f"{os.path.splitext(os.path.basename(p))[0]}.nii.gz"),
                                args.img_size, transpose=mask_transpose)
              for p in img_paths}
        return img_paths, gt

    val_paths, val_gt = load_split("val", os.path.join(args.data_dir, "train", "CXR_label"))
    test_paths, test_gt = load_split("test", args.nnunet_labels_dir, mask_transpose=False)

    # ultralytics applies conf filtering inside model.predict() itself, so
    # sweeping the threshold means re-predicting per grid point -- val is only
    # ~89 images, cheap enough not to warrant hand-rolling per-instance-
    # confidence reuse across the grid.
    best_t, best_dice = lowest_conf, -1.0
    for t in grid:
        preds = predict_union_masks(model, val_paths, args.img_size, float(t), args.device)
        dices = [dice_score_np(preds[i], val_gt[i]) for i in preds]
        d = float(np.nanmean(dices))
        print(f"[yolo_seg] val conf={t:.3f} dice={d:.4f}")
        if d > best_dice:
            best_dice, best_t = d, float(t)

    print(f"[yolo_seg] VAL best conf*={best_t:.3f} dice={best_dice:.4f}")

    test_preds = predict_union_masks(model, test_paths, args.img_size, best_t, args.device)
    test_dices = [dice_score_np(test_preds[i], test_gt[i]) for i in test_preds]
    test_dice = float(np.nanmean(test_dices))
    print(f"[yolo_seg] TEST dice={test_dice:.4f} (real held-out, conf={best_t:.3f} from val, "
          f"n={len(test_preds)}) -- compare against nnU-Net baseline 0.306")


if __name__ == "__main__":
    main()
