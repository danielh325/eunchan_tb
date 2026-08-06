#!/usr/bin/env python
"""Convert Task1's cavity masks into YOLOv8-seg's expected data layout
(images/{train,val,test}/*.png + labels/{train,val,test}/*.txt polygons +
data.yaml), so a structurally different segmentation architecture (box/
instance-based YOLOv8-seg, vs. nnU-Net/foundation-decoder's dense pixel-wise
prediction) can be trained and ensembled for real architectural diversity --
see the paper-derived idea this implements (PMC11992683: nnU-Net + YOLOv8-seg
ensemble was their most stable segmentation result).

Split: same patient-grouped fold-0-as-dev split every other model in this
project uses (common.py's make_folds, seed=42, n_folds=5) for train/val, so
val numbers are comparable in spirit to train_segmentation.py's dev split.
Test images/masks: same source as everything else -- Data/test/CXR images,
masks borrowed from nnU-Net's nnUNet_raw/labelsTr (Data/test/CXR_label
doesn't exist in this project; see train_segmentation.py's own docstring for
why that substitution is used elsewhere too).

Usage:
    python prepare_yolo_seg_data.py --img-size 640
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ID_COL, find_file, make_folds, sitk_read

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))
DEFAULT_OUT_DIR = os.path.join(HERE, "yolo_seg_data")
_DEFAULT_NNUNET_LABELS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data",
    "nnUNet_raw", "Dataset001_Task1", "labelsTr"))


class CFG:
    def __init__(self, args):
        self.img_size = args.img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip, self.clahe_grid = 2.0, 8
        self.n_folds = args.n_folds


def image_to_u8(arr, cfg):
    """Same clip+CLAHE+resize pipeline as common.py's preprocess_image, minus
    the final mean/std normalization (YOLO wants plain 0-255 pixels, not a
    normalized float tensor -- ultralytics normalizes internally)."""
    lo, hi = np.percentile(arr, cfg.clip_lo), np.percentile(arr, cfg.clip_hi)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    u8 = (arr * 255).astype(np.uint8)
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        u8 = clahe.apply(u8)
    return cv2.resize(u8, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)


def mask_to_polygons(mask_path, img_size, transpose=False):
    """Binary mask -> list of YOLO-seg polygons (each a flat list of
    normalized x,y pairs, one polygon per connected foreground contour).
    Returns [] for a no-cavity case (correct/expected -- an empty label file
    is YOLO's own convention for "no objects", not an error).

    transpose defaults to False and should stay that way, including for TEST
    masks sourced from nnU-Net's labelsTr. An earlier version of this
    function (and of train_segmentation.py's load_mask(), which this was
    copied from) set transpose=True for those masks based on a single noisy
    grid-search reading. train_segmentation.py later re-swept this offline
    against job 117713's saved test_probs.pt across all 8 dihedral
    transforms for all three encoders: identity (transpose=False) was best by
    a wide margin (dice 0.27-0.29), transpose was one of the worst (dice
    ~0.05, matching the ~0.04 collapse originally seen with it). No transform
    is needed -- the mask and image are already read via the same
    sitk.GetArrayFromImage row/col convention. See train_segmentation.py's
    load_mask() docstring for the full account."""
    m = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    m = np.squeeze(m)
    if transpose:
        m = m.T
    m = (m > 0).astype(np.uint8)
    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c) < 4:
            continue  # degenerate/near-zero-area contour, not a real lesion
        pts = c.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= img_size
        pts[:, 1] /= img_size
        polys.append(pts.flatten().tolist())
    return polys


def write_split(df, image_dir, mask_dir, image_ext, cfg, out_dir, split_name, skip_missing_mask,
                 mask_transpose=False):
    img_out = os.path.join(out_dir, "images", split_name)
    lbl_out = os.path.join(out_dir, "labels", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n_written, n_positive = 0, 0
    for _, row in df.iterrows():
        _id = row[ID_COL]
        ip = find_file(image_dir, _id, image_ext)
        if ip is None:
            print(f"!! {split_name}: no image for {ID_COL}={_id}, skipping")
            continue
        mp = os.path.join(mask_dir, f"{_id}.nii.gz")
        if not os.path.exists(mp):
            if skip_missing_mask:
                print(f"!! {split_name}: no mask for {ID_COL}={_id}, skipping")
                continue
            polys = []
        else:
            polys = mask_to_polygons(mp, cfg.img_size, transpose=mask_transpose)

        u8 = image_to_u8(sitk_read(ip), cfg)
        cv2.imwrite(os.path.join(img_out, f"{_id}.png"), u8)
        with open(os.path.join(lbl_out, f"{_id}.txt"), "w") as f:
            for poly in polys:
                f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

        n_written += 1
        n_positive += 1 if polys else 0

    print(f"[{split_name}] wrote {n_written} images ({n_positive} with >=1 cavity polygon) -> {img_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--dev-fold", type=int, default=0,
                    help="fold used as val -- matches train_segmentation.py's dev split choice")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--nnunet-labels-dir", default=_DEFAULT_NNUNET_LABELS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    cfg = CFG(args)

    df_tr_all = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_tr_all = make_folds(df_tr_all, cfg, group_col=None, seed=args.seed)
    tr = df_tr_all[df_tr_all.fold != args.dev_fold]
    dev = df_tr_all[df_tr_all.fold == args.dev_fold]

    train_image_dir = os.path.join(args.data_dir, "train", "CXR")
    train_mask_dir = os.path.join(args.data_dir, "train", "CXR_label")
    write_split(tr, train_image_dir, train_mask_dir, ".dcm", cfg, args.out_dir, "train", skip_missing_mask=False)
    write_split(dev, train_image_dir, train_mask_dir, ".dcm", cfg, args.out_dir, "val", skip_missing_mask=False)

    df_te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    test_image_dir = os.path.join(args.data_dir, "test", "CXR")
    write_split(df_te, test_image_dir, args.nnunet_labels_dir, ".dcm", cfg, args.out_dir, "test",
                skip_missing_mask=False, mask_transpose=False)

    yaml_path = os.path.join(args.out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n  0: cavity\n")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
