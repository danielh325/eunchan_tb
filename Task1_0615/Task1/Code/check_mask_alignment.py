#!/usr/bin/env python
"""One-off diagnostic: overlay nnU-Net's GT mask on the raw image for specific
case IDs, to check for an image/mask misalignment bug (flip, crop offset,
wrong resample).

Why: baseline nnU-Net (0.306 Dice) predicts a COMPLETELY EMPTY mask for 16+
cases that have a real cavity (see fold_0/validation/summary.json,
n_pred=0 and n_ref>0). Two of those misses are large lesions, not small
ones -- case 597 (n_ref=14399 px, ~6% of the image) and case 82
(n_ref=12270 px) -- which argues against pure class-imbalance/small-lesion
underweighting as the sole explanation and for checking basic image/mask
alignment first, per common.py's own documented pydicom/nibabel flip risk.

Usage (run inside the project's docker container -- CPU only, no GPU needed):
    python check_mask_alignment.py --case-id 597
    python check_mask_alignment.py --case-id 597 82 216 1
"""
import argparse
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NNUNET_ROOT = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data"))
IMAGES_DIR = os.path.join(NNUNET_ROOT, "nnUNet_raw", "Dataset001_Task1", "imagesTr")
LABELS_DIR = os.path.join(NNUNET_ROOT, "nnUNet_raw", "Dataset001_Task1", "labelsTr")


def load_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay(case_id, out_dir):
    img_path = os.path.join(IMAGES_DIR, f"{case_id}_0000.nii.gz")
    mask_path = os.path.join(LABELS_DIR, f"{case_id}.nii.gz")
    if not os.path.exists(img_path):
        print(f"!! case {case_id}: image not found at {img_path}")
        return
    if not os.path.exists(mask_path):
        print(f"!! case {case_id}: mask not found at {mask_path}")
        return

    img = np.squeeze(load_array(img_path))
    mask = np.squeeze(load_array(mask_path)) > 0

    print(f"case {case_id}: image shape={img.shape}, mask shape={mask.shape}, "
          f"mask pixel count={int(mask.sum())}")
    if img.shape != mask.shape:
        print(f"  !! SHAPE MISMATCH -- image and mask are not even the same "
              f"size, that alone would break everything downstream.")

    base = normalize_to_uint8(img)
    rgb = np.stack([base, base, base], axis=-1)
    # paint the mask in red at 60% opacity so the underlying anatomy is
    # still visible through it -- makes a misalignment obvious at a glance.
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    rgb = np.where(mask[..., None], (0.4 * rgb + 0.6 * red).astype(np.uint8), rgb)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"overlay_{case_id}.png")
    Image.fromarray(rgb).save(out_path)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", nargs="+", required=True)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "alignment_check"))
    args = ap.parse_args()
    for cid in args.case_id:
        overlay(cid, args.out_dir)


if __name__ == "__main__":
    main()
