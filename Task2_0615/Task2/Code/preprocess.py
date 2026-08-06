"""CLI preprocessing for Task 2 -- precomputes only the expensive, GPU-bound
step (lung-field segmentation), once, up front. CLAHE and bone-suppression
are deliberately NOT baked in here -- they're cheap CPU filters applied
on the fly by common.preprocess_image (matching Task1's established
pattern: only the segmenter's forward pass is worth caching to disk;
baking CLAHE here too would double-apply it on top of the on-the-fly pass
that every training run already does via Cfg.use_clahe).

Output layout:
  ch0 = lung-cropped raw grayscale (min-max normalized only)
  ch2 = lung mask (0/255 uint8), pixel-aligned 1:1 with ch0, only written
        when --mask-channel is set. Consumed by dataset.py's Cfg.mask_dir to
        replace TBDataset's otherwise-redundant 3rd duplicate channel (see
        common.preprocess_image's mask_arr handling).
--lung-crop is required whenever --mask-channel is set (both derive from the
same segmenter pass, and the mask is only meaningful paired with a
consistently-cropped ch0).
"""
import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def save_u8(arr, path):
    out = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir of raw {id}.png images")
    ap.add_argument("--out-dir", required=True, help="output base dir (writes ch0[/ch2] subdirs)")
    ap.add_argument("--lung-crop", action="store_true", help="crop to lung-field bounding box")
    ap.add_argument("--mask-channel", action="store_true",
                     help="also write the lung mask into ch2, pixel-aligned with ch0 "
                          "(requires --lung-crop -- both come from the same segmenter pass)")
    args = ap.parse_args()

    if args.mask_channel and not args.lung_crop:
        raise SystemExit("--mask-channel requires --lung-crop")

    out_ch0 = os.path.join(args.out_dir, "ch0")
    for p in [out_ch0]:
        os.makedirs(p, exist_ok=True)
    out_ch2 = None
    if args.mask_channel:
        out_ch2 = os.path.join(args.out_dir, "ch2")
        os.makedirs(out_ch2, exist_ok=True)

    cropper = None
    if args.lung_crop:
        from lung_crop import LungCropper
        cropper = LungCropper()

    png_files = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Preprocessing {len(png_files)} images "
          f"(lung_crop={args.lung_crop}, mask_channel={args.mask_channel})")

    for png_path in tqdm(png_files):
        new_id = os.path.splitext(os.path.basename(png_path))[0]
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARN: failed to read {png_path}, skipping")
            continue
        raw = img.astype(np.float32)

        crop_mask = None
        if cropper is not None:
            full_mask = cropper.segment(raw)
            y0, y1, x0, x1 = cropper.box_from_mask(full_mask)
            raw = raw[y0:y1, x0:x1]
            crop_mask = full_mask[y0:y1, x0:x1]

        base_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        ch0_u8 = (base_norm * 255).astype(np.uint8)
        # No _0000-style suffix: ch0/ch2 already live in separate
        # directories, and common.find_file() (dataset.py's lookup path)
        # looks for a plain "{id}.png" -- a suffix here would silently
        # never match and every __getitem__ would raise FileNotFoundError.
        save_u8(ch0_u8, os.path.join(out_ch0, f"{new_id}.png"))

        if out_ch2 is not None:
            ch2_u8 = crop_mask.astype(np.uint8) * 255
            save_u8(ch2_u8, os.path.join(out_ch2, f"{new_id}.png"))

    print("Done")


if __name__ == "__main__":
    main()
