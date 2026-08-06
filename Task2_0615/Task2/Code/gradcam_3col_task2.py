"""3-column Grad-CAM grid: original (raw, uncropped) | cropped (PSPNet
lung-crop, the actual model input) | Grad-CAM overlay, one row per image.
Uses our recommended model (rad_dino, strong-aug/no-mixup DG-ablation
config -- see memory: task2_dg_ablation_results), scored with the correct
leave-one-modality-out held-out fold for each image's OWN Modality_DICOM
(not one fixed fold for both rows).

Reuses gradcam_task2.py's model/Grad-CAM functions and
gradcam_grid_task2.py's plain_image_tile/save_grid layout code verbatim --
no new Grad-CAM logic here, just assembling the 3-column layout neither
existing script produces directly (gradcam_grid_task2.py's "original"
column is already the cropped input, not the true pre-crop original).

Usage:
  cd Task2_0615/Task2/Code
  python gradcam_3col_task2.py --image-ids 1193 3 \
      --ckpt-dir ../checkpoints/checkpoints_dgablation_strong_aug_no_mixup \
      --raw-dir ../Data/test --crop-dir ../Data/Preprocessed/test_images/ch0 \
      --csv ../Data/test.csv --out gradcam_3col.png
"""
import argparse

import cv2
import pandas as pd
import torch

from common import find_file, sitk_read
from dataset import Cfg
from gradcam_grid_task2 import plain_image_tile, save_grid
from gradcam_task2 import build_model, gradcam_rad_dino, load_checkpoint, load_input, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-ids", nargs="+", required=True)
    ap.add_argument("--ckpt-dir", default="../checkpoints/checkpoints_dgablation_strong_aug_no_mixup",
                     help="default: our recommended model (rad_dino, strong aug, no mixup)")
    ap.add_argument("--encoder", default="rad_dino")
    ap.add_argument("--raw-dir", default="../Data/test",
                     help="RAW, uncropped images -- the true 'original' column")
    ap.add_argument("--crop-dir", default="../Data/Preprocessed/test_images/ch0",
                     help="PSPNet lung-cropped images -- the actual model input, and the 'cropped' column")
    ap.add_argument("--csv", default="../Data/test.csv",
                     help="used to look up each image's true label + Modality_DICOM (picks the "
                          "correct held-out LOMO fold per image, same rule as gradcam_grid_task2.py)")
    ap.add_argument("--tile-size", type=int, default=320)
    ap.add_argument("--out", default="gradcam_3col.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df["new_id"] = df["new_id"].astype(str)
    df = df.set_index("new_id")

    fold_cache = {}
    columns = ["original", "cropped", "gradcam"]
    grid_rows = []

    for image_id in args.image_ids:
        row = df.loc[image_id] if image_id in df.index else None
        label = str(row["TB/Normal"]) if row is not None else "?"
        modality = str(row["Modality_DICOM"]) if row is not None and "Modality_DICOM" in df.columns else None
        fold_tag = modality or "CR"

        print(f"--- image {image_id} (true label={label}, fold={fold_tag}) ---")

        if fold_tag not in fold_cache:
            ckpt = load_checkpoint(args.ckpt_dir, args.encoder, fold_tag)
            fold_cache[fold_tag] = build_model(ckpt)
        model = fold_cache[fold_tag]

        raw_path = find_file(args.raw_dir, image_id)
        if raw_path is None:
            raise FileNotFoundError(f"no raw image for id={image_id} under {args.raw_dir}")
        raw_arr = sitk_read(raw_path)

        cfg = Cfg(args.encoder, image_dir=args.crop_dir)
        x, crop_arr = load_input(args.crop_dir, image_id, cfg)

        cam, prob_tb = gradcam_rad_dino(model, x)
        print(f"    P(TB) = {prob_tb:.4f}")

        original_vis = plain_image_tile(raw_arr, args.tile_size)
        cropped_vis = plain_image_tile(crop_arr, args.tile_size)
        gradcam_vis = overlay(crop_arr, cam, out_size=args.tile_size)

        row_label = [f"id={image_id}", f"true={label}", f"fold={fold_tag}"]
        cells = [
            ("original", None, original_vis),
            ("cropped", None, cropped_vis),
            ("gradcam", prob_tb, gradcam_vis),
        ]
        grid_rows.append((row_label, cells))

    save_grid(grid_rows, columns, args.out, args.tile_size)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
