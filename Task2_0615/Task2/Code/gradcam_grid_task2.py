"""Batch Grad-CAM: run gradcam_task2's per-encoder heatmaps over N images at
once and lay them all out in one grid (rows=images, columns=encoders), so
patterns across cases are visible together instead of one image at a time.
Reuses every model/heatmap function from gradcam_task2.py verbatim -- no
separate Grad-CAM logic here, just batching + a bigger grid layout.

Each image is scored against the checkpoint whose held-out modality matches
THAT image's OWN Modality_DICOM (the genuinely held-out fold for that case),
not one fixed fold for everything -- otherwise most images would be scored
by a checkpoint that actually trained on them, defeating the point of a
held-out check. Override with --fold-tag to force one fold for every image
instead (e.g. to specifically inspect one modality).

Models are loaded once per (encoder, fold) pair and reused across every
image sharing that modality, not reloaded per image.

Usage:
  cd Task2_0615/Task2/Code
  # 10 random images (balanced TB/Normal), reproducible via --seed:
  python gradcam_grid_task2.py --n-images 10 --seed 0 \
      --ckpt-dir ../checkpoints --image-dir ../Data/Preprocessed/train_images/ch0 \
      --csv ../Data/train.csv --out gradcam_grid.png

  # or specific ids:
  python gradcam_grid_task2.py --image-ids 1193 51 3261 5467 7731 \
      --out gradcam_grid.png
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

from common import find_file, sitk_read
from gradcam_task2 import (ALL_ENCODERS, ENCODERS, build_model, gradcam_rad_dino,
                            gradcam_timm, load_checkpoint, load_input, overlay)
from dataset import Cfg


def pick_images(csv_path, n, seed, label=None, exclude_ids=None):
    """Random sample of size n. label=None: balanced (as close to half TB /
    half Normal as n allows), so a mixed grid isn't accidentally all-one-
    class by luck of the draw. label="tb"/"normal": every row that class
    only. exclude_ids: skip these (e.g. images already shown in an earlier
    grid, so a follow-up request for "more" cases doesn't repeat any)."""
    df = pd.read_csv(csv_path)
    df["new_id"] = df["new_id"].astype(str)
    df["TB/Normal"] = df["TB/Normal"].astype(str).str.strip().str.lower()
    df = df[df["TB/Normal"].isin(["tb", "normal"])]
    if exclude_ids:
        df = df[~df["new_id"].isin({str(i) for i in exclude_ids})]

    if label:
        pool = df[df["TB/Normal"] == label.lower()]
        return pool.sample(min(n, len(pool)), random_state=seed)

    n_tb = n // 2
    n_normal = n - n_tb
    tb = df[df["TB/Normal"] == "tb"]
    normal = df[df["TB/Normal"] == "normal"]
    picked = pd.concat([
        tb.sample(min(n_tb, len(tb)), random_state=seed),
        normal.sample(min(n_normal, len(normal)), random_state=seed),
    ]).sample(frac=1, random_state=seed)  # shuffle row order so TB/Normal isn't grouped
    return picked


def plain_image_tile(raw_img, out_size):
    """Same base-image normalization as gradcam_task2.overlay(), just without
    the heatmap blend -- an 'original' column cell, in the identical RGB
    convention overlay() returns (save_grid converts every cell's RGB->BGR
    uniformly, heatmap or not)."""
    base = cv2.resize(raw_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    return cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def save_grid(grid_rows, columns, out_path, tile_size):
    """Pure-OpenCV grid (see gradcam_task2.save_panel_grid's docstring for
    why: avoids matplotlib entirely). One row per image, one column per
    entry in `columns` (typically ["original", <encoder names>]); column
    headers once at the top; row label (image id / true label / modality) on
    the left of each row; each cell gets its own small strip -- P(TB) for a
    heatmap cell, blank for the plain "original" cell (no probability to
    show there), "n/a" for a missing checkpoint."""
    header_h, strip_h, row_label_w, pad = 26, 20, 170, 4
    white = lambda h, w: np.full((h, w, 3), 255, dtype=np.uint8)

    col_header = white(header_h, row_label_w)
    for col_name in columns:
        cell = white(header_h, tile_size)
        cv2.putText(cell, col_name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        col_header = np.hstack([col_header, white(header_h, pad), cell])

    rows_img = [col_header]
    for row_label_lines, cells in grid_rows:
        label_tile = white(strip_h + tile_size, row_label_w)
        for i, line in enumerate(row_label_lines):
            cv2.putText(label_tile, line, (4, strip_h + 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        row_img = label_tile
        for col_name, prob_tb, vis in cells:
            strip = white(strip_h, tile_size)
            if vis is None:
                tile = np.full((tile_size, tile_size, 3), 225, dtype=np.uint8)
                cv2.putText(strip, "n/a", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                tile = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                strip_text = f"P(TB)={prob_tb:.3f}" if prob_tb is not None else ""
                cv2.putText(strip, strip_text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cell = np.vstack([strip, tile])
            row_img = np.hstack([row_img, white(strip_h + tile_size, pad), cell])
        rows_img.append(row_img)

    sep = white(pad, rows_img[0].shape[1])
    grid = rows_img[0]
    for r in rows_img[1:]:
        grid = np.vstack([grid, sep, r])
    cv2.imwrite(out_path, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-ids", nargs="*", default=None,
                     help="explicit list of new_id values; omit to auto-sample --n-images")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", choices=["tb", "normal"], default=None,
                     help="restrict random sampling to one class; default: balanced TB+Normal mix "
                          "(no effect when --image-ids is given)")
    ap.add_argument("--exclude-ids", nargs="*", default=None,
                     help="skip these new_id values when random-sampling -- e.g. pass a previous "
                          "run's image-ids here so a follow-up 'show me more' grid never repeats one")
    ap.add_argument("--csv", default="../Data/train.csv",
                     help="used to look up each image's true label/modality, and to sample "
                          "--n-images at random if --image-ids isn't given")
    ap.add_argument("--no-original", action="store_true",
                     help="drop the plain-original-image column (included by default)")
    ap.add_argument("--fold-tag", default=None,
                     help="force one held-out fold for every image; default: auto-pick per "
                          "image from its own Modality_DICOM column (the genuinely held-out "
                          "checkpoint for that case)")
    ap.add_argument("--ckpt-dir", default="../checkpoints")
    ap.add_argument("--image-dir", default="../Data/Preprocessed/train_images/ch0")
    ap.add_argument("--encoders", nargs="*", default=ENCODERS, choices=ALL_ENCODERS,
                     help="default: densenet121/swin_tiny/convnext_tiny (rad_dino needs "
                          "transformers+peft+a working torchaudio import; pass explicitly "
                          "once available)")
    ap.add_argument("--tile-size", type=int, default=200, help="per-cell heatmap size in px")
    ap.add_argument("--out", default="gradcam_grid.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    df = pd.read_csv(args.csv)
    df["new_id"] = df["new_id"].astype(str)
    df = df.set_index("new_id")

    if args.image_ids:
        image_ids = [str(i) for i in args.image_ids]
    else:
        picked = pick_images(args.csv, args.n_images, args.seed,
                              label=args.label, exclude_ids=args.exclude_ids)
        image_ids = picked["new_id"].astype(str).tolist()

    rows_meta = []
    for iid in image_ids:
        if iid in df.index:
            row = df.loc[iid]
            label = str(row["TB/Normal"]).strip()
            modality = str(row["Modality_DICOM"]).strip()
        else:
            label, modality = "?", "?"
        fold_tag = args.fold_tag or modality
        rows_meta.append((iid, label, modality, fold_tag))

    model_cache = {}

    def get_model(enc_name, fold_tag):
        key = (enc_name, fold_tag)
        if key not in model_cache:
            print(f"loading {enc_name} / fold {fold_tag} ...")
            ckpt = load_checkpoint(args.ckpt_dir, enc_name, fold_tag)
            model_cache[key] = build_model(ckpt)
        return model_cache[key]

    grid_rows = []
    for iid, label, modality, fold_tag in rows_meta:
        print(f"--- image {iid} (true={label}, modality={modality}, held-out fold={fold_tag}) ---")
        cells = []
        if not args.no_original:
            img_path = find_file(args.image_dir, iid, ".png")
            if img_path is None:
                cells.append(("original", None, None))
            else:
                orig_vis = plain_image_tile(sitk_read(img_path), args.tile_size)
                cells.append(("original", None, orig_vis))
        for enc_name in args.encoders:
            try:
                model = get_model(enc_name, fold_tag)
            except FileNotFoundError as e:
                print(f"    !! {e}")
                cells.append((enc_name, None, None))
                continue
            cfg = Cfg(enc_name, image_dir=args.image_dir)
            x, raw = load_input(args.image_dir, iid, cfg)
            if enc_name == "rad_dino":
                cam, prob_tb = gradcam_rad_dino(model, x)
            else:
                cam, prob_tb = gradcam_timm(model, x)
            vis = overlay(raw, cam, out_size=args.tile_size)
            cells.append((enc_name, prob_tb, vis))
            print(f"    {enc_name}: P(TB)={prob_tb:.3f}")
        grid_rows.append(([f"{iid}  true={label}", f"modality={modality}"], cells))

    columns = (["original"] if not args.no_original else []) + args.encoders
    save_grid(grid_rows, columns, args.out, args.tile_size)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
