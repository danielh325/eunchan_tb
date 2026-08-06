"""Builds a common.py-compatible CSV + flat image dir from the Mendeley
"Dataset of Tuberculosis Chest X-rays Images" (Pakistan hospital collection;
2494 TB.jpg + 514 "others (N).jpg" normal images), moved into this project
at Data/external/mendeley_tb/{normal,tb}/ on 2026-08-04.

UNLIKE the Belarus cohort (build_qu_tb_csv.py), this dataset HAS both
classes, so it supports full accuracy/AUC/F1 evaluation, not recall-only.

PROVENANCE -- RESOLVED 2026-08-05: this is Mendeley Data DOI
10.17632/8j2g3csprk.2, "Dataset of Tuberculosis Chest X-rays Images",
contributors Saira Kiran and Dr Ishrat Jabeen, CC BY 4.0, published
2024-05-28 (https://data.mendeley.com/datasets/8j2g3csprk/2) -- confirmed by
its listed counts (2494 TB + 514 Normal, "collected from a local hospital in
Pakistan") matching this project's copy exactly. No peer-reviewed paper is
linked to it (unlike Belarus's arXiv:2007.14895), so it's a weaker citation
than Belarus/Shenzhen/Montgomery/TBX11K, but it IS a citable, DOI'd primary
source, not an untraceable pull -- the "others (N).jpg" naming turned out to
be just this dataset's own file-naming convention, not evidence of a
filtered-out third class from a bigger release (the Mendeley listing itself
is exactly Normal+TB, 2 classes, matching what we have). Cite as: Kiran, S.,
Jabeen, I. "Dataset of Tuberculosis Chest X-rays Images." Mendeley Data,
V2, 2024, doi: 10.17632/8j2g3csprk.2.

Two overlap risks this script actively checks for, same reasoning as
build_qu_tb_csv.py:
  1. Overlap with Shenzhen/Montgomery/TBX11K/Belarus (already ours) --
     checked via perceptual-hash dedup (--dedupe-against), same dHash
     approach validated on Belarus (threshold=1, since threshold=5 was
     shown there to produce smooth-unimodal false positives on CXR images).
  2. Overlap with RAD-DINO's pretraining corpus (BRAX/CheXpert/MIMIC-CXR/
     ChestX-ray14/PadChest) -- NOT checked here; those source images are
     not available locally to hash against. Treat any surprisingly strong
     result on this cohort with the same skepticism applied to the QU/
     Kaggle dataset's RSNA-derived Normal images.

USAGE
-----
  python build_mendeley_tb_csv.py \
      --normal-dir ../Data/external/mendeley_tb/normal \
      --tb-dir ../Data/external/mendeley_tb/tb \
      --out-csv ../Data/external/mendeley_tb/mendeley_tb.csv \
      --out-image-dir ../Data/external/mendeley_tb/raw \
      --dedupe-against ../Data/external/shenzhen/CXR_png \
                        ../Data/external/montgomery/CXR_png \
                        ../Data/external/qu_tb/raw

Run with --dry-run first to see the dedup breakdown before copying 3008
images across the network.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image


def dhash(path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_reference_hashes(dirs):
    refs = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  !! {d} not found -- skipping")
            continue
        n = 0
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        refs.append(dhash(os.path.join(root, f)))
                        n += 1
                    except Exception as e:
                        print(f"  !! {f}: {e}")
        print(f"  hashed {n} reference images from {d}")
    return refs


def list_images(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-dir", required=True)
    ap.add_argument("--tb-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-image-dir", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts and dedup breakdown, write nothing")
    ap.add_argument("--dedupe-against", nargs="*", default=[
                        "../Data/external/shenzhen/CXR_png",
                        "../Data/external/montgomery/CXR_png"],
                    help="dirs of images we already evaluate on; candidates "
                         "matching one by perceptual hash are dropped. Pass "
                         "no values to skip (not recommended -- provenance "
                         "here is unverified, see this script's header).")
    ap.add_argument("--hash-threshold", type=int, default=1,
                    help="max Hamming distance (of 64) to count as a "
                         "duplicate. Default 1: threshold=5 was shown on "
                         "the Belarus cohort to false-positive-match ~35%% "
                         "of genuinely distinct chest X-rays (smooth "
                         "unimodal distance distribution, no gap) -- see "
                         "memory: task2_qu_dataset_composition.")
    args = ap.parse_args()

    normals = list_images(args.normal_dir)
    tbs = list_images(args.tb_dir)
    print(f"{len(normals)} normal images in {args.normal_dir}")
    print(f"{len(tbs)} TB images in {args.tb_dir}")

    refs = []
    if args.dedupe_against:
        print("\nhashing already-used cohorts (overlap check):")
        refs = build_reference_hashes(args.dedupe_against)
        if not refs:
            raise SystemExit("!! --dedupe-against matched no readable images; "
                             "fix the paths or pass --dedupe-against with no "
                             "values to deliberately skip the check")

    def scan(src_dir, files, label):
        kept, dropped, missing = [], 0, 0
        for f in files:
            p = os.path.join(src_dir, f)
            if not os.path.exists(p):
                missing += 1
                continue
            if refs:
                try:
                    h = dhash(p)
                    if min(hamming(h, r) for r in refs) <= args.hash_threshold:
                        dropped += 1
                        continue
                except Exception as e:
                    print(f"  !! could not hash {p}: {e} -- keeping it")
            kept.append((p, label))
        print(f"{label}: kept {len(kept)} / {len(files)} "
              f"(dropped {dropped} as dupes, {missing} missing)")
        return kept

    normal_kept = scan(args.normal_dir, normals, "Normal")
    tb_kept = scan(args.tb_dir, tbs, "TB")

    print(f"\nTOTAL: {len(normal_kept)} Normal + {len(tb_kept)} TB = "
          f"{len(normal_kept) + len(tb_kept)} images")
    if args.dry_run:
        print("(dry run -- nothing written)")
        return

    os.makedirs(args.out_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    rows = []
    for i, (p, label) in enumerate(normal_kept):
        new_id = f"mendeley_normal_{i:05d}"
        shutil.copyfile(p, os.path.join(args.out_image_dir, new_id + ".png"))
        rows.append({"new_id": new_id, "TB/Normal": "Normal"})
    for i, (p, label) in enumerate(tb_kept):
        new_id = f"mendeley_tb_{i:05d}"
        shutil.copyfile(p, os.path.join(args.out_image_dir, new_id + ".png"))
        rows.append({"new_id": new_id, "TB/Normal": "TB"})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")
    print(f"copied images -> {args.out_image_dir}")
    print("\nNEXT: preprocess (--lung-crop --mask-channel), then score with "
          "predict_task2.py --score. Has both classes -- accuracy/AUC/F1 are "
          "all meaningful, unlike the Belarus (TB-only) cohort. Provenance "
          "resolved (Mendeley DOI 10.17632/8j2g3csprk.2, Kiran & Jabeen "
          "2024) -- cite it, no peer-reviewed paper is linked to it though.")


if __name__ == "__main__":
    main()
