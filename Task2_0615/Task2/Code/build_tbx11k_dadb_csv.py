"""Builds a train.csv-compatible CSV (new_id, TB/Normal) from TBX11K's
imgs/extra/da+db/ subfolder -- the DA+DB dataset (National Institute of
Tuberculosis and Respiratory Diseases, New Delhi, India, two X-ray
machines), bundled inside the TBX11K zip we already downloaded for
build_tbx11k_full_csv.py.

This is a genuinely separate external source from everything else scored so
far (Shenzhen/Montgomery/TBX11K's own health+sick+tb pool are all China/
various; DA+DB is India) -- NOT reachable via build_tbx11k_full_csv.py,
which deliberately excludes ALL of imgs/extra/ (that folder also bundles an
mc+shenzhen/ sibling subfolder, which WOULD double-count data already
scored). da+db/ is a clean, distinct sibling of mc+shenzhen/ inside the
same extra/ folder, so it needs its own small builder rather than just
loosening the eval_tbx11k_full.sbatch exclusion.

Label convention (filename prefix, confirmed by directory listing -- two
machines' worth of each class): n*.png / nx*.png -> Normal, p*.png /
px*.png -> TB. train/ and val/ subfolders are pooled together since neither
represents held-out data for OUR model (we never trained on this set at
all).

Usage:
    python build_tbx11k_dadb_csv.py \
        --dadb-root ../Data/external/tbx11k/full/TBX11K/imgs/extra/da+db \
        --raw-out ../Data/external/tbx11k/raw_dadb \
        --csv-out ../Data/external/tbx11k/tbx11k_dadb.csv
"""
import argparse
import glob
import os
import re
import shutil

import cv2
import pandas as pd

_PREFIX_RE = re.compile(r"^(nx?|px?)\d+\.(png|jpg|jpeg)$", re.IGNORECASE)
_LABEL = {"n": "Normal", "nx": "Normal", "p": "TB", "px": "TB"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dadb-root", required=True, help="dir with train/ and val/ subfolders")
    ap.add_argument("--raw-out", required=True, help="flat dir to copy included images into")
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()

    os.makedirs(args.raw_out, exist_ok=True)
    rows = []
    skipped = []
    for split in ("train", "val"):
        split_dir = os.path.join(args.dadb_root, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: {split_dir} not found, skipping")
            continue
        paths = sorted(
            glob.glob(os.path.join(split_dir, "*.png"))
            + glob.glob(os.path.join(split_dir, "*.jpg"))
            + glob.glob(os.path.join(split_dir, "*.jpeg"))
        )
        for path in paths:
            fname = os.path.basename(path)
            m = _PREFIX_RE.match(fname)
            if not m:
                skipped.append(fname)
                continue
            label = _LABEL[m.group(1).lower()]
            # split+filename as new_id -- val/ and train/ each have their own
            # n1.png/p1.png etc, so the bare filename alone isn't unique
            # across the pooled set.
            stem, ext = os.path.splitext(fname)
            new_id = f"{split}_{stem}"
            out_path = os.path.join(args.raw_out, f"{new_id}.png")
            if ext.lower() == ".png":
                shutil.copy2(path, out_path)
            else:
                # nx/px files are actually .jpg, not .png, despite living
                # alongside .png files in the same folder -- preprocess.py's
                # --in-dir handling only globs *.png (confirmed by job 118526:
                # it silently preprocessed 72/176 images, dropping every
                # .jpg), so these need to be real PNG-encoded files, not just
                # renamed. A bare rename would leave JPEG bytes under a .png
                # extension, which cv2/SimpleITK would still happily
                # misdecode without error -- re-encode instead.
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    skipped.append(fname)
                    continue
                cv2.imwrite(out_path, img)
            rows.append(dict(new_id=new_id, **{"TB/Normal": label}))

    if skipped:
        print(f"WARNING: {len(skipped)} file(s) didn't match the expected "
              f"{{n,nx,p,px}}####.png pattern, skipped: {skipped[:5]}"
              f"{' ...' if len(skipped) > 5 else ''}")
    if not rows:
        raise SystemExit(f"no matching files found under {args.dadb_root}/{{train,val}}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    n_tb = (df["TB/Normal"] == "TB").sum()
    print(f"wrote {args.csv_out}: {len(df)} images ({n_tb} TB, {len(df) - n_tb} Normal)")


if __name__ == "__main__":
    main()
