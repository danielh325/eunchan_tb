"""Builds a train.csv-compatible CSV (new_id, TB/Normal) from the FULL
TBX11K download (fetch_tbx11k_full.sh), covering the entire imgs/tb +
imgs/health + imgs/sick pool -- a superset of the 500-image balanced subset
already scored by eval_tbx11k.sbatch (Data/external/tbx11k/tbx11k_500.csv).

Deliberately EXCLUDES imgs/extra/, same reasoning as the 500-subset: that
folder is drawn from Montgomery/Shenzhen/DA/DB, datasets already used in
run_external_validation.sbatch, so including it would double-count data
already validated on rather than testing on genuinely new images.

Also copies every included image into --raw-out (flat dir, filenames
unchanged) since predict_task2.py / preprocess.py expect a single flat
image directory, not TBX11K's per-class subfolders.

Usage:
    python build_tbx11k_full_csv.py --tbx-root Data/external/tbx11k/full \
        --raw-out Data/external/tbx11k/raw_full \
        --csv-out Data/external/tbx11k/tbx11k_full.csv
"""
import argparse
import glob
import os
import shutil

import pandas as pd

# folder name -> label, per the layout eval_tbx11k.sbatch's header comment
# already documented for the 500-subset (imgs/tb, imgs/health, imgs/sick).
_CLASS_DIRS = {"tb": "TB", "health": "Normal", "sick": "Normal"}


def _find_imgs_root(tbx_root):
    hits = glob.glob(os.path.join(tbx_root, "**", "imgs"), recursive=True)
    if not hits:
        raise SystemExit(f"no 'imgs' directory found under {tbx_root} -- "
                          f"check the extracted tree / README before retrying")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbx-root", required=True, help="dir the TBX11K zip was extracted into")
    ap.add_argument("--raw-out", required=True, help="flat dir to copy included images into")
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()

    imgs_root = _find_imgs_root(args.tbx_root)
    print(f"using imgs root: {imgs_root}")
    os.makedirs(args.raw_out, exist_ok=True)

    rows = []
    for subdir, label in _CLASS_DIRS.items():
        class_dir = os.path.join(imgs_root, subdir)
        if not os.path.isdir(class_dir):
            print(f"WARNING: {class_dir} not found, skipping")
            continue
        paths = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            shutil.copy2(path, os.path.join(args.raw_out, os.path.basename(path)))
            rows.append(dict(new_id=stem, **{"TB/Normal": label}))
        print(f"{subdir}/ -> {label}: {len(paths)} images")

    if not rows:
        raise SystemExit(f"no images found under {imgs_root}/{{tb,health,sick}} -- "
                          f"check the extracted tree matches the expected layout")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    n_tb = (df["TB/Normal"] == "TB").sum()
    print(f"wrote {args.csv_out}: {len(df)} images ({n_tb} TB, {len(df) - n_tb} Normal)")


if __name__ == "__main__":
    main()
