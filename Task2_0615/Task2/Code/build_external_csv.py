"""Builds a train.csv-compatible CSV from the Shenzhen/Montgomery public TB
CXR datasets, for external-validation (genuine domain generalization, not
just the internal leave-one-modality-out proxy). Same two datasets your own
SSRN paper (Domain Generalization of Tuberculosis Detection in Chest
X-Rays through MixStyle and Multi-Level Augmentation) validated on, so this
gives a directly comparable number: 66%->89% Shenzhen with plain
DenseNet-121+MixStyle+aug -- this Task2 build's real bar to clear or beat.

Filename convention (Jaeger et al. 2014, the datasets' original paper):
  {PREFIX}CXR_####_X.png, X=0 -> Normal, X=1 -> abnormal (TB-consistent)
  Shenzhen prefix: CHN. Montgomery prefix: MCU.

Usage:
    python build_external_csv.py --image-dir /path/to/ChinaSet_AllFiles/CXR_png \
        --out shenzhen.csv
    python build_external_csv.py --image-dir /path/to/MontgomerySet/CXR_png \
        --out montgomery.csv

Output columns match common.py's expectations (ID_COL, LABEL_COL) --
Modality_DICOM is deliberately NOT fabricated: predict_task2.py/
predict_dual_stream.py already fall back to a placeholder "unknown" value
when that column is absent (TBDataset reads it but only for domain-aware-
mixup bookkeeping, which never runs at inference anyway), so there's no
need to invent a fake domain label for external data that doesn't have one.
"""
import argparse
import glob
import os
import re

import pandas as pd

_PATTERN = re.compile(r"^(CHN|MCU)CXR_\d+_([01])$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True, help="dir of {PREFIX}CXR_####_X.png files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    skipped = []
    for path in sorted(glob.glob(os.path.join(args.image_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = _PATTERN.match(stem)
        if not m:
            skipped.append(stem)
            continue
        label = "TB" if m.group(2) == "1" else "Normal"
        rows.append(dict(new_id=stem, **{"TB/Normal": label}))

    if skipped:
        print(f"WARNING: {len(skipped)} file(s) didn't match the expected "
              f"{{PREFIX}}CXR_####_X.png pattern, skipped: {skipped[:5]}"
              f"{' ...' if len(skipped) > 5 else ''}")
    if not rows:
        raise SystemExit(f"no matching files found under {args.image_dir} -- "
                          f"check --image-dir points at the CXR_png subfolder, not the zip root")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    n_tb = (df["TB/Normal"] == "TB").sum()
    print(f"wrote {args.out}: {len(df)} images ({n_tb} TB, {len(df) - n_tb} Normal)")


if __name__ == "__main__":
    main()
