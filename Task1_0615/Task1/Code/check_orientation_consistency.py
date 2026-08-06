#!/usr/bin/env python
"""Check whether the 90-degree rotation seen in check_mask_alignment.py's
overlays is a UNIFORM property of the whole dataset (cosmetically wrong but
harmless to training) or VARIES case-to-case (which would actively break
nnU-Net's learning, since the model needs consistent image/mask geometry
across training examples, not consistent-with-reality geometry).

For each case, prints:
  - the raw NIfTI header (Direction, Origin, Spacing) for the image and mask
    -- if Direction matrices differ across cases, orientation handling was
       not applied consistently during the DICOM/PNG -> NIfTI conversion.
  - the case's series_modality_cd (from test.csv/train.csv) -- checks
    whether inconsistent orientation correlates with acquisition modality
    (XC vs CR etc.), a plausible root cause if the conversion pipeline
    didn't normalize orientation per modality.
  - saves an overlay PNG (same raw, uncorrected dump as check_mask_alignment.py)
    labeled with case id / Dice / modality, so cases can be compared side by
    side.

Auto-selects a comparison set: the worst cases (total misses: n_pred=0 with
n_ref>0, i.e. real cavity predicted as nothing) and the best cases (highest
Dice) from nnU-Net's own fold_0 validation summary.json -- no need to type
out case IDs by hand.

Usage (same environment as check_mask_alignment.py -- CPU only):
    python check_orientation_consistency.py
    python check_orientation_consistency.py --n-worst 6 --n-best 4
    python check_orientation_consistency.py --case-id 62 86 11   # override auto-selection
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NNUNET_ROOT = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data"))
IMAGES_DIR = os.path.join(NNUNET_ROOT, "nnUNet_raw", "Dataset001_Task1", "imagesTr")
LABELS_DIR = os.path.join(NNUNET_ROOT, "nnUNet_raw", "Dataset001_Task1", "labelsTr")
SUMMARY_JSON = os.path.join(
    NNUNET_ROOT, "nnUNet_results", "Dataset001_Task1",
    "nnUNetTrainer__nnUNetPlans__2d", "fold_0", "validation", "summary.json")
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))


def case_id_from_pred_path(path):
    return os.path.basename(path).replace(".nii.gz", "")


def load_summary():
    with open(SUMMARY_JSON) as f:
        data = json.load(f)
    rows = []
    for entry in data["metric_per_case"]:
        m = entry["metrics"]["1"]
        rows.append({
            "case_id": case_id_from_pred_path(entry["prediction_file"]),
            "dice": m["Dice"],
            "n_pred": m["n_pred"],
            "n_ref": m["n_ref"],
        })
    return pd.DataFrame(rows)


def pick_cases(df, n_worst, n_best):
    misses = df[(df["n_pred"] == 0) & (df["n_ref"] > 0)].sort_values("n_ref", ascending=False)
    worst = misses.head(n_worst)
    best = df.sort_values("dice", ascending=False).head(n_best)
    picked = pd.concat([worst, best]).drop_duplicates(subset="case_id")
    return picked


def load_modality_lookup():
    lookup = {}
    for split in ("test.csv", "train.csv"):
        path = os.path.join(DEFAULT_DATA_DIR, split)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "series_modality_cd" in df.columns:
                for our_id, modality in zip(df["our_id"].astype(str), df["series_modality_cd"]):
                    lookup.setdefault(our_id, modality)
    return lookup


def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def inspect_case(case_id, dice, modality, out_dir):
    img_path = os.path.join(IMAGES_DIR, f"{case_id}_0000.nii.gz")
    mask_path = os.path.join(LABELS_DIR, f"{case_id}.nii.gz")
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"!! case {case_id}: image or mask missing, skipping")
        return

    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)

    print(f"\n=== case {case_id}  (dice={dice:.3f}, modality={modality}) ===")
    print(f"  image: size={img_sitk.GetSize()} spacing={img_sitk.GetSpacing()} "
          f"origin={img_sitk.GetOrigin()} direction={img_sitk.GetDirection()}")
    print(f"  mask : size={mask_sitk.GetSize()} spacing={mask_sitk.GetSpacing()} "
          f"origin={mask_sitk.GetOrigin()} direction={mask_sitk.GetDirection()}")
    if img_sitk.GetDirection() != mask_sitk.GetDirection():
        print(f"  !! image/mask DIRECTION MISMATCH for this case -- image and "
              f"mask headers disagree on orientation.")

    img = np.squeeze(sitk.GetArrayFromImage(img_sitk))
    mask = np.squeeze(sitk.GetArrayFromImage(mask_sitk)) > 0

    base = normalize_to_uint8(img)
    rgb = np.stack([base, base, base], axis=-1)
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    rgb = np.where(mask[..., None], (0.4 * rgb + 0.6 * red).astype(np.uint8), rgb)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"dice{dice:.2f}_mod{modality}".replace(" ", "")
    out_path = os.path.join(out_dir, f"case{case_id}_{tag}.png")
    Image.fromarray(rgb).save(out_path)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-id", nargs="+", default=None,
                    help="override auto-selection with explicit case IDs")
    ap.add_argument("--n-worst", type=int, default=6,
                    help="number of total-miss cases (n_pred=0, n_ref>0) to include, "
                         "largest n_ref first")
    ap.add_argument("--n-best", type=int, default=4,
                    help="number of highest-Dice cases to include, for comparison")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "orientation_check"))
    args = ap.parse_args()

    modality_lookup = load_modality_lookup()
    df = load_summary()

    if args.case_id:
        picked = df[df["case_id"].isin(args.case_id)]
    else:
        picked = pick_cases(df, args.n_worst, args.n_best)

    n_misses = int(((picked["n_pred"] == 0) & (picked["n_ref"] > 0)).sum())
    print(f"checking {len(picked)} cases ({n_misses} misses, "
          f"rest are comparison/high-Dice cases)")

    for _, row in picked.sort_values("dice").iterrows():
        modality = modality_lookup.get(str(row["case_id"]), "UNKNOWN")
        inspect_case(row["case_id"], row["dice"], modality, args.out_dir)


if __name__ == "__main__":
    main()
