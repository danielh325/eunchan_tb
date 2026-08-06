"""Step 1 diagnostic (Task1 improvement plan, see ~/.claude/plans/lovely-
meandering-lecun.md): where do the ensemble's errors actually live, and how
tightly does cavity mask area track the ordinal severity label? Both are
read-only analyses -- no training, no GPU needed -- that gate whether Step 6
(area-mediated model) is worth building at all.

Usage:
    python analyze_errors.py
    python analyze_errors.py --oof densenet201:oof/densenet201_oof.csv \\
        --oof tf_efficientnetv2_s:oof/tf_efficientnetv2_s_oof.csv \\
        --oof resnet50d:oof/resnet50d_oof.csv \\
        --oof eva_x_base_notab:oof/eva_x_base_notab_oof.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GRADE2ORD, ID_COL, find_file, sitk_read, sweep_tau_accuracy
from ensemble_submissions import _parse_tag_path_pairs, weighted_merge

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))

# Matches build_valid_ensemble.sbatch's actual member set (uniform weights,
# the shipped submission_valid_ensemble_v1.csv) -- override via --oof if
# analyzing a different combination.
DEFAULT_OOF = {
    "densenet201": "oof/densenet201_oof.csv",
    "tf_efficientnetv2_s": "oof/tf_efficientnetv2_s_oof.csv",
    "resnet50d": "oof/resnet50d_oof.csv",
    "eva_x_base_notab": "oof/eva_x_base_notab_oof.csv",
}


def error_breakdown(args):
    oof_paths = _parse_tag_path_pairs(args.oof) if args.oof else DEFAULT_OOF
    oof_frames = {tag: pd.read_csv(path) for tag, path in oof_paths.items()}

    df_tr = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    y = (df_tr.cavity != "none").astype(int)
    y_by_id = dict(zip(df_tr[ID_COL], y))
    grade_by_id = dict(zip(df_tr[ID_COL], df_tr.cavity))

    combined = weighted_merge(oof_frames, "oof_prob", {t: 1.0 for t in oof_frames})
    combined = combined[combined.index.isin(y_by_id)]
    y_true = np.array([y_by_id[i] for i in combined.index])
    tau, oof_acc = sweep_tau_accuracy(y_true, combined.to_numpy())
    y_pred = (combined.to_numpy() >= tau).astype(int)

    print(f"Combined OOF: n={len(combined)}  tau*={tau:.3f}  accuracy={oof_acc:.4f}\n")

    grades = ["none", "small", "medium", "large"]
    rows = []
    for g in grades:
        ids = [i for i in combined.index if grade_by_id[i] == g]
        if not ids:
            continue
        idx = [combined.index.get_loc(i) for i in ids]
        yt = y_true[idx]
        yp = y_pred[idx]
        n = len(ids)
        wrong = int((yt != yp).sum())
        direction = "false-positive (predicted cavity, none present)" if g == "none" \
            else "false-negative (missed the cavity)"
        rows.append({"grade": g, "n": n, "wrong": wrong,
                      "error_rate": wrong / n if n else float("nan"), "error_type": direction})
    err_df = pd.DataFrame(rows)
    print("OOF error breakdown by severity grade:")
    print(err_df.to_string(index=False))
    print()
    return err_df


def mask_area_correlation(args):
    df_tr = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    mask_dir = os.path.join(args.data_dir, "train", "CXR_label")

    areas, grades_ord = [], []
    missing = 0
    for _, row in df_tr.iterrows():
        _id = row[ID_COL]
        mp = find_file(mask_dir, _id, ".nii.gz")
        if mp is None or not os.path.exists(mp):
            missing += 1
            continue
        mask = sitk_read(mp)
        areas.append(float((mask > 0).sum()))
        grades_ord.append(GRADE2ORD[row.cavity])

    if missing:
        print(f"WARNING: {missing} train images had no mask file under {mask_dir}, skipped")

    rho, pval = spearmanr(areas, grades_ord)
    print(f"\nMask-area vs. ordinal-grade correlation (n={len(areas)}):")
    print(f"  Spearman rho = {rho:.4f}  (p = {pval:.2e})")
    if rho > 0.8:
        print("  -> strong: grade is close to a discretization of mask area. "
              "Area-mediated ordinal head (Step 6) is well-motivated.")
    elif rho > 0.5:
        print("  -> moderate: area carries real signal about grade, but grade "
              "isn't purely a function of area -- Step 6 may still help but "
              "expect it to be an imperfect mediator.")
    else:
        print("  -> weak: grade is not well explained by mask area alone. "
              "Reconsider Step 6 before investing in it.")
    return rho, pval


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", action="append", default=None, metavar="TAG:PATH",
                    help="repeatable; TAG:PATH to a --dump-oof CSV. Defaults to the "
                         "valid_ensemble_v1 member set if omitted.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    error_breakdown(args)
    mask_area_correlation(args)


if __name__ == "__main__":
    main()
