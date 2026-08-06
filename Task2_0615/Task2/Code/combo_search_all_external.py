"""Task 2 -- find the ensemble combination that performs best ACROSS ALL
THREE external datasets jointly (Shenzhen, Montgomery, TBX11K-full), not
just the best combo per-dataset (that's what combo_search_test.py already
gives you, run separately per --pred-prefix).

Ranks every non-empty subset of the given tags by MEAN F1@0.5 across the
three datasets -- a combo that's merely good on one dataset and mediocre on
the other two scores worse here than one that's solidly good on all three,
which is the more honest "does this generalize" answer than optimizing any
single external set alone.

Pure pandas/numpy over already-computed prob_tb CSVs -- no GPU needed.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from common import ID_COL, LABEL_COL, LABEL_MAP, detection_accuracy, safe_auc, safe_f1


DATASET_REL_PATHS = {
    "shenzhen": "Data/external/shenzhen.csv",
    "montgomery": "Data/external/montgomery.csv",
    "tbx11k_full": "Data/external/tbx11k/tbx11k_full.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace",
                     help="prefix for DATASET_REL_PATHS -- /workspace inside the docker "
                          "container, or the Task2 project root for a local run")
    ap.add_argument("--preds-dir", default="/workspace")
    ap.add_argument("--pred-prefix", default="submission_ext",
                     help="filename prefix pattern: <prefix>_<dataset>_<tag>.csv")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="/workspace/external_eval_joint_combos.csv")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    # load true labels + prob_tb per (dataset, tag)
    y_true_by_ds = {}
    order_by_ds = {}
    probs = {}  # probs[(dataset, tag)] -> np.array
    missing = []
    for ds, rel_path in DATASET_REL_PATHS.items():
        csv_path = os.path.join(args.data_root, rel_path)
        df = pd.read_csv(csv_path)
        y_true_by_ds[ds] = df[LABEL_COL].astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy()
        order_by_ds[ds] = df[ID_COL].tolist()
        for tag in args.tags:
            path = os.path.join(args.preds_dir, f"{args.pred_prefix}_{ds}_{tag}.csv")
            if not os.path.exists(path):
                missing.append((ds, tag))
                continue
            pdf = pd.read_csv(path).set_index(ID_COL).loc[order_by_ds[ds]]
            probs[(ds, tag)] = pdf["prob_tb"].to_numpy()

    if missing:
        print(f"WARNING: {len(missing)} (dataset,tag) prediction files missing, skipped: {missing}")

    # only keep tags with predictions available on ALL three datasets -- a
    # combo including a tag missing on any dataset can't get a fair joint score
    usable_tags = [t for t in args.tags if all((ds, t) in probs for ds in DATASET_REL_PATHS)]
    dropped = set(args.tags) - set(usable_tags)
    if dropped:
        print(f"dropping tags missing on at least one dataset: {sorted(dropped)}")
    print(f"searching over {len(usable_tags)} tags, {2**len(usable_tags) - 1} combinations")

    rows = []
    for r in range(1, len(usable_tags) + 1):
        for combo in itertools.combinations(usable_tags, r):
            per_ds = {}
            for ds in DATASET_REL_PATHS:
                prob = np.mean([probs[(ds, t)] for t in combo], axis=0)
                pred = (prob >= 0.5).astype(int)
                per_ds[f"{ds}_f1"] = safe_f1(y_true_by_ds[ds], pred)
                per_ds[f"{ds}_acc"] = detection_accuracy(y_true_by_ds[ds], pred)
                per_ds[f"{ds}_auc"] = safe_auc(y_true_by_ds[ds], prob)
            mean_f1 = np.mean([per_ds[f"{ds}_f1"] for ds in DATASET_REL_PATHS])
            min_f1 = np.min([per_ds[f"{ds}_f1"] for ds in DATASET_REL_PATHS])
            rows.append({
                "combo": "|".join(combo),
                "n_models": r,
                "mean_f1": mean_f1,
                "min_f1": min_f1,
                **per_ds,
            })

    out_df = pd.DataFrame(rows).sort_values("mean_f1", ascending=False)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} combinations -> {args.out}")
    print(f"\n=== TOP {args.top_n} BY MEAN F1@0.5 ACROSS ALL 3 EXTERNAL DATASETS ===")
    cols = ["combo", "n_models", "mean_f1", "min_f1", "shenzhen_f1", "montgomery_f1", "tbx11k_full_f1"]
    print(out_df[cols].head(args.top_n).to_string(index=False))

    print(f"\n=== TOP {min(10, args.top_n)} BY WORST-CASE (MIN) F1@0.5 ACROSS THE 3 DATASETS ===")
    print(out_df.sort_values("min_f1", ascending=False)[cols].head(min(10, args.top_n)).to_string(index=False))


if __name__ == "__main__":
    main()
