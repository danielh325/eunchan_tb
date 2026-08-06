"""Task 2 -- brute-force ensemble-combination search over Data/test.csv,
across ALL individually-scored models from ALL THREE mask-usage variants
(baseline / mask-channel-ablation / mask-guided-attention), not just one
checkpoint dir's encoders like eval_test_combinations.sbatch.

Each input CSV (one per --preds tag) already holds a per-image prob_tb
column written by predict_task2.py --encoders <single-encoder> --ckpt-dir
<variant-dir> (that invocation is the only step that needs the GPU/model
forward pass, done once per model up front by
eval_test_all_variants.sbatch's Stage 1). Combining already-computed
probabilities is pure numpy averaging -- no GPU needed for anything in this
script, only pandas/numpy/sklearn (via common.py's metric helpers).

With N tags this is 2**N - 1 non-empty subsets; kept fast because each
subset costs only a mean() + a few metric calls over ~1941 rows.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from common import ID_COL, LABEL_COL, LABEL_MAP, detection_accuracy, safe_auc, safe_f1, sweep_tau_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", default="/workspace/Data/test.csv")
    ap.add_argument("--preds-dir", default="/workspace",
                     help="dir containing <pred-prefix>_<tag>.csv files")
    ap.add_argument("--pred-prefix", default="submission_test",
                     help="filename prefix before _<tag>.csv -- e.g. 'submission_test' for "
                          "Data/test.csv, or 'submission_ext_shenzhen' for an external dataset, "
                          "so the same script works across test.csv and every external dataset "
                          "without filename collisions")
    ap.add_argument("--tags", nargs="+", required=True,
                     help="tags matching <pred-prefix>_<tag>.csv filenames, "
                          "e.g. baseline_densenet121 maskch_rad_dino maskguided_swin_tiny")
    ap.add_argument("--max-combo-size", type=int, default=None,
                     help="cap subset size (default: no cap, i.e. up to len(tags)) "
                          "-- use this to bound runtime/output size if tags is large")
    ap.add_argument("--out", default="/workspace/test_eval_all_variants_combos.csv")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_csv)
    y_true = test_df[LABEL_COL].astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy()
    order = test_df[ID_COL].tolist()

    probs = {}
    missing = []
    for tag in args.tags:
        path = os.path.join(args.preds_dir, f"{args.pred_prefix}_{tag}.csv")
        if not os.path.exists(path):
            missing.append(tag)
            continue
        pdf = pd.read_csv(path).set_index(ID_COL).loc[order]
        probs[tag] = pdf["prob_tb"].to_numpy()

    if missing:
        print(f"WARNING: {len(missing)} tag(s) missing prediction CSVs, skipping them "
              f"(combos will only be searched over the remaining {len(probs)}): {missing}")
    tags = list(probs.keys())
    if not tags:
        raise SystemExit("no tags with prediction CSVs found -- nothing to search")

    max_size = args.max_combo_size or len(tags)
    rows = []
    for r in range(1, max_size + 1):
        for combo in itertools.combinations(tags, r):
            prob = np.mean([probs[t] for t in combo], axis=0)
            pred = (prob >= 0.5).astype(int)
            acc = detection_accuracy(y_true, pred)
            auc = safe_auc(y_true, prob)
            f1 = safe_f1(y_true, pred)
            tau_star, acc_tau = sweep_tau_accuracy(y_true, prob)
            rows.append({
                "combo": "|".join(combo),
                "n_models": r,
                "accuracy@0.5": acc,
                "auc": auc,
                "f1@0.5": f1,
                "best_tau": tau_star,
                "accuracy@best_tau": acc_tau,
            })

    out_df = pd.DataFrame(rows).sort_values("f1@0.5", ascending=False)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} combinations -> {args.out}")
    print(f"\n=== TOP {args.top_n} BY F1@0.5 ===")
    print(out_df.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
