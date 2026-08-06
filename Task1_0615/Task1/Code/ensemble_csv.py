#!/usr/bin/env python
"""CSV-level ensembling across ANY already-produced submission_*.csv files,
regardless of which pipeline trained them (OrdFused vs plain CNNDetector
baselines use different checkpoint formats, so predict_ordfused.py's own
--ensemble can't mix them -- this sidesteps that entirely by averaging the
already-computed prob_cavity column, no model loading needed).

Motivated by the evaluate_binary.py results: ordfused_chexfound_vitl16
(best accuracy/F1/sensitivity, 52 TP/6 FN) and densenet201 (best AUC/
specificity, 45 TP/13 FN but only 6 FP vs chexfound's 10) have visibly
complementary error patterns -- worth testing whether averaging their
probabilities beats both on accuracy AND AUC simultaneously, which nothing
in the current lineup does alone.

Usage:
    python ensemble_csv.py submission_ordfused_chexfound_vitl16.csv submission_densenet201.csv --name top2
    python ensemble_csv.py submission_*.csv --name all
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def sweep_tau_accuracy(y_true, prob, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 181)
    best = max((accuracy_score(y_true, (prob >= t).astype(int)), t) for t in grid)
    return float(best[1]), float(best[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="submission_*.csv files to average (equal weight)")
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "submissions"))
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.csvs]
    base_ids = dfs[0][dfs[0].columns[0]]  # id column, whatever it's named
    for p, df in zip(args.csvs[1:], dfs[1:]):
        if len(df) != len(dfs[0]) or not (df[df.columns[0]] == base_ids).all():
            raise SystemExit(f"row order/id mismatch between {args.csvs[0]} and {p} -- "
                              f"can't average positionally, they must be the same test set in the same order")
        if "cavity_true" in df.columns and "cavity_true" in dfs[0].columns:
            if not (df["cavity_true"] == dfs[0]["cavity_true"]).all():
                raise SystemExit(f"cavity_true mismatch between {args.csvs[0]} and {p} -- not the same test set")

    prob = np.mean([df["prob_cavity"].to_numpy() for df in dfs], axis=0)
    out = dfs[0][[dfs[0].columns[0]]].copy()
    out["prob_cavity"] = prob

    if "cavity_true" in dfs[0].columns:
        y_true = (dfs[0]["cavity_true"] != "none").astype(int).to_numpy()
        tau, _ = sweep_tau_accuracy(y_true, prob)
        pred = (prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_true, pred)
        sens = recall_score(y_true, pred, zero_division=0)
        spec = tn / max(tn + fp, 1)
        f1 = f1_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, prob)
        except ValueError:
            auc = float("nan")

        print(f"\n=== ensemble of {len(dfs)} model(s): {', '.join(os.path.basename(p) for p in args.csvs)} ===")
        print(f"tau*={tau:.3f}  accuracy={acc:.4f}  precision={prec:.4f}  sensitivity={sens:.4f}  "
              f"specificity={spec:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        print(f"tp={tp} tn={tn} fp={fp} fn={fn}")

        out["cavity_true"] = dfs[0]["cavity_true"]
        out["pred_cavity"] = pred
    else:
        print(f"no cavity_true column in {args.csvs[0]} -- wrote probabilities only, no scoring")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"submission_{args.name}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
