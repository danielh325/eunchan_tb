#!/usr/bin/env python
"""Binary cavity-presence evaluation for every submission_*.csv we've produced.

The competition's own challenge_score (0.7*acc + 0.3*dice) is what training
optimizes tau against, but "how good is this at plain binary cavity
detection" is its own question worth seeing in isolation: accuracy,
precision/recall, specificity, F1, and AUC, side by side across every model
we've trained so far.

Reads every submissions/submission_*.csv (each already has prob_cavity,
pred_cavity, and cavity_true from predict.py/predict_ordfused.py) and
binarizes cavity_true as (grade != "none").

Usage:
    python evaluate_binary.py
    python evaluate_binary.py --submissions-dir ../submissions
"""
import argparse
import glob
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


def evaluate_one(path):
    df = pd.read_csv(path)
    if "cavity_true" not in df.columns:
        return None  # no ground truth in this file (e.g. a pure inference-only run)

    y_true = (df["cavity_true"] != "none").astype(int).to_numpy()
    y_pred = df["pred_cavity"].astype(int).to_numpy()
    y_prob = df["prob_cavity"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # = recall on positive class
    specificity = tn / max(tn + fp, 1)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in y_true

    return {
        "model": os.path.basename(path).replace("submission_", "").replace(".csv", ""),
        "n": len(df),
        "n_pos": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submissions-dir", default=os.path.join(HERE, "..", "submissions"))
    ap.add_argument("--sort-by", default="accuracy",
                    choices=["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"])
    ap.add_argument("--out-csv", default=None, help="optionally write the results table to this CSV")
    ap.add_argument("--baseline", default=None,
                    help="model name (as it appears in the table) to diff every other row against, "
                         "e.g. --baseline densenet201")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.submissions_dir, "submission_*.csv")))
    if not paths:
        print(f"no submission_*.csv found under {args.submissions_dir}")
        return

    rows = []
    for p in paths:
        r = evaluate_one(p)
        if r is None:
            print(f"skipping {os.path.basename(p)} -- no cavity_true column (no ground truth)")
            continue
        rows.append(r)

    if not rows:
        print("no submissions had ground truth to evaluate against.")
        return

    results = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\nBinary cavity-presence evaluation ({len(results)} models, "
          f"n={results['n'].iloc[0]} test images, {results['n_pos'].iloc[0]} positive)\n")
    print(results[["model", "accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]]
          .to_string(index=False))
    print("\nconfusion counts (tp/tn/fp/fn):")
    print(results[["model", "tp", "tn", "fp", "fn"]].to_string(index=False))

    if args.baseline:
        base_rows = results[results["model"] == args.baseline]
        if base_rows.empty:
            print(f"\n!! --baseline {args.baseline!r} not found among evaluated models")
        else:
            b = base_rows.iloc[0]
            print(f"\nvs. baseline '{args.baseline}' (acc={b['accuracy']:.4f}, auc={b['auc']:.4f}):")
            diff = results.copy()
            diff["d_accuracy"] = diff["accuracy"] - b["accuracy"]
            diff["d_auc"] = diff["auc"] - b["auc"]
            diff["beats_baseline"] = (diff["d_accuracy"] > 0) & (diff["d_auc"] > 0)
            print(diff[["model", "d_accuracy", "d_auc", "beats_baseline"]].to_string(index=False))

    if args.out_csv:
        results.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
