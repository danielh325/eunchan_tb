#!/usr/bin/env python
"""STEP 4 -- CPU only, no GPU, no retraining. Task2 external-validation reanalysis.

FINDING THIS SCRIPT REPRODUCES
------------------------------
task2_paper / internal_validation_results.md report the external-domain gap as
a representation-quality failure:

    "Shenzhen F1@0.5=0.7707, Montgomery F1@0.5=0.8348 ... this external gap is
     substantially larger than the internal-validation number suggests"

Running this script on the submission CSVs already committed in
Task2/submissions/ (they carry `prob_tb`, so no GPU and no retraining is
needed) shows that most of that gap is threshold miscalibration, not
discrimination:

    Montgomery   AUC 0.95-0.98 for every variant,
                 but F1@0.5 spans 0.68-0.90 and F1@tau* spans 0.87-0.93
    Shenzhen     AUC 0.93-0.97, F1@0.5 0.82-0.92, F1@tau* 0.86-0.93
    TBX11K-full  AUC 0.91 (!), F1@0.5 0.32, F1@tau* 0.60
                 prevalence 9.5% (800 TB / 3800 healthy / 3800 sick-non-TB)
                 vs 47.5% in training -- a 5x prior shift

TBX11K in particular is not a model failure. AUC 0.91 is a model that ranks TB
above non-TB perfectly respectably; F1 collapses because a threshold tuned at
47% prevalence is wildly permissive at 9.5% prevalence, and the 3800
sick-but-not-TB negatives are exactly the hard negatives that a permissive
threshold converts into false positives.

This also undercuts the domain-generalisation ablation table: ranking the
variants by AUC gives a different winner than ranking them by F1@0.5 on the
same predictions, so the ablation is partly measuring which run happened to
land its threshold well, not which regulariser generalises.

WHAT TO REPORT INSTEAD
----------------------
  * AUC/AUPRC as the domain-shift metric (threshold-free), and
  * F1 at a prevalence-corrected operating point, using the Saerens-Latinne-
    Decock prior-shift adjustment when the target prevalence is known, or a
    small labelled calibration slice when it is not.

Usage (from Task2_0615/Task2/):
    python ../../kaggle/40_task2_threshold_analysis.py --sub-dir submissions
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

TRAIN_PREVALENCE = 0.4746          # 3681 TB / 7757, from Data/train.csv


def labels_for(dataset, ids):
    """Ground truth recovered from the id naming convention of each source."""
    ids = pd.Series(ids).astype(str)
    if dataset == "tbx11k_full":
        # TBX11K: tb* = TB, h* = healthy, s* = sick-but-not-TB
        return ids.str.startswith("tb").astype(int).values
    if dataset in ("shenzhen", "montgomery"):
        # NLM sets: CHNCXR_0001_0.png -> 0 normal, _1 -> TB
        lab = ids.str.extract(r"_(\d)(?:\D|$)")[0]
        return None if lab.isna().any() else lab.astype(int).values
    return None


def prior_correct(p, pi_src, pi_tgt):
    r = pi_tgt / pi_src
    q = (1 - pi_tgt) / (1 - pi_src)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (r * p) / (r * p + q * (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", default="submissions")
    ap.add_argument("--datasets", nargs="+",
                    default=["shenzhen", "montgomery", "tbx11k_full"])
    a = ap.parse_args()

    grid = np.linspace(0.005, 0.995, 199)
    for ds in a.datasets:
        rows = []
        for f in sorted(glob.glob(os.path.join(a.sub_dir, f"*_{ds}.csv"))):
            d = pd.read_csv(f)
            if "prob_tb" not in d.columns or "new_id" not in d.columns:
                continue
            y = labels_for(ds, d.new_id)
            if y is None:
                continue
            p = d.prob_tb.values
            pi_t = float(y.mean())
            pc = prior_correct(p, TRAIN_PREVALENCE, pi_t)
            best = max((f1_score(y, (p >= t).astype(int)), t) for t in grid)
            rows.append(dict(
                model=os.path.basename(f).replace("submission_", "").replace(f"_{ds}.csv", ""),
                AUC=roc_auc_score(y, p),
                AUPRC=average_precision_score(y, p),
                F1_at_05=f1_score(y, (p >= 0.5).astype(int)),
                F1_prior_corrected=f1_score(y, (pc >= 0.5).astype(int)),
                F1_at_best_tau=best[0],
                tau_star=best[1]))
        if not rows:
            print(f"[{ds}] no usable submission CSVs in {a.sub_dir}")
            continue
        t = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        print(f"\n===== {ds}  (n={len(d)}, prevalence={pi_t:.3f}) =====")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  rank correlation AUC vs F1@0.5: "
              f"{t.AUC.corr(t.F1_at_05, method='spearman'):.3f}"
              "   <- if this is low, the ablation table is ranking thresholds, not methods")


if __name__ == "__main__":
    main()
