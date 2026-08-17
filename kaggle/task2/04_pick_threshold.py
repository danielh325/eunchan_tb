#!/usr/bin/env python
"""STEP 4 -- CPU only, seconds to run. Choose the threshold baked into the container.

WHY THIS IS NOT OPTIONAL
------------------------
The submission contract writes a BINARY column:

    /output/prediction.csv  ->  filename,TB/Normal

(`submissions/submission_predict_task2.py` line 80). So whatever probabilities
the model produces, the graded object is a hard label, and tau is a decision
you own. It is currently 0.5 by default -- a value that was never chosen, only
inherited.

Under F1 grading across countries with unknown prevalence and unknown negative
composition, tau is worth more than any 3-day modelling change. Measured on the
current model, moving 0.5 -> 0.80:

    balanced country, mixed negatives      F1 0.8031 -> 0.8231
    30% prevalence, mixed negatives        F1 0.6474 -> 0.7043
    10% prevalence, healthy negatives      F1 0.6660 -> 0.8369
    Montgomery (real)                      F1 0.8976 -> 0.9204
    Shenzhen (real)                        F1 0.8966 -> 0.8762   <- the one loss

WHAT IT DOES
------------
Builds synthetic "countries" by resampling the real cohorts at a specified TB
prevalence and a specified healthy/sick mix of negatives, then reports F1 for
each candidate tau under:

  * worst-case over scenarios (minimax -- very conservative, drags tau up high)
  * scenario-weighted expectation (recommended: put weight where you actually
    believe the 14 countries sit)

The sick/healthy negative axis comes from TBX11K, which is the only cohort here
that labels its negatives by type: 3,800 healthy and 3,800 sick-but-not-TB.

HONEST LIMIT
------------
No tau reaches 0.90 F1 in a low-prevalence country whose negatives carry other
lung pathology. That is not a tuning failure, it is the AUC: TB-vs-sick-negative
discrimination is 0.8364, so even an oracle threshold caps that population at
about 0.60 F1. Threshold choice buys real points in the mid-range scenarios and
cannot rescue the hard one -- only the hard-negative retrain moves that.

Usage:
    python 04_pick_threshold.py --pred-dir submissions --model dgablation_strong_aug_no_mixup
    python 04_pick_threshold.py --prevalence 0.4 0.5 0.6 --sick-frac 0.3 0.5
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def read(path):
    d = pd.read_csv(path)
    col = next(c for c in d.columns if "prob" in c.lower())
    return d.set_index(d.columns[0])[col]


def f1_counts(pos, neg, t):
    tp = int((pos >= t).sum())
    fp = int((neg >= t).sum())
    fn = int((pos < t).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="submissions")
    ap.add_argument("--model", default="dgablation_strong_aug_no_mixup")
    ap.add_argument("--prevalence", nargs="+", type=float, default=[0.10, 0.30, 0.50])
    ap.add_argument("--sick-frac", nargs="+", type=float, default=[0.0, 0.5, 1.0],
                    help="fraction of NEGATIVES that are sick-but-not-TB")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="one weight per (prevalence x sick-frac) cell, row-major; "
                         "default uniform. Put mass where you think the 14 countries are.")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # TBX11K is the only cohort labelling negatives by type -- it supplies the
    # healthy vs sick pools the simulation draws from.
    tbx = [f for f in glob.glob(os.path.join(a.pred_dir, "*tbx11k_full.csv")) if a.model in f]
    if not tbx:
        raise SystemExit(f"need a TBX11K prediction CSV matching '{a.model}' in {a.pred_dir}")
    T = read(tbx[0])
    ids = pd.Series(T.index).astype(str)
    g = np.where(ids.str.startswith("tb"), "TB",
                 np.where(ids.str.startswith("h"), "H", "S"))
    tb, he, si = T.values[g == "TB"], T.values[g == "H"], T.values[g == "S"]
    print(f"pools: TB={len(tb)} healthy={len(he)} sick={len(si)}")

    real = {}
    for ds in ["shenzhen", "montgomery"]:
        c = [f for f in glob.glob(os.path.join(a.pred_dir, f"*{ds}.csv")) if a.model in f]
        if not c:
            continue
        p = read(c[0])
        y = pd.Series(p.index).astype(str).str.extract(r"_(\d)(?:\D|$)")[0]
        if y.isna().any():
            continue
        real[ds] = (y.astype(int).values, p.values)

    cells = [(pi, sf) for pi in a.prevalence for sf in a.sick_frac]
    w = np.array(a.weights, float) if a.weights else np.ones(len(cells))
    if len(w) != len(cells):
        raise SystemExit(f"--weights needs {len(cells)} values (got {len(w)})")
    w = w / w.sum()

    def sim(t, pi, sf):
        out = []
        for _ in range(a.reps):
            npos = max(int(a.n * pi), 5)
            nneg = a.n - npos
            ns = int(nneg * sf)
            pos = rng.choice(tb, npos)
            neg = np.r_[rng.choice(si, ns), rng.choice(he, nneg - ns)]
            out.append(f1_counts(pos, neg, t))
        return float(np.mean(out))

    taus = np.round(np.arange(0.35, 0.96, 0.05), 2)
    print(f"\n{'tau':>5} " + " ".join(f"{k[:9]:>9s}" for k in real) +
          " | " + " ".join(f"p{int(pi*100):02d}s{int(sf*100):03d}" for pi, sf in cells) +
          " |   WORST   WEIGHTED")
    rows = []
    for t in taus:
        r = [f1_score(y, (p >= t).astype(int)) for y, p in real.values()]
        s = [sim(t, pi, sf) for pi, sf in cells]
        worst = min(r + s)
        wt = float(np.dot(w, s))
        rows.append((t, worst, wt))
        print(f"{t:5.2f} " + " ".join(f"{x:9.4f}" for x in r) + " | " +
              " ".join(f"{x:8.4f}" for x in s) + f" | {worst:7.4f}  {wt:8.4f}")

    bw = max(rows, key=lambda x: x[1])
    be = max(rows, key=lambda x: x[2])
    print(f"\n  minimax tau            = {bw[0]:.2f}  (worst-case F1 {bw[1]:.4f})")
    print(f"  scenario-weighted tau  = {be[0]:.2f}  (expected F1 {be[2]:.4f})   <- recommended")
    print("\n  Set it in submissions/submission_predict_task2.py before building the image.")
    if bw[1] < 0.90:
        print(f"\n  NOTE: worst-case F1 is {bw[1]:.4f} at ANY tau. If the private set includes a\n"
              f"  low-prevalence country with clinic-type negatives, 0.90 is not reachable by\n"
              f"  thresholding -- that ceiling is AUC 0.8364 on TB-vs-sick-negative, and only\n"
              f"  the hard-negative retrain moves it.")


if __name__ == "__main__":
    main()
