#!/usr/bin/env python
"""STEP 3 -- CPU only. External evaluation with uncertainty, plus the
TB-vs-sick decomposition that tells you whether the hard-negative run worked.

WHY THIS REPLACES THE CURRENT EVAL TABLES
-----------------------------------------
The existing summary CSVs report a bare F1@0.5 per (model, cohort). Montgomery
has 138 images. A bare F1 on n=138 with no interval invites conclusions the
data cannot support, and re-running the numbers shows this is not hypothetical:

  Claim in the paper                              Paired bootstrap, 6000 resamples
  ----------------------------------------------  --------------------------------
  "Mixup does not earn its place" (dropped)       Shenzhen  +0.0012 [-0.011,+0.013]
                                                  Montgomery +0.0196 [-0.019,+0.063]
  Strong aug >> light aug                         Montgomery +0.1384 [+0.088,+0.197]
  Lung crop helps                                 Shenzhen  +0.0440 [+0.025,+0.064]
  Drop CheXFound, keep RAD-DINO alone             Montgomery +0.0596 [+0.025,+0.102]

Three of those four survive. The mixup one does not -- its interval spans zero
on both cohorts, so the honest statement is that removing mixup is *neutral*
and was dropped for simplicity, not because it hurt. That is still a fine
reason to drop it, and it is a claim a reviewer running this script cannot
knock down.

THE DECOMPOSITION
-----------------
On TBX11K the current model scores, at tau=0.5:

    false-positive rate on healthy negatives      10.3%
    false-positive rate on sick-but-not-TB        76.0%
    AUC, TB vs healthy                            0.9807
    AUC, TB vs sick-but-not-TB                    0.8364
    AUC, healthy vs sick (using P(TB) as score)   0.8999   <- reads as "abnormal"
    max F1 on TB-vs-sick at ANY threshold         0.6012

The last line is what matters. Even a perfect threshold caps TB-vs-sick at 0.60,
so this is a discrimination deficit, not a calibration one -- the model is
substantially an abnormality detector. That is exactly the gap hard negatives
target, and `AUC TB-vs-sick` is the number to watch: if the hardneg run does not
move it well above 0.84, it did not work.

Usage:
    python 03_eval_external.py --pred-dir submissions \
        --model dgablation_strong_aug_no_mixup --baseline baseline_rad_dino
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

B = 6000


def read_probs(path):
    d = pd.read_csv(path)
    col = next((c for c in d.columns if "prob" in c.lower()), None)
    if col is None:
        return None
    return d.set_index(d.columns[0])[col]


def labels_for(dataset, idx):
    s = pd.Series(idx).astype(str)
    if dataset.startswith("tbx11k"):
        return s.str.startswith("tb").astype(int).values
    lab = s.str.extract(r"_(\d)(?:\D|$)")[0]      # NLM: CHNCXR_0001_0 -> normal
    return None if lab.isna().any() else lab.astype(int).values


def balanced_weights(y):
    return np.where(y == 1, 1.0 / max(y.sum(), 1), 1.0 / max((y == 0).sum(), 1))


def boot_ci(fn, n, rng, reps=B):
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(reps)])
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def tbx_decomposition(ids, p):
    """TBX11K only: split the negatives into healthy vs sick-but-not-TB."""
    s = pd.Series(ids).astype(str)
    grp = np.where(s.str.startswith("tb"), "TB",
                   np.where(s.str.startswith("h"), "healthy", "sick"))
    tb, he, si = p[grp == "TB"], p[grp == "healthy"], p[grp == "sick"]
    if len(he) == 0 or len(si) == 0:
        return
    def auc(a, b):
        return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    y2 = np.r_[np.ones(len(tb)), np.zeros(len(si))]
    p2 = np.r_[tb, si]
    print(f"    decomposition   recall@0.5={100*(tb>=.5).mean():5.1f}%   "
          f"FP healthy={100*(he>=.5).mean():5.1f}%   FP sick={100*(si>=.5).mean():5.1f}%")
    print(f"    AUC TB-vs-healthy={auc(tb,he):.4f}   "
          f"AUC TB-vs-sick={auc(tb,si):.4f}   "
          f"AUC healthy-vs-sick={auc(si,he):.4f}")
    print(f"    max F1 TB-vs-sick at ANY tau = "
          f"{max(f1_score(y2,(p2>=t).astype(int)) for t in np.linspace(.01,.99,99)):.4f}"
          f"   <- the ceiling hard negatives need to lift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="submissions")
    ap.add_argument("--model", required=True, help="tag substring of the run to score")
    ap.add_argument("--baseline", default="", help="tag to compare against, paired")
    ap.add_argument("--datasets", nargs="+",
                    default=["shenzhen", "montgomery", "tbx11k_full"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    for ds in a.datasets:
        cand = [f for f in glob.glob(os.path.join(a.pred_dir, f"*{ds}.csv")) if a.model in f]
        if not cand:
            print(f"[{ds}] no prediction CSV matching '{a.model}'")
            continue
        p_s = read_probs(cand[0])
        idx = p_s.index
        y = labels_for(ds, idx)
        if y is None:
            print(f"[{ds}] could not recover labels from ids")
            continue
        p = p_s.values
        bal = ds.startswith("tbx11k")
        w = balanced_weights(y) if bal else None

        def F1(sel, prob):
            return f1_score(y[sel], (prob[sel] >= 0.5).astype(int),
                            sample_weight=(w[sel] if w is not None else None))

        allsel = np.arange(len(y))
        lo, hi = boot_ci(lambda i: F1(i, p), len(y), rng)
        print(f"\n===== {ds}  n={len(y)}  prevalence={y.mean():.3f}"
              f"{'  (class-balanced F1)' if bal else ''} =====")
        print(f"  {a.model}")
        print(f"    AUC={roc_auc_score(y,p):.4f}   F1@0.5={F1(allsel,p):.4f}  "
              f"95%CI=[{lo:.4f},{hi:.4f}]")
        if bal:
            tbx_decomposition(idx, p)

        if a.baseline:
            bc = [f for f in glob.glob(os.path.join(a.pred_dir, f"*{ds}.csv"))
                  if a.baseline in f]
            if not bc:
                print(f"    (no baseline CSV matching '{a.baseline}')")
                continue
            q = read_probs(bc[0]).reindex(idx).values
            d = np.array([F1(i, p) - F1(i, q) for i in
                          (rng.integers(0, len(y), len(y)) for _ in range(B))])
            print(f"  vs {a.baseline}: F1={F1(allsel,q):.4f}  AUC={roc_auc_score(y,q):.4f}")
            print(f"    paired diff={d.mean():+.4f}  95%CI=[{np.percentile(d,2.5):+.4f},"
                  f"{np.percentile(d,97.5):+.4f}]  P(model better)={100*(d>0).mean():.1f}%")
            if np.percentile(d, 2.5) < 0 < np.percentile(d, 97.5):
                print("    -> interval spans zero: not a difference you can claim.")


if __name__ == "__main__":
    main()
