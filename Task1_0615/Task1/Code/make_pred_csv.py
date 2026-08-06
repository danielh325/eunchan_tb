#!/usr/bin/env python
"""Adapter: submission_*.csv (our_id, prob_cavity, pred_cavity, cavity_true) ->
the our_id,cavity format evaluate.py's read_cavity_csv expects.

Usage:
    python make_pred_csv.py ../submissions/submission_densenet201.csv pred_densenet201.csv
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
out = df[["our_id"]].copy()
out["cavity"] = df["pred_cavity"].astype(int)
out.to_csv(dst, index=False)
print(f"wrote {dst} ({len(out)} rows)")
