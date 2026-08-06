#!/usr/bin/env python
"""Copy only the test-set case masks out of nnUNet_raw/labelsTr (which holds
GT masks for all 555 train+test images combined) into a filtered directory,
so evaluate.py's --gt-mask-dir only iterates over the 111 test cases that
actually have a matching prediction.

Usage:
    python filter_gt_masks.py ../Data/test.csv \
        ../../../Code/baseline/nnUNetv2/nnUNet_data/nnUNet_raw/Dataset001_Task1/labelsTr \
        /tmp/gt_masks_test
"""
import os
import shutil
import sys

import pandas as pd

test_csv, src_dir, dst_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst_dir, exist_ok=True)

ids = pd.read_csv(test_csv)["our_id"].astype(str).tolist()
missing, copied = [], 0
for i in ids:
    src = os.path.join(src_dir, f"{i}.nii.gz")
    if not os.path.exists(src):
        missing.append(i)
        continue
    shutil.copy(src, os.path.join(dst_dir, f"{i}.nii.gz"))
    copied += 1

print(f"copied {copied}/{len(ids)} test-case GT masks to {dst_dir}")
if missing:
    print(f"missing GT masks for: {missing}")
