#!/usr/bin/env python
"""Copy just the 111 test-case images out of nnUNet_raw/imagesTr (which holds
all 555 train+test images combined, 3 channel files each: _0000/_0001/_0002)
into a filtered folder -- the input nnUNetv2_predict needs.

Usage:
    python filter_test_images.py ../Data/test.csv \
        <nnUNet_raw>/Dataset001_Task1/imagesTr \
        /tmp/test_images_for_predict
"""
import glob
import os
import shutil
import sys

import pandas as pd

test_csv, src_dir, dst_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst_dir, exist_ok=True)

ids = pd.read_csv(test_csv)["our_id"].astype(str).tolist()
missing, copied = [], 0
for i in ids:
    chans = sorted(glob.glob(os.path.join(src_dir, f"{i}_*.nii.gz")))
    if not chans:
        missing.append(i)
        continue
    for c in chans:
        shutil.copy(c, os.path.join(dst_dir, os.path.basename(c)))
    copied += 1

print(f"copied {copied}/{len(ids)} test-case image sets to {dst_dir}")
if missing:
    print(f"missing images for: {missing}")
