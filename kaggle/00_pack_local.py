#!/usr/bin/env python
"""STEP 0 -- run this on the Mac, NOT on Kaggle.

Packs the raw data (4.0G of DICOM for Task1 + 1.0G of PNG for Task2) into a
compact upload bundle so you can put it in a Kaggle Dataset without waiting on
a 5 GB upload. Output is ~250 MB for Task1 and ~600 MB for Task2.

What it does, and why each choice:

Task1
  * DICOM -> uint8 PNG at a fixed 1024x1024. Percentile clip (0.5/99.5) +
    min-max, exactly the preamble common.py::_clahe_u8 already uses, so nothing
    about the intensity pipeline changes. CLAHE is deliberately NOT baked in --
    it stays on-the-fly so you can still ablate it on Kaggle.
  * Masks come from Data/train/CXR_label (train) and Data/CXR_label (test).
    Verified locally: these are stored at NATIVE resolution and their shapes
    match their images exactly, for all 555 cases. They need no transpose and
    no resampling-aware handling. Do NOT use the nnUNet_raw/labelsTr copies --
    those are the fixed-512 resampled store that caused the whole
    transpose/no-transpose confusion between task1_paper.tex and
    train_segmentation.py's docstring.
  * native_h / native_w are recorded in meta.csv so predictions can be mapped
    back to native resolution for official-metric scoring.

Task2
  * PNG -> JPEG q92 at 512. Only used as *pretraining* data for the Task1
    cavity classifier (7757 labelled same-cohort CXRs vs Task1's 444), so mild
    JPEG loss is irrelevant; it cuts the upload roughly in half.

Usage:
    python kaggle/00_pack_local.py \
        --task1-data ../Task1_0615/Task1/Data \
        --task2-data ../Task2_0615/Task2/Data \
        --out ~/mmtb_kaggle_pack
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

T1_SIZE = 1024
T2_SIZE = 512


def sitk_read(path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
    return arr.astype(np.float32)


def to_u8(arr, lo=0.5, hi=99.5):
    a, b = np.percentile(arr, lo), np.percentile(arr, hi)
    arr = np.clip(arr, a, b)
    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    return (arr * 255).astype(np.uint8)


def pack_task1(data_dir, out_dir):
    img_out = os.path.join(out_dir, "task1", "img")
    msk_out = os.path.join(out_dir, "task1", "mask")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    rows = []
    for split, img_dir, msk_dir in [
        ("train", f"{data_dir}/train/CXR", f"{data_dir}/train/CXR_label"),
        ("test", f"{data_dir}/test/CXR", f"{data_dir}/CXR_label"),
    ]:
        df = pd.read_csv(f"{data_dir}/{split}.csv")
        for n, row in enumerate(df.itertuples(), 1):
            _id = row.our_id
            raw = sitk_read(f"{img_dir}/{_id}.dcm")
            h, w = raw.shape
            u8 = cv2.resize(to_u8(raw), (T1_SIZE, T1_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{img_out}/{_id}.png", u8)

            m = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(f"{msk_dir}/{_id}.nii.gz")))
            m = (m > 0).astype(np.uint8)
            if m.shape != (h, w):
                # Audited across all 555 cases: the ONLY masks whose shape
                # disagrees with their image are the 247+53 cavity-negative
                # placeholders, stored as fixed 512x512 all-black arrays (this
                # is documented in info.txt: "빈 데이터는 Black(512x512)").
                # Every one of the 235 cavity-POSITIVE masks is stored at native
                # resolution and matches its image exactly -- no transpose, no
                # resampling. So the transpose question that task1_paper.tex and
                # train_segmentation.py's docstring disagree about does not
                # arise for this label store at all; it only ever applied to the
                # nnUNet_raw/labelsTr copies, which should not be used as GT.
                assert m.sum() == 0, (
                    f"id={_id}: mask {m.shape} != image {(h, w)} AND is non-empty "
                    f"({int(m.sum())} fg px). That contradicts the audit -- stop and check.")
                m = np.zeros((h, w), np.uint8)
            fg_native = int(m.sum())
            m = cv2.resize(m, (T1_SIZE, T1_SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{msk_out}/{_id}.png", m * 255)

            rows.append(dict(our_id=_id, split=split, cavity=row.cavity,
                             native_h=h, native_w=w, fg_native=fg_native,
                             modality=row.series_modality_cd))
            if n % 50 == 0:
                print(f"  [{split}] {n}/{len(df)}", flush=True)

    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(out_dir, "task1", "meta.csv"), index=False)
    print(f"[task1] {len(meta)} cases -> {out_dir}/task1")
    print(meta.groupby(["split", "cavity"]).size())


def pack_task2(data_dir, out_dir):
    if not os.path.isdir(data_dir):
        print(f"[task2] {data_dir} not found -- skipping")
        return
    for split in ["train", "test"]:
        src = os.path.join(data_dir, split)
        dst = os.path.join(out_dir, "task2", split)
        os.makedirs(dst, exist_ok=True)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
        for n, f in enumerate(files, 1):
            img = cv2.imread(os.path.join(src, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (T2_SIZE, T2_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dst, f.replace(".png", ".jpg")), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if n % 500 == 0:
                print(f"  [task2/{split}] {n}/{len(files)}", flush=True)
        pd.read_csv(f"{data_dir}/{split}.csv").to_csv(
            os.path.join(out_dir, "task2", f"{split}.csv"), index=False)
        print(f"[task2/{split}] {len(files)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1-data", required=True)
    ap.add_argument("--task2-data", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-task2", action="store_true")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    pack_task1(os.path.expanduser(a.task1_data), out)
    if a.task2_data and not a.skip_task2:
        pack_task2(os.path.expanduser(a.task2_data), out)
    print(f"\nDone. Upload {out} as a PRIVATE Kaggle Dataset named e.g. 'mmtb-2026-pack'.")


if __name__ == "__main__":
    main()
