"""CLI preprocessing for Task1's densenet201-lungcrop experiment -- precomputes
only the expensive, GPU-bound step (lung-field segmentation), once, up front,
mirroring Task2/Code/preprocess.py's pattern. CLAHE/clip/resize are NOT baked
in here -- they stay on-the-fly in common.py's preprocess_image (applied
identically to every model), so this script's only job is to replace each raw
DICOM frame with its lung-cropped grayscale PNG at the same id.

Output layout (mirrors Data/'s split/CXR structure so --data-dir can just
point at --out-dir):
  <out-dir>/train.csv          copied unchanged from <in-dir>/train.csv
  <out-dir>/test.csv           copied unchanged from <in-dir>/test.csv
  <out-dir>/train/CXR/{id}.png lung-cropped, min-max-normalized grayscale
  <out-dir>/test/CXR/{id}.png  same, for the held-out test split

Cavity ground-truth masks (CXR_label/) are NOT touched or copied -- the
classifier scripts (train.py/predict.py) never read them.
"""
import argparse
import os
import shutil

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def sitk_read(path):
    # Same loader as common.py's sitk_read (SimpleITK-first, per the
    # organizers' warning about pydicom/nibabel silently flipping images).
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
    return arr.astype(np.float32)


def save_u8(arr, path):
    out = np.clip(arr, 0, 255).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(out), path)


def _to_u8_for_segmenter(arr, clip_lo=1.0, clip_hi=99.0):
    """xrv.utils.normalize() (called inside LungCropper.segment) asserts the
    input's max is already <= its maxval=255 -- true for Task2's cv2.imread'd
    8-bit PNGs, but NOT for Task1's raw DICOM arrays (SimpleITK returns native
    pixel intensities, e.g. up to ~900+ here, not capped at 255). Percentile
    clip + min-max normalize to 0-255 first, same preamble as common.py's
    _clahe_u8, so the segmenter gets a properly bounded image -- crop bbox
    coordinates are unaffected by this remap, so the actual crop below still
    reads from `raw`'s original dynamic range."""
    lo, hi = np.percentile(arr, clip_lo), np.percentile(arr, clip_hi)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    return (arr * 255).astype(np.float32)


def process_split(split, in_dir, out_dir, cropper):
    csv_src = os.path.join(in_dir, f"{split}.csv")
    csv_dst = os.path.join(out_dir, f"{split}.csv")
    if os.path.exists(csv_src):
        shutil.copyfile(csv_src, csv_dst)

    img_in_dir = os.path.join(in_dir, split, "CXR")
    img_out_dir = os.path.join(out_dir, split, "CXR")
    os.makedirs(img_out_dir, exist_ok=True)

    dcm_files = sorted(f for f in os.listdir(img_in_dir) if f.endswith(".dcm"))
    print(f"[{split}] preprocessing {len(dcm_files)} images", flush=True)
    for fname in tqdm(dcm_files, desc=split):
        _id = os.path.splitext(fname)[0]
        raw = sitk_read(os.path.join(img_in_dir, fname))

        full_mask = cropper.segment(_to_u8_for_segmenter(raw))
        y0, y1, x0, x1 = cropper.box_from_mask(full_mask)
        cropped = raw[y0:y1, x0:x1]

        base_norm = (cropped - cropped.min()) / (cropped.max() - cropped.min() + 1e-8)
        ch0_u8 = (base_norm * 255).astype(np.uint8)
        # Plain "{id}.png": common.find_file() looks for {id}{ext} first, then
        # falls back to {id}.png -- no suffix needed, and a suffix here would
        # silently never match, making every __getitem__ raise FileNotFoundError.
        save_u8(ch0_u8, os.path.join(img_out_dir, f"{_id}.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="Task1 Data dir (has train/CXR, test/CXR, train.csv, test.csv)")
    ap.add_argument("--out-dir", required=True, help="output base dir, mirrors --in-dir's split/CXR layout")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    from lung_crop import LungCropper
    cropper = LungCropper()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        process_split(split, args.in_dir, args.out_dir, cropper)

    print("Done")


if __name__ == "__main__":
    main()
