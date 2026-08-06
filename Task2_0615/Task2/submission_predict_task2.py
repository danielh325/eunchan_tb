"""TREAT-MMTB 2026 -- Task 2 EXTERNAL PHASE submission entrypoint.

Matches the organizers' FIXED I/O contract exactly (see
https://github.com/mi2rl-challenge/treat-mmtb.miccai2026/blob/main/Task2/README_task2.md):

  python predict_task2.py --input /input --output /output

  Input  (read-only): /input/*.png   -- raw, unpreprocessed test images
  Output:              /output/prediction.csv  -- columns: filename,TB/Normal
                        filename = the PNG's own basename (e.g. "abc.png")
                        TB/Normal = "TB" or "Normal"

This wraps our actual pipeline (lung-crop preprocessing via Code/preprocess.py,
then rad_dino ALONE via Code/predict_task2.py -- switched from the internal-
phase rad_dino+chexfound_vitl16 ensemble because rad_dino alone generalizes
better on real external data (Shenzhen/Montgomery), while chexfound's
internal-phase checkpoints were trained with a LoRA adapter-targeting bug --
that file is OUR internal inference script and has a different CLI, --ckpt-dir/
--csv/--image-dir/--out, hence this separate wrapper rather than editing it
directly to speak two incompatible interfaces at once) and translates our
internal `new_id` (filename stem) convention back to the exact original PNG
filename the organizers require in the output CSV.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "Code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--output", default="/output")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
    if not image_paths:
        raise RuntimeError(f"No PNG images found under: {args.input}")

    # new_id (filename stem, matches preprocess.py's own convention) -> original filename
    id_to_filename = {
        os.path.splitext(os.path.basename(p))[0]: os.path.basename(p)
        for p in image_paths
    }

    manifest_csv = "/workspace/_input_manifest.csv"
    pd.DataFrame({"new_id": sorted(id_to_filename.keys())}).to_csv(manifest_csv, index=False)

    preproc_dir = "/workspace/_preprocessed"
    subprocess.run(
        [sys.executable, "preprocess.py",
         "--in-dir", args.input, "--out-dir", preproc_dir, "--lung-crop"],
        cwd=CODE_DIR, check=True,
    )

    raw_pred_csv = "/workspace/_raw_prediction.csv"
    subprocess.run(
        [sys.executable, "predict_task2.py",
         "--ckpt-dir", "/workspace/checkpoints",
         "--csv", manifest_csv,
         "--image-dir", os.path.join(preproc_dir, "ch0"),
         "--out", raw_pred_csv,
         "--encoders", "rad_dino"],
        cwd=CODE_DIR, check=True,
    )

    df = pd.read_csv(raw_pred_csv)
    out_path = os.path.join(args.output, "prediction.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "TB/Normal"])
        for _, row in df.iterrows():
            new_id = str(row["new_id"])
            filename = id_to_filename[new_id]
            writer.writerow([filename, row["pred_TB/Normal"]])

    print(f"wrote {out_path} ({len(df)} predictions)")


if __name__ == "__main__":
    main()
