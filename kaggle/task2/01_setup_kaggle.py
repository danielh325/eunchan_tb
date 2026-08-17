#!/usr/bin/env python
"""STEP 1 -- Kaggle GPU notebook. One-time setup for the Task 2 port.

Deliberately a THIN wrapper. Task2/Code/ is already portable: every cluster
path in it (`/workspace/Data/train.csv`, `/workspace/checkpoints`, ...) is an
argparse *default*, and the foundation-model weights directory is read from
the `TASK2_WEIGHTS_DIR` environment variable. Nothing in the training code
needs editing to run on Kaggle -- it needs the right flags and this setup.

That matters more than convenience: `train_task2.py` is the code that produced
every number in the DG ablation table. Rewriting it for Kaggle would silently
change the thing whose effect you measured. So it runs unmodified.

What this does:
  1. installs the handful of packages Kaggle does not ship
  2. pulls RAD-DINO's weights from the HF hub into the layout encoders.py wants
  3. runs Task2/Code/preprocess.py to produce the lung-cropped `ch0` images that
     `--image-dir` expects (the lung crop is worth +0.044 F1 on Shenzhen with a
     bootstrap CI excluding zero -- it is not optional)
  4. optionally fetches TBX11K for the hard-negative experiment

KAGGLE SETUP
------------
  Accelerator: GPU T4 x2      Internet: ON
  Datasets: your uploaded copy of Task2/Data (raw, ~1.0 GB -- upload it as-is,
            no packing needed) and a copy of Task2/Code.

  !python 01_setup_kaggle.py --stage deps
  !python 01_setup_kaggle.py --stage weights
  !python 01_setup_kaggle.py --stage preprocess
  !python 01_setup_kaggle.py --stage tbx11k     # only for the hardneg run
"""
import argparse
import os
import subprocess
import sys

CODE = os.environ.get("TASK2_CODE", "/kaggle/input/mmtb-task2-code/Code")
DATA = os.environ.get("TASK2_DATA", "/kaggle/input/mmtb-task2-data/Data")
WORK = os.environ.get("TASK2_WORK", "/kaggle/working")
WEIGHTS = os.path.join(WORK, "weights")


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def stage_deps():
    # torchxrayvision pulls the PSPNet lung segmenter; transformers loads
    # RAD-DINO (a Dinov2Model). peft is NOT needed -- encoders.py implements
    # its own LoRA injection.
    sh(f"{sys.executable} -m pip -q install torchxrayvision transformers "
       f"huggingface_hub scikit-image")
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
          f"devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} {p.total_memory/2**30:.1f}GB sm_{p.major}{p.minor}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        print("  -> sm<80: fp16 autocast only, no bf16. train_task2.py already "
              "uses fp16 + GradScaler, so nothing to change.")


def stage_weights():
    from huggingface_hub import snapshot_download
    d = os.path.join(WEIGHTS, "rad_dino")
    os.makedirs(d, exist_ok=True)
    snapshot_download("microsoft/rad-dino", local_dir=d,
                      allow_patterns=["*.json", "*.safetensors", "*.bin"])
    print(f"RAD-DINO -> {d}")
    print(f"Set TASK2_WEIGHTS_DIR={WEIGHTS} for every later step.")
    # CheXFound is deliberately not fetched: it loses to RAD-DINO on both
    # external cohorts under both adaptation recipes tried, and dropping it is
    # one of the few ablation results with a bootstrap CI that excludes zero
    # (Shenzhen +0.018, Montgomery +0.060). Nothing here needs it.


def stage_preprocess():
    env = dict(os.environ, TASK2_WEIGHTS_DIR=WEIGHTS)
    for split in ["train", "test"]:
        src = os.path.join(DATA, split)
        dst = os.path.join(WORK, "preprocessed", split)
        os.makedirs(dst, exist_ok=True)
        cmd = (f"{sys.executable} {CODE}/preprocess.py --in-dir {src} "
               f"--out-dir {dst} --lung-crop")
        print(f"$ {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True, env=env)
        print(f"[{split}] -> {dst}/ch0")


def stage_tbx11k():
    """TBX11K for the hard-negative experiment.

    The repo's own fetch script pulls a ~4 GB zip from Google Drive through the
    MI2RL proxy. On Kaggle, check for an existing public TBX11K dataset first
    and attach it rather than re-downloading into a 20 GB working directory --
    search Kaggle Datasets for "TBX11K". Only fall back to gdown if none is
    usable.

    What the hardneg run actually needs is narrow: the `sick` images listed in
    TBX11K's OFFICIAL train split (lists/TBX11K_train.txt), ~3000 files. It does
    NOT need the TB positives, and it must NOT touch the val split, which stays
    a held-out benchmark.
    """
    print(stage_tbx11k.__doc__)
    sh(f"{sys.executable} -m pip -q install gdown")
    print("\nThen point 20_build_hardneg at it:")
    print("  python build_hardneg_train_csv.py --tbx-root <TBX11K_ROOT> --no-test ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["deps", "weights", "preprocess", "tbx11k"])
    a = ap.parse_args()
    {"deps": stage_deps, "weights": stage_weights,
     "preprocess": stage_preprocess, "tbx11k": stage_tbx11k}[a.stage]()


if __name__ == "__main__":
    main()
