#!/usr/bin/env python
"""STEP 2 -- Kaggle GPU notebook. Launches Task2 training across both T4s.

Copies Task2/Code into /kaggle/working (input datasets are read-only), applies
ONE surgical patch, and launches folds in parallel.

THE PATCH, AND WHY IT IS THE ONLY ONE
-------------------------------------
train_task2.py trains its leave-one-modality-out folds in a single sequential
loop (`for f, tag in enumerate(fold_names)`), so it uses one GPU no matter how
many are present. The patch adds an `--only-fold` selector and skips every
other fold. It changes no training behaviour whatsoever -- same data, same
folds, same seed, same loss -- it only lets fold 0 run on cuda:0 while fold 1
runs on cuda:1. Everything else in Code/ runs byte-identical to the cluster
runs that produced the ablation table, which is the point: those numbers stay
comparable.

WHY 3 FOLDS, NOT 4
------------------
The XA modality has 11 training images. A leave-one-modality-out fold that
holds out 11 images is not a validation fold. This was already measured --
CSV/rad_dino_4fold_vs_3fold_summary.csv, 3fold_no_xa vs 4fold_full:

    Shenzhen    0.8919 vs 0.8966      internal   0.9907 vs 0.9913
    Montgomery  0.9048 vs 0.8976      TBX11K     0.3187 vs 0.3172

Indistinguishable, with Montgomery slightly favouring 3-fold. Dropping XA is a
free 25% of the compute budget, which matters a lot inside a 12 h session.

RUN ORDER
---------
    !python 02_launch_train.py --run baseline   # reproduce the paper's model
    !python 02_launch_train.py --run hardneg    # the experiment that has never run

Run `baseline` first even though its result is already known. It is the control:
without it retrained under this exact environment, a hardneg number cannot be
attributed to hard negatives rather than to Kaggle-vs-cluster differences.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

SRC_CODE = os.environ.get("TASK2_CODE", "/kaggle/input/mmtb-task2-code/Code")
DATA = os.environ.get("TASK2_DATA", "/kaggle/input/mmtb-task2-data/Data")
WORK = os.environ.get("TASK2_WORK", "/kaggle/working")
CODE = os.path.join(WORK, "Code")
WEIGHTS = os.path.join(WORK, "weights")
IMG_DIR = os.path.join(WORK, "preprocessed", "train", "ch0")

PATCH_ANCHOR = "    for f, tag in enumerate(fold_names):\n"
PATCH_BODY = (
    "    for f, tag in enumerate(fold_names):\n"
    "        # --only-fold: added for multi-GPU Kaggle launching. Selects a\n"
    "        # single fold so N processes can cover N folds concurrently, one\n"
    "        # per GPU. No effect on what any fold trains on.\n"
    "        if args.only_fold is not None and f != args.only_fold:\n"
    "            continue\n"
)
ARG_ANCHOR = '    ap.add_argument("--seed", type=int, default=42)\n'
ARG_BODY = (
    '    ap.add_argument("--seed", type=int, default=42)\n'
    '    ap.add_argument("--only-fold", type=int, default=None,\n'
    '                     help="train only this fold index (multi-GPU launching)")\n'
)


def prepare_code():
    if os.path.exists(CODE):
        shutil.rmtree(CODE)
    shutil.copytree(SRC_CODE, CODE)
    p = os.path.join(CODE, "train_task2.py")
    with open(p) as f:
        src = f.read()
    if "--only-fold" in src:
        print("[patch] already applied")
        return
    for anchor, body, what in [(ARG_ANCHOR, ARG_BODY, "argparse"),
                               (PATCH_ANCHOR, PATCH_BODY, "fold loop")]:
        if src.count(anchor) != 1:
            raise SystemExit(
                f"[patch] anchor for {what} matched {src.count(anchor)} times, expected 1. "
                f"train_task2.py has changed -- apply --only-fold by hand rather than "
                f"letting this guess.")
        src = src.replace(anchor, body)
    with open(p, "w") as f:
        f.write(src)
    print("[patch] --only-fold added to train_task2.py")


def drop_xa(in_csv, out_csv):
    import pandas as pd
    df = pd.read_csv(in_csv)
    n = len(df)
    df = df[df.Modality_DICOM != "XA"].reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[folds] dropped {n - len(df)} XA rows -> {len(df)} rows, "
          f"{df.Modality_DICOM.nunique()} modality folds")
    return sorted(df.Modality_DICOM.unique())


def launch(run, csv, epochs, extra):
    mods = drop_xa(csv, os.path.join(WORK, f"train_{run}.csv"))
    csv = os.path.join(WORK, f"train_{run}.csv")
    out = os.path.join(WORK, f"checkpoints_{run}")
    os.makedirs(out, exist_ok=True)
    base = (f"{sys.executable} {CODE}/train_task2.py --encoder rad_dino "
            f"--csv {csv} --image-dir {IMG_DIR} --out-dir {out} "
            f"--fold-mode leave_one_modality_out --aug-strength strong "
            f"--domain-mixup-alpha 0.0 --epochs {epochs} --skip-existing-folds {extra}")

    # Two at a time, one per card. With 3 folds that is one pair then a single.
    env = dict(os.environ, TASK2_WEIGHTS_DIR=WEIGHTS)
    pending = list(range(len(mods)))
    while pending:
        batch, pending = pending[:2], pending[2:]
        procs = []
        for gpu, fold in enumerate(batch):
            e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            log = open(os.path.join(WORK, f"train_{run}_fold{fold}.log"), "w")
            cmd = f"{base} --only-fold {fold}"
            print(f"[launch] gpu{gpu} fold{fold} ({mods[fold]}): {cmd}", flush=True)
            procs.append((fold, subprocess.Popen(cmd, shell=True, env=e,
                                                 stdout=log, stderr=subprocess.STDOUT), log))
        t0 = time.time()
        for fold, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"[launch] fold{fold} exited {rc} after {(time.time()-t0)/60:.1f} min", flush=True)
            if rc != 0:
                raise SystemExit(f"fold {fold} failed -- see {WORK}/train_{run}_fold{fold}.log")
    print(f"\n[done] checkpoints in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["baseline", "hardneg"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hardneg-csv", default=os.path.join(WORK, "train_hardneg.csv"))
    a = ap.parse_args()
    prepare_code()
    if a.run == "baseline":
        launch("baseline", os.path.join(DATA, "train.csv"), a.epochs, "")
    else:
        if not os.path.exists(a.hardneg_csv):
            raise SystemExit(
                f"{a.hardneg_csv} not found. Build it first:\n"
                f"  python {CODE}/build_hardneg_train_csv.py --no-test \\\n"
                f"      --train-csv {DATA}/train.csv --tbx-root <TBX11K_ROOT> \\\n"
                f"      --main-image-dir {IMG_DIR} --out-csv {a.hardneg_csv} \\\n"
                f"      --out-image-dir {WORK}/hardneg_images\n"
                f"  (--no-test keeps Data/test.csv out of training so the internal\n"
                f"   benchmark survives and baseline-vs-hardneg stays comparable)")
        launch("hardneg", a.hardneg_csv, a.epochs, "")


if __name__ == "__main__":
    main()
