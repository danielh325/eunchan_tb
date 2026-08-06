#!/bin/bash
# ============================================================================
# Task 1 — nnU-Net RETRAIN: ResEnc planner + Focal Tversky loss (plan §12).
#
# Goal: push real Dice up from the existing 0.306 baseline toward the ~0.7
# target. Two changes stacked on the SAME fold/split (splits_final.json is
# never touched, per info.txt's explicit warning):
#   1. nnU-Net's Residual Encoder (ResEnc) planner -- documented as a
#      substantial upgrade over the plain-vanilla config that produced 0.306.
#   2. A custom trainer (Code/nnunet_custom/nnUNetTrainerFocalTversky.py)
#      swapping the default Dice+CE loss for Focal Tversky+CE, which
#      specifically counteracts small-lesion underweighting.
#
# TRAINER/--time note (2026-07-21): the default nnUNetTrainerFocalTversky
# inherits nnU-Net's 1000-epoch schedule. Observed steady-state speed on
# this ResEnc-M config is ~23 min/epoch (from job 117570's log) -- 1000
# epochs would take ~16 days, but the old --time=3-00:00:00 only covers
# ~185 epochs before SLURM kills it. Since the LR schedule is a polynomial
# decay tied to the full epoch count, getting killed at epoch 185 would
# leave the LR barely annealed -- likely WORSE than the fully-converged
# 0.306 baseline, not better. Switched to nnUNetTrainerFocalTversky_250epochs
# (same loss, properly rescaled 250-epoch schedule) with --time bumped to
# fit it with margin. If 250 epochs undershoots the target Dice once it
# completes, nnUNetTrainerFocalTversky_500epochs is the next step up (add
# it to nnUNetTrainerFocalTversky.py the same way).
#
# GPU memory note (2026-07-21): nnU-Net's planner does NOT auto-detect GPU
# memory -- by default it plans batch/patch size against a conservative
# reference budget, not this node's actual 32GB V100. Stage 3 below passes
# -gpu_memory_target_in_gb to let the planner size batch/patch up to
# actually use the available memory, but only after verifying that flag
# exists in the installed nnunetv2 version (checked via `-h` output) --
# falls back to planner defaults with a clear warning if not, rather than
# hard-failing the whole job over an optional optimization. IMPORTANT: a
# larger batch/patch size does MORE compute per iteration (each nnU-Net
# "epoch" is a fixed 250 iterations regardless of batch size), so this can
# make each epoch slower in wall-clock time even though it processes more
# of the 444 training images per epoch -- the ~23 min/epoch figure above
# was measured WITHOUT this flag and may not hold once it's applied. --time
# is set with extra margin below to absorb that uncertainty rather than
# assume the old timing still applies.
#
# PREREQUISITE: this needs the GPU that the classification job
# (ordfused_foundation.sbatch) is using -- submit this AFTER that finishes.
# If job 117570 (the old 1000-epoch run) is still going, cancel it first
# (scancel 117570) -- letting it continue wastes GPU time on a schedule
# that can't complete in the allotted --time.
#
# Stages: sanity-check the custom trainer/loss on a dummy batch first (same
# philosophy as smoke_encoders.py for the classification side) -- catches an
# nnU-Net API mismatch before burning hours of training on it -- then plan +
# preprocess with ResEnc, then the real training run.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-retrain-nnunet
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"          # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"            # .../Task1_0615 (has Dockerfile_nnunet)
NNUNET_DATA_ROOT="$(cd "$PROJECT_ROOT/../../Code/baseline/nnUNetv2/nnUNet_data" && pwd)"
IMAGE="lisa-task1:nnunet"
CONTAINER_NAME="task1_retrain_nnunet_${SLURM_JOB_ID}"

DATASET_ID=1
DATASET="Dataset001_Task1"
FOLD=0
RESENC_PLANNER="nnUNetPlannerResEncM"     # bump to ResEncL/ResEncXL if compute allows
RESENC_PLANS_ID="nnUNetResEncUNetMPlans"  # nnU-Net's own default naming for the M planner
TRAINER="nnUNetTrainerFocalTversky_250epochs"
# V100 has 32GB; target 28 to leave headroom for CUDA context/fragmentation
# overhead rather than cutting it exactly at the physical limit.
GPU_MEMORY_TARGET_GB=28
# Matches --cpus-per-task=8 exactly (nnU-Net's default n_proc_DA can exceed
# this on a higher-core node, oversubscribing CPUs allocated to this job --
# same class of problem as the earlier OpenBLAS thread-explosion fix below,
# just at the multiprocessing-worker level instead of the BLAS-thread level).
NNUNET_N_PROC_DA=8

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "nnUNet data root: $NNUNET_DATA_ROOT"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_nnunet" "$DOCKER_CONTEXT"
fi

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=16gb \
  -v "$PROJECT_ROOT":/workspace \
  -v "$NNUNET_DATA_ROOT":/nnunet_data \
  -e nnUNet_raw=/nnunet_data/nnUNet_raw \
  -e nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed \
  -e nnUNet_results=/nnunet_data/nnUNet_results \
  -e DATASET_ID="$DATASET_ID" -e DATASET="$DATASET" -e FOLD="$FOLD" \
  -e RESENC_PLANNER="$RESENC_PLANNER" -e RESENC_PLANS_ID="$RESENC_PLANS_ID" \
  -e TRAINER="$TRAINER" -e GPU_MEMORY_TARGET_GB="$GPU_MEMORY_TARGET_GB" \
  -e nnUNet_n_proc_DA="$NNUNET_N_PROC_DA" \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e VECLIB_MAXIMUM_THREADS=1 \
  "$IMAGE" \
  bash -c '
    set -euo pipefail

    echo "############################################################"
    echo "### STAGE 1/4: install custom trainer into nnunetv2 package ###"
    echo "############################################################"
    NNUNET_TRAINER_DIR="$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), \"training\", \"nnUNetTrainer\", \"variants\", \"custom\"))")"
    mkdir -p "$NNUNET_TRAINER_DIR"
    cp /workspace/Code/nnunet_custom/nnUNetTrainerFocalTversky.py "$NNUNET_TRAINER_DIR/"
    # nnU-Net trainer discovery (recursive_find_python_class) walks this tree
    # via pkgutil and only recurses into proper packages -- a freshly created
    # "custom/" dir has no __init__.py, so nnUNetv2_train silently fails to
    # find the class here even though a direct sys.path import (Stage 2 below)
    # works fine. Always ensure it exists, regardless of whether the repo copy
    # has one.
    touch "$NNUNET_TRAINER_DIR/__init__.py"
    echo "installed to $NNUNET_TRAINER_DIR"

    echo "############################################################"
    echo "### STAGE 2/4: sanity-check the custom trainer + loss ###"
    echo "############################################################"
    python - <<PYEOF
import sys, os, torch
import nnunetv2

trainer_dir = os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer", "variants", "custom")
sys.path.insert(0, trainer_dir)
from nnUNetTrainerFocalTversky import (FocalTversky_and_CE_loss, nnUNetTrainerFocalTversky,
                                        nnUNetTrainerFocalTversky_250epochs)
print(f"imported OK from {trainer_dir}: {nnUNetTrainerFocalTversky}, {nnUNetTrainerFocalTversky_250epochs}")

# The import above only proves the file itself is valid Python -- it bypasses
# the actual nnU-Net trainer lookup, which is what Stage 4 nnUNetv2_train CLI
# uses. Exercise that real discovery path too, for the EXACT class name Stage
# 4 actually passes via -tr, so a missing __init__.py (or any other
# discovery-only failure) is caught here instead of after hours of
# preprocessing in Stage 3.
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
found = recursive_find_python_class(
    os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer"),
    "nnUNetTrainerFocalTversky_250epochs", "nnunetv2.training.nnUNetTrainer")
assert found is not None, (
    "recursive_find_python_class did NOT find nnUNetTrainerFocalTversky_250epochs -- "
    "nnUNetv2_train would fail at Stage 4 even though the direct import above "
    "worked. Check for a missing __init__.py under variants/custom/.")
print(f"nnU-Net own trainer discovery found it OK: {found}")

loss_fn = FocalTversky_and_CE_loss(
    tversky_kwargs=dict(batch_dice=True, smooth=1e-5, do_bg=False, alpha=0.3, beta=0.7, gamma=1.33),
    ce_kwargs={}, weight_ce=1.0, weight_tversky=1.0, ignore_label=None)
dummy_logits = torch.randn(2, 2, 64, 64)   # (B, C=2 [bg,cavity], H, W)
dummy_target = torch.randint(0, 2, (2, 1, 64, 64))
loss_val = loss_fn(dummy_logits, dummy_target)
print(f"dummy loss forward OK: {loss_val.item():.4f}")
PYEOF

    echo "############################################################"
    echo "### STAGE 3/4: ResEnc plan + preprocess (splits_final.json untouched) ###"
    echo "############################################################"
    echo "nnUNet_n_proc_DA=${nnUNet_n_proc_DA} (matches --cpus-per-task=8)"

    # Resume-safety: if this job was cancelled and is being resubmitted to
    # continue (scancel + resubmit, per the pause/resume workflow), the
    # plans file + preprocessed data from the earlier run are already on
    # disk (bind-mounted from the host, survives across job submissions) --
    # re-running plan_experiment/preprocess would just redo several minutes
    # of idempotent work, harmless but wasted. Skip when already present.
    PLANS_JSON="/nnunet_data/nnUNet_preprocessed/${DATASET}/${RESENC_PLANS_ID}.json"
    if [ -f "$PLANS_JSON" ]; then
        echo "skip: $PLANS_JSON already exists (resuming a previous run)"
    else
        PLAN_ARGS="-d ${DATASET_ID} -pl ${RESENC_PLANNER}"
        if nnUNetv2_plan_experiment -h 2>&1 | grep -q "gpu_memory_target_in_gb"; then
            echo "gpu_memory_target_in_gb supported -- targeting ${GPU_MEMORY_TARGET_GB}GB of this V100'\''s 32GB"
            PLAN_ARGS="$PLAN_ARGS -gpu_memory_target_in_gb ${GPU_MEMORY_TARGET_GB}"
        else
            echo "!! -gpu_memory_target_in_gb not found in this nnunetv2 version'\''s CLI --" >&2
            echo "!! falling back to planner defaults (job will still run correctly," >&2
            echo "!! just won'\''t fully utilize the V100'\''s 32GB)" >&2
        fi
        nnUNetv2_plan_experiment $PLAN_ARGS
        nnUNetv2_preprocess -d ${DATASET_ID} -c 2d -p ${RESENC_PLANS_ID} -np 8
    fi

    echo "############################################################"
    echo "### STAGE 4/4: train fold ${FOLD} (${TRAINER}, ${RESENC_PLANS_ID}) ###"
    echo "############################################################"
    # Resume-safety: nnUNet writes checkpoint_latest.pth every epoch (verified
    # by its mtime tracking the training log in real time, not just every
    # save_every=50 epochs as nnUNet'\''s docs might suggest at a glance) --
    # if this job is a resubmission after a scancel/pause, auto-detect that
    # checkpoint and pass --c (continue_training) so training resumes from
    # exactly where it left off (correct epoch, correctly-annealed LR
    # schedule) instead of silently restarting from epoch 0 and overwriting
    # ~180 epochs of progress with a fresh run under the same trainer name.
    RESULTS_DIR="/nnunet_data/nnUNet_results/${DATASET}/${TRAINER}__${RESENC_PLANS_ID}__2d/fold_${FOLD}"
    TRAIN_ARGS="${DATASET} 2d ${FOLD} -p ${RESENC_PLANS_ID} -tr ${TRAINER}"
    if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
        echo "found $RESULTS_DIR/checkpoint_latest.pth -- resuming with --c"
        TRAIN_ARGS="$TRAIN_ARGS --c"
    else
        echo "no existing checkpoint at $RESULTS_DIR -- starting fresh"
    fi
    nnUNetv2_train $TRAIN_ARGS
  '

echo "=== RETRAIN DONE. Results in nnUNet_results/${DATASET}/${TRAINER}__${RESENC_PLANS_ID}__2d/fold_${FOLD}/ ==="
echo "Run sbatch/test_nnunet.sbatch-style inference (adjust CONFIG/trainer name) to score it via evaluate.py."
