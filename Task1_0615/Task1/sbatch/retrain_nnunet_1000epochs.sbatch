#!/bin/bash
# ============================================================================
# Task 1 — nnU-Net RETRAIN: ResEnc planner + Focal Tversky loss, FULL
# 1000-epoch schedule (not the rescaled 250-epoch variant).
#
# Why this exists / corrects: retrain_nnunet.sbatch switched to
# nnUNetTrainerFocalTversky_250epochs because job 117570's log suggested
# ~23 min/epoch, making the real 1000-epoch default (~16 days) infeasible.
# That estimate was WRONG -- 117570 was stuck in an NFS I/O stall, not
# representative steady-state speed. The actual measured speed, confirmed
# repeatedly across epochs 130-204 of job 117597 (same ResEnc-M config,
# same GPU class), is ~79s/epoch. 1000 epochs at that rate is ~22 hours,
# not 16 days -- comfortably fits a normal walltime, and gets nnU-Net's
# real, undistorted polynomial LR annealing (PolyLRScheduler decays against
# self.num_epochs, so a compressed 250-epoch schedule anneals ~4x faster/
# more aggressively than the paper's actual design intends).
#
# 250-epoch result for comparison: real validation Dice was 0.2912 (using
# checkpoint_final.pth) -- essentially even with the plain 0.306 baseline,
# not a clear win. This run is the test of whether that was schedule-length
# related (250-epoch anneal too fast/short) or a more fundamental limit.
#
# Same fold, same untouched splits_final.json, same custom loss + the
# oversample_foreground_percent=0.66 tweak already baked into the base
# nnUNetTrainerFocalTversky class -- only the epoch count differs from the
# 250-epoch run.
#
# PREREQUISITE: needs the GPU -- check nothing else (train_segmentation.sbatch,
# the 250-epoch run if still active) is using it first.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-retrain-nnunet-1000ep
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"          # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"            # .../Task1_0615 (has Dockerfile_nnunet)
NNUNET_DATA_ROOT="$(cd "$PROJECT_ROOT/../../Code/baseline/nnUNetv2/nnUNet_data" && pwd)"
IMAGE="lisa-task1:nnunet"
CONTAINER_NAME="task1_retrain_nnunet_1000ep_${SLURM_JOB_ID}"

DATASET_ID=1
DATASET="Dataset001_Task1"
FOLD=0
RESENC_PLANNER="nnUNetPlannerResEncM"
RESENC_PLANS_ID="nnUNetResEncUNetMPlans"
TRAINER="nnUNetTrainerFocalTversky"   # base class -- inherits nnU-Net's real 1000-epoch default
GPU_MEMORY_TARGET_GB=28
NNUNET_N_PROC_DA=8

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "nnUNet data root: $NNUNET_DATA_ROOT"
echo "trainer: $TRAINER (full 1000-epoch schedule, ~22h at measured 79s/epoch)"

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
from nnUNetTrainerFocalTversky import FocalTversky_and_CE_loss, nnUNetTrainerFocalTversky
print(f"imported OK from {trainer_dir}: {nnUNetTrainerFocalTversky}")

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
found = recursive_find_python_class(
    os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer"),
    "nnUNetTrainerFocalTversky", "nnunetv2.training.nnUNetTrainer")
assert found is not None, (
    "recursive_find_python_class did NOT find nnUNetTrainerFocalTversky -- "
    "nnUNetv2_train would fail at Stage 4 even though the direct import above "
    "worked. Check for a missing __init__.py under variants/custom/.")
print(f"nnU-Net own trainer discovery found it OK: {found}")

loss_fn = FocalTversky_and_CE_loss(
    tversky_kwargs=dict(batch_dice=True, smooth=1e-5, do_bg=False, alpha=0.3, beta=0.7, gamma=1.33),
    ce_kwargs={}, weight_ce=1.0, weight_tversky=1.0, ignore_label=None)
dummy_logits = torch.randn(2, 2, 64, 64)
dummy_target = torch.randint(0, 2, (2, 1, 64, 64))
loss_val = loss_fn(dummy_logits, dummy_target)
print(f"dummy loss forward OK: {loss_val.item():.4f}")
PYEOF

    echo "############################################################"
    echo "### STAGE 3/4: ResEnc plan + preprocess (splits_final.json untouched) ###"
    echo "############################################################"
    echo "nnUNet_n_proc_DA=${nnUNet_n_proc_DA}"

    # Same plans/preprocessed data as the 250-epoch run (identical planner,
    # same dataset) -- already on disk, skip redoing it.
    PLANS_JSON="/nnunet_data/nnUNet_preprocessed/${DATASET}/${RESENC_PLANS_ID}.json"
    if [ -f "$PLANS_JSON" ]; then
        echo "skip: $PLANS_JSON already exists"
    else
        PLAN_ARGS="-d ${DATASET_ID} -pl ${RESENC_PLANNER}"
        if nnUNetv2_plan_experiment -h 2>&1 | grep -q "gpu_memory_target_in_gb"; then
            echo "gpu_memory_target_in_gb supported -- targeting ${GPU_MEMORY_TARGET_GB}GB"
            PLAN_ARGS="$PLAN_ARGS -gpu_memory_target_in_gb ${GPU_MEMORY_TARGET_GB}"
        else
            echo "!! -gpu_memory_target_in_gb not found -- using planner defaults" >&2
        fi
        nnUNetv2_plan_experiment $PLAN_ARGS
        nnUNetv2_preprocess -d ${DATASET_ID} -c 2d -p ${RESENC_PLANS_ID} -np 8
    fi

    echo "############################################################"
    echo "### STAGE 4/4: train fold ${FOLD} (${TRAINER}, ${RESENC_PLANS_ID}, 1000 epochs) ###"
    echo "############################################################"
    # Resume-safety: if this job gets cancelled/preempted and resubmitted,
    # auto-detect checkpoint_latest.pth and continue with --c instead of
    # silently restarting from epoch 0.
    RESULTS_DIR="/nnunet_data/nnUNet_results/${DATASET}/${TRAINER}__${RESENC_PLANS_ID}__2d/fold_${FOLD}"
    TRAIN_ARGS="${DATASET} 2d ${FOLD} -p ${RESENC_PLANS_ID} -tr ${TRAINER}"
    if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
        echo "found $RESULTS_DIR/checkpoint_latest.pth -- resuming with --c"
        TRAIN_ARGS="$TRAIN_ARGS --c"
    else
        echo "no checkpoint_latest.pth at $RESULTS_DIR -- starting fresh "
        echo "(note: a stale checkpoint_best.pth may exist there from an earlier "
        echo "abandoned attempt under this same trainer name -- nnUNetv2_train "
        echo "without --c overwrites it, which is correct/intended here)"
    fi
    nnUNetv2_train $TRAIN_ARGS
  '

echo "=== RETRAIN DONE. Results in nnUNet_results/${DATASET}/${TRAINER}__${RESENC_PLANS_ID}__2d/fold_${FOLD}/ ==="
echo "Score it via sbatch/test_nnunet.sbatch -- update its TRAINER var to nnUNetTrainerFocalTversky (no _250epochs suffix)."
