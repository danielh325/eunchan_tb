#!/bin/bash
# ============================================================================
# Task 2 — mask-channel ablation (option 3 from the mask-usage discussion):
# same training loop as train_task2.sbatch, but passes --mask-dir so
# common.preprocess_image swaps the classifier's redundant 3rd (duplicate)
# channel for the actual PSPNet lung mask (Data/Preprocessed/*/ch2), giving
# the model an explicit "this pixel is lung" prior instead of making it
# infer lung boundaries implicitly from intensity within the bbox crop.
#
# Nothing new to implement here -- train_task2.py/predict_task2.py/dataset.py
# /common.py already support --mask-dir end to end (dataset.py's Cfg.mask_dir
# -> common.preprocess_image's mask_arr path). This script is just a ready-
# to-fire ablation run: same encoders/folds/hyperparameters as the main run,
# writing to a SEPARATE checkpoint dir so it never overwrites/mixes with the
# no-mask checkpoints already trained.
#
# PREREQUISITE: run AFTER the external validation numbers
# (run_external_validation.sbatch) come back for the main (no-mask) run --
# if internal leave-one-modality-out accuracy (already ~0.99+) doesn't
# survive external validation, that tells you whether it's worth spending
# GPU time on this ablation at all before you launch it.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch2 must already exist
# (train_task2.sbatch's Stage 1 always runs preprocess.py with
# --mask-channel, so it should be there already from the main run).
#
# Usage:
#   sbatch train_task2_mask_ablation.sbatch
#   ENCODERS="densenet121 chexfound_vitl16" sbatch train_task2_mask_ablation.sbatch   # cheaper subset
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-mask-ablation
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script elsewhere before running it, so derive
# paths from $SLURM_SUBMIT_DIR (where you ran sbatch), not BASH_SOURCE.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # reuse Task1's already-downloaded rad_dino/chexfound checkpoints
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_mask_ablation_${SLURM_JOB_ID}"

# Same 5-encoder default as the main run, for a directly comparable
# per-encoder/per-fold table via evaluate.py -- override with the env var
# for a cheaper spot-check (e.g. just densenet121, the fastest backbone) if
# you want a quick signal before committing the full 4-day budget again.
ENCODERS="${ENCODERS:-densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16}"

# Separate from the main run's checkpoints/ -- never overwrite the no-mask
# checkpoints, both stay on disk so evaluate.py can compare them side by side.
OUT_DIR="${OUT_DIR:-checkpoints_mask_ablation}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "encoders: $ENCODERS"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ ! -f "$TASK1_WEIGHTS/rad_dino/config.json" ] || [ ! -f "$TASK1_WEIGHTS/chexfound/teacher_checkpoint.pth" ]; then
    echo "!! rad_dino/chexfound weights missing under $TASK1_WEIGHTS -- these" >&2
    echo "!! should already exist from Task1's foundation-model job; if not," >&2
    echo "!! run Task1_0615/Task1/sbatch/fetch_foundation_weights.sh first." >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch2" ] || [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch2" ]; then
    echo "!! Data/Preprocessed/{train,test}_images/ch2 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 first (preprocess.py --lung-crop" >&2
    echo "!! --mask-channel), this ablation needs the precomputed masks, not" >&2
    echo "!! just the ch0 crops." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights)

echo "############################################################"
echo "### full leave-one-modality-out run, mask channel ON ($ENCODERS) ###"
echo "############################################################"

for ENC in $ENCODERS; do
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_full_${ENC}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
        --csv /workspace/Data/train.csv \
        --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
        --mask-dir /workspace/Data/Preprocessed/train_images/ch2 \
        --out-dir /workspace/$OUT_DIR \
        --fold-mode leave_one_modality_out --epochs 15"
done

echo "=== MASK ABLATION DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="
echo "Compare against the no-mask run:"
echo "  python Code/evaluate.py --ckpt-dir $OUT_DIR"
echo "  python Code/evaluate.py --ckpt-dir checkpoints"
echo ""
echo "For the ensembled prediction, --mask-dir at predict time MUST match"
echo "training (mandatory here, not optional -- an on/off mismatch feeds the"
echo "model a different-looking input than it was trained on, silently):"
echo "  python Code/predict_task2.py --ckpt-dir $OUT_DIR \\"
echo "      --csv Data/test.csv --image-dir Data/Preprocessed/test_images/ch0 \\"
echo "      --mask-dir Data/Preprocessed/test_images/ch2 --score"
