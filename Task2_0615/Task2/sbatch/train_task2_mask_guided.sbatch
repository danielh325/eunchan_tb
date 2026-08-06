#!/bin/bash
# ============================================================================
# Task 2 — mask-guided attention training: trains TBClassifier's optional
# use_mask_attention gate (model.py) via train_task2.py's --mask-guide-dir,
# ACTIVELY supervising attention against the real PSPNet lung mask
# (Data/Preprocessed/*/ch2) with a Dice loss, instead of just cropping to the
# lung bbox and hoping (train_task2.sbatch) or passively offering the mask
# as an extra input channel and hoping the model uses it
# (train_task2_mask_ablation.sbatch, --mask-dir).
#
# CNN backbones ONLY -- densenet121/swin_tiny/convnext_tiny expose a spatial
# forward_features map to gate; rad_dino/chexfound's frozen+LoRA wrappers
# only expose pooled output (train_task2.py itself now hard-errors if you
# pass those two encoders here, rather than crashing deep in forward()).
#
# Validated locally first (CPU, mask_guided_demo.py): warm-started an
# existing densenet121_CR checkpoint, froze everything except a fresh gate,
# fit it 60 steps on 12 images (dice loss 0.52->0.21) -- one held-out image
# went from a scattered vessel-crossing Grad-CAM to cleanly tracking the
# left lung field (P(TB) 0.62->0.18, likely correcting a shortcut false
# positive), but another still partly attended the same corner marker as
# baseline, since only the gate (not the frozen encoder) was trained in that
# demo. This run is the real test: full joint encoder+gate training over
# the whole leave-one-modality-out schedule, not a frozen-backbone demo.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch2 must already exist
# (train_task2.sbatch's Stage 1 always runs preprocess.py with
# --mask-channel, so it should be there already from the main run).
#
# Usage:
#   sbatch train_task2_mask_guided.sbatch
#   ENCODERS=densenet121 MASK_GUIDE_LAMBDA=0.5 sbatch train_task2_mask_guided.sbatch   # cheap spot-check
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-mask-guided
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
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # unused here (no foundation encoders), kept for a consistent docker mount
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_mask_guided_${SLURM_JOB_ID}"

# Only the 3 timm CNN backbones support the gate -- see train_task2.py's own
# guard (hard-errors on rad_dino/chexfound_vitl16 with --mask-guide-dir).
ENCODERS="${ENCODERS:-densenet121 swin_tiny convnext_tiny}"

# Weight on the Dice attention-supervision loss (added to BCE classification
# loss) -- 1.0 matches mask_guided_demo.py's default. Higher pulls attention
# harder toward the mask at potential cost to classification accuracy if
# pushed too far; this is exactly the kind of tradeoff --n-images/--steps
# couldn't probe in the CPU demo (frozen encoder, 12 images, 60 steps) --
# first real signal on it comes from this run's val_acc/val_auc vs the
# no-mask-guide run's.
MASK_GUIDE_LAMBDA="${MASK_GUIDE_LAMBDA:-1.0}"

# Separate from checkpoints/ and checkpoints_mask_ablation/ -- never
# overwrite either, all three stay on disk for a 3-way evaluate.py comparison.
OUT_DIR="${OUT_DIR:-checkpoints_mask_guided}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "encoders: $ENCODERS"
echo "mask-guide-lambda: $MASK_GUIDE_LAMBDA"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch2" ] || [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch2" ]; then
    echo "!! Data/Preprocessed/{train,test}_images/ch2 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 first (preprocess.py --lung-crop" >&2
    echo "!! --mask-channel), this run needs the precomputed masks, not" >&2
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
echo "### full leave-one-modality-out run, mask-guided attention ON ($ENCODERS) ###"
echo "############################################################"

for ENC in $ENCODERS; do
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_full_${ENC}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
        --csv /workspace/Data/train.csv \
        --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
        --mask-guide-dir /workspace/Data/Preprocessed/train_images/ch2 \
        --mask-guide-lambda $MASK_GUIDE_LAMBDA \
        --out-dir /workspace/$OUT_DIR \
        --fold-mode leave_one_modality_out --epochs 15"
done

echo "=== MASK-GUIDED TRAINING DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="
echo "Compare against both prior runs:"
echo "  python Code/evaluate.py --ckpt-dir $OUT_DIR"
echo "  python Code/evaluate.py --ckpt-dir checkpoints"
echo "  python Code/evaluate.py --ckpt-dir checkpoints_mask_ablation   # if that run exists"
echo ""
echo "For Grad-CAM inspection of the trained gate on real (non-demo) checkpoints,"
echo "point mask_guided_demo.py or gradcam_task2.py's TBClassifier construction at"
echo "use_mask_attention=True and --ckpt-dir $OUT_DIR."