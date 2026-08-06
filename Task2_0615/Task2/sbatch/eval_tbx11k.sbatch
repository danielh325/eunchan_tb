#!/bin/bash
# ============================================================================
# Task 2 — external validation on a 500-image balanced TBX11K subset (250 TB
# from imgs/tb/, 125 Normal from imgs/health/, 125 Normal from imgs/sick/ --
# "sick" = sick-but-non-TB, included so Normal isn't only clean healthy
# images, a harder and more realistic negative class).
#
# Deliberately EXCLUDES TBX11K's imgs/extra/ folder -- that subset is drawn
# from the same Montgomery/Shenzhen/DA/DB datasets already used in
# run_external_validation.sbatch, so including it would double-count data
# we've already validated on rather than testing on genuinely new images.
#
# Scores ONLY the final submitted model (rad_dino + chexfound_vitl16), not
# the full 5-encoder ensemble or a per-encoder breakdown -- this is a
# pre-submission sanity check on the actual shipped model, not a research
# sweep.
#
# PREREQUISITE: Data/external/tbx11k/raw/*.png (500 raw images) and
# Data/external/tbx11k/tbx11k_500.csv already exist (extracted/built on the
# host Mac from the official TBX11K Google Drive zip). Needs checkpoints/
# to exist.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-eval-tbx11k
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
EXT_DIR="$PROJECT_ROOT/Data/external/tbx11k"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_tbx11k_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -f "$EXT_DIR/tbx11k_500.csv" ] || [ ! -d "$EXT_DIR/raw" ]; then
    echo "!! $EXT_DIR/tbx11k_500.csv or raw/ missing." >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/checkpoints" ]; then
    echo "!! $PROJECT_ROOT/checkpoints missing -- nothing to score." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

echo "############################################################"
echo "### STAGE 1/2: lung-crop preprocessing (500 images) ###"
echo "############################################################"
if [ -d "$EXT_DIR/preprocessed/ch0" ]; then
  echo "skip: $EXT_DIR/preprocessed already exists"
else
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_preprocess" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/Data/external/tbx11k/raw \
        --out-dir /workspace/Data/external/tbx11k/preprocessed \
        --lung-crop"
fi

echo "############################################################"
echo "### STAGE 2/2: score with rad_dino + chexfound_vitl16 ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_score" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python predict_task2.py \
      --ckpt-dir /workspace/checkpoints \
      --csv /workspace/Data/external/tbx11k/tbx11k_500.csv \
      --image-dir /workspace/Data/external/tbx11k/preprocessed/ch0 \
      --out /workspace/submission_tbx11k_500.csv --score \
      --encoders rad_dino chexfound_vitl16"

echo "=== TBX11K (500-image subset) EVAL DONE ==="
