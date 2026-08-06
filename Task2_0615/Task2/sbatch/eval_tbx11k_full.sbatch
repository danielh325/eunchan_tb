#!/bin/bash
# ============================================================================
# Task 2 — external validation on the FULL TBX11K pool (imgs/tb + imgs/health
# + imgs/sick, minus imgs/extra/ -- same exclusion reasoning as the 500-image
# subset in eval_tbx11k.sbatch: imgs/extra/ is drawn from Montgomery/
# Shenzhen/DA/DB, already covered by run_external_validation.sbatch, so
# including it would double-count data already validated on).
#
# Supersedes eval_tbx11k.sbatch's 500-image number with the full ~7-8k pool
# for a tighter external-val estimate. Scores ONLY the final submitted model
# (rad_dino + chexfound_vitl16), matching eval_tbx11k.sbatch.
#
# PREREQUISITE: run these two on the login/build node first --
#   bash sbatch/fetch_tbx11k_full.sh
#   python3 Code/build_tbx11k_full_csv.py --tbx-root Data/external/tbx11k/full \
#       --raw-out Data/external/tbx11k/raw_full \
#       --csv-out Data/external/tbx11k/tbx11k_full.csv
# so that Data/external/tbx11k/raw_full/*.png and tbx11k_full.csv exist.
# Needs checkpoints/ to exist.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-eval-tbx11k-full
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
EXT_DIR="$PROJECT_ROOT/Data/external/tbx11k"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_tbx11k_full_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -f "$EXT_DIR/tbx11k_full.csv" ] || [ ! -d "$EXT_DIR/raw_full" ]; then
    echo "!! $EXT_DIR/tbx11k_full.csv or raw_full/ missing -- run fetch_tbx11k_full.sh + build_tbx11k_full_csv.py first." >&2
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
echo "### STAGE 1/2: lung-crop preprocessing (full pool) ###"
echo "############################################################"
if [ -d "$EXT_DIR/preprocessed_full/ch0" ]; then
  echo "skip: $EXT_DIR/preprocessed_full already exists"
else
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_preprocess" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/Data/external/tbx11k/raw_full \
        --out-dir /workspace/Data/external/tbx11k/preprocessed_full \
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
      --csv /workspace/Data/external/tbx11k/tbx11k_full.csv \
      --image-dir /workspace/Data/external/tbx11k/preprocessed_full/ch0 \
      --out /workspace/submission_tbx11k_full.csv --score \
      --encoders rad_dino chexfound_vitl16"

echo "=== TBX11K (full pool) EVAL DONE ==="
