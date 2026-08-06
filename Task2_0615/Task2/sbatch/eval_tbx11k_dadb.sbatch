#!/bin/bash
# ============================================================================
# Task 2 — external validation on TBX11K's bundled DA+DB dataset (National
# Institute of Tuberculosis and Respiratory Diseases, New Delhi, India, two
# X-ray machines) -- a genuinely separate external source from everything
# scored so far (Shenzhen/Montgomery/TBX11K's own health+sick+tb pool), not
# reachable via eval_tbx11k_full.sbatch (which excludes ALL of imgs/extra/,
# since that folder also bundles an mc+shenzhen/ sibling that WOULD
# double-count already-scored data). Only 176 images (102 Normal, 74 TB) --
# small, but free: the data is already sitting on disk from the TBX11K full
# download, no new fetch needed.
#
# Scores ONLY the final submitted model (rad_dino + chexfound_vitl16),
# matching eval_tbx11k.sbatch / eval_tbx11k_full.sbatch.
#
# PREREQUISITE: run this first on the login/build node --
#   python3 Code/build_tbx11k_dadb_csv.py \
#       --dadb-root Data/external/tbx11k/full/TBX11K/imgs/extra/da+db \
#       --raw-out Data/external/tbx11k/raw_dadb \
#       --csv-out Data/external/tbx11k/tbx11k_dadb.csv
# so that Data/external/tbx11k/raw_dadb/*.png and tbx11k_dadb.csv exist.
# Needs checkpoints/ to exist.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-eval-tbx11k-dadb
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
EXT_DIR="$PROJECT_ROOT/Data/external/tbx11k"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_tbx11k_dadb_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -f "$EXT_DIR/tbx11k_dadb.csv" ] || [ ! -d "$EXT_DIR/raw_dadb" ]; then
    echo "!! $EXT_DIR/tbx11k_dadb.csv or raw_dadb/ missing -- run Code/build_tbx11k_dadb_csv.py first." >&2
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
echo "### STAGE 1/2: lung-crop preprocessing (DA+DB, 176 images) ###"
echo "############################################################"
if [ -d "$EXT_DIR/preprocessed_dadb/ch0" ]; then
  echo "skip: $EXT_DIR/preprocessed_dadb already exists"
else
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_preprocess" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/Data/external/tbx11k/raw_dadb \
        --out-dir /workspace/Data/external/tbx11k/preprocessed_dadb \
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
      --csv /workspace/Data/external/tbx11k/tbx11k_dadb.csv \
      --image-dir /workspace/Data/external/tbx11k/preprocessed_dadb/ch0 \
      --out /workspace/submission_tbx11k_dadb.csv --score \
      --encoders rad_dino chexfound_vitl16"

echo "=== TBX11K DA+DB (India) EVAL DONE ==="
