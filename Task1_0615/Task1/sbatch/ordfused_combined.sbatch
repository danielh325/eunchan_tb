#!/bin/bash
# ============================================================================
# Task 1 — OrdFused-CXR: smoke test + full run, combined into a single job
# on a single GPU. Runs the smoke test first (fast sanity check of the whole
# pipeline); only if it prints "=== SMOKE OK ===" does the full multi-day
# 5-fold CV + ensemble run start. Both stages share the same container image
# (built once) and the same GPU allocation, so nothing runs concurrently on
# the GPU -- they're sequential stages of one job.
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-ordfused-combined
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5-00:30:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script into /var/spool/slurmd/job<id>/ before
# running it, so derive paths from $SLURM_SUBMIT_DIR (where you ran sbatch),
# not from BASH_SOURCE.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
IMAGE="lisa-task1:latest"
CONTAINER_NAME="task1_ordfused_combined_${SLURM_JOB_ID}"

# Which backbones to train OrdFused on for the FULL run (space-separated).
# First is primary (also gets the image-only ablation). Smoke test always
# uses just the first one.
ENCODERS="tf_efficientnetv2_s densenet201"
SMOKE_ENC="$(echo "$ENCODERS" | awk '{print $1}')"

# Cluster proxy — build-time only (apt/pip); Docker auto-passes these through.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "image: $IMAGE"
echo "smoke encoder: $SMOKE_ENC"
echo "full encoders: $ENCODERS"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (first run only, proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

echo "############################################################"
echo "### STAGE 1/2: smoke test ($SMOKE_ENC, 1 epoch, 1 fold) ###"
echo "############################################################"

docker run --rm \
  --name "${CONTAINER_NAME}_smoke" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  -e SMOKE_ENC="$SMOKE_ENC" \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code
    ENC="$SMOKE_ENC"
    OUT=runs_smoke
    SUB=/workspace/submissions_smoke
    rm -rf "$OUT" "$SUB"

    echo "### 1/4 train (tabular, fold 0, 1 epoch) ###"
    python train_ordfused.py --encoder "$ENC" --fold 0 --epochs 1 \
        --max-retries 0 --num-workers 2 --out-dir "$OUT"

    echo "### 2/4 train (--no-tabular ablation, fold 0, 1 epoch) ###"
    python train_ordfused.py --encoder "$ENC" --no-tabular --fold 0 --epochs 1 \
        --max-retries 0 --num-workers 2 --out-dir "$OUT"

    echo "### 3/4 predict (single, tabular) ###"
    python predict_ordfused.py --encoder "$ENC" --num-workers 2 \
        --out-dir "$OUT" --submissions-dir "$SUB"

    echo "### 4/4 predict (--ensemble path) ###"
    python predict_ordfused.py --ensemble "ordfused_${ENC}" --name ordfused_smoke \
        --num-workers 2 --out-dir "$OUT" --submissions-dir "$SUB"

    echo "--- smoke submission head ---"
    head -5 "$SUB/submission_ordfused_smoke.csv"
    rm -rf "$OUT" "$SUB"
    echo "=== SMOKE OK ==="
  '

echo "############################################################"
echo "### STAGE 2/2: full run ($ENCODERS) ###"
echo "############################################################"

docker run --rm \
  --name "${CONTAINER_NAME}_full" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c "cd /workspace/Code && bash run_ordfused.sh $ENCODERS"

echo "=== ORDFUSED COMBINED DONE -> $PROJECT_ROOT/submissions/submission_ordfused.csv ==="
