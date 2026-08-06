#!/bin/bash
# ============================================================================
# Task 1 — BASELINE model only (convnext_tiny), 5-fold CV + ensemble.
# One GPU, plain Docker (no pyxis/enroot/singularity on this cluster).
# Companion job: sbatch/other_cnns.sbatch (resnet/densenet/effnet).
#
# The image is built automatically below if it isn't already present on
# this node (first run only — later submissions reuse it, since Docker
# caches by tag on the node's local storage). No manual build step needed.
#
# NOTE (job 117221 postmortem): the first run of this job had 2 of 5 folds
# collapse to predicting the majority class for all 25 epochs (flat LR,
# no warmup -- see Code/train.py's MODEL_HP for the fix: lower LR, LR
# warmup, discriminative encoder/head LR, grad clipping, collapse-retry).
# This script now wipes that run's stale fold checkpoints/history before
# retraining so old collapsed-fold artifacts can't linger.
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-baseline
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script into /var/spool/slurmd/job<id>/ before
# running it, so `dirname "${BASH_SOURCE[0]}"` resolves to the spool dir, not
# this file's real location — always derive paths from $SLURM_SUBMIT_DIR
# (where you ran `sbatch` from) instead.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
IMAGE="lisa-task1:latest"
CONTAINER_NAME="task1_baseline_${SLURM_JOB_ID}"

# Cluster proxy — only used at build time (for apt/pip); Docker auto-passes
# these env vars through as build args, no --build-arg needed. Not baked into
# the image itself, so runtime containers aren't forced through the proxy.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "image: $IMAGE"

# Clear out the previous (collapsed-fold) convnext_tiny run so retraining
# can't mix in stale fold checkpoints/history from job 117221.
rm -rf "$PROJECT_ROOT/Code/runs/convnext_tiny"
rm -f "$PROJECT_ROOT/submissions/submission_convnext_tiny.csv"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (first run only, proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c "cd /workspace/Code && bash run_models.sh convnext_tiny"

echo "=== BASELINE DONE -> $PROJECT_ROOT/submissions/submission_convnext_tiny.csv ==="
