#!/bin/bash
# ============================================================================
# Task 1 — the other 3 CNN backbones (resnet50d, densenet201,
# tf_efficientnetv2_s), 5-fold CV + ensemble each, run sequentially on ONE
# GPU, plain Docker (no pyxis/enroot/singularity on this cluster).
# Companion job: sbatch/baseline.sbatch (convnext_tiny).
#
# The image is built automatically below if it isn't already present on
# this node. If baseline.sbatch already ran and built it, this job reuses it.
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-other-cnns
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
CONTAINER_NAME="task1_other_cnns_${SLURM_JOB_ID}"

# Cluster proxy — only used at build time (for apt/pip); Docker auto-passes
# these env vars through as build args, no --build-arg needed. Not baked into
# the image itself, so runtime containers aren't forced through the proxy.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

# Best-in-family CNNs per family (balanced accuracy/compute pick for a
# ~444-image training set within the 5-day budget):
#   ResNet family:       resnet50d
#   DenseNet family:      densenet201  (largest official DenseNet variant)
#   EfficientNet family:  tf_efficientnetv2_s
MODELS=(resnet50d densenet201 tf_efficientnetv2_s)

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "image: $IMAGE"
echo "models (sequential, same GPU): ${MODELS[*]}"

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
  bash -c "cd /workspace/Code && bash run_models.sh ${MODELS[*]}"

echo "=== OTHER CNNs DONE -> $PROJECT_ROOT/submissions/submission_<model>.csv for each of: ${MODELS[*]} ==="
