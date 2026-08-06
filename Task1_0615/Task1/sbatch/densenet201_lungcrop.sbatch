#!/bin/bash
# ============================================================================
# Task 1 — densenet201, lung-field-cropped input, single-model experiment.
#
# Ports Task2's lung_crop.py preprocessing lever (torchxrayvision's frozen
# PSPNet lung segmenter, crop to bounding box before CLAHE/resize) onto
# Task1's best single valid (image-only) model, densenet201 (acc=0.8288 on
# the raw-frame baseline, see task1-classifier-decision memory). Scoped to
# densenet201 ONLY -- does not touch resnet50d/tf_efficientnetv2_s/convnext_tiny
# or any of the ordfused/hybrid runs.
#
# Uses a SEPARATE image (lisa-task1-lungcrop:v2, built FROM the existing
# lisa-task1:latest + torchxrayvision) and a SEPARATE data dir
# (Task1/Data_lungcrop/) and out-dir (Code/runs_lungcrop/) so this experiment
# cannot clobber the existing densenet201 checkpoint/submission that
# submission_valid_ensemble_v1.csv depends on.
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-densenet201-lungcrop
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script into /var/spool/slurmd/job<id>/ before
# running it, so `dirname "${BASH_SOURCE[0]}"` resolves to the spool dir, not
# this file's real location — always derive paths from $SLURM_SUBMIT_DIR
# (where you ran `sbatch` from) instead.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final, Dockerfile_lungcrop)
BASE_IMAGE="lisa-task1:latest"
IMAGE="lisa-task1-lungcrop:v2"
CONTAINER_NAME="task1_densenet201_lungcrop_${SLURM_JOB_ID}"

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

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "base image '$BASE_IMAGE' not found locally on $SLURM_NODELIST -- building it now..."
    docker build -t "$BASE_IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (adds torchxrayvision on top of $BASE_IMAGE)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_lungcrop" "$DOCKER_CONTEXT"
fi

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code
    if [ -d /workspace/Data_lungcrop/train/CXR ] && [ -d /workspace/Data_lungcrop/test/CXR ] \
       && [ "$(ls /workspace/Data_lungcrop/train/CXR | wc -l)" = "$(ls /workspace/Data/train/CXR | wc -l)" ] \
       && [ "$(ls /workspace/Data_lungcrop/test/CXR | wc -l)" = "$(ls /workspace/Data/test/CXR | wc -l)" ]; then
        echo "=== Data_lungcrop already complete -- skipping preprocessing ==="
    else
        echo "=== lung-crop preprocessing (train + test) ==="
        python preprocess_lung_crop.py --in-dir /workspace/Data --out-dir /workspace/Data_lungcrop \
            2>&1 | tee preprocess_lung_crop.log
    fi

    echo "=== [densenet201/lungcrop] training 5 folds ==="
    python train.py --model densenet201 --fold all \
        --data-dir /workspace/Data_lungcrop --out-dir runs_lungcrop \
        2>&1 | tee train_densenet201_lungcrop.log

    echo "=== [densenet201/lungcrop] 5-fold ensemble inference ==="
    python predict.py --model densenet201 --out-tag densenet201_lungcrop \
        --data-dir /workspace/Data_lungcrop --out-dir runs_lungcrop \
        --submissions-dir ../submissions \
        2>&1 | tee predict_densenet201_lungcrop.log
  '

echo "=== DENSENET201 LUNGCROP DONE -> $PROJECT_ROOT/submissions/submission_densenet201_lungcrop.csv ==="
echo "compare its TEST score/acc against submission_densenet201.csv (0.7802/0.8288, uncropped baseline)"
