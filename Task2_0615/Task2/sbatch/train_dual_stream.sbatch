#!/bin/bash
# ============================================================================
# Task 2 — global+local dual-stream fusion model (train_dual_stream.py /
# GlobalLocalTBClassifier): a second, architecturally different model from
# the main 5-backbone ensemble (train_task2.sbatch), built to try to beat
# it rather than just add to it. Two independent encoder branches -- one
# sees the full uncropped image (global context), one sees the lung-cropped
# image (local anatomical detail) -- fused before the binary head. Per
# CheXFound's own "Global and Local Representations Integration" design.
#
# Default pairing is convnext_tiny (global) + densenet121 (local) -- both
# cheap 224px CNNs, so running two backbones per sample stays affordable.
# Override --global-encoder/--local-encoder for other combinations (e.g.
# swap in rad_dino/chexfound_vitl16 for a heavier pairing, once the cheap
# pairing's leave-one-modality-out number is in hand and worth the extra
# compute to try to beat).
#
# PREREQUISITE: run AFTER train_task2.sbatch (or whatever else holds the
# GPU) finishes. Reuses the same Docker image and Task1 weight mount.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-dual-stream
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_dual_${SLURM_JOB_ID}"

GLOBAL_ENC="${GLOBAL_ENC:-convnext_tiny}"
LOCAL_ENC="${LOCAL_ENC:-densenet121}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "pairing: global=$GLOBAL_ENC local=$LOCAL_ENC"

if [ ! -d "$PROJECT_ROOT/Data/train" ]; then
    echo "!! $PROJECT_ROOT/Data/train missing -- extract train.zip/test.zip first" >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch0" ]; then
    echo "!! $PROJECT_ROOT/Data/Preprocessed/train_images/ch0 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 (preprocess.py --lung-crop) first," >&2
    echo "!! this model needs the same precomputed lung-crop as the main ensemble." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

# NOTE: docker requires `docker run [OPTIONS] IMAGE [COMMAND]` -- see the
# identical comment in train_task2.sbatch. This array is a pure OPTIONS
# prefix (no image, no --name); each call site inserts --name and "$IMAGE"
# itself, in that order, before the actual bash -c command.
RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights)

echo "############################################################"
echo "### smoke test (1 epoch, stratified) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_smoke" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_dual_stream.py \
      --global-encoder $GLOBAL_ENC --local-encoder $LOCAL_ENC \
      --csv /workspace/Data/train.csv \
      --global-image-dir /workspace/Data/train \
      --local-image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --out-dir /workspace/checkpoints_dual_smoke \
      --fold-mode stratified --n-folds 5 --epochs 1 --num-workers 2"
rm -rf "$PROJECT_ROOT/checkpoints_dual_smoke"

echo "############################################################"
echo "### full leave-one-modality-out run ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_dual_stream.py \
      --global-encoder $GLOBAL_ENC --local-encoder $LOCAL_ENC \
      --csv /workspace/Data/train.csv \
      --global-image-dir /workspace/Data/train \
      --local-image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --out-dir /workspace/checkpoints_dual \
      --fold-mode leave_one_modality_out --epochs 15"

echo "=== DUAL-STREAM TRAIN DONE -> $PROJECT_ROOT/checkpoints_dual/ ==="
echo "Score with: python predict_dual_stream.py --ckpt-dir checkpoints_dual \\"
echo "    --csv Data/test.csv --global-image-dir Data/test \\"
echo "    --local-image-dir Data/Preprocessed/test_images/ch0 --score"
