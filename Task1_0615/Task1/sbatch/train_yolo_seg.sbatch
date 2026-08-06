#!/bin/bash
# ============================================================================
# Task 1 — YOLOv8-seg cavity segmentation: a second, structurally different
# architecture family to ensemble with nnU-Net (dense pixel-wise) and the
# foundation-decoder models (also dense pixel-wise).
#
# Why this exists: per PMC11992683 (TB lesion segmentation on CXR), their
# nnU-Net(ResEnc) + YOLOv8m-seg ensemble was their most stable result across
# segmentation/classification/detection -- box/instance-based YOLO makes
# different mistakes than dense pixel-wise models, so it adds real diversity
# rather than another correlated dense predictor. Uses the same patient-
# grouped fold-0-as-dev split as train_segmentation.py (see
# prepare_yolo_seg_data.py) and the same never-tune-on-test discipline
# (confidence threshold swept on val only -- see eval_yolo_seg.py).
#
# Uses a DISTINCT image tag (lisa-task1:yolo, not :foundation) so this never
# silently reuses an already-built :foundation image that predates
# Dockerfile_final's ultralytics addition -- new tag forces a fresh build.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task1-train-yolo-seg
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
IMAGE="lisa-task1:yolo"
CONTAINER_NAME="task1_train_yolo_seg_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=16gb \
  -v "$PROJECT_ROOT":/workspace \
  -v "$(cd "$PROJECT_ROOT/../../Code/baseline/nnUNetv2/nnUNet_data" && pwd)":/nnunet_data \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code

    echo "############################################################"
    echo "### STAGE 1/3: convert masks -> YOLOv8-seg format ###"
    echo "############################################################"
    python prepare_yolo_seg_data.py --img-size 640 \
        --nnunet-labels-dir /nnunet_data/nnUNet_raw/Dataset001_Task1/labelsTr

    echo "############################################################"
    echo "### STAGE 2/3: train YOLOv8m-seg ###"
    echo "############################################################"
    yolo segment train \
        data=yolo_seg_data/data.yaml \
        model=yolov8m-seg.pt \
        epochs=100 \
        imgsz=640 \
        batch=16 \
        patience=20 \
        project=runs/segment \
        name=cavity_yolov8m \
        exist_ok=True

    echo "############################################################"
    echo "### STAGE 3/3: real held-out Dice (conf swept on val, never test) ###"
    echo "############################################################"
    python eval_yolo_seg.py \
        --weights runs/segment/cavity_yolov8m/weights/best.pt \
        --img-size 640 \
        --nnunet-labels-dir /nnunet_data/nnUNet_raw/Dataset001_Task1/labelsTr
  '

echo "=== DONE. Checkpoint: Code/runs/segment/cavity_yolov8m/weights/best.pt ==="
echo "TEST dice printed above -- compare against nnU-Net baseline (0.306) and"
echo "the foundation-decoder ensemble (train_segmentation.sbatch, still running)."
