#!/bin/bash
# ============================================================================
# Task 1 — Foundation-model segmentation decoder (plan §12.5), a parallel
# track to the nnU-Net retrain (sbatch/retrain_nnunet.sbatch). Trains a
# lightweight DPT-style decoder on top of FROZEN dense patch features from
# RAD-DINO and CheXFound (both already integrated, frozen+LoRA, for
# classification) -- the bet being that their pretraining on hundreds of
# thousands of real chest X-rays gives richer features than a segmentation
# CNN trained from scratch on 444 images (what nnU-Net does).
#
# Uses the classification image (lisa-task1:foundation) since it already has
# transformers/peft/the vendored CheXFound code + the downloaded weights --
# no new Dockerfile needed.
#
# PREREQUISITE: needs the GPU the classification job (ordfused_foundation.sbatch)
# is using -- submit after that finishes, or on a different GPU node in parallel.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-train-segmentation
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
NNUNET_DATA_ROOT="$(cd "$PROJECT_ROOT/../../Code/baseline/nnUNetv2/nnUNet_data" && pwd)"
IMAGE="lisa-task1:foundation"
CONTAINER_NAME="task1_train_seg_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  -v "$NNUNET_DATA_ROOT":/nnunet_data \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code

    echo "############################################################"
    echo "### STAGE 0/4: smoke test (random-init, catches shape/wiring ###"
    echo "### bugs in the multi-scale decoder before burning GPU time) ###"
    echo "############################################################"
    python smoke_segmentation.py

    # TEST-set masks are not in Task1/Data/test/ -- they live under nnU-Net
    # own raw data dir, mounted above at /nnunet_data. Without this flag,
    # train_segmentation.py default --nnunet-labels-dir is computed via a
    # relative-path formula that assumes a HOST directory layout and resolves
    # to a nonexistent path inside the container -- SegDataset then silently
    # falls back to an all-zero mask for every test case, and every TEST dice
    # printed comes out as a meaningless 0.0000 (exactly what job 117660 hit).
    NNUNET_LABELS=/nnunet_data/nnUNet_raw/Dataset001_Task1/labelsTr

    echo "############################################################"
    echo "### STAGE 1/4: RAD-DINO segmentation decoder ###"
    echo "############################################################"
    python train_segmentation.py --encoder rad_dino --batch-size 4 --accum-steps 4 --epochs 50 \
        --nnunet-labels-dir "$NNUNET_LABELS"

    echo "############################################################"
    echo "### STAGE 2/4: CheXFound segmentation decoder ###"
    echo "############################################################"
    python train_segmentation.py --encoder chexfound_vitl16 --batch-size 2 --accum-steps 8 --epochs 50 \
        --nnunet-labels-dir "$NNUNET_LABELS"

    echo "############################################################"
    echo "### STAGE 3/4: EVA-X segmentation decoder ###"
    echo "############################################################"
    # 224px, vs the 512-518px used by the other two -- much cheaper per step,
    # same batch/accum as its classification config (ENCODER_FAMILIES["eva_x_base"]).
    python train_segmentation.py --encoder eva_x_base --batch-size 16 --accum-steps 1 --epochs 50 \
        --nnunet-labels-dir "$NNUNET_LABELS"

    echo "############################################################"
    echo "### STAGE 4/4: RAD-DINO + CheXFound + EVA-X ensemble ###"
    echo "############################################################"
    python ensemble_seg_predictions.py --nnunet-labels-dir "$NNUNET_LABELS"
  '

echo "=== DONE. Checkpoints in Code/runs_seg/<encoder>/best.pt ==="
echo "TEST dice printed at the end of each stage's log above -- directly comparable"
echo "to nnUNet_results/.../summary.json's 0.306 and to the ResEnc+Tversky retrain."
echo "The ensemble's TEST dice (Stage 4, last line) is the number that matters most --"
echo "only worth adopting over the best of the three standalone models if it actually wins."
