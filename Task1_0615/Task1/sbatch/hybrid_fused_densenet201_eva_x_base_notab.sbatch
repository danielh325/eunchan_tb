#!/bin/bash
# ============================================================================
# Task 1 — HybridFused-CXR: joint CNN+ViT fusion (densenet201 + eva_x_base),
# image-only (--no-tabular) so its OOF/test probs are valid ensemble inputs.
#
# Why this exists: every existing checkpoint (train_ordfused.py) is a SINGLE
# encoder -- either a plain CNN or one foundation ViT -- never both. This is
# the first architecture that fuses a CNN's local-feature stream with a ViT's
# global-context stream at the FEATURE level, trained jointly (see
# hybrid_fused.py's docstring for the paper motivating this: Alhafdhi et al.
# 2026, Diagnostics 16(7):999, ViT-CNN hybrids for thyroid ultrasound). Task1's
# current best submission is late/logit fusion of independently trained
# single-stream models -- this is meant as a genuinely different, more
# complementary ensemble member, not just another correlated CNN or ViT.
#
# densenet201 (224px, ImageNet stats, full fine-tune) + eva_x_base (224px,
# foundation stats, frozen+LoRA) chosen because both run at 224px -- cheapest
# pairing, avoids RAD-DINO/CheXFound's much larger attention memory footprint.
#
# PREREQUISITE: weights/eva_x/eva_x_base_patch16_merged520k_mim.pt must exist
# (already fetched for ordfused_eva_x_base). Check no other job holds the GPU
# first -- single V100 partition.
#
# TODO — confirm --partition / --gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task1-hybrid-densenet201-eva_x_base-notab
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
IMAGE="lisa-task1:foundation"
CONTAINER_NAME="task1_hybrid_densenet201_eva_x_base_notab_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/weights/eva_x/eva_x_base_patch16_merged520k_mim.pt" ]; then
    echo "!! eva_x weights missing under $PROJECT_ROOT/weights/eva_x/ -- run" >&2
    echo "!! sbatch/fetch_foundation_weights.sh on the host first." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "!! image '$IMAGE' not found locally on $SLURM_NODELIST -- build it first" >&2
    echo "!! (see ordfused_foundation.sbatch STAGE 0, or run the docker build manually)." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

echo "############################################################"
echo "### HybridFused densenet201+eva_x_base (--no-tabular), 5 folds ###"
echo "############################################################"

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code
    python train_hybrid_fused.py --cnn-encoder densenet201 --vit-encoder eva_x_base \
        --no-tabular --fold all \
        2>&1 | tee train_hybrid_densenet201_eva_x_base_notab.log
    python predict_hybrid_fused.py --cnn-encoder densenet201 --vit-encoder eva_x_base \
        --no-tabular --dump-oof oof/hybrid_densenet201_eva_x_base_notab_oof.csv \
        2>&1 | tee predict_hybrid_densenet201_eva_x_base_notab.log
  '
