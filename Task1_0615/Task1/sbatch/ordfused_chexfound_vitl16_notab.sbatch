#!/bin/bash
# ============================================================================
# Task 1 — OrdFused-CXR: image-only ablation for chexfound_vitl16 (--no-tabular).
#
# Why this exists: chexfound_vitl16 is currently the strongest classifier we
# have (TEST score 0.7996, acc 0.8559) -- but that number was measured WITH
# tabular metadata at inference (Data/test.csv includes it). The real
# challenge test set gives images only, no metadata, so that 0.7996/0.8559
# number is NOT valid for the real submission. eva_x_base already got a
# proper from-scratch --no-tabular ablation (ordfused_eva_x_base_notab,
# TEST score 0.7596) -- this job gives chexfound_vitl16 the same treatment,
# since it's the single most valuable missing number in the whole plan: is
# our strongest architecture still strongest once metadata is removed?
#
# Only trains+predicts chexfound_vitl16 --no-tabular -- does not touch
# eva_x_base, rad_dino, or re-run the tabular chexfound_vitl16 checkpoint.
# Cheap: 1 encoder x 5 folds, same cost as the eva_x_base_notab run.
#
# PREREQUISITE: weights/chexfound/teacher_checkpoint.pth must exist (already
# fetched for the tabular run, per ordfused_foundation.sbatch). Submit after
# whatever else is holding the GPU (nnU-Net retrain investigation, etc.)
# finishes -- single V100 partition.
#
# TODO — confirm --partition / --gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task1-ordfused-chexfound_vitl16-notab
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
CONTAINER_NAME="task1_ordfused_chexfound_vitl16_notab_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/weights/chexfound/teacher_checkpoint.pth" ]; then
    echo "!! chexfound weights missing under $PROJECT_ROOT/weights/chexfound/ -- run" >&2
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
echo "### chexfound_vitl16 image-only ablation (--no-tabular), 5 folds ###"
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
    mkdir -p oof
    python train_ordfused.py --encoder chexfound_vitl16 --no-tabular --fold all \
        --max-retries 3 \
        2>&1 | tee train_ordfused_chexfound_vitl16_notab.log
    python predict_ordfused.py --encoder chexfound_vitl16 --no-tabular \
        --dump-oof oof/chexfound_vitl16_notab_oof.csv \
        2>&1 | tee predict_ordfused_chexfound_vitl16_notab.log
  '

echo "=== DONE -> Task1/submissions/submission_ordfused_chexfound_vitl16_notab.csv ==="
echo "Compare predict_ordfused_chexfound_vitl16_notab.log TEST score against:"
echo "  - ordfused_chexfound_vitl16 (tabular, NOT valid for real submission): 0.7996"
echo "  - densenet201 (current best valid, image-only-by-construction): 0.7802"
echo "  - ordfused_eva_x_base_notab (other proper image-only ablation): 0.7596"
