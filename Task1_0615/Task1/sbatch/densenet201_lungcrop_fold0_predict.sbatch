#!/bin/bash
# ============================================================================
# Task 1 — densenet201/lungcrop, single-fold (fold0) test-set inference.
#
# fold0 had the best VALIDATION accuracy among the 5 folds (0.9101, on its
# own held-out 89-image slice during training -- see
# runs_lungcrop/densenet201/fold0_history.csv). This job checks whether that
# also holds up on the actual 111-image test set, using predict_single_fold_cnn.py
# (a sibling of predict_single_fold.py that matches train.py's plain-CNN
# checkpoint format instead of OrdFused's). Inference only, no training --
# reuses the already-trained runs_lungcrop/densenet201/fold0_best.pt and the
# already lung-cropped Data_lungcrop/test/CXR from the earlier preprocessing
# pass, so this is a quick job (~minutes, not hours).
#
# Caveat worth remembering when reading the result: fold0 was picked by its
# own validation score, not by test performance -- if it also wins on test,
# that's one more data point on n=111, not proof it's "the best model."
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-densenet201-lungcrop-fold0-predict
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script into /var/spool/slurmd/job<id>/ before
# running it, so `dirname "${BASH_SOURCE[0]}"` resolves to the spool dir, not
# this file's real location — always derive paths from $SLURM_SUBMIT_DIR
# (where you ran `sbatch` from) instead.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
IMAGE="lisa-task1-lungcrop:v2"
CONTAINER_NAME="task1_densenet201_lungcrop_fold0_predict_${SLURM_JOB_ID}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "image: $IMAGE"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- run densenet201_lungcrop.sbatch first to build it." >&2
    exit 1
fi
if [ ! -f "$PROJECT_ROOT/Code/runs_lungcrop/densenet201/fold0_best.pt" ]; then
    echo "runs_lungcrop/densenet201/fold0_best.pt not found -- run densenet201_lungcrop.sbatch first." >&2
    exit 1
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
    python predict_single_fold_cnn.py \
        --ckpt runs_lungcrop/densenet201/fold0_best.pt \
        --data-dir /workspace/Data_lungcrop \
        --name densenet201_lungcrop_fold0 \
        --submissions-dir ../submissions \
        2>&1 | tee predict_densenet201_lungcrop_fold0.log
  '

echo "=== DONE -> $PROJECT_ROOT/submissions/submission_densenet201_lungcrop_fold0.csv ==="
