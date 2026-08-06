#!/bin/bash
# ============================================================================
# Task 1 — build the SUBMISSION docker image (Dockerfile.submission: self-
# contained, classification + segmentation ensembles baked in, no network
# needed at runtime), smoke-test it against a handful of real images, then
# docker save + gzip into a .tar.gz for upload (e.g. Google Drive).
#
# Build context MUST be the common parent of Task1_0615 and Task2_0615
# (this script's PARENT_ROOT below) -- Dockerfile.submission COPYs paths
# relative to that root (same convention as Task2's build script).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — segmentation postprocessing in predict_task1_submission.py is still
# a placeholder pending the dev-only threshold/connected-component sweep;
# rebuild after that lands, before treating this as the final submission.
# ============================================================================
#SBATCH --job-name=task1-build-submission
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"                 # .../Task1
TASK1_0615_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"                  # .../Task1_0615
PARENT_ROOT="$(cd "$TASK1_0615_ROOT/.." && pwd)"                   # .../eunchanhwang (has both Task1_0615 + Task2_0615)
DOCKERFILE="$PROJECT_ROOT/Dockerfile.submission"
IMAGE="lisa-task1-submission:v1"
TAR_OUT="$PROJECT_ROOT/lisa-task1-submission_v1.tar.gz"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "build context: $PARENT_ROOT"

if [ ! -f "$DOCKERFILE" ]; then
    echo "!! $DOCKERFILE missing." >&2
    exit 1
fi
if [ ! -f "$PARENT_ROOT/Task1_0615/Task1/weights/rad_dino/config.json" ]; then
    echo "!! Task1_0615/Task1/weights/rad_dino/config.json missing under $PARENT_ROOT" >&2
    exit 1
fi

echo "############################################################"
echo "### building $IMAGE ###"
echo "############################################################"
docker build -t "$IMAGE" -f "$DOCKERFILE" "$PARENT_ROOT"

echo "############################################################"
echo "### smoke test: score a handful of real test images ###"
echo "############################################################"
SMOKE_INPUT="$PROJECT_ROOT/.smoke_input"
SMOKE_OUTPUT="$PROJECT_ROOT/.smoke_output"
rm -rf "$SMOKE_INPUT" "$SMOKE_OUTPUT"
mkdir -p "$SMOKE_INPUT" "$SMOKE_OUTPUT"
# grab 5 real images straight from Data/test (raw .dcm, unpreprocessed) so
# this exercises the image's OWN preprocessing step too, not just prediction.
# `find | head` under `set -o pipefail` is a SIGPIPE trap -- see Task2's
# build_submission_image.sbatch for the exact incident (job 118018) this
# guards against. Disable pipefail for just this one pipeline.
set +o pipefail
find "$PROJECT_ROOT/Data/test/CXR" -maxdepth 1 -iname "*.dcm" | head -5 | xargs -I{} cp {} "$SMOKE_INPUT/"
set -o pipefail
N_SMOKE=$(ls "$SMOKE_INPUT" | wc -l | tr -d ' ')
echo "smoke-testing with $N_SMOKE images from Data/test/CXR"

# PYTHONUNBUFFERED=1 -- see Task2's build script for why this matters (python
# stdout buffering can make a slow/hung run look like zero output for
# minutes).
docker run --rm --device nvidia.com/gpu=all \
  -e PYTHONUNBUFFERED=1 \
  -v "$SMOKE_INPUT":/input:ro \
  -v "$SMOKE_OUTPUT":/output \
  "$IMAGE"

if [ -f "$SMOKE_OUTPUT/submission.csv" ]; then
    echo "--- smoke test OK, output: ---"
    cat "$SMOKE_OUTPUT/submission.csv"
    ls -la "$SMOKE_OUTPUT/masks" 2>/dev/null || echo "(no masks/ dir -- fine if all smoke cases were classifier-negative)"
else
    echo "!! smoke test FAILED -- no $SMOKE_OUTPUT/submission.csv produced" >&2
    exit 1
fi
rm -rf "$SMOKE_INPUT" "$SMOKE_OUTPUT"

echo "############################################################"
echo "### docker save + gzip -> $TAR_OUT ###"
echo "############################################################"
docker save "$IMAGE" | gzip > "$TAR_OUT"
ls -lh "$TAR_OUT"

echo "=== DONE ==="
echo "Image: $IMAGE"
echo "Archive: $TAR_OUT"
echo "Upload $TAR_OUT to Google Drive for submission."
echo "To load elsewhere: gunzip -c $TAR_OUT | docker load"
