#!/bin/bash
# ============================================================================
# Task 2 — build the EXTERNAL PHASE SUBMISSION docker image
# (Dockerfile.submission: self-contained, rad_dino ALONE baked in -- see
# Dockerfile.submission header for why the chexfound half was dropped --
# no network needed at runtime), smoke-test it against a handful of real
# images, then docker save + gzip into a .tar.gz for upload (e.g. email).
#
# Build context MUST be the common parent of Task1_0615 and Task2_0615
# (this script's PARENT_ROOT below) -- Dockerfile.submission COPYs
# Task1_0615/Task1/weights/rad_dino/config.json alongside
# Task2_0615/Task2/checkpoints/*.pth, so a narrower context (e.g. just
# Task2_0615) won't see the Task1 file at all.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-build-submission
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"                 # .../Task2
TASK2_0615_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"                  # .../Task2_0615
PARENT_ROOT="$(cd "$TASK2_0615_ROOT/.." && pwd)"                   # .../eunchanhwang (has both Task1_0615 + Task2_0615)
DOCKERFILE="$PROJECT_ROOT/Dockerfile.submission"
IMAGE="lisa-task2-submission:v2-raddino"
TAR_OUT="$PROJECT_ROOT/lisa-task2-submission_v2-raddino.tar.gz"

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
# grab 5 real images straight from Data/test (raw, unpreprocessed) so this
# exercises the image's OWN preprocessing step too, not just prediction.
# `find | head` under `set -o pipefail` is a classic SIGPIPE trap: head -5
# closes its stdin after 5 lines, find gets killed writing image #6 (Data/
# test has 1941), and pipefail treats that as a pipeline failure -- set -e
# then kills the WHOLE script right here, before the smoke test container
# ever runs (this is exactly what silently killed job 118018 -- the image
# built fine, 5 files got copied since that happened before the SIGPIPE,
# but the script died before printing anything further or launching
# docker run at all). Disable pipefail for just this one pipeline.
set +o pipefail
find "$PROJECT_ROOT/Data/test" -maxdepth 1 -iname "*.png" | head -5 | xargs -I{} cp {} "$SMOKE_INPUT/"
set -o pipefail
N_SMOKE=$(ls "$SMOKE_INPUT" | wc -l | tr -d ' ')
echo "smoke-testing with $N_SMOKE images from Data/test"

# PYTHONUNBUFFERED=1 -- without it, python's own stdout buffering (not just
# docker's) means preprocess.py/predict_task2.py's print() calls only flush
# once fully buffered or at process exit, so a slow or hung run shows NOTHING
# in this log until it finishes (or is killed, silently losing whatever
# hadn't flushed yet) -- exactly what made an earlier smoke test look like
# it produced zero output for 10+ minutes with no way to tell hung from slow.
docker run --rm --device nvidia.com/gpu=all \
  -e PYTHONUNBUFFERED=1 \
  -v "$SMOKE_INPUT":/input:ro \
  -v "$SMOKE_OUTPUT":/output \
  "$IMAGE"

# Organizers' fixed contract (Task2/README_task2.md) requires exactly
# /output/prediction.csv, not submission.csv -- an earlier version of this
# check looked for the wrong filename entirely.
if [ -f "$SMOKE_OUTPUT/prediction.csv" ]; then
    echo "--- smoke test OK, output: ---"
    cat "$SMOKE_OUTPUT/prediction.csv"
else
    echo "!! smoke test FAILED -- no $SMOKE_OUTPUT/prediction.csv produced" >&2
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
echo "External phase requires either the .tar.gz emailed directly, or a docker pull command -- upload/hosting choice is yours."
echo "To load elsewhere: gunzip -c $TAR_OUT | docker load"
