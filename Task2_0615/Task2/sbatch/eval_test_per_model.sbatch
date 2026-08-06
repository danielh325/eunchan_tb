#!/bin/bash
# ============================================================================
# Task 2 — score the real held-out Data/test.csv (1941 labeled rows,
# organizer-held-out same-source-pool test set -- distinct from both the
# plain stratified split and true external validation) per INDIVIDUAL
# encoder, plus the full 5-encoder ensemble as a reference point.
#
# predict_task2.py has no built-in "break down by encoder" mode -- it always
# ensembles whatever checkpoints match --encoders (each encoder's own 4
# leave-one-modality-out fold checkpoints, its "implicit k-fold ensemble").
# So getting a per-model number means one invocation per encoder, restricted
# via --encoders to that encoder alone, plus one more with no restriction
# for the full-ensemble comparison point.
#
# PREREQUISITE: Data/Preprocessed/test_images/ch0 must already exist
# (train_task2.sbatch's Stage 1 already preprocesses BOTH train and test, so
# it should be there already -- this script errors out early if not, rather
# than silently launching a redundant preprocessing pass).
# Needs train_task2.sbatch's checkpoints/ to exist. Run this AFTER that job
# finishes, not concurrently (shares the same lisa-task2:latest image + GPU).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-eval-test
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_evaltest_${SLURM_JOB_ID}"

# Individually-scored encoders (each restricted to its own 4 LOMO-fold
# checkpoints) plus "all" (the full ensemble, no --encoders restriction) for
# a reference point against the per-encoder numbers.
ENCODERS_LIST="densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16 all"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -f "$PROJECT_ROOT/Data/test.csv" ]; then
    echo "!! $PROJECT_ROOT/Data/test.csv missing." >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch0" ]; then
    echo "!! $PROJECT_ROOT/Data/Preprocessed/test_images/ch0 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 (preprocess.py --lung-crop) first." >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/checkpoints" ]; then
    echo "!! $PROJECT_ROOT/checkpoints missing -- train_task2.sbatch hasn't produced" >&2
    echo "!! any checkpoints yet. Nothing to score." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

# NOTE: docker images are local to each node's daemon, not cluster-wide --
# train_task2.sbatch building the image on gpu101 doesn't make it available
# on gpu102 or any other node this job might land on. Build it here too
# (same as train_task2.sbatch/run_external_validation.sbatch should also do
# -- this was the actual cause of job 117992's "image not found" failure on
# gpu102) rather than assuming it already exists.
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"   # .../Task2_0615 (has Dockerfile_final)
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights)

echo "############################################################"
echo "### scoring Data/test.csv per encoder + full ensemble ###"
echo "############################################################"

# Collects all 4 scores (accuracy@0.5, AUC, F1@0.5, accuracy@best-tau) per
# encoder into one clean table at the end, instead of leaving them scattered
# across 6 separate docker/CUDA-boilerplate-heavy blocks -- exactly what made
# job 117994's log tedious to compare by eye.
SUMMARY_CSV="$PROJECT_ROOT/test_eval_summary.csv"
echo "encoder,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for ENC in $ENCODERS_LIST; do
  # Plain string, not a bash array -- an earlier array-based version of this
  # (${ENCODERS_FLAG[@]} interpolated into the docker `bash -c "..."` string)
  # silently failed to restrict the ensemble at runtime: job 117994's
  # "densenet121" stage still logged "wrote submission_test_densenet121.csv
  # (20 models ensembled)", i.e. --encoders never reached predict_task2.py
  # even though the same construct checked out fine in isolated bash/argparse
  # tests. Root cause not fully pinned down (likely something about array
  # expansion crossing the docker/bash-in-bash boundary) -- a plain string
  # sidesteps it rather than chasing it further.
  ENCODERS_ARG=""
  if [ "$ENC" != "all" ]; then
    ENCODERS_ARG="--encoders $ENC"
  fi
  echo "--- test.csv: $ENC ---"
  RAW_LOG="$PROJECT_ROOT/.eval_test_${ENC}.raw.log"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_${ENC}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/checkpoints \
        --csv /workspace/Data/test.csv \
        --image-dir /workspace/Data/Preprocessed/test_images/ch0 \
        --out /workspace/submission_test_${ENC}.csv --score \
        $ENCODERS_ARG" 2>&1 | tee "$RAW_LOG"

  SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
  TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
  ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
  AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
  F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
  BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
  ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
  echo "${ENC},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
  rm -f "$RAW_LOG"
done

echo "=== TEST-SET EVAL DONE ==="
echo "=== SUMMARY: all 4 scores per encoder (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"
