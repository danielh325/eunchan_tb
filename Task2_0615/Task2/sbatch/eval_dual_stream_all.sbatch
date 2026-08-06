#!/bin/bash
# ============================================================================
# Task 2 — score the dual-stream (GlobalLocalTBClassifier) ensemble against
# Data/test.csv AND all 3 external datasets (Shenzhen, Montgomery,
# TBX11K-full), the same rigor already applied to every other variant
# (see eval_test_all_variants.sbatch / eval_external_all_variants.sbatch).
#
# train_dual_stream.sbatch (job 119181) already completed with a strong LOMO
# number (mean val_acc=0.9879) -- but LOMO/internal numbers have repeatedly
# NOT predicted external performance for every other model tried so far
# (see memory: task2_submission_vs_recommended_model, task2_training_status).
# This is the run that actually tells us whether dual-stream generalizes, or
# is another internal-only number.
#
# Unlike predict_task2.py's single-image-per-checkpoint design, dual-stream
# needs TWO image dirs per dataset: --global-image-dir (RAW, uncropped --
# the whole point of the "global" branch is seeing context a lung-crop
# throws away) and --local-image-dir (preprocess.py's ch0, same lung-crop
# every other variant uses). No --mask-dir here at all -- this architecture
# doesn't use the PSPNet mask in any form (neither channel nor guided
# attention), it substitutes "give the model the uncropped image too" as its
# own way of injecting broader context, a third distinct design point from
# the mask-usage ablation.
#
# PREREQUISITE: checkpoints_dual/ must exist (train_dual_stream.sbatch, job
# 119181, already done). PREREQUISITE: the usual ch0 preprocessed dirs for
# each dataset (Data/Preprocessed/test_images/ch0, Data/external/{shenzhen,
# montgomery}/preprocessed/ch0, Data/external/tbx11k/preprocessed_full/ch0)
# must already exist -- they do, from earlier eval jobs.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-eval-dual-stream-all
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_evaldual_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -d "$PROJECT_ROOT/checkpoints_dual" ]; then
    echo "!! $PROJECT_ROOT/checkpoints_dual missing -- train_dual_stream.sbatch hasn't run." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

# --- dataset table: name, csv, global(raw)-dir, local(ch0)-dir (relative to $PROJECT_ROOT) ---
DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_GLOBAL=(Data/test Data/external/shenzhen/CXR_png Data/external/montgomery/CXR_png Data/external/tbx11k/raw_full)
DATASETS_LOCAL=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

for i in "${!DATASETS_NAME[@]}"; do
  if [ ! -f "$PROJECT_ROOT/${DATASETS_CSV[$i]}" ] || [ ! -d "$PROJECT_ROOT/${DATASETS_GLOBAL[$i]}" ] || [ ! -d "$PROJECT_ROOT/${DATASETS_LOCAL[$i]}" ]; then
    echo "!! ${DATASETS_NAME[$i]}: missing csv/global-dir/local-dir -- check paths above." >&2
    exit 1
  fi
done

echo "############################################################"
echo "### score dual-stream (checkpoints_dual/) against test.csv + 3 external datasets ###"
echo "############################################################"

SUMMARY_CSV="$PROJECT_ROOT/dual_stream_eval_summary.csv"
echo "dataset,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for i in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$i]}"
  echo "--- $DS ---"
  RAW_LOG="$PROJECT_ROOT/.eval_dual_${DS}.raw.log"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_dual_stream.py \
        --ckpt-dir /workspace/checkpoints_dual \
        --csv /workspace/${DATASETS_CSV[$i]} \
        --global-image-dir /workspace/${DATASETS_GLOBAL[$i]} \
        --local-image-dir /workspace/${DATASETS_LOCAL[$i]} \
        --out /workspace/submission_dual_${DS}.csv --score" 2>&1 | tee "$RAW_LOG"

  SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
  TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
  ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
  AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
  F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
  BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
  ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
  echo "${DS},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
  rm -f "$RAW_LOG"
done

echo "=== DUAL-STREAM FULL EVAL DONE ==="
echo "=== SUMMARY (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "Compare F1@0.5 here against baseline_rad_dino's numbers (Shenzhen 0.8954,"
echo "Montgomery 0.8780, TBX11K-full 0.3225, test.csv 0.9890 -- see memory:"
echo "task2_submission_vs_recommended_model) to see whether the extra global-"
echo "context branch actually helps generalization or is another internal-only win."
