#!/bin/bash
# ============================================================================
# Task 2 -- compares the current 4-fold LOMO ensemble (CR/DX/XA/XC) against a
# 3-fold version with the XA checkpoint dropped, for rad_dino
# strong_aug_no_mixup (our recommended config -- see memory:
# task2_dg_ablation_results), on test.csv + shenzhen + montgomery + tbx11k_full.
#
# WHY: XA has only 11 training rows, all TB-positive -- its own held-out LOMO
# score is meaningless (single-class -> safe_auc() returns NaN; accuracy/F1
# on it just measures "did the model call TB positive", trivially gameable).
# test.csv itself has ZERO XA rows (CR=866/DX=537/XC=537 only), so this isn't
# even a domain the submission is ever scored against. The 11 XA images are
# NOT lost by dropping the fold -- they're still inside every OTHER fold's
# training set (CR/DX/XC's train_df excludes only its own held-out modality),
# so rad_dino_XA.pth is a near-duplicate bagging member of the other three
# (trained on ~99.9% the same data), not a genuinely different domain view.
# This job checks whether dropping it changes ensemble performance at all.
#
# 4-fold baseline numbers already exist in
# CSV/master_results_all_models.csv (model=dgablation_strong_aug_no_mixup)
# from an earlier run -- this job RE-scores the 4-fold ensemble fresh
# (rather than trusting that file) so both rows come from the exact same
# code path/environment and are directly comparable.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-4fold-vs-3fold-raddino
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
CONTAINER_NAME="task2_4v3fold_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true; rm -rf "$PROJECT_ROOT/.ckpt_3fold_${SLURM_JOB_ID}"' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

FULL_CKPT_DIR="checkpoints/checkpoints_dgablation_strong_aug_no_mixup"

# 3-fold dir: symlink every checkpoint EXCEPT rad_dino_XA.pth. Built on the
# host (not in-container) so it's a plain local dir predict_task2.py's glob
# sees like any other --ckpt-dir.
#
# MUST be a RELATIVE symlink target. An earlier version used `ln -s "$f" ...`
# with $f an absolute HOST path (e.g. /mnt/nas125/.../checkpoints/...) --
# that path doesn't exist inside the container, which only has $PROJECT_ROOT
# bind-mounted at /workspace, so predict_task2.py's torch.load() hit
# FileNotFoundError on every checkpoint. A relative target
# (../checkpoints/.../rad_dino_CR.pth) still resolves correctly once the
# whole PROJECT_ROOT tree -- symlink and real file both -- is remounted
# under /workspace.
THREEFOLD_DIR="$PROJECT_ROOT/.ckpt_3fold_${SLURM_JOB_ID}"
mkdir -p "$THREEFOLD_DIR"
for f in "$PROJECT_ROOT/$FULL_CKPT_DIR"/rad_dino_*.pth; do
  base="$(basename "$f")"
  if [ "$base" != "rad_dino_XA.pth" ]; then
    ln -s "../$FULL_CKPT_DIR/$base" "$THREEFOLD_DIR/$base"
  fi
done
echo "3-fold dir contents:"
ls -la "$THREEFOLD_DIR"

DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

SUMMARY_CSV="$PROJECT_ROOT/rad_dino_4fold_vs_3fold_summary.csv"
echo "ensemble,dataset,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for ENSEMBLE in "4fold_full" "3fold_no_xa"; do
  if [ "$ENSEMBLE" = "4fold_full" ]; then
    CKPT_MOUNT_DIR="$FULL_CKPT_DIR"
  else
    CKPT_MOUNT_DIR=".ckpt_3fold_${SLURM_JOB_ID}"
  fi

  echo "############################################################"
  echo "### ensemble: $ENSEMBLE ($CKPT_MOUNT_DIR) ###"
  echo "############################################################"

  for d in "${!DATASETS_NAME[@]}"; do
    DS="${DATASETS_NAME[$d]}"
    echo "--- $ENSEMBLE / $DS ---"

    RAW_LOG="$PROJECT_ROOT/.eval_4v3fold_${ENSEMBLE}_${DS}.raw.log"
    "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_${ENSEMBLE}_${DS}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$CKPT_MOUNT_DIR \
          --csv /workspace/${DATASETS_CSV[$d]} \
          --image-dir /workspace/${DATASETS_IMG[$d]} \
          --encoders rad_dino \
          --out /workspace/submission_4v3fold_${ENSEMBLE}_${DS}.csv --score" 2>&1 | tee "$RAW_LOG"

    N_MODELS="$(grep -m1 '^wrote ' "$RAW_LOG" | sed -n 's/.*(\([0-9]*\) models ensembled).*/\1/p')"
    echo "    ($N_MODELS model(s) ensembled)"

    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
    ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
    echo "${ENSEMBLE},${DS},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
    rm -f "$RAW_LOG"
  done
done

echo "=== 4-FOLD vs 3-FOLD (XA DROPPED) DONE ==="
echo "=== SUMMARY (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "F1@0.5 is the metric that matters (see memory: task2_f1_is_the_metric)."
echo "If 3fold_no_xa is within noise of 4fold_full on every dataset, the XA"
echo "checkpoint was a redundant bagging member and dropping it (and its"
echo "meaningless single-class held-out LOMO score) is a clean simplification."
