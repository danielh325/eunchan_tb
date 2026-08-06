#!/bin/bash
# ============================================================================
# Task 2 -- scores baseline_rad_dino ("everything on": leave-one-modality-out
# + strong augmentation + domain-aware mixup, all defaults) plus every
# domain-generalization-machinery ablation config from
# train_task2_rad_dino_dg_ablation.sbatch against Data/test.csv and all 3
# external datasets, one consolidated table -- isolating whether the LOMO
# fold-mode, the augmentation bundle, and domain-aware mixup each actually
# contribute anything on real external data, rather than trusting the
# bundled "everything on" result alone.
#
# GRACEFUL SKIP: any ablation config not yet trained is skipped with a
# warning, not a hard failure -- run this progressively as each of the 6
# train_task2_rad_dino_dg_ablation.sbatch configs finishes (stratified_fold,
# no_dg, light_aug_no_mixup, strong_aug_no_mixup, mixup_only, no_crop --
# 7 rows per dataset once "full"/baseline_rad_dino is counted).
#
# NOTE: configs trained from 2026-08-02 onward self-score against the same
# 4 datasets in their own training job (STAGE 2 of the train script), so this
# script is for (a) the three configs trained before that -- stratified_fold,
# no_dg, light_aug_no_mixup -- and (b) one consolidated table across all of
# them plus the baseline.
#
# PREREQUISITE: the usual ch0 preprocessed dirs for test.csv + all 3
# external datasets must already exist (they do, from earlier eval jobs).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-rad-dino-dg-ablation-eval
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_dgablation_eval_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

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

# --- dataset table (relative to $PROJECT_ROOT) -- CROPPED (ch0) versions,
# used by every config except no_crop. ---
DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

# --- SAME datasets, RAW (uncropped) versions -- ONLY for no_crop, whose
# checkpoint was trained on raw images and would get badly mismatched input
# if scored against the cropped ch0 dirs instead. ---
DATASETS_IMG_RAW=(Data/test Data/external/shenzhen/CXR_png Data/external/montgomery/CXR_png Data/external/tbx11k/raw_full)

# --- config table: name, ckpt-dir. "full" = baseline_rad_dino, the
# already-trained "everything on" reference point -- not retrained. ---
CONFIG_NAME=(full stratified_fold no_dg light_aug_no_mixup strong_aug_no_mixup mixup_only no_crop)
CONFIG_CKPT_DIR=(checkpoints checkpoints_dgablation_stratified_fold checkpoints_dgablation_no_dg checkpoints_dgablation_light_aug_no_mixup checkpoints_dgablation_strong_aug_no_mixup checkpoints_dgablation_mixup_only checkpoints_dgablation_no_crop)

# A CONFIGS-filtered run writes its own file so it can't truncate a full
# consolidated table down to the subset it happened to score.
if [ -n "${CONFIGS:-}" ]; then
  SUMMARY_CSV="$PROJECT_ROOT/rad_dino_dg_ablation_summary_$(echo "$CONFIGS" | tr ' ' '+').csv"
else
  SUMMARY_CSV="$PROJECT_ROOT/rad_dino_dg_ablation_summary.csv"
fi
echo "config,dataset,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for c in "${!CONFIG_NAME[@]}"; do
  CNAME="${CONFIG_NAME[$c]}"
  CCKPT="${CONFIG_CKPT_DIR[$c]}"

  # CONFIGS="a b c" scores only those configs. Without it every config is
  # scored, which is what you want for the final consolidated table but NOT
  # while a config is still training -- predict_task2.py ensembles whatever
  # folds it finds, so a half-trained dir yields a plausible-looking row built
  # from 1 of 4 folds, and the running job may drop another fold mid-eval.
  if [ -n "${CONFIGS:-}" ] && ! echo " $CONFIGS " | grep -q " $CNAME "; then
    continue
  fi

  # Configs trained before the out-dir convention changed live one level
  # deeper, under checkpoints/ (stratified_fold, no_dg, light_aug_no_mixup).
  # Resolve either layout rather than hardcoding, so this keeps working
  # whenever those dirs get moved up to match the newer ones. The resolved
  # path is relative to PROJECT_ROOT, which is what gets mounted at
  # /workspace, so it stays valid inside the container too.
  if [ ! -d "$PROJECT_ROOT/$CCKPT" ] && [ -d "$PROJECT_ROOT/checkpoints/$CCKPT" ]; then
    CCKPT="checkpoints/$CCKPT"
  fi

  if [ ! -d "$PROJECT_ROOT/$CCKPT" ]; then
    echo "!! skipping config '$CNAME' -- $PROJECT_ROOT/$CCKPT does not exist yet" >&2
    continue
  fi

  N_FOLDS="$(ls -1 "$PROJECT_ROOT/$CCKPT"/rad_dino_*.pth 2>/dev/null | wc -l | tr -d ' ')"
  echo "config '$CNAME': $N_FOLDS fold checkpoint(s) in $CCKPT"

  echo "############################################################"
  echo "### config: $CNAME ($CCKPT) ###"
  echo "############################################################"

  for d in "${!DATASETS_NAME[@]}"; do
    DS="${DATASETS_NAME[$d]}"
    echo "--- $CNAME / $DS ---"

    # no_crop's checkpoint was trained on RAW images -- score it against
    # the matching raw dirs, not the lung-cropped ch0 ones every other
    # config uses, or the input distribution silently won't match training.
    IMG_DIR="${DATASETS_IMG[$d]}"
    if [ "$CNAME" = "no_crop" ]; then
      IMG_DIR="${DATASETS_IMG_RAW[$d]}"
    fi

    RAW_LOG="$PROJECT_ROOT/.eval_dgablation_${CNAME}_${DS}.raw.log"
    "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_${CNAME}_${DS}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$CCKPT \
          --csv /workspace/${DATASETS_CSV[$d]} \
          --image-dir /workspace/$IMG_DIR \
          --encoders rad_dino \
          --out /workspace/submission_dgablation_${CNAME}_${DS}.csv --score" 2>&1 | tee "$RAW_LOG"

    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
    ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
    echo "${CNAME},${DS},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
    rm -f "$RAW_LOG"
  done
done

echo "=== DG-MACHINERY ABLATION STUDY DONE ==="
echo "=== SUMMARY (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "Read F1@0.5 per (config, dataset), external datasets (shenzhen/montgomery/"
echo "tbx11k_full) are what matter -- test.csv is same-distribution and known to"
echo "be misleading (see memory: task2_training_status). If 'no_dg' (zero"
echo "augmentation/mixup, just LOMO folding) scores close to 'full', the strong-"
echo "aug+mixup bundle isn't earning its complexity; if 'stratified_fold' scores"
echo "close to 'full' too, even the LOMO split itself isn't doing much of the work."
