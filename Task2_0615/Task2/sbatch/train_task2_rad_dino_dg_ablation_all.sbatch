#!/bin/bash
# ============================================================================
# Task 2 -- runs ALL 6 rad_dino domain-generalization-machinery ablation
# configs SEQUENTIALLY in one job (stratified_fold, no_dg, light_aug_no_mixup,
# strong_aug_no_mixup, mixup_only, no_crop -- see
# train_task2_rad_dino_dg_ablation.sbatch's header for what each isolates).
#
# COST WARNING, READ BEFORE SUBMITTING: this is 6 full rad_dino
# leave-one-modality-out (or 5-fold stratified, for stratified_fold) training
# runs back to back on a SINGLE GPU -- if the earlier ~79s/epoch reference
# point for a rad_dino-class ResEnc config from Task1's nnU-Net work is any
# guide, each 15-epoch/4-fold run is a real multi-day job on its own, so 6 of
# them sequentially is plausibly 1-2+ WEEKS of wall-clock time. If your
# cluster has more than one GPU available, or a queue where jobs run in
# parallel, submitting the 6 configs as SEPARATE jobs (see
# train_task2_rad_dino_dg_ablation.sbatch's own Usage section, one
# `ABLATION=<name> sbatch ...` per config) finishes far sooner in wall-clock
# terms, since they don't depend on each other -- this script exists for the
# case where only one GPU slot is realistically available and running
# unattended-sequential is preferred over juggling 6 manual submissions.
#
# Each config writes to its own checkpoints_dgablation_<name>/, exactly like
# submitting them individually -- this script is just a convenience wrapper
# that loops the same per-config training call train_task2_rad_dino_dg_
# ablation.sbatch uses, IN-PROCESS (one docker container per config, one
# after another), so a mid-list failure doesn't lose the configs that
# already finished (each one's checkpoint is saved to disk before moving on
# to the next).
#
# If ANY config in the list fails (non-zero exit), this script logs it and
# CONTINUES to the next one rather than aborting the whole job -- you don't
# want one bad config (e.g. a transient docker/GPU hiccup) to silently skip
# training the other 5 you actually wanted.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch0 must exist (every
# config except no_crop); Data/train must exist (no_crop only -- already
# present, matches train.csv row-for-row).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — confirm your cluster's max walltime allows a request this long, or
# split this into fewer configs per job / use separate submissions instead.
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-rad-dino-dg-ablation-all
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -uo pipefail   # NOT -e: one config failing must not abort the rest (see header)

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # rad_dino weights live here
IMAGE="lisa-task2:latest"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST -- running ALL 6 DG-ablation configs sequentially"

if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch0" ]; then
    echo "!! Data/Preprocessed/train_images/ch0 missing -- run train_task2.sbatch's Stage 1 first." >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/Data/train" ]; then
    echo "!! Data/train (raw, uncropped) missing -- needed for the no_crop config." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

ABLATIONS=(stratified_fold no_dg light_aug_no_mixup strong_aug_no_mixup mixup_only no_crop)
FAILED=()

for ABLATION in "${ABLATIONS[@]}"; do
  echo "############################################################"
  echo "### STARTING config: $ABLATION ###"
  echo "############################################################"

  # Same per-config parameter table as train_task2_rad_dino_dg_ablation.sbatch --
  # kept in sync manually since bash has no clean way to source another
  # script's case-statement without also re-running its SBATCH/env setup.
  IMAGE_DIR="/workspace/Data/Preprocessed/train_images/ch0"
  case "$ABLATION" in
    stratified_fold)     FOLD_MODE="stratified";              AUG_STRENGTH="strong"; MIXUP_ALPHA="0.2" ;;
    no_dg)                FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="none";   MIXUP_ALPHA="0.0" ;;
    light_aug_no_mixup)   FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="light";  MIXUP_ALPHA="0.0" ;;
    strong_aug_no_mixup)  FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="strong"; MIXUP_ALPHA="0.0" ;;
    mixup_only)           FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="none";   MIXUP_ALPHA="0.2" ;;
    no_crop)              FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="strong"; MIXUP_ALPHA="0.2"
                          IMAGE_DIR="/workspace/Data/train" ;;
  esac

  OUT_DIR="checkpoints_dgablation_${ABLATION}"
  FOLD_ARGS="--fold-mode $FOLD_MODE"
  if [ "$FOLD_MODE" = "stratified" ]; then
    FOLD_ARGS="$FOLD_ARGS --n-folds 5"
  fi

  CONTAINER_NAME="task2_dgablation_all_${ABLATION}_${SLURM_JOB_ID}"
  trap "docker rm -f '$CONTAINER_NAME' 2>/dev/null || true" EXIT

  "${RUN_DOCKER[@]}" \
    --name "$CONTAINER_NAME" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python train_task2.py --encoder rad_dino \
        --csv /workspace/Data/train.csv \
        --image-dir $IMAGE_DIR \
        $FOLD_ARGS \
        --aug-strength $AUG_STRENGTH \
        --domain-mixup-alpha $MIXUP_ALPHA \
        --out-dir /workspace/$OUT_DIR \
        --epochs 15"

  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "!! config '$ABLATION' FAILED (exit $STATUS) -- continuing to the next config" >&2
    FAILED+=("$ABLATION")
  else
    echo "=== config '$ABLATION' DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="
  fi
done

echo "############################################################"
echo "### ALL 6 DG-ABLATION CONFIGS ATTEMPTED ###"
echo "############################################################"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "!! ${#FAILED[@]} config(s) FAILED: ${FAILED[*]} -- check this log's output above for each," >&2
  echo "!! then re-run just those with: ABLATION=<name> sbatch train_task2_rad_dino_dg_ablation.sbatch" >&2
else
  echo "All 6 configs completed successfully."
fi
echo ""
echo "Next: sbatch eval_rad_dino_dg_ablation.sbatch to score everything"
echo "(baseline_rad_dino + all 6 configs) against test.csv and all 3 external datasets."
