#!/bin/bash
# ============================================================================
# Task 2 -- ablation of the domain-generalization MACHINERY itself for
# rad_dino: leave-one-modality-out fold-mode, the "strong" augmentation
# bundle (RandConv, spectral amplitude perturbation, resolution jitter,
# affine, color jitter, blur -- see domain_gen.py's build_augmentation),
# and domain-aware cross-modality mixup. All three have been ON, bundled
# together, in EVERY rad_dino run this session (including baseline_rad_dino,
# the standing recommendation) -- nobody has isolated whether any ONE of
# them actually contributes anything on real external validation, the same
# gap flagged for Task1's ResEnc-planner+FocalTversky-loss combo earlier
# (changing multiple things at once means a win/loss doesn't say which
# change mattered).
#
# baseline_rad_dino (checkpoints/) IS the "everything on" config already
# (fold-mode=leave_one_modality_out, aug-strength=strong, domain-mixup-
# alpha=0.2 -- all train_task2.py defaults) -- NOT retrained here, just
# used as the reference point in eval_rad_dino_dg_ablation.sbatch.
#
# This script trains ONE named ablation config per invocation (selected via
# the ABLATION env var), writing to its own checkpoints_dgablation_<name>/ --
# submit it multiple times (one per config) rather than one long sequential
# job, so they can run in parallel across whatever GPU queue slots are
# available instead of serially for weeks.
#
# Configs (each isolates a different piece from the "everything on" baseline):
#   stratified_fold      -- fold-mode=stratified instead of leave_one_modality_out,
#                            aug-strength=strong, mixup=0.2 (isolates: does the
#                            domain-aware LOMO split itself matter, vs a plain
#                            random split with the same augmentation/mixup?)
#   no_dg                -- fold-mode=leave_one_modality_out, aug-strength=none,
#                            mixup=0 (the floor: zero DG-specific techniques,
#                            only LOMO splitting + the base preprocessing)
#   light_aug_no_mixup    -- fold-mode=leave_one_modality_out, aug-strength=light,
#                            mixup=0 (isolates: does light augmentation alone help,
#                            without the strong DG-specific transforms or mixup?)
#   strong_aug_no_mixup   -- fold-mode=leave_one_modality_out, aug-strength=strong,
#                            mixup=0 (isolates: the strong augmentation bundle
#                            ALONE, without domain-aware mixup)
#   mixup_only            -- fold-mode=leave_one_modality_out, aug-strength=none,
#                            mixup=0.2 (isolates: domain-aware mixup ALONE,
#                            without any image-level augmentation)
#   no_crop               -- same as "everything on" (LOMO, strong aug, mixup=0.2)
#                            EXCEPT --image-dir points to Data/train (RAW,
#                            uncropped -- matches train.csv row-for-row) instead
#                            of the lung-cropped ch0 dir. Isolates: does the
#                            PSPNet lung-crop bounding box itself matter for
#                            rad_dino's generalization? Dual-stream already
#                            showed a catastrophic external collapse (Shenzhen
#                            F1=0.41, Montgomery F1=0.21) with an uncropped
#                            branch FUSED with a cropped one -- this tests the
#                            crop's effect alone, on our actual best single-
#                            stream model, without that fusion-architecture
#                            confound. IMPORTANT: this checkpoint must be
#                            scored against RAW images at predict time too
#                            (Data/test, Data/external/*/CXR_png,
#                            Data/external/tbx11k/raw_full), not ch0 --
#                            eval_rad_dino_dg_ablation.sbatch already handles
#                            this per-config.
#
# RUN SO FAR: stratified_fold (job 119897), no_dg (119898),
# light_aug_no_mixup (119967) -- all three trained BEFORE this script grew
# its STAGE 2 eval, so they have checkpoints but no per-config eval CSV;
# score them with eval_rad_dino_dg_ablation.sbatch instead.
# NOT YET RUN: strong_aug_no_mixup, mixup_only, no_crop.
#
# STAGE 2 (added after those first three runs): every invocation now scores
# its own checkpoint against Data/test.csv AND all 3 external datasets
# (shenzhen, montgomery, tbx11k_full) as soon as training finishes, writing
# rad_dino_dgablation_<name>_eval_summary.csv. External F1@0.5 is the number
# that actually ranks these configs -- LOMO val_acc came out 0.9868-0.9997
# for all three finished configs, i.e. completely non-discriminative -- so
# waiting on a separate eval job just delays the only useful signal. Set
# SKIP_EVAL=1 to train only. The standalone eval_rad_dino_dg_ablation.sbatch
# is still the way to get one consolidated table across ALL configs plus the
# baseline; this stage is the per-config early read, and the two agree
# because they invoke predict_task2.py identically.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch0 must exist (for
# every config except no_crop, which instead needs Data/train -- already
# present, matches train.csv row-for-row). STAGE 2 additionally needs the
# preprocessed external dirs (present since the earlier external-eval jobs);
# any dataset whose dir is missing is skipped with a warning rather than
# failing the job -- training is already banked by then.
#
# Usage:
#   ABLATION=stratified_fold sbatch train_task2_rad_dino_dg_ablation.sbatch
#   ABLATION=no_dg sbatch train_task2_rad_dino_dg_ablation.sbatch
#   ABLATION=light_aug_no_mixup sbatch train_task2_rad_dino_dg_ablation.sbatch
#   ABLATION=strong_aug_no_mixup sbatch train_task2_rad_dino_dg_ablation.sbatch
#   ABLATION=mixup_only sbatch train_task2_rad_dino_dg_ablation.sbatch
#   ABLATION=no_crop sbatch train_task2_rad_dino_dg_ablation.sbatch
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-rad-dino-dg-ablation
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # rad_dino weights live here
IMAGE="lisa-task2:latest"

ABLATION="${ABLATION:?set ABLATION to one of: stratified_fold, no_dg, light_aug_no_mixup, strong_aug_no_mixup, mixup_only, no_crop}"

# IMAGE_DIR defaults to the lung-cropped ch0 dir every other config uses;
# no_crop overrides it to Data/train (RAW, uncropped -- matches train.csv
# row-for-row, same raw dir dual-stream's global branch already used).
# Isolates: does the PSPNet lung-crop bounding box itself matter for
# rad_dino's generalization, independent of any fusion architecture (dual-
# stream already showed a catastrophic external collapse when an uncropped
# branch was fused with a cropped one -- this tests the crop alone, on our
# actual best single-stream model, without that confound).
IMAGE_DIR="/workspace/Data/Preprocessed/train_images/ch0"

case "$ABLATION" in
  stratified_fold)    FOLD_MODE="stratified";              AUG_STRENGTH="strong"; MIXUP_ALPHA="0.2" ;;
  no_dg)               FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="none";   MIXUP_ALPHA="0.0" ;;
  light_aug_no_mixup)  FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="light";  MIXUP_ALPHA="0.0" ;;
  strong_aug_no_mixup) FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="strong"; MIXUP_ALPHA="0.0" ;;
  mixup_only)          FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="none";   MIXUP_ALPHA="0.2" ;;
  no_crop)             FOLD_MODE="leave_one_modality_out";  AUG_STRENGTH="strong"; MIXUP_ALPHA="0.2"
                        IMAGE_DIR="/workspace/Data/train" ;;
  *) echo "!! unknown ABLATION='$ABLATION' -- see this script's header for valid names" >&2; exit 1 ;;
esac

CONTAINER_NAME="task2_dgablation_${ABLATION}_${SLURM_JOB_ID}"
OUT_DIR="checkpoints_dgablation_${ABLATION}"
FOLD_ARGS="--fold-mode $FOLD_MODE"
if [ "$FOLD_MODE" = "stratified" ]; then
  FOLD_ARGS="$FOLD_ARGS --n-folds 5"
fi

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "ablation config: $ABLATION"
echo "  fold-mode=$FOLD_MODE  aug-strength=$AUG_STRENGTH  domain-mixup-alpha=$MIXUP_ALPHA  image-dir=$IMAGE_DIR"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ "$ABLATION" = "no_crop" ]; then
  if [ ! -d "$PROJECT_ROOT/Data/train" ]; then
    echo "!! Data/train (raw, uncropped training images) missing." >&2
    exit 1
  fi
elif [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch0" ]; then
    echo "!! Data/Preprocessed/train_images/ch0 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 first." >&2
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

echo "############################################################"
echo "### full run, rad_dino, DG-ablation config: $ABLATION ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py --encoder rad_dino \
      --csv /workspace/Data/train.csv \
      --image-dir $IMAGE_DIR \
      $FOLD_ARGS \
      --aug-strength $AUG_STRENGTH \
      --domain-mixup-alpha $MIXUP_ALPHA \
      --out-dir /workspace/$OUT_DIR \
      --epochs 15"

echo "=== rad_dino DG-ablation ($ABLATION) TRAINING DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="

# ============================================================================
# STAGE 2 -- score this config on test.csv + all 3 external datasets.
# ============================================================================
if [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "SKIP_EVAL=1 -- stopping after training; score later with eval_rad_dino_dg_ablation.sbatch"
  exit 0
fi

DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

# no_crop trained on RAW images, so it must be scored against raw dirs too --
# scoring it against the lung-cropped ch0 dirs would silently mismatch the
# input distribution and make the crop look worse than it is. Mirrors the
# same per-config switch in eval_rad_dino_dg_ablation.sbatch.
DATASETS_IMG_RAW=(Data/test Data/external/shenzhen/CXR_png Data/external/montgomery/CXR_png Data/external/tbx11k/raw_full)

EVAL_CSV="$PROJECT_ROOT/rad_dino_dgablation_${ABLATION}_eval_summary.csv"
echo "config,dataset,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$EVAL_CSV"

echo "############################################################"
echo "### STAGE 2: external validation, config: $ABLATION ###"
echo "############################################################"

for d in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$d]}"

  IMG_DIR="${DATASETS_IMG[$d]}"
  if [ "$ABLATION" = "no_crop" ]; then
    IMG_DIR="${DATASETS_IMG_RAW[$d]}"
  fi

  if [ ! -f "$PROJECT_ROOT/${DATASETS_CSV[$d]}" ] || [ ! -d "$PROJECT_ROOT/$IMG_DIR" ]; then
    echo "!! skipping $DS -- ${DATASETS_CSV[$d]} or $IMG_DIR missing" >&2
    echo "${ABLATION},${DS},NA,NA,NA,NA,NA" >> "$EVAL_CSV"
    continue
  fi

  echo "--- $ABLATION / $DS ---"
  RAW_LOG="$PROJECT_ROOT/.eval_dgablation_${ABLATION}_${DS}.raw.log"

  # Training is already banked by now, so one failed dataset must not take the
  # job down with it: record NA and carry on to the next.
  if "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_eval_${DS}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$OUT_DIR \
          --csv /workspace/${DATASETS_CSV[$d]} \
          --image-dir /workspace/$IMG_DIR \
          --encoders rad_dino \
          --out /workspace/submission_dgablation_${ABLATION}_${DS}.csv --score" > "$RAW_LOG" 2>&1; then
    cat "$RAW_LOG"
    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
    ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
    echo "${ABLATION},${DS},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$EVAL_CSV"
  else
    echo "!! scoring FAILED on $DS -- output follows" >&2
    cat "$RAW_LOG" >&2
    echo "${ABLATION},${DS},NA,NA,NA,NA,NA" >> "$EVAL_CSV"
  fi
  rm -f "$RAW_LOG"
done

echo "=== rad_dino DG-ablation ($ABLATION) EVAL DONE -> $EVAL_CSV ==="
column -s, -t "$EVAL_CSV"
echo ""
echo "Read F1@0.5 on the EXTERNAL rows (shenzhen/montgomery/tbx11k_full) -- test.csv"
echo "is same-distribution and saturated at ~0.99 for every variant. Compare against"
echo "baseline_rad_dino (the 'everything on' config: LOMO + strong aug + mixup 0.2):"
echo "  shenzhen=0.8954  montgomery=0.8780  tbx11k_full=0.3225  test=0.9890"
echo "If this config lands close to baseline, the piece it removed wasn't earning its"
echo "complexity. Run eval_rad_dino_dg_ablation.sbatch once all 6 configs exist for"
echo "one consolidated table including the three configs trained before this stage."
