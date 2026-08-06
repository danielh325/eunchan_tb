#!/bin/bash
# ============================================================================
# Task 2 -- consolidated ablation study for rad_dino: scores EVERY rad_dino
# variant tried this session against Data/test.csv + all 3 external datasets
# (Shenzhen, Montgomery, TBX11K-full), in one pass, one summary table.
#
# Variants covered:
#   baseline      checkpoints/                    -- no mask, CLAHE-tripled input (the
#                                                     standing recommendation, see memory:
#                                                     task2_submission_vs_recommended_model)
#   maskch        checkpoints_mask_ablation/       -- passive PSPNet mask spliced into ch3
#   maskguidedvit checkpoints_mask_guided_vit/     -- Dice-supervised per-patch-token
#                                                     attention gate, no mask at inference
#   irm           checkpoints_irm/                 -- Invariant Risk Minimization over
#                                                     Modality_DICOM environments
#   multiview     checkpoints_rad_dino_multiview/  -- channel 0=original, 1=CLAHE,
#                                                     2=bone-suppressed (--channel-mode
#                                                     multi_view), instead of one view tripled
#
# Each variant needing its own preprocessing flag is baked into the table
# below (mask-dir for maskch, channel-mode for multiview) so this is a
# single, correct invocation per (variant, dataset) pair -- no manual
# flag-matching required at the call site, which is exactly the kind of
# thing that silently produces wrong numbers if done by hand per
# predict_task2.py's own repeated train/predict-consistency warnings.
#
# GRACEFUL SKIP: any variant whose checkpoint dir doesn't exist yet is
# skipped with a warning, not a hard failure -- run this progressively as
# each training job (train_task2_irm_rad_dino.sbatch,
# train_task2_rad_dino_multiview.sbatch) finishes, rather than needing
# every variant done up front.
#
# PREREQUISITE: the usual ch0 (and ch2, for maskch) preprocessed dirs for
# test.csv + all 3 external datasets must already exist (they do, from
# earlier eval jobs this session).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-rad-dino-ablation-all
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
CONTAINER_NAME="task2_raddino_ablation_${SLURM_JOB_ID}"

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

# --- dataset table (relative to $PROJECT_ROOT) ---
DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)
DATASETS_MASK=(Data/Preprocessed/test_images/ch2 Data/external/shenzhen/preprocessed/ch2 Data/external/montgomery/preprocessed/ch2 Data/external/tbx11k/preprocessed_full/ch2)

# --- variant table: tag, ckpt-dir, needs-mask-dir(0/1), channel-mode ---
VARIANT_NAME=(baseline maskch maskguidedvit irm multiview)
VARIANT_CKPT_DIR=(checkpoints checkpoints_mask_ablation checkpoints_mask_guided_vit checkpoints_irm checkpoints_rad_dino_multiview)
VARIANT_NEEDS_MASK=(0 1 0 0 0)
VARIANT_CHANNEL_MODE=(duplicate duplicate duplicate duplicate multi_view)

SUMMARY_CSV="$PROJECT_ROOT/rad_dino_ablation_summary.csv"
echo "variant,dataset,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for v in "${!VARIANT_NAME[@]}"; do
  VNAME="${VARIANT_NAME[$v]}"
  VCKPT="${VARIANT_CKPT_DIR[$v]}"

  if [ ! -d "$PROJECT_ROOT/$VCKPT" ]; then
    echo "!! skipping variant '$VNAME' -- $PROJECT_ROOT/$VCKPT does not exist yet" >&2
    continue
  fi

  echo "############################################################"
  echo "### variant: $VNAME ($VCKPT) ###"
  echo "############################################################"

  for d in "${!DATASETS_NAME[@]}"; do
    DS="${DATASETS_NAME[$d]}"
    echo "--- $VNAME / $DS ---"

    MASK_DIR_ARG=""
    if [ "${VARIANT_NEEDS_MASK[$v]}" = "1" ]; then
      MASK_DIR_ARG="--mask-dir /workspace/${DATASETS_MASK[$d]}"
    fi

    RAW_LOG="$PROJECT_ROOT/.eval_ablation_${VNAME}_${DS}.raw.log"
    "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_${VNAME}_${DS}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$VCKPT \
          --csv /workspace/${DATASETS_CSV[$d]} \
          --image-dir /workspace/${DATASETS_IMG[$d]} \
          --encoders rad_dino \
          --channel-mode ${VARIANT_CHANNEL_MODE[$v]} \
          $MASK_DIR_ARG \
          --out /workspace/submission_ablation_${VNAME}_${DS}.csv --score" 2>&1 | tee "$RAW_LOG"

    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
    ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
    echo "${VNAME},${DS},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
    rm -f "$RAW_LOG"
  done
done

echo "=== RAD_DINO ABLATION STUDY DONE ==="
echo "=== SUMMARY (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "Read F1@0.5 per (variant, dataset). Standing baseline (no mask, CLAHE-tripled)"
echo "numbers for reference: test=0.9890 shenzhen=0.8954 montgomery=0.8780"
echo "tbx11k_full=0.3225 (see memory: task2_submission_vs_recommended_model)."
echo "A variant only beats it if it wins on the EXTERNAL sets, not test.csv --"
echo "that's the whole lesson of this session's mask-channel/dual-stream results."
