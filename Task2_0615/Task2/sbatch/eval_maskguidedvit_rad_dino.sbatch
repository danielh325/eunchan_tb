#!/bin/bash
# ============================================================================
# Task 2 -- score the new ViT mask-guided rad_dino (checkpoints_mask_guided_vit/,
# job 119464, LOMO mean val_acc=0.9888/val_auc=0.9966) against Data/test.csv
# and all 3 external datasets, for direct comparison against:
#   baseline_rad_dino   (checkpoints/)              -- no mask at all
#   maskch_rad_dino     (checkpoints_mask_ablation/) -- passive mask channel
#   maskguidedvit_rad_dino (checkpoints_mask_guided_vit/) -- THIS run: active
#     Dice-supervised per-patch-token attention gate, no mask needed at
#     inference (see memory: task2_submission_vs_recommended_model for why
#     this specific comparison matters -- mask-channel helped internally but
#     hurt rad_dino externally; this tests whether the guided-attention
#     variant, which never touches the mask at test time, avoids that same
#     failure).
#
# No --mask-dir needed for this variant at predict time -- the gate
# reconstructs itself from the checkpoint (use_mask_attention stored in the
# .pth), same as the CNN mask-guided variants.
#
# PREREQUISITE: checkpoints_mask_guided_vit/ must exist (job 119464, done).
# PREREQUISITE: the usual ch0 preprocessed dirs for test.csv + all 3 external
# datasets must already exist (they do, from earlier eval jobs).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-eval-maskguidedvit-rad-dino
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_evalmgvit_${SLURM_JOB_ID}"

CKPT_DIR="checkpoints_mask_guided_vit"
TAG="maskguidedvit_rad_dino"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -d "$PROJECT_ROOT/$CKPT_DIR" ]; then
    echo "!! $PROJECT_ROOT/$CKPT_DIR missing -- train_task2_mask_guided_vit.sbatch hasn't run." >&2
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

# --- dataset table: name, csv, image-dir (ch0, relative to $PROJECT_ROOT) ---
DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

SUMMARY_CSV="$PROJECT_ROOT/maskguidedvit_rad_dino_eval_summary.csv"
echo "dataset,tag,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for i in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$i]}"
  echo "--- $DS ---"
  RAW_LOG="$PROJECT_ROOT/.eval_mgvit_${DS}.raw.log"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/$CKPT_DIR \
        --csv /workspace/${DATASETS_CSV[$i]} \
        --image-dir /workspace/${DATASETS_IMG[$i]} \
        --encoders rad_dino \
        --out /workspace/submission_${TAG}_${DS}.csv --score" 2>&1 | tee "$RAW_LOG"

  SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
  TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
  ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
  AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
  F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
  BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
  ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
  echo "${DS},${TAG},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
  rm -f "$RAW_LOG"
done

echo "=== DONE ==="
echo "=== SUMMARY (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "Compare F1@0.5 against:"
echo "  baseline_rad_dino:   test=0.9890 shenzhen=0.8954 montgomery=0.8780 tbx11k_full=0.3225"
echo "  maskch_rad_dino:     test=0.9934 shenzhen=0.8318 montgomery=0.7943 tbx11k_full=0.3097"
echo "(see memory: task2_submission_vs_recommended_model / external_eval_all_variants_summary.csv)"
