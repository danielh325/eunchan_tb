#!/bin/bash
# ============================================================================
# Task 2 -- learning-rate sweep for chexfound_vitl16, around the untouched
# default (1e-4). Every training run in this project (grep confirms: no
# sbatch script ever passes an explicit --lr) has used train_task2.py's
# flat default for every encoder, including chexfound_vitl16 despite it
# having ~2.37x more effective LoRA parameters than rad_dino at the same
# rank (see encoders.py's apply_lora comment) and a structurally different
# architecture (24 vs 12 layers, fused qkv, SwiGLU FFN, register tokens). An
# LR tuned (informally, by trial and error) against rad_dino was never
# re-checked for chexfound specifically.
#
# The 1e-4 arm is NOT retrained here -- baseline_chexfound_vitl16 in
# CSV/master_results_all_models.csv already IS that exact config, trained
# for the full 15 epochs and already scored on test/shenzhen/montgomery/
# tbx11k. Retraining it at a shortened epoch count would waste compute AND
# not even be comparable to the real baseline (a 5-epoch rerun vs. a
# 15-epoch reference is an apples-to-oranges comparison). Those numbers are
# read directly from that CSV into this job's summary table instead.
#
# The two NEW arms (3e-5, 3e-4) run the FULL 15 epochs (not a shortened
# sweep), specifically so they're directly comparable to the reused 1e-4
# baseline rather than a rough directional-only signal -- this is the
# compute the 1e-4 reuse frees up. Expect this to take meaningfully longer
# than a shortened sweep would have; --time bumped accordingly.
#
# Scores each new LR's ensemble against test.csv + shenzhen + montgomery
# (the two CLEAN external cohorts -- see memory: task2_dg_ablation_results,
# task2_qu_dataset_composition -- skipping tbx11k/mendeley here to keep this
# focused; rerun the full eval on the winning LR afterward if it changes).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-sweep-chexfound-lr
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_sweep_chexfound_lr_${SLURM_JOB_ID}"

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

SUMMARY_CSV="$PROJECT_ROOT/chexfound_lr_sweep_summary.csv"
echo "lr,dataset,accuracy@0.5,auc,f1@0.5,source" > "$SUMMARY_CSV"

# NOTE (2026-08-05): an earlier version of this script REUSED
# baseline_chexfound_vitl16's rows from CSV/master_results_all_models.csv as
# the 1e-4 arm, to avoid retraining a config that already existed. That is no
# longer valid: encoders.py's chexfound lora_targets was fixed the same day
# (SwiGLU FFN w12/w3 never matched "fc1"/"fc2", and "proj" was also wrapping
# patch_embed.proj -- see that file's comment and diagnostic job 121285).
# Every pre-fix chexfound checkpoint therefore trained a DIFFERENT adapter
# set (2,387,968 trainable, no FFN) than this sweep's arms will
# (6,297,600 trainable, FFN included). Mixing them would confound learning
# rate with the LoRA fix, so 1e-4 is retrained here under the corrected
# config like every other arm.
LR_VALUES=(3e-5 1e-4 3e-4)
for LR in "${LR_VALUES[@]}"; do
  OUT_DIR="checkpoints/checkpoints_chexfound_lr_sweep_${LR}"
  echo "############################################################"
  echo "### LR=$LR -> $OUT_DIR (full 15 epochs) ###"
  echo "############################################################"

  if [ -d "$PROJECT_ROOT/$OUT_DIR" ] && [ "$(ls "$PROJECT_ROOT/$OUT_DIR"/chexfound_vitl16_*.pth 2>/dev/null | wc -l)" -ge 4 ]; then
    echo "skip training: $OUT_DIR already has 4 fold checkpoints"
  else
    "${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_train_${LR}" "$IMAGE" \
      bash -c "cd /workspace/Code && python train_task2.py \
          --encoder chexfound_vitl16 \
          --csv /workspace/Data/train.csv \
          --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
          --out-dir /workspace/$OUT_DIR \
          --fold-mode leave_one_modality_out \
          --epochs 15 \
          --lr $LR \
          --aug-strength strong \
          --skip-existing-folds"
  fi

  for DS_NAME in test shenzhen montgomery; do
    case "$DS_NAME" in
      test)       DS_CSV="Data/test.csv";                     DS_IMG="Data/Preprocessed/test_images/ch0" ;;
      shenzhen)   DS_CSV="Data/external/shenzhen.csv";         DS_IMG="Data/external/shenzhen/preprocessed/ch0" ;;
      montgomery) DS_CSV="Data/external/montgomery.csv";       DS_IMG="Data/external/montgomery/preprocessed/ch0" ;;
    esac

    RAW_LOG="$PROJECT_ROOT/.sweep_chexfound_${LR}_${DS_NAME}.raw.log"
    "${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_eval_${LR}_${DS_NAME}" "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$OUT_DIR \
          --csv /workspace/$DS_CSV \
          --image-dir /workspace/$DS_IMG \
          --encoders chexfound_vitl16 --score \
          --out /workspace/submission_chexfound_lrsweep_${LR}_${DS_NAME}.csv" 2>&1 | tee "$RAW_LOG"

    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    echo "${LR},${DS_NAME},${ACC:-NA},${AUC:-NA},${F1:-NA},this_job_15epoch" >> "$SUMMARY_CSV"
    rm -f "$RAW_LOG"
  done
done

echo "=== CHEXFOUND LR SWEEP DONE (all 3 arms trained here, 15 epochs, post-LoRA-fix config) ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "All three LR values now directly comparable (same epoch count, same"
echo "eval protocol). Read F1@0.5 on shenzhen/montgomery -- if 3e-5 or 3e-4"
echo "clearly beats 1e-4, that's a real, trustworthy signal, not a"
echo "shortened-run artifact."
