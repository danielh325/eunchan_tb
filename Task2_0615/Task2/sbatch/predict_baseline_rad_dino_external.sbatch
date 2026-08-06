#!/bin/bash
# ============================================================================
# Task 2 -- dump PER-SAMPLE baseline_rad_dino probabilities for test.csv +
# all 3 external datasets.
#
# WHY: every external comparison so far exists only as aggregate metrics
# (external_eval_*_combos.csv holds accuracy/AUC/F1 per combo, not the
# underlying prob_tb columns), and the per-sample external CSVs were never
# kept -- only CSV/submission_ext_shenzhen_baseline_densenet121.csv survives.
# That makes it impossible to ensemble baseline_rad_dino OFFLINE against any
# newly-trained variant (e.g. checkpoints_chexfound_glori/), because offline
# combo-averaging needs both models' prob_tb per new_id.
#
# With these files on disk, any future "what if we ensemble X with the
# baseline?" question is answered locally in seconds by averaging prob_tb
# columns -- no GPU, no retraining. The offline average reproduces
# predict_task2.py's own multi-encoder output exactly: recomputing
# rad_dino + chexfound_vitl16 from the two single-model test CSVs gives
# F1@0.5=0.9929 / AUC=0.9999, byte-matching the submitted
# CSV/submission_test_final_rad_dino_chexfound_vitl16.csv.
#
# Writes CSV/submission_ext_<dataset>_baseline_rad_dino.csv (new_id, prob_tb,
# pred_TB/Normal). Inference only -- no checkpoints are written or modified.
#
# PREREQUISITE: the ch0 preprocessed dirs for test + all 3 external datasets
# (present since the earlier external-eval jobs). Any dataset whose csv/dir
# is missing is skipped with a warning rather than failing the job.
#
# Usage:
#   sbatch predict_baseline_rad_dino_external.sbatch
# ============================================================================
#SBATCH --job-name=task2-predict-baseline-rad-dino-ext
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_predbase_${SLURM_JOB_ID}"

CKPT_DIR="checkpoints"          # baseline_rad_dino lives here (rad_dino_{CR,DX,XA,XC}.pth)

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "dumping per-sample baseline_rad_dino probs from $CKPT_DIR"

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

DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

mkdir -p "$PROJECT_ROOT/CSV"

for d in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$d]}"
  OUT="CSV/submission_ext_${DS}_baseline_rad_dino.csv"

  if [ ! -f "$PROJECT_ROOT/${DATASETS_CSV[$d]}" ] || [ ! -d "$PROJECT_ROOT/${DATASETS_IMG[$d]}" ]; then
    echo "!! skipping $DS -- ${DATASETS_CSV[$d]} or ${DATASETS_IMG[$d]} missing" >&2
    continue
  fi

  echo "############################################################"
  echo "### baseline_rad_dino / $DS -> $OUT ###"
  echo "############################################################"

  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/$CKPT_DIR \
        --csv /workspace/${DATASETS_CSV[$d]} \
        --image-dir /workspace/${DATASETS_IMG[$d]} \
        --encoders rad_dino \
        --out /workspace/$OUT --score"
done

echo "=== DONE -> $PROJECT_ROOT/CSV/submission_ext_*_baseline_rad_dino.csv ==="
echo "The single-model F1@0.5 printed above must match the known"
echo "baseline_rad_dino numbers (shenzhen=0.8954 montgomery=0.8780"
echo "tbx11k_full=0.3225 test=0.9890) -- if not, the preprocessing or"
echo "checkpoint set used here differs from the earlier eval run and the"
echo "per-sample files should NOT be trusted for offline ensembling."
