#!/bin/bash
# ============================================================================
# Task 2 -- score Data/test.csv with EVERY individual model across ALL THREE
# mask-usage variants (baseline / mask-channel-ablation / mask-guided-
# attention: CNN + ViT encoders where each variant has them), then brute-
# force every ensemble combination of those models' probabilities.
#
#   baseline (checkpoints/):            densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16
#   mask-channel (checkpoints_mask_ablation/): same 5 encoders, --mask-dir required (ch2)
#   mask-guided  (checkpoints_mask_guided/):   CNN-only -- densenet121 swin_tiny convnext_tiny
#                                               (no --mask-dir -- see model comparison notes:
#                                               the gate reconstructs itself from the checkpoint)
#   = 13 individual models total.
#
# Two stages:
#   STAGE 1 (needs the GPU) -- one predict_task2.py --score call per model,
#     restricted via --encoders to a single encoder (so each call ensembles
#     only that encoder's own 4 leave-one-modality-out fold checkpoints --
#     its "individual model" score), writing submission_test_<tag>.csv.
#   STAGE 2 (CPU only, no GPU work) -- Code/combo_search_test.py reads all 13
#     already-computed prob_tb CSVs and brute-forces every non-empty subset
#     (2**13 - 1 = 8191 combinations) by averaging already-computed
#     probabilities -- pure numpy, no forward pass, no GPU needed. Runs in
#     the same container/job right after Stage 1 so nothing has to be
#     re-launched separately.
#
# PREREQUISITE: all three checkpoint dirs must exist (train_task2.sbatch,
# train_task2_mask_ablation.sbatch, train_task2_mask_guided.sbatch all done).
# PREREQUISITE: Data/Preprocessed/test_images/{ch0,ch2} must both already
# exist (train_task2.sbatch's Stage 1 always runs preprocess.py with
# --lung-crop --mask-channel, so ch2 should already be there -- unlike the
# external datasets, no extra preprocessing pass is needed here).
# ============================================================================
#SBATCH --job-name=task2-eval-test-allvariants
#SBATCH --partition=TitanRTX
#SBATCH --gres=gpu:TitanRTX:1
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
CONTAINER_NAME="task2_evalall_${SLURM_JOB_ID}"

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
if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch0" ] || [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch2" ]; then
    echo "!! Data/Preprocessed/test_images/{ch0,ch2} missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 (preprocess.py --lung-crop --mask-channel) first." >&2
    exit 1
fi
for CKPT_DIR in checkpoints checkpoints_mask_ablation checkpoints_mask_guided; do
  if [ ! -d "$PROJECT_ROOT/$CKPT_DIR" ]; then
    echo "!! $PROJECT_ROOT/$CKPT_DIR missing -- train that variant first." >&2
    exit 1
  fi
done

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

# --- model table: tag, ckpt-dir, encoder, needs-mask-dir(0/1) ---
TAGS=()
CKPT_DIRS=()
ENCODERS=()
NEEDS_MASK=()

for ENC in densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16; do
  TAGS+=("baseline_${ENC}");      CKPT_DIRS+=("checkpoints");              ENCODERS+=("$ENC"); NEEDS_MASK+=(0)
  TAGS+=("maskch_${ENC}");        CKPT_DIRS+=("checkpoints_mask_ablation"); ENCODERS+=("$ENC"); NEEDS_MASK+=(1)
done
for ENC in densenet121 swin_tiny convnext_tiny; do
  TAGS+=("maskguided_${ENC}");    CKPT_DIRS+=("checkpoints_mask_guided");   ENCODERS+=("$ENC"); NEEDS_MASK+=(0)
done

echo "############################################################"
echo "### STAGE 1: score Data/test.csv per individual model (${#TAGS[@]} models) ###"
echo "############################################################"

SUMMARY_CSV="$PROJECT_ROOT/test_eval_all_variants_summary.csv"
echo "tag,ckpt_dir,encoder,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"
  CDIR="${CKPT_DIRS[$i]}"
  ENC="${ENCODERS[$i]}"

  # Plain string, not a bash array, in the docker bash -c command below --
  # an array-based ${..[@]} interpolated into a docker `bash -c "..."`
  # string silently dropped flags at runtime in an earlier script (job
  # 117994) even though the same construct checked out in isolated tests.
  MASK_DIR_ARG=""
  if [ "${NEEDS_MASK[$i]}" = "1" ]; then
    MASK_DIR_ARG="--mask-dir /workspace/Data/Preprocessed/test_images/ch2"
  fi

  echo "--- test.csv: $TAG ($CDIR / $ENC) ---"
  RAW_LOG="$PROJECT_ROOT/.eval_test_${TAG}.raw.log"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_${TAG}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/$CDIR \
        --csv /workspace/Data/test.csv \
        --image-dir /workspace/Data/Preprocessed/test_images/ch0 \
        --encoders $ENC \
        $MASK_DIR_ARG \
        --out /workspace/submission_test_${TAG}.csv --score" 2>&1 | tee "$RAW_LOG"

  SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
  TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
  ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
  AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
  F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
  BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
  ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
  echo "${TAG},${CDIR},${ENC},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
  rm -f "$RAW_LOG"
done

echo "=== STAGE 1 DONE -- per-model scores (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"

echo "############################################################"
echo "### STAGE 2: brute-force every ensemble combination (no GPU work) ###"
echo "############################################################"
TAGS_ARG="$(printf '%s ' "${TAGS[@]}")"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_combos" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python combo_search_test.py \
      --test-csv /workspace/Data/test.csv \
      --preds-dir /workspace \
      --tags $TAGS_ARG \
      --out /workspace/test_eval_all_variants_combos.csv \
      --top-n 40"

echo "=== ALL DONE ==="
echo "Per-model scores: $SUMMARY_CSV"
echo "All 2^13-1 = 8191 combination scores (sorted by F1@0.5 in the log above,"
echo "full table): $PROJECT_ROOT/test_eval_all_variants_combos.csv"
