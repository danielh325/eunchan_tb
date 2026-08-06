#!/bin/bash
# ============================================================================
# Task 2 -- the external-validation counterpart to eval_test_all_variants.sbatch:
# score EVERY individual model across ALL THREE mask-usage variants (baseline /
# mask-channel-ablation / mask-guided-attention), against ALL THREE external
# datasets (Shenzhen, Montgomery, TBX11K-full), then brute-force every
# ensemble combination PER DATASET.
#
# Why this run matters more than the test.csv version: test.csv is same-
# source-pool data (organizer-held-out, but same distribution as train.csv).
# The test.csv sweep (job 119022) found maskch_rad_dino as the best single
# model (F1=0.9934) and a 3-model cross-variant combo as the best ensemble
# (F1=0.9940) -- but internal numbers are known to be inflated: baseline's
# own external F1@0.5 was only 0.77 (Shenzhen) / 0.83 (Montgomery) vs ~0.99
# internally. This run is what tells us whether "mask-channel + rad_dino"
# is a real win or just another internal-test artifact.
#
#   baseline (checkpoints/):            densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16
#   mask-channel (checkpoints_mask_ablation/): same 5 encoders, --mask-dir required (ch2)
#   mask-guided  (checkpoints_mask_guided/):   CNN-only -- densenet121 swin_tiny convnext_tiny
#   = 13 individual models x 3 datasets = 39 predict_task2.py calls.
#
# Three stages:
#   STAGE 0 (GPU, segmentation pass) -- add the missing --mask-channel (ch2)
#     output to each external dataset's preprocessed dir (currently ch0-only;
#     needed for the mask-channel-ablation variant). Skipped per-dataset if
#     ch2 already exists.
#   STAGE 1 (GPU) -- one predict_task2.py --score call per (dataset, model)
#     pair, restricted via --encoders to a single encoder, writing
#     submission_ext_<dataset>_<tag>.csv.
#   STAGE 2 (CPU only, no GPU work) -- combo_search_test.py brute-forces all
#     2**13 - 1 = 8191 combinations PER DATASET (3 separate searches) by
#     averaging already-computed prob_tb -- pure numpy, no forward pass.
#
# COST WARNING: TBX11K-full is ~8400 images vs test.csv's ~1941 -- expect
# this job to run considerably longer than eval_test_all_variants.sbatch
# (which took under an hour). --time is set generously below.
#
# PREREQUISITE: all three checkpoint dirs must exist. PREREQUISITE:
# sbatch/fetch_external_validation.sh (Shenzhen+Montgomery) and
# sbatch/fetch_tbx11k_full.sh + Code/build_tbx11k_full_csv.py (TBX11K) must
# already have been run on the host.
# ============================================================================
#SBATCH --job-name=task2-eval-external-allvariants
#SBATCH --partition=TitanRTX
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_extall_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

# --- dataset table: name, raw-in-dir, preprocessed-out-dir, csv (relative to $PROJECT_ROOT) ---
DATASETS_NAME=(shenzhen montgomery tbx11k_full)
DATASETS_RAW=(Data/external/shenzhen/CXR_png Data/external/montgomery/CXR_png Data/external/tbx11k/raw_full)
DATASETS_OUT=(Data/external/shenzhen/preprocessed Data/external/montgomery/preprocessed Data/external/tbx11k/preprocessed_full)
DATASETS_CSV=(Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)

for i in "${!DATASETS_NAME[@]}"; do
  if [ ! -d "$PROJECT_ROOT/${DATASETS_RAW[$i]}" ] || [ ! -f "$PROJECT_ROOT/${DATASETS_CSV[$i]}" ]; then
    echo "!! ${DATASETS_NAME[$i]}: ${DATASETS_RAW[$i]} or ${DATASETS_CSV[$i]} missing." >&2
    echo "!! Run fetch_external_validation.sh (shenzhen/montgomery) or" >&2
    echo "!! fetch_tbx11k_full.sh + build_tbx11k_full_csv.py (tbx11k) first." >&2
    exit 1
  fi
done
for CKPT_DIR in checkpoints checkpoints_mask_ablation checkpoints_mask_guided; do
  if [ ! -d "$PROJECT_ROOT/$CKPT_DIR" ]; then
    echo "!! $PROJECT_ROOT/$CKPT_DIR missing -- train that variant first." >&2
    exit 1
  fi
done

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

echo "############################################################"
echo "### STAGE 0: (re)preprocess external datasets WITH --mask-channel ###"
echo "############################################################"
for i in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$i]}"
  OUT="${DATASETS_OUT[$i]}"
  if [ -d "$PROJECT_ROOT/$OUT/ch2" ]; then
    echo "skip: $PROJECT_ROOT/$OUT/ch2 already exists"
    continue
  fi
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_preprocess_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/${DATASETS_RAW[$i]} \
        --out-dir /workspace/$OUT \
        --lung-crop --mask-channel"
done

# --- model table: tag, ckpt-dir, encoder, needs-mask-dir(0/1) -- same 13 models as eval_test_all_variants.sbatch ---
TAGS=()
CKPT_DIRS=()
ENCODERS=()
NEEDS_MASK=()
for ENC in densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16; do
  TAGS+=("baseline_${ENC}");   CKPT_DIRS+=("checkpoints");               ENCODERS+=("$ENC"); NEEDS_MASK+=(0)
  TAGS+=("maskch_${ENC}");     CKPT_DIRS+=("checkpoints_mask_ablation");  ENCODERS+=("$ENC"); NEEDS_MASK+=(1)
done
for ENC in densenet121 swin_tiny convnext_tiny; do
  TAGS+=("maskguided_${ENC}"); CKPT_DIRS+=("checkpoints_mask_guided");    ENCODERS+=("$ENC"); NEEDS_MASK+=(0)
done

echo "############################################################"
echo "### STAGE 1: score each of 3 external datasets per individual model (${#TAGS[@]} models each) ###"
echo "############################################################"

SUMMARY_CSV="$PROJECT_ROOT/external_eval_all_variants_summary.csv"
echo "dataset,tag,ckpt_dir,encoder,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$SUMMARY_CSV"

for d in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$d]}"
  OUT="${DATASETS_OUT[$d]}"
  CSV="${DATASETS_CSV[$d]}"

  for i in "${!TAGS[@]}"; do
    TAG="${TAGS[$i]}"
    CDIR="${CKPT_DIRS[$i]}"
    ENC="${ENCODERS[$i]}"

    # Plain string, not a bash array, in the docker bash -c command below --
    # see eval_test_all_variants.sbatch's note on why (job 117994's array
    # flag-dropping bug).
    MASK_DIR_ARG=""
    if [ "${NEEDS_MASK[$i]}" = "1" ]; then
      MASK_DIR_ARG="--mask-dir /workspace/$OUT/ch2"
    fi

    echo "--- $DS: $TAG ($CDIR / $ENC) ---"
    RAW_LOG="$PROJECT_ROOT/.eval_ext_${DS}_${TAG}.raw.log"
    "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_${DS}_${TAG}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$CDIR \
          --csv /workspace/$CSV \
          --image-dir /workspace/$OUT/ch0 \
          --encoders $ENC \
          $MASK_DIR_ARG \
          --out /workspace/submission_ext_${DS}_${TAG}.csv --score" 2>&1 | tee "$RAW_LOG"

    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
    ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
    echo "${DS},${TAG},${CDIR},${ENC},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$SUMMARY_CSV"
    rm -f "$RAW_LOG"
  done
done

echo "=== STAGE 1 DONE -- per-model per-dataset scores (also written to $SUMMARY_CSV) ==="
column -s, -t "$SUMMARY_CSV"

echo "############################################################"
echo "### STAGE 2: brute-force every ensemble combination, PER DATASET (no GPU work) ###"
echo "############################################################"
TAGS_ARG="$(printf '%s ' "${TAGS[@]}")"
for i in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$i]}"
  CSV="${DATASETS_CSV[$i]}"
  echo "--- combo search: $DS ---"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_combos_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python combo_search_test.py \
        --test-csv /workspace/$CSV \
        --preds-dir /workspace \
        --pred-prefix submission_ext_${DS} \
        --tags $TAGS_ARG \
        --out /workspace/external_eval_${DS}_combos.csv \
        --top-n 40"
done

echo "=== ALL DONE ==="
echo "Per-model per-dataset scores: $SUMMARY_CSV"
echo "Per-dataset combo sweeps (8191 rows each): external_eval_{shenzhen,montgomery,tbx11k_full}_combos.csv"
echo "Compare F1@0.5 here against test.csv's job-119022 results (best single model there:"
echo "maskch_rad_dino F1=0.9934; best combo: maskch_rad_dino+baseline_chexfound_vitl16+maskguided_swin_tiny"
echo "F1=0.9940) to see whether that ranking survives real external validation."
