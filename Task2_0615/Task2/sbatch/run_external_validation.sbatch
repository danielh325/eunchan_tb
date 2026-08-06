#!/bin/bash
# ============================================================================
# Task 2 — genuine external-domain validation: scores the already-trained
# main ensemble (train_task2.sbatch) AND the dual-stream model
# (train_dual_stream.sbatch), if its checkpoints exist, against the
# Shenzhen + Montgomery public TB CXR datasets (sbatch/fetch_external_
# validation.sh must be run first, on the host). This is the number that
# actually answers "do our domain-generalization techniques work" -- the
# leave-one-modality-out metric used during training is still same-source-
# pool data; this is real cross-country, cross-institution, never-seen data.
#
# Directly comparable to the user's own SSRN paper's result on these same
# two datasets (66%->89% Shenzhen with plain DenseNet-121+MixStyle+aug) --
# that's the number this build needs to match or beat to say the added
# techniques (RandConv, spectral perturbation, domain-aware mixup, the two
# foundation-model backbones, lung-crop, dual-stream fusion) earned their
# complexity.
#
# PREREQUISITE: run sbatch/fetch_external_validation.sh on the host FIRST
# (needs internet, no GPU). Needs train_task2.sbatch's checkpoints/ to
# exist -- run this AFTER that job finishes, not concurrently (shares the
# same lisa-task2:latest image and GPU).
#
# Can also be run PARTWAY through train_task2.sbatch, against whichever
# encoders have already finished ALL 4 leave-one-modality-out folds --
# restrict via ENCODERS (e.g. an in-progress chexfound_vitl16 with only 2/4
# fold checkpoints saved would otherwise be silently under-represented in
# the ensemble average relative to fully-trained backbones):
#   ENCODERS="densenet121 swin_tiny convnext_tiny rad_dino" sbatch run_external_validation.sbatch
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-external-val
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
EXT_DIR="$PROJECT_ROOT/Data/external"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_extval_${SLURM_JOB_ID}"

# Empty (default) = predict_task2.py's own default: every encoder found
# under --ckpt-dir. Set to restrict the ensemble to encoders with complete
# fold coverage (see the ENCODERS note above) -- passed through as
# --encoders, and folded into the output filename so a restricted run never
# overwrites a full-ensemble run's submission_*.csv.
# Plain string, not a bash array -- see eval_test_per_model.sbatch's note:
# an array-based ${ENCODERS_FLAG[@]} interpolated into a docker `bash -c
# "..."` string silently failed to reach predict_task2.py at runtime in
# that script (job 117994), even though this exact ENCODERS restriction
# path here has never actually been exercised in a completed run to
# confirm it works. Using a plain string instead to avoid the same risk.
ENCODERS="${ENCODERS:-}"
ENCODERS_TAG="all"
ENCODERS_ARG=""
if [ -n "$ENCODERS" ]; then
  ENCODERS_TAG="$(echo "$ENCODERS" | tr ' ' '_')"
  ENCODERS_ARG="--encoders $ENCODERS"
fi

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

for DS in shenzhen montgomery; do
  if [ ! -f "$EXT_DIR/$DS.csv" ] || [ ! -d "$EXT_DIR/$DS/CXR_png" ]; then
    echo "!! $EXT_DIR/$DS missing -- run sbatch/fetch_external_validation.sh on the host first." >&2
    exit 1
  fi
done
if [ ! -d "$PROJECT_ROOT/checkpoints" ]; then
    echo "!! $PROJECT_ROOT/checkpoints missing -- train_task2.sbatch hasn't produced" >&2
    echo "!! any checkpoints yet. Nothing to validate." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

# NOTE: docker images are local to each node's daemon, not cluster-wide --
# train_task2.sbatch building the image on one node doesn't make it
# available on whichever node THIS job lands on (job 117992 hit exactly
# this: "image not found" on gpu102 after the image was only ever built on
# gpu101). Build it here too rather than assuming it already exists.
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"   # .../Task2_0615 (has Dockerfile_final)
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights)

echo "############################################################"
echo "### STAGE 1/2: preprocess external datasets (lung-crop) ###"
echo "############################################################"
for DS in shenzhen montgomery; do
  if [ -d "$EXT_DIR/$DS/preprocessed/ch0" ]; then
    echo "skip: $EXT_DIR/$DS/preprocessed already exists"
    continue
  fi
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_preprocess_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/Data/external/${DS}/CXR_png \
        --out-dir /workspace/Data/external/${DS}/preprocessed \
        --lung-crop"
done

echo "############################################################"
echo "### STAGE 2/2: score main ensemble + dual-stream (if present) ###"
echo "############################################################"
echo "encoders: ${ENCODERS:-<all found under checkpoints/>}"

for DS in shenzhen montgomery; do
  echo "--- $DS: ensemble ($ENCODERS_TAG) ---"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_score_${DS}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/checkpoints \
        --csv /workspace/Data/external/${DS}.csv \
        --image-dir /workspace/Data/external/${DS}/preprocessed/ch0 \
        --out /workspace/submission_${DS}_ensemble_${ENCODERS_TAG}.csv --score \
        $ENCODERS_ARG"

  if [ -d "$PROJECT_ROOT/checkpoints_dual" ]; then
    echo "--- $DS: dual-stream model ---"
    "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_score_dual_${DS}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_dual_stream.py \
          --ckpt-dir /workspace/checkpoints_dual \
          --csv /workspace/Data/external/${DS}.csv \
          --global-image-dir /workspace/Data/external/${DS}/CXR_png \
          --local-image-dir /workspace/Data/external/${DS}/preprocessed/ch0 \
          --out /workspace/submission_${DS}_dual.csv --score"
  else
    echo "--- $DS: skipping dual-stream (checkpoints_dual/ not found yet) ---"
  fi
done

echo "=== EXTERNAL VALIDATION DONE ==="
echo "Compare the printed accuracy@0.5/AUC numbers above against your SSRN"
echo "paper's own Shenzhen result (66%->89% with plain DenseNet-121+MixStyle+aug)"
echo "-- that is the real bar for whether this build's added techniques help."
