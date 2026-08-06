#!/bin/bash
# ============================================================================
# Task 1 — OrdFused-CXR: the real foundation-model build (EVA-X + RAD-DINO +
# CheXFound, each frozen backbone + LoRA), fused through the existing gated-
# tabular CORN head, ensembled 5-fold x 3-encoder. One GPU, sequential stages
# (encoder sanity check -> per-encoder smoke test -> full run) so nothing
# runs concurrently on the GPU.
#
# PREREQUISITES (do these first, on the host, NOT inside sbatch):
#   1. bash sbatch/fetch_foundation_weights.sh
#      (needs CHEXFOUND_FILE_ID set — see that script's TODO comment)
#   2. Make sure job 117374 (or any other job holding the GPU) has finished.
#
# NOTE on the image tag: this uses a DIFFERENT tag ("lisa-task1:foundation")
# from the other sbatch scripts ("lisa-task1:latest"). The existing scripts
# only rebuild when the tag is completely missing locally -- reusing
# "lisa-task1:latest" here would silently skip picking up Dockerfile_final's
# new transformers/peft/gdown layers if that tag was already built by an
# earlier job (which it was, by 117374), and every foundation-encoder import
# would fail at runtime. The distinct tag forces exactly one real rebuild;
# Docker's layer cache still reuses every unchanged base layer, so it's fast.
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-ordfused-foundation
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script into /var/spool/slurmd/job<id>/ before
# running it, so derive paths from $SLURM_SUBMIT_DIR (where you ran sbatch),
# not from BASH_SOURCE.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
IMAGE="lisa-task1:foundation"
CONTAINER_NAME="task1_ordfused_foundation_${SLURM_JOB_ID}"

ENCODERS="eva_x_base rad_dino chexfound_vitl16"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "image: $IMAGE"
echo "encoders: $ENCODERS"

if [ ! -f "$PROJECT_ROOT/weights/eva_x/eva_x_base_patch16_merged520k_mim.pt" ] || \
   [ ! -f "$PROJECT_ROOT/weights/rad_dino/config.json" ] || \
   [ ! -f "$PROJECT_ROOT/weights/chexfound/teacher_checkpoint.pth" ]; then
    echo "!! foundation-model weights missing under $PROJECT_ROOT/weights/ -- run" >&2
    echo "!! sbatch/fetch_foundation_weights.sh on the host first." >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

echo "############################################################"
echo "### STAGE 1/3: encoder sanity check (smoke_encoders.py) ###"
echo "############################################################"

docker run --rm \
  --name "${CONTAINER_NAME}_sanity" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python smoke_encoders.py"

echo "############################################################"
echo "### STAGE 2/3: per-encoder smoke test (1 fold, 1 epoch each) ###"
echo "############################################################"

for ENC in $ENCODERS; do
  echo "--- smoke: $ENC ---"
  docker run --rm \
    --name "${CONTAINER_NAME}_smoke_${ENC}" \
    --device nvidia.com/gpu=all \
    --shm-size=8gb \
    -v "$PROJECT_ROOT":/workspace \
    -e ENC="$ENC" \
    "$IMAGE" \
    bash -c '
      set -euo pipefail
      cd /workspace/Code
      OUT=runs_smoke_$ENC
      SUB=/workspace/submissions_smoke_$ENC
      rm -rf "$OUT" "$SUB"

      python train_ordfused.py --encoder "$ENC" --fold 0 --epochs 1 \
          --max-retries 0 --num-workers 2 --out-dir "$OUT"
      python predict_ordfused.py --encoder "$ENC" --num-workers 2 \
          --out-dir "$OUT" --submissions-dir "$SUB"

      head -5 "$SUB/submission_ordfused_${ENC}.csv"
      rm -rf "$OUT" "$SUB"
      echo "=== SMOKE OK: $ENC ==="
    '
done

echo "############################################################"
echo "### STAGE 3/3: full run ($ENCODERS) ###"
echo "############################################################"

docker run --rm \
  --name "${CONTAINER_NAME}_full" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c "cd /workspace/Code && bash run_ordfused.sh $ENCODERS"

echo "=== ORDFUSED FOUNDATION DONE -> $PROJECT_ROOT/submissions/submission_ordfused.csv ==="
echo "NOTE: this overwrites submission_ordfused.csv from the CNN-only run (117374)."
echo "Per the plan, the final ensemble composition should be picked by comparing"
echo "OOF scores across CNN-only / foundation-only / combined subsets -- see the"
echo "Verification section of the saved plan before treating this as final."
