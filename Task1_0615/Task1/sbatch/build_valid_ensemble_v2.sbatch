#!/bin/bash
# ============================================================================
# Task 1 — build the valid classifier ensemble, v2: CPU-only, TTA + optimized
# weights.
#
# Why this exists: this is pure INFERENCE on already-trained checkpoints
# (predict.py / predict_ordfused.py, no backprop) over ~555 images total
# (444 train OOF + 111 test) -- ensemble_submissions.py itself is explicitly
# CPU-only per its own docstring. None of this needs a GPU, so it doesn't
# have to queue behind the nnU-Net retrain / segmentation-decoder jobs
# occupying the V100. Combines three separate fixes made 2026-07-22 that
# were previously only exercised individually:
#   1. flip-TTA added to predict.py's ensemble_probs (was already default-on
#      in predict_ordfused.py, missing from the plain-CNN pipeline until now)
#   2. augment_u8() in common.py -- real geometric/photometric augmentation
#      replacing the old flip-only training augmentation (affects future
#      training runs, not this inference-only job, but noted here since the
#      checkpoints this job scores were trained WITHOUT it -- see TODO below)
#   3. ensemble_submissions.py --optimize-weights -- coordinate-ascent search
#      over per-member weights on OOF accuracy, instead of flat 1.0 each
#
# TODO: the checkpoints under Code/runs/* were all trained before the
# augment_u8() fix landed -- this job only benefits from TTA (item 1) and
# optimized weights (item 3) using those old checkpoints. To benefit from
# item 2 as well, retrain densenet201/tf_efficientnetv2_s/resnet50d/
# eva_x_base_notab first (GPU-bound), then rerun this job.
#
# Uses lisa-task1:foundation (same image build_valid_ensemble.sbatch used --
# has both timm CNN deps and the OrdFused/peft/LoRA deps for eva_x_base).
#
# TODO — confirm there's a CPU-only partition on this cluster (sinfo -o "%P
# %G") and set --partition below; left unset for now so it goes to the
# scheduler default. If your cluster's GPU partition refuses jobs with no
# --gres request, fall back to requesting the V100 partition with no --gres
# (reserves the node's CPUs only, not the GPU) or, worst case, add back
# --gres=gpu:V100:1 -- this job is CPU-bound regardless and just won't use it.
# ============================================================================
#SBATCH --job-name=task1-build-valid-ensemble-v2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --partition=TODO_CPU_PARTITION
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
IMAGE="lisa-task1:foundation"
CONTAINER_NAME="task1_build_valid_ensemble_v2_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST (CPU-only -- no --gres requested)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

# NOTE: no --device nvidia.com/gpu=all -- CPU inference only, deliberately,
# so this never competes with GPU-bound training jobs.
docker run --rm \
  --name "$CONTAINER_NAME" \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code
    mkdir -p oof

    echo "############################################################"
    echo "### STAGE 1/2: dump OOF+test probs, TTA on, for each valid member ###"
    echo "############################################################"

    echo "--- densenet201 ---"
    python predict.py --model densenet201 --dump-oof oof/densenet201_oof.csv

    echo "--- tf_efficientnetv2_s ---"
    python predict.py --model tf_efficientnetv2_s --dump-oof oof/tf_efficientnetv2_s_oof.csv

    echo "--- resnet50d ---"
    python predict.py --model resnet50d --dump-oof oof/resnet50d_oof.csv

    echo "--- ordfused_eva_x_base_notab ---"
    python predict_ordfused.py --encoder eva_x_base --no-tabular \
        --dump-oof oof/eva_x_base_notab_oof.csv

    echo "############################################################"
    echo "### STAGE 2/2: build + score the ensemble with OPTIMIZED weights ###"
    echo "### (threshold + weights both swept on dev/OOF only, never test) ###"
    echo "############################################################"
    python ensemble_submissions.py \
        --optimize-weights \
        --oof densenet201:oof/densenet201_oof.csv \
        --oof tf_efficientnetv2_s:oof/tf_efficientnetv2_s_oof.csv \
        --oof resnet50d:oof/resnet50d_oof.csv \
        --oof eva_x_base_notab:oof/eva_x_base_notab_oof.csv \
        --test densenet201:../submissions/submission_densenet201.csv \
        --test tf_efficientnetv2_s:../submissions/submission_tf_efficientnetv2_s.csv \
        --test resnet50d:../submissions/submission_resnet50d.csv \
        --test eva_x_base_notab:../submissions/submission_ordfused_eva_x_base_notab.csv \
        --name valid_ensemble_v2
  '

echo "=== DONE. See submissions/submission_valid_ensemble_v2.csv ==="
echo "Compare its printed TEST score/acc against valid_ensemble_v1 (score=0.7960,"
echo "acc=0.8468) -- only keep v2 if TTA + optimized weights actually win."
