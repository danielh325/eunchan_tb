#!/bin/bash
# ============================================================================
# Task 1 — build the best METADATA-FREE classifier ensemble.
#
# Why this exists: submission_top3.csv (the earlier "87.39% accuracy"
# ensemble) used submission_ordfused_chexfound_vitl16.csv -- a TABULAR-FUSION
# checkpoint that assumed metadata was present at inference. The real
# challenge test set gives images only. This job builds the honest
# equivalent: an ensemble of ONLY models that are valid under no-metadata
# inference (plain CNNs, which never see tabular data at all, plus the
# --no-tabular OrdFused ablations), using ensemble_submissions.py -- which
# refuses tabular-fusion submissions by filename pattern, so this can't
# accidentally repeat that mistake.
#
# Member selection: densenet201 (0.7802), tf_efficientnetv2_s (0.7712),
# resnet50d (0.7665) -- the three strongest valid models -- plus
# ordfused_eva_x_base_notab (0.7596), the one proper from-scratch
# image-only fusion ablation that exists. convnext_tiny (0.5387) is
# deliberately EXCLUDED -- it's a clear underperformer, and the earlier
# 3-way foundation ensemble already demonstrated that naive averaging lets
# one weak member drag the whole ensemble below the best single model.
#
# Needs --dump-oof runs first (none exist yet -- checked, only
# submission_*.csv test-side outputs exist from earlier runs) so
# ensemble_submissions.py can sweep its threshold on dev, never on test.
# These are INFERENCE ONLY on already-trained checkpoints -- no training,
# ~minutes total, not hours.
#
# Uses lisa-task1:foundation for everything (confirmed a superset build of
# lisa-task1:latest via the same Dockerfile_final -- see
# ordfused_foundation.sbatch's own comment on why the distinct tag exists),
# so both the plain-CNN and EVA-X predict calls run from one image, one
# docker run block.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task1-build-valid-ensemble
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"   # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"     # .../Task1_0615 (has Dockerfile_final)
IMAGE="lisa-task1:foundation"
CONTAINER_NAME="task1_build_valid_ensemble_${SLURM_JOB_ID}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally on $SLURM_NODELIST -- building it now (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    cd /workspace/Code
    mkdir -p oof

    echo "############################################################"
    echo "### STAGE 1/2: dump OOF probabilities for each valid member ###"
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
    echo "### STAGE 2/2: build + score the ensemble (threshold swept ###"
    echo "### on dev only, never on test -- see ensemble_submissions.py) ###"
    echo "############################################################"
    python ensemble_submissions.py \
        --oof densenet201:oof/densenet201_oof.csv \
        --oof tf_efficientnetv2_s:oof/tf_efficientnetv2_s_oof.csv \
        --oof resnet50d:oof/resnet50d_oof.csv \
        --oof eva_x_base_notab:oof/eva_x_base_notab_oof.csv \
        --test densenet201:../submissions/submission_densenet201.csv \
        --test tf_efficientnetv2_s:../submissions/submission_tf_efficientnetv2_s.csv \
        --test resnet50d:../submissions/submission_resnet50d.csv \
        --test eva_x_base_notab:../submissions/submission_ordfused_eva_x_base_notab.csv \
        --name valid_ensemble_v1
  '

echo "=== DONE. See submissions/submission_valid_ensemble_v1.csv ==="
echo "Compare its printed TEST score/acc against densenet201 alone (0.7802/0.8288,"
echo "the current best single valid model) -- only keep the ensemble if it actually wins."
