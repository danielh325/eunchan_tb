#!/bin/bash
# ============================================================================
# Task 2 -- train rad_dino with --channel-mode multi_view: channel 0 =
# original (percentile-clipped only), channel 1 = CLAHE-enhanced, channel 2
# = bone-suppressed (applied on top of the CLAHE'd view) -- 3 genuinely
# different views instead of one view (CLAHE, on by default) copied into
# all 3 channels, which is what every existing checkpoint (including
# baseline_rad_dino) actually uses.
#
# WHY: "Chest X-ray Bone Suppression for Improving Classification of
# Tuberculosis-Consistent Findings" (PMC8151767 / arXiv 2104.04518) found
# bone suppression ALONE gave 5-11 AUC points on these EXACT Shenzhen/
# Montgomery external benchmarks -- this codebase already has a
# `bone_suppression.py` implementing that idea (a classical Gaussian-
# homomorphic approximation, since no trained bone-suppression network is
# vendored here) but it was only ever wired as an all-or-nothing whole-image
# replacement (--bone-suppress), never actually turned on by default, and
# never combined with CLAHE the way other work reports working well
# together. Giving the model all three views (original contrast, CLAHE-
# enhanced, bone-suppressed) simultaneously as separate channels -- rather
# than silently discarding two of them by tripling one processed view --
# is a different, cheap (no new preprocessing infrastructure, no GPU
# segmenter pass) lever from anything else tried this session (mask-
# channel, mask-guided attention, dual-stream, IRM).
#
# Applied to rad_dino specifically since it's the current best-generalizing
# model (see memory: task2_submission_vs_recommended_model) -- this tests
# whether a genuinely-informative 3-channel input helps our actual best
# encoder, rather than testing it on a weaker one first.
#
# NOT YET RUN END TO END. Verified locally (no GPU) with a synthetic image:
# multi_view mode produces 3 measurably different channels (duplicate mode
# produces 3 identical channels, as expected), and --mask-dir's channel-2
# override still works correctly on top of multi_view mode.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch0 must exist.
#
# Usage:
#   sbatch train_task2_rad_dino_multiview.sbatch
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-rad-dino-multiview
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # rad_dino weights live here
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_raddino_mv_${SLURM_JOB_ID}"

ENC="rad_dino"

# Separate from checkpoints/ (original duplicate-CLAHE-x3 rad_dino) -- never
# overwrite, both stay on disk for a direct comparison.
OUT_DIR="${OUT_DIR:-checkpoints_rad_dino_multiview}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "encoder: $ENC   channel-mode: multi_view (original+CLAHE+bone-suppressed)"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch0" ] || [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch0" ]; then
    echo "!! Data/Preprocessed/{train,test}_images/ch0 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 first." >&2
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

echo "############################################################"
echo "### STAGE 0: sanity-check multi_view preprocessing + forward+backward (no training) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_smoke" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python3 -c \"
import glob
import numpy as np
import torch
from common import preprocess_image, sitk_read
from dataset import Cfg
from model import TBClassifier

cfg = Cfg('rad_dino', image_dir='/workspace/Data/Preprocessed/train_images/ch0', channel_mode='multi_view')
sample_path = sorted(glob.glob('/workspace/Data/Preprocessed/train_images/ch0/*.png'))[0]
arr = sitk_read(sample_path)
x = preprocess_image(arr, cfg)
print(f'preprocessed shape: {x.shape}')
assert x.shape == (3, 518, 518)
d01 = abs(x[0]-x[1]).mean(); d02 = abs(x[0]-x[2]).mean()
print(f'channel diffs on a REAL image: ch0-1={d01:.4f} ch0-2={d02:.4f}')
assert d01 > 0.005 and d02 > 0.005, 'channels look suspiciously similar on a real image'

model = TBClassifier('rad_dino', pretrained=True).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
xb = torch.from_numpy(x).unsqueeze(0).repeat(2,1,1,1).to(device)
logit = model(xb)
assert logit.shape == (2,), logit.shape
logit.sum().backward()
grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
assert grad_norm > 0
print('forward+backward OK on real multi_view input, grad_norm=', grad_norm)
print('SMOKE TEST PASSED')
\""

echo "############################################################"
echo "### full leave-one-modality-out run, rad_dino with channel-mode=multi_view ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
      --csv /workspace/Data/train.csv \
      --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --channel-mode multi_view \
      --out-dir /workspace/$OUT_DIR \
      --fold-mode leave_one_modality_out --epochs 15"

echo "=== rad_dino (multi_view) TRAINING DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="
echo "IMPORTANT: score with --channel-mode multi_view at predict time too, e.g.:"
echo "  python Code/predict_task2.py --ckpt-dir $OUT_DIR --channel-mode multi_view \\"
echo "      --csv Data/test.csv --image-dir Data/Preprocessed/test_images/ch0 --score"
echo ""
echo "Compare F1@0.5 against baseline_rad_dino (Shenzhen 0.8954, Montgomery 0.8780,"
echo "TBX11K-full 0.3225, test.csv 0.9890 -- see memory:"
echo "task2_submission_vs_recommended_model) on all 3 external datasets + test.csv"
echo "to see whether giving the model original+CLAHE+bone-suppressed simultaneously"
echo "beats the current CLAHE-tripled default."
