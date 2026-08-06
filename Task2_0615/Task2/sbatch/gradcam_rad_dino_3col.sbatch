#!/bin/bash
# ============================================================================
# Task 2 -- regenerates the 3-column original/cropped/gradcam grid for
# rad_dino (strong_aug_no_mixup, our recommended config -- see memory:
# task2_dg_ablation_results) on ids 1193, 3, and 4, using the FIXED
# gradcam_rad_dino() in gradcam_task2.py.
#
# id=4 added on request: true=TB, Modality_DICOM=DX (a third, different
# fold/checkpoint from 1193's CR and 3's XC -- rad_dino_DX.pth already
# exists under CKPT_DIR, no retraining needed).
#
# Why this reran: the previous version of gradcam_rad_dino() returned the
# last-layer CLS-to-patch self-attention map -- gradient-free, so it's
# class-agnostic and can't support a "the model localizes TB" claim (id=1193,
# true=Normal, P(TB)=0.001, and id=3, true=TB, P(TB)=0.999, produced
# visually similar heatmaps). Fixed 2026-08-05 to a real Grad-CAM hooked at
# hidden_states[-2] (patch tokens one block before the CLS-only pooler read,
# which is where the true gradient dies -- see gradcam_task2.py's docstring
# on gradcam_rad_dino for the full chain-rule argument). Also went through
# two failed clip attempts (flat 99th-percentile, then interior-only
# percentile) for a register-token edge-stripe artifact before landing on a
# median+MAD robust clip -- see that same docstring for why.
#
# This job also prints a quick sanity check first: the mean |grad| at
# hidden_states[-2] for a real image/checkpoint, to confirm it's actually
# nonzero (couldn't run this locally -- no `transformers` install outside
# this project's docker image).
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-gradcam-rad-dino-3col
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_gradcam_3col_${SLURM_JOB_ID}"

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

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=8gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

CKPT_DIR="checkpoints/checkpoints_dgablation_strong_aug_no_mixup"

echo "############################################################"
echo "### gradient sanity check: hidden_states[-2] vs [-1] on id=1193/CR ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_probe" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python -c \"
import torch
from dataset import Cfg
from gradcam_task2 import build_model, load_checkpoint, load_input

m = build_model(load_checkpoint('/workspace/$CKPT_DIR', 'rad_dino', 'CR'))
cfg = Cfg('rad_dino', image_dir='/workspace/Data/Preprocessed/test_images/ch0')
x, _ = load_input('/workspace/Data/Preprocessed/test_images/ch0', '1193', cfg)
bb = m.encoder.backbone
out = bb(pixel_values=x, output_hidden_states=True)
hs = out.hidden_states
for k in [len(hs)-1, len(hs)-2]:
    hs[k].retain_grad()
pooled = out.pooler_output
logit = m.head(pooled).squeeze(1)
m.zero_grad(set_to_none=True)
logit.backward()
for k in [len(hs)-1, len(hs)-2]:
    g = hs[k].grad
    print(f'hidden_states[{k}] patch-grad |mean| =', None if g is None else g[:,1:].abs().mean().item())
print('P(TB) =', torch.sigmoid(logit).item())
\""

echo "############################################################"
echo "### 3-column grid: original | cropped | gradcam (ids 1193, 3, 4) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python gradcam_3col_task2.py --image-ids 1193 3 4 \
      --ckpt-dir /workspace/$CKPT_DIR \
      --raw-dir /workspace/Data/test \
      --crop-dir /workspace/Data/Preprocessed/test_images/ch0 \
      --csv /workspace/Data/test.csv \
      --out /workspace/gradcam_3col_1193_3_4_v2.png"

echo "=== DONE -- output: $PROJECT_ROOT/gradcam_3col_1193_3_4_v2.png ==="
echo "Compare against the OLD figure (Code/gradcam_3col_1193_3.png, CLS-attention,"
echo "class-agnostic) -- the new one should show visibly different hotspots between"
echo "the Normal (P(TB)=0.001) and TB (P(TB)=0.999/0.99xx) rows if the fix worked,"
echo "and the median+MAD clip should have killed the register-token edge stripe."
