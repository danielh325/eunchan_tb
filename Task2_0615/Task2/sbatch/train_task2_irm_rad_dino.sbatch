#!/bin/bash
# ============================================================================
# Task 2 -- Layer 1 of the redesigned pipeline (see memory/conversation
# 2026-07-30): train rad_dino with an Invariant Risk Minimization (IRM,
# Arjovsky et al. 2019) penalty over Modality_DICOM as the environment
# variable, instead of (or alongside) domain-aware mixup.
#
# WHY: rad_dino alone is already our best-generalizing model (Shenzhen
# F1=0.8954, Montgomery F1=0.8780 -- see memory:
# task2_submission_vs_recommended_model), but still drops hard from ~0.99
# internal. Every mask-usage variant we've tried (mask-channel, mask-guided
# attention) and the dual-stream architecture point at the same underlying
# problem: the model latches onto acquisition/site-specific signal that
# correlates with the label WITHIN the training distribution but doesn't
# hold up across real external hospitals. IRM directly targets this: it
# penalizes the classifier for learning anything that predicts TB
# differently depending on which of the training modalities (CR/DX/XA/XC,
# 3 of 4 present under leave-one-modality-out) produced the image, forcing
# it toward features that are genuinely invariant across those environments
# rather than modality-specific shortcuts. Precedent: "Learning Invariant
# Feature Representation to Improve Generalization across Chest X-ray
# Datasets" (MICCAI 2020, arXiv 2008.04152) does exactly this for CXR.
#
# NOT a replacement for domain-aware mixup -- mutually exclusive by
# necessity (train_task2.py forces mixup off when --use-irm is set, since
# mixup's cross-domain blending breaks IRM's clean per-environment split),
# not by design preference. This run tests IRM as an alternative regularizer
# on the SAME backbone/data, so its result is directly comparable to the
# existing baseline_rad_dino checkpoints (checkpoints/rad_dino_*.pth).
#
# NOT YET RUN END TO END. STAGE 0 below smoke-tests the actual
# forward+irm_loss+backward path (not just the mock-tensor check already
# done locally) before committing to the full multi-day job.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch0 must exist
# (train_task2.sbatch's Stage 1 already produces this).
#
# Usage:
#   sbatch train_task2_irm_rad_dino.sbatch
#   IRM_LAMBDA=5 IRM_ANNEAL_EPOCHS=3 sbatch train_task2_irm_rad_dino.sbatch   # weaker penalty
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-irm-rad-dino
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
CONTAINER_NAME="task2_irm_rad_dino_${SLURM_JOB_ID}"

ENC="rad_dino"
IRM_LAMBDA="${IRM_LAMBDA:-10.0}"
IRM_ANNEAL_EPOCHS="${IRM_ANNEAL_EPOCHS:-5}"

# Separate from checkpoints/ (baseline, no IRM) -- never overwrite, both
# stay on disk for a direct evaluate.py / external-validation comparison.
OUT_DIR="${OUT_DIR:-checkpoints_irm}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "encoder: $ENC (IRM only, rad_dino)"
echo "irm-lambda: $IRM_LAMBDA  irm-anneal-epochs: $IRM_ANNEAL_EPOCHS"
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
echo "### STAGE 0: sanity-check IRM forward+loss+backward on the real model (no training) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_smoke" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python3 -c \"
import torch
import torch.nn as nn
from model import TBClassifier
from train_task2 import irm_loss

enc = 'rad_dino'
img_size = 518
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TBClassifier(enc, pretrained=True, use_mixstyle=False, use_mask_attention=False).to(device)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{enc}: {n_trainable} trainable params (LoRA + head)')
assert n_trainable > 0

x = torch.randn(8, 3, img_size, img_size, device=device)
y = torch.randint(0, 2, (8,), device=device).float()
dom = ['CR','CR','CR','DX','DX','XA','XA','XA']
criterion = nn.BCEWithLogitsLoss()

logits = model(x)
loss = irm_loss(logits, y, dom, criterion, penalty_weight=1.0)
loss.backward()
grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f'{enc}: IRM loss={loss.item():.4f}, backward OK, total abs grad sum={grad_norm:.4f}')
assert grad_norm > 0, 'zero gradient -- something is disconnected'
print(f'{enc}: SMOKE TEST PASSED')
\""

echo "############################################################"
echo "### full leave-one-modality-out run, IRM ON (rad_dino) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
      --csv /workspace/Data/train.csv \
      --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --use-irm --irm-lambda $IRM_LAMBDA --irm-anneal-epochs $IRM_ANNEAL_EPOCHS \
      --out-dir /workspace/$OUT_DIR \
      --fold-mode leave_one_modality_out --epochs 15"

echo "=== IRM TRAINING DONE (rad_dino) -> $PROJECT_ROOT/$OUT_DIR/ ==="
echo "Compare against the baseline (no-IRM) rad_dino checkpoints:"
echo "  python Code/evaluate.py --ckpt-dir $OUT_DIR --encoder rad_dino"
echo "  python Code/evaluate.py --ckpt-dir checkpoints --encoder rad_dino"
echo ""
echo "Then score against Data/test.csv and all 3 external datasets the same way"
echo "as every other variant (tag it 'irm_rad_dino' in eval_test_all_variants.sbatch /"
echo "eval_external_all_variants.sbatch-style scoring) to see whether IRM actually"
echo "closes the internal-vs-external gap for rad_dino, or is another internal-only"
echo "artifact like mask-channel and dual-stream both turned out to be."
