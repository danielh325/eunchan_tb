#!/bin/bash
# ============================================================================
# Task 2 — mask-guided attention training for rad_dino ONLY, extending
# train_task2_mask_guided.sbatch's CNN-only gate to a per-patch-token gate.
# Was previously hard-blocked -- train_task2.py used to raise SystemExit on
# rad_dino/chexfound_vitl16 with --mask-guide-dir, because TBClassifier's
# gate only knew how to hook a spatial (B,C,H,W)/(B,H,W,C) CNN feature map,
# and rad_dino's frozen+LoRA wrapper only exposed pooled CLS output, no
# spatial map to gate.
#
# rad_dino specifically (not chexfound_vitl16 too) per the decision that it's
# the best-generalizing encoder overall (see memory:
# task2_submission_vs_recommended_model -- baseline_rad_dino alone beat every
# other model/ensemble on real external validation, Shenzhen F1=0.8954 /
# Montgomery F1=0.8780).
#
# What changed (2026-07-30), to lift that block:
#   - encoders.py: _RadDinoWrapper.forward_features now exposes the raw
#     (B,N,C) patch-token sequence (N=1369, 37x37 grid @ 518px/patch14)
#     instead of only the pooled CLS output.
#   - model.py: TBClassifier._forward_vit_gated -- a per-token nn.Linear(feat,1)
#     gate (the token-sequence equivalent of the CNN path's 1x1 conv),
#     sigmoid'd, multiplied elementwise into each token, then mean-pooled
#     (replacing CLS-token pooling) before the classification head. The
#     token-grid attn map reshapes to (B,37,37) -- gradcam_task2.
#     gradcam_rad_dino already used this exact reshape for visualization;
#     this is the training-time counterpart.
#   - train_task2.py's Dice-loss upsampling code (attn -> bilinear-interpolate
#     -> dice_loss against mask_t) is UNCHANGED and already generic to any
#     grid size, so no changes were needed there once the gate itself worked.
#
# NOT YET SMOKE-TESTED end to end (no local GPU/rad_dino weights this
# session) -- the one real remaining risk is whether calling
# self.encoder.forward_features(x) actually reaches the underlying
# _RadDinoWrapper's custom method THROUGH peft's LoRA wrapper (PeftModel
# needs to attribute-delegate an arbitrary custom method down to the wrapped
# module -- documented peft behavior, but unverified in this exact codebase).
# STAGE 0 below runs a cheap (~seconds) forward+backward sanity check before
# committing to the full multi-day run -- do not comment it out to save
# time; if it fails, the full run would too, just days later.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch2 must already exist
# (train_task2.sbatch's Stage 1 always runs preprocess.py with
# --mask-channel, so it should be there already from the main run).
#
# Usage:
#   sbatch train_task2_mask_guided_vit.sbatch
#   MASK_GUIDE_LAMBDA=0.5 sbatch train_task2_mask_guided_vit.sbatch   # cheap spot-check
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-mask-guided-vit-rad-dino
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
CONTAINER_NAME="task2_mask_guided_vit_rad_dino_${SLURM_JOB_ID}"

ENC="rad_dino"
IMG_SIZE=518

MASK_GUIDE_LAMBDA="${MASK_GUIDE_LAMBDA:-1.0}"

# Separate dir from the CNN-only checkpoints_mask_guided/ -- different gate
# mechanism (per-token linear vs 1x1 conv) on a different encoder entirely,
# never overwrite either.
OUT_DIR="${OUT_DIR:-checkpoints_mask_guided_vit}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "encoder: $ENC (rad_dino only)"
echo "mask-guide-lambda: $MASK_GUIDE_LAMBDA"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch2" ] || [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch2" ]; then
    echo "!! Data/Preprocessed/{train,test}_images/ch2 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 first (preprocess.py --lung-crop" >&2
    echo "!! --mask-channel), this run needs the precomputed masks, not" >&2
    echo "!! just the ch0 crops." >&2
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
  -e TASK2_WEIGHTS_DIR=/weights)

echo "############################################################"
echo "### STAGE 0: sanity-check the ViT mask-guided gate (forward+backward, no training) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_smoke" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python3 -c \"
import torch
import torch.nn.functional as F
from model import TBClassifier

enc = 'rad_dino'
img_size = $IMG_SIZE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TBClassifier(enc, pretrained=True, use_mask_attention=True).to(device)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f'{enc}: {n_trainable}/{n_total} trainable params (LoRA + gate + head)')
assert n_trainable > 0, 'nothing trainable -- LoRA/gate not wired correctly'

x = torch.randn(2, 3, img_size, img_size, device=device)
y = torch.randint(0, 2, (2,), device=device).float()
mask_t = torch.randint(0, 2, (2, img_size, img_size), device=device).float()

logit, attn = model(x, return_attn=True)
print(f'{enc}: logit shape={tuple(logit.shape)} attn shape={tuple(attn.shape)}')
assert logit.shape == (2,), logit.shape
assert attn.dim() == 3 and attn.shape[0] == 2, attn.shape

# exactly mirrors train_task2.py's real loss: BCE(logits,y) + lambda*dice_loss(attn_up, mask_t)
attn_up = F.interpolate(attn.unsqueeze(1), size=mask_t.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
attn_up = attn_up.clamp(0, 1)
inter = (attn_up.flatten(1) * mask_t.flatten(1)).sum(1)
union = attn_up.flatten(1).sum(1) + mask_t.flatten(1).sum(1)
dice = (1 - (2 * inter + 1e-6) / (union + 1e-6)).mean()
loss = F.binary_cross_entropy_with_logits(logit, y) + dice
loss.backward()

grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f'{enc}: backward OK, dice={dice.item():.4f}, total abs grad sum={grad_norm:.4f}')
assert grad_norm > 0, 'zero gradient -- forward_features likely bypassing LoRA/gate'
print(f'{enc}: SMOKE TEST PASSED')
\""

echo "############################################################"
echo "### full leave-one-modality-out run, ViT mask-guided attention ON (rad_dino) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
      --csv /workspace/Data/train.csv \
      --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --mask-guide-dir /workspace/Data/Preprocessed/train_images/ch2 \
      --mask-guide-lambda $MASK_GUIDE_LAMBDA \
      --out-dir /workspace/$OUT_DIR \
      --fold-mode leave_one_modality_out --epochs 15"

echo "=== ViT MASK-GUIDED TRAINING DONE (rad_dino) -> $PROJECT_ROOT/$OUT_DIR/ ==="
echo "Compare against the existing baseline/mask-channel rad_dino runs:"
echo "  python Code/evaluate.py --ckpt-dir $OUT_DIR"
echo "  python Code/evaluate.py --ckpt-dir checkpoints"
echo "  python Code/evaluate.py --ckpt-dir checkpoints_mask_ablation"
echo ""
echo "Then score against Data/test.csv and the external datasets the same way"
echo "as the other variants (see eval_test_all_variants.sbatch /"
echo "eval_external_all_variants.sbatch -- add checkpoints_mask_guided_vit as"
echo "a 4th variant, tag 'maskguidedvit_rad_dino') to see whether ViT"
echo "mask-guidance actually beats plain baseline_rad_dino externally, the"
echo "way it did NOT for mask-channel (see memory:"
echo "task2_submission_vs_recommended_model)."
