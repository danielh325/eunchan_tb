#!/bin/bash
# ============================================================================
# Task 2 -- diagnostic (no training): checks whether apply_lora()'s
# target_modules=["qkv", "proj", "fc1", "fc2"] for chexfound_vitl16
# (encoders.py) accidentally wraps patch_embed.proj -- a Conv2d, not the
# intended attention-output Linear -- because peft matches target names by
# path SUFFIX, and vendor/chexfound/layers/patch_embed.py:65 also defines
# `self.proj` (a Conv2d) under the `self.patch_embed` attribute
# (vision_transformer.py:106), giving it the dotted path `patch_embed.proj`,
# which also ends in ".proj".
#
# rad_dino's own target list (["query","key","value","dense","fc1","fc2"])
# has no such collision -- HF's Dinov2Model names its patch-embedding conv
# `projection`, not `dense`, so "dense" never matches it. If chexfound's
# config IS wrapping patch_embed.proj, that's a real, previously-unexamined
# asymmetry between the two encoders' LoRA configs, not an inherent property
# of CheXFound -- worth ruling in/out before trusting any chexfound-vs-
# rad_dino comparison in this project.
#
# Prints named_modules() for both encoders (chexfound_vitl16 AND rad_dino,
# for a same-job sanity comparison) filtered to anything containing "lora",
# so you can grep for "patch_embed" in the output directly.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-diagnose-chexfound-lora
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_diag_lora_${SLURM_JOB_ID}"

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

"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python -c \"
import torch
from encoders import build_encoder, apply_lora

for enc_name in ['chexfound_vitl16', 'rad_dino']:
    print(f'==================== {enc_name} ====================')
    backbone, feat_dim, family = build_encoder(enc_name, pretrained=True)
    peft_model = apply_lora(backbone, enc_name, r=16, alpha=16)

    lora_names = [n for n, _ in peft_model.named_modules() if 'lora' in n.lower()]
    print(f'total modules with \\\"lora\\\" in name: {len(lora_names)}')

    patch_embed_hits = [n for n in lora_names if 'patch_embed' in n]
    print(f'DEFECT 1 -- patch_embed-related lora modules: {len(patch_embed_hits)}  (want 0)')
    for n in patch_embed_hits:
        print('  !!', n)
    if not patch_embed_hits:
        print('  OK (none -- patch_embed.proj was NOT wrapped)')

    # DEFECT 2: the feed-forward stack must actually be adapted. chexfound
    # uses SwiGLUFFNFused (w12/w3); rad_dino uses a standard Mlp (fc1/fc2).
    # An adapter count of 0 here means the FFN target names never matched.
    ffn_hits = set()
    for n in lora_names:
        for tok in ('w12', 'w3', 'fc1', 'fc2'):
            if f'.{tok}.' in n:
                ffn_hits.add(tok)
    print(f'DEFECT 2 -- FFN adapter names present: {sorted(ffn_hits) or \\\"NONE\\\"}  (want non-empty)')

    # unique wrapped target modules = lora_names / 8 names per target
    print(f'unique wrapped target modules: {len(lora_names)//8}')

    other_hits = [n for n in lora_names if 'patch_embed' not in n][:4]
    print('sample of other lora modules (sanity):')
    for n in other_hits:
        print('  ', n)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f'trainable params: {trainable} / {total} ({100*trainable/total:.3f}%)')
    print()
\""

echo "=== DONE ==="
echo "EXPECTED AFTER THE 2026-08-05 encoders.py FIX (chexfound_vitl16):"
echo "  DEFECT 1 patch_embed lora modules -> 0"
echo "  DEFECT 2 FFN adapter names        -> ['w12','w3']"
echo "  unique wrapped target modules     -> 96  (24 layers x 4: qkv/proj/w12/w3)"
echo "  trainable params                  -> 6297600  (was 2387968 pre-fix)"
echo ""
echo "rad_dino is unchanged by the fix and acts as the control:"
echo "  patch_embed 0, FFN ['fc1','fc2'], 72 targets, 2654208 trainable."
echo ""
echo "NOTE: post-fix chexfound (6.30M trainable) now has ~2.37x rad_dino's"
echo "2.65M -- which is what the old 2026-07-31 encoders.py note ASSUMED was"
echo "already true. It only becomes true now. If chexfound looks over-adapted"
echo "at this budget, THAT is the point at which lowering lora_r for it is a"
echo "real question -- on measured counts, not a theoretical one."
