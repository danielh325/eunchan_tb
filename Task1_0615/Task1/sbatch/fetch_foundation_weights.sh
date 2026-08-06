#!/bin/bash
# ============================================================================
# One-time host-side download of the three foundation-model checkpoints used
# by OrdFused-CXR's new encoders (eva_x_base, rad_dino, chexfound_vitl16).
#
# Run this directly on the login/build node (NOT inside sbatch/docker) — it
# just needs outbound internet via the cluster proxy, no GPU. Downloads land
# in Task1/weights/, which is bind-mounted into every training container via
# the existing `-v "$PROJECT_ROOT":/workspace` in the sbatch scripts, so no
# Dockerfile changes or extra mounts are needed for the containers to see
# these weights.
#
# Usage: bash sbatch/fetch_foundation_weights.sh
# Safe to re-run — skips any file/dir that already exists.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"   # .../Task1
WEIGHTS_DIR="$PROJECT_ROOT/weights"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

mkdir -p "$WEIGHTS_DIR/eva_x" "$WEIGHTS_DIR/rad_dino" "$WEIGHTS_DIR/chexfound"

echo "=== 1/3: EVA-X base (ViT-B/16, HF direct file) ==="
EVA_X_CKPT="$WEIGHTS_DIR/eva_x/eva_x_base_patch16_merged520k_mim.pt"
if [ -s "$EVA_X_CKPT" ]; then
    echo "already present -> $EVA_X_CKPT"
else
    curl -fL --retry 5 --retry-delay 15 \
        "https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_base_patch16_merged520k_mim.pt" \
        -o "$EVA_X_CKPT"
fi

echo "=== 2/3: RAD-DINO (microsoft/rad-dino, HF snapshot) ==="
if [ -s "$WEIGHTS_DIR/rad_dino/config.json" ]; then
    echo "already present -> $WEIGHTS_DIR/rad_dino"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/rad-dino', local_dir='$WEIGHTS_DIR/rad_dino')
" || pip install --user huggingface_hub && python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/rad-dino', local_dir='$WEIGHTS_DIR/rad_dino')
"
fi

echo "=== 3/3: CheXFound teacher checkpoint (Google Drive, via gdown) ==="
CHEXFOUND_CKPT="$WEIGHTS_DIR/chexfound/teacher_checkpoint.pth"
if [ -s "$CHEXFOUND_CKPT" ]; then
    echo "already present -> $CHEXFOUND_CKPT"
else
    # TODO: fill in the exact Google Drive file ID for teacher_checkpoint.pth
    # from https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806
    # (the folder link from the CheXFound README) before running this step.
    CHEXFOUND_FILE_ID="${CHEXFOUND_FILE_ID:-}"
    if [ -z "$CHEXFOUND_FILE_ID" ]; then
        echo "!! CHEXFOUND_FILE_ID not set — open the Google Drive folder, find" >&2
        echo "!! teacher_checkpoint.pth, copy its file ID, and re-run as:" >&2
        echo "!!   CHEXFOUND_FILE_ID=<id> bash $0" >&2
        exit 1
    fi
    python3 -m pip show gdown >/dev/null 2>&1 || pip install --user gdown
    python3 -m gdown "https://drive.google.com/uc?id=${CHEXFOUND_FILE_ID}" -O "$CHEXFOUND_CKPT"
fi

echo "=== all foundation-model weights present under $WEIGHTS_DIR ==="
