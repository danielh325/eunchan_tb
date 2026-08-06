#!/bin/bash
# ============================================================================
# One-time host-side download of TBX11K, a third public TB CXR dataset for
# external-domain validation (alongside Shenzhen/Montgomery, already fetched
# by fetch_external_validation.sh). 11,200 images across 5 categories
# (healthy, sick_but_non-tb, active_tb, latent_tb, active_and_latent_tb) --
# by far the largest and most independent external source available, so it
# should tighten the external-val numbers (Shenzhen+Montgomery combined are
# only ~800 images, a lot of variance behind the 0.74/0.86 accuracy gap
# already found).
#
# Source: official repo https://github.com/yun-liu/Tuberculosis -- NOT a
# plain curl-able URL like Shenzhen/Montgomery (those are direct NLM zips).
# TBX11K is hosted on Google Drive, so this uses `gdown` (handles Google's
# large-file virus-scan confirmation page that a bare curl/wget chokes on).
#
# Run directly on the login/build node (NOT inside sbatch/docker) -- just
# needs outbound internet via the cluster proxy, no GPU.
#
# IMPORTANT -- this script only downloads and unzips. It deliberately does
# NOT build a classification CSV yet: TBX11K's exact post-extraction folder
# layout and label-file format are only documented in the README.md bundled
# inside the dataset zip itself (every external source -- the GitHub repo,
# mmcheng.net, review papers -- just says "see the included README",
# without repeating the details on the page). Once this script finishes,
# READ Data/external/tbx11k/README.md and inspect the extracted tree
# (the "STAGE 2" listing below prints it) before writing/adjusting
# build_tbx11k_csv.py -- don't assume the folder names sight-unseen.
#
# Usage: bash sbatch/fetch_tbx11k.sh
# Safe to re-run -- skips the download if the zip's extracted dir already
# exists.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"   # .../Task2
EXT_DIR="$PROJECT_ROOT/Data/external"
TBX_DIR="$EXT_DIR/tbx11k"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

mkdir -p "$EXT_DIR"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found -- installing (pip install --user gdown)..."
    pip install --user gdown
fi

echo "=== 1/2: downloading TBX11K (Google Drive, official repo link) ==="
if [ -d "$TBX_DIR" ] && [ -n "$(ls -A "$TBX_DIR" 2>/dev/null)" ]; then
    echo "already present -> $TBX_DIR (skipping download)"
else
    mkdir -p "$TBX_DIR"
    # --fuzzy accepts the full "view?usp=sharing" share URL directly and
    # handles Google's large-file confirm-download interstitial itself.
    gdown --fuzzy \
        "https://drive.google.com/file/d/1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR/view?usp=sharing" \
        -O "$EXT_DIR/tbx11k.zip"
    python3 -m zipfile -e "$EXT_DIR/tbx11k.zip" "$TBX_DIR"
    rm -f "$EXT_DIR/tbx11k.zip"
fi

echo "=== 2/2: extracted tree (inspect this + README.md before writing the CSV builder) ==="
find "$TBX_DIR" -maxdepth 3 | sort
echo
if [ -f "$TBX_DIR/README.md" ]; then
    echo "--- $TBX_DIR/README.md ---"
    cat "$TBX_DIR/README.md"
else
    echo "!! no top-level README.md found -- check one level down (find above) for it."
fi

echo
echo "=== NEXT STEP (manual) ==="
echo "Read the README/tree above, confirm the label format (folder-per-class"
echo "vs TBX11K_train.txt/val.txt list files with 'path label' lines), then"
echo "write/adjust Code/build_tbx11k_csv.py to emit a train.csv-compatible"
echo "CSV (new_id, TB/Normal) the same way build_external_csv.py does for"
echo "Shenzhen/Montgomery. Do not guess the folder names blind."
