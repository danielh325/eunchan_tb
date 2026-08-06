#!/bin/bash
# ============================================================================
# Full-pool re-download of TBX11K, superseding the one-off 500-image subset
# built by fetch_tbx11k.sh (Data/external/tbx11k/raw/ + tbx11k_500.csv). The
# original full extraction was deleted after that subset was sampled, so
# there is no local copy of imgs/{health,sick,tb,extra} anymore -- this
# re-downloads the whole zip and keeps the extraction around this time
# (only the zip itself is removed after extraction, to save space).
#
# Downloads into Data/external/tbx11k/full/ (separate from raw/, so the
# existing 500-subset raw/ + preprocessed/ch0 + tbx11k_500.csv are left
# untouched -- eval_tbx11k.sbatch keeps working against the old subset
# unchanged; eval_tbx11k_full.sbatch is the new script for the expanded run).
#
# Source: official repo https://github.com/yun-liu/Tuberculosis, hosted on
# Google Drive -- uses gdown (handles Google's large-file virus-scan
# confirmation page a bare curl/wget chokes on).
#
# Run directly on the login/build node (NOT inside sbatch/docker) -- just
# needs outbound internet via the cluster proxy, no GPU.
#
# Usage: bash sbatch/fetch_tbx11k_full.sh
# Safe to re-run -- skips the download if Data/external/tbx11k/full/ already
# has content.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"   # .../Task2
EXT_DIR="$PROJECT_ROOT/Data/external"
TBX_DIR="$EXT_DIR/tbx11k"
FULL_DIR="$TBX_DIR/full"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

mkdir -p "$FULL_DIR"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found -- installing (pip install --user gdown)..."
    pip install --user gdown
fi

echo "=== 1/2: downloading TBX11K (Google Drive, official repo link) ==="
if [ -n "$(find "$FULL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
    echo "already present -> $FULL_DIR (skipping download)"
else
    # bare file ID, not the full share URL -- gdown >=6 dropped the --fuzzy
    # CLI flag (fuzzy URL parsing became automatic for some URL forms but
    # not reliably for all of them), so pass the ID directly for
    # cross-version compatibility instead of depending on that flag.
    gdown "1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR" -O "$EXT_DIR/tbx11k_full.zip"
    python3 -m zipfile -e "$EXT_DIR/tbx11k_full.zip" "$FULL_DIR"
    rm -f "$EXT_DIR/tbx11k_full.zip"
fi

echo "=== 2/2: extracted tree (top levels) ==="
find "$FULL_DIR" -maxdepth 3 | sort

echo
echo "=== NEXT STEP ==="
echo "Confirm imgs/{health,sick,tb,extra} exist somewhere under $FULL_DIR"
echo "(same layout the original 500-subset comment in eval_tbx11k.sbatch"
echo "describes), then run:"
echo "  python3 Code/build_tbx11k_full_csv.py --tbx-root $FULL_DIR \\"
echo "      --raw-out $TBX_DIR/raw_full --csv-out $TBX_DIR/tbx11k_full.csv"
echo "then submit sbatch/eval_tbx11k_full.sbatch."
