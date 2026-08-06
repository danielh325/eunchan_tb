#!/bin/bash
# ============================================================================
# One-time host-side download of the Shenzhen + Montgomery public TB CXR
# datasets, for genuine external-domain validation of Task 2's models -- NOT
# the internal leave-one-modality-out proxy (which is still same-source-pool
# data), but real cross-country, cross-institution data the models have
# never seen. Same two datasets the user's own SSRN paper (Domain
# Generalization of Tuberculosis Detection in Chest X-Rays through MixStyle
# and Multi-Level Augmentation) validated on, so results are directly
# comparable to that paper's own numbers (66%->89% Shenzhen with plain
# DenseNet-121+MixStyle+aug) -- this build's real bar to clear.
#
# Sources (NLM/NIH official, per Jaeger et al. 2014, the datasets' own paper):
#   Shenzhen:   https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
#               (662 images: 326 Normal, 336 TB)
#   Montgomery: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
#               (138 images: 80 Normal, 58 TB)
#
# Run directly on the login/build node (NOT inside sbatch/docker) -- just
# needs outbound internet via the cluster proxy, no GPU.
#
# Usage: bash sbatch/fetch_external_validation.sh
# Safe to re-run -- skips any file/dir that already exists.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"   # .../Task2
EXT_DIR="$PROJECT_ROOT/Data/external"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

mkdir -p "$EXT_DIR"

echo "=== 1/2: Shenzhen Hospital CXR Set ==="
if [ -d "$EXT_DIR/shenzhen/CXR_png" ]; then
    echo "already present -> $EXT_DIR/shenzhen/CXR_png"
else
    curl -fL --retry 5 --retry-delay 15 \
        "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip" \
        -o "$EXT_DIR/shenzhen.zip"
    mkdir -p "$EXT_DIR/shenzhen"
    python3 -m zipfile -e "$EXT_DIR/shenzhen.zip" "$EXT_DIR/shenzhen"
    # zip layout is ChinaSet_AllFiles/{CXR_png,ClinicalReadings}/ -- flatten
    # one level so CXR_png sits directly under $EXT_DIR/shenzhen/. Uses
    # `rm -rf` instead of `mv * ; rmdir` -- bash's bare `*` glob doesn't
    # match dotfiles, so a stray .DS_Store (common in these NLM zips, which
    # look Mac-originated given the __MACOSX metadata folder they also
    # contain) gets left behind, making `rmdir` fail on a "non-empty"
    # directory and killing the whole script under set -e. We've already
    # moved everything we actually want, so just force-remove what's left.
    if [ -d "$EXT_DIR/shenzhen/ChinaSet_AllFiles" ]; then
        mv "$EXT_DIR/shenzhen/ChinaSet_AllFiles"/* "$EXT_DIR/shenzhen/"
        rm -rf "$EXT_DIR/shenzhen/ChinaSet_AllFiles"
    fi
    rm -rf "$EXT_DIR/shenzhen/__MACOSX"
    rm -f "$EXT_DIR/shenzhen.zip"
fi

echo "=== 2/2: Montgomery County CXR Set ==="
if [ -d "$EXT_DIR/montgomery/CXR_png" ]; then
    echo "already present -> $EXT_DIR/montgomery/CXR_png"
else
    curl -fL --retry 5 --retry-delay 15 \
        "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip" \
        -o "$EXT_DIR/montgomery.zip"
    mkdir -p "$EXT_DIR/montgomery"
    python3 -m zipfile -e "$EXT_DIR/montgomery.zip" "$EXT_DIR/montgomery"
    if [ -d "$EXT_DIR/montgomery/MontgomerySet" ]; then
        mv "$EXT_DIR/montgomery/MontgomerySet"/* "$EXT_DIR/montgomery/"
        rm -rf "$EXT_DIR/montgomery/MontgomerySet"
    fi
    rm -rf "$EXT_DIR/montgomery/__MACOSX"
    rm -f "$EXT_DIR/montgomery.zip"
fi

echo "=== building CSVs (new_id, TB/Normal) ==="
cd "$HERE/../Code"
python3 build_external_csv.py --image-dir "$EXT_DIR/shenzhen/CXR_png" --out "$EXT_DIR/shenzhen.csv"
python3 build_external_csv.py --image-dir "$EXT_DIR/montgomery/CXR_png" --out "$EXT_DIR/montgomery.csv"

echo "=== all external validation data present under $EXT_DIR ==="
echo "Next: run preprocess.py --lung-crop on each CXR_png dir (see"
echo "sbatch/run_external_validation.sbatch), then predict_task2.py --score"
echo "against each CSV once real Task2 checkpoints exist."
