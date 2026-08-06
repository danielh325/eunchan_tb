#!/bin/bash
# ============================================================================
# One-time host-side download of the NIH ChestX-ray14 dataset, to serve as a
# SECOND, source-diverse pool of hard negatives (abnormal but TB-negative) for
# Task 2 training.
#
# WHY A SECOND POOL
# -----------------
# The first hard-negative source is TBX11K's `sick` split, which is already on
# disk and costs nothing (see Code/build_hardneg_train_csv.py). But it is
# single-source -- Chinese hospitals, one imaging pipeline -- so it teaches
# "abnormal is not TB" without teaching "abnormal is not TB ACROSS VENDORS AND
# COUNTRIES". ChestX-ray14 is US-sourced (NIH Clinical Center), 112,120 images
# from 30,805 patients, with 14 pathology labels. Its abnormal-but-not-TB cases
# are exactly the negatives a worldwide external test set will be full of.
#
# ChestX-ray14 has NO tuberculosis label. Verified directly against
# Data_Entry_2017_v2020.csv on 2026-08-06 -- the label vocabulary is exactly 15
# values and TB is not among them:
#   No Finding 60361 | Infiltration 19894 | Effusion 13317 | Atelectasis 11559
#   Nodule 6331 | Mass 5782 | Pneumothorax 5302 | Consolidation 4667
#   Pleural_Thickening 3385 | Cardiomegaly 2776 | Emphysema 2516 | Edema 2303
#   Fibrosis 1686 | Pneumonia 1431 | Hernia 227
# 51,759 of the 112,120 images (46.2%) carry >=1 finding, and the cohort is not
# a TB cohort, so those are usable abnormal TB-negatives. Treat that as
# "overwhelmingly TB-negative", not "guaranteed" -- a handful of undiagnosed TB
# cases in a 112k US cohort is a far smaller error than the "all negatives are
# healthy" bias this exists to fix.
#
# WHY NOT CheXpert / PadChest / VinDr-CXR / MIMIC-CXR
# ---------------------------------------------------
# All four are gated behind an account plus a data use agreement that a named
# human has to sign (Stanford AIMI, BIMCV, and PhysioNet credentialing
# respectively). No script can obtain them. If you register and get credentials
# for any of them, they are better still than ChestX-ray14 (more sites, more
# vendors) and this script can be extended.
#
# SIZE / TIME: ~42 GB across 12 tar.gz volumes. Expect hours on a shared link.
# The images are 1024x1024 PNGs and still need preprocess.py --lung-crop before
# training, which is a second, separate multi-hour pass.
#
# Source: the official NIH Box mirror at
# https://nihcc.app.box.com/v/ChestXray-NIHCC. The 12 image URLs below were
# verified byte-for-byte against that page's own batch_download_zips.py on
# 2026-08-06, and a ranged GET on images_001 returned HTTP 206 with gzip magic
# (1f8b). NOTE: `curl -I` (HEAD) against these Box links returns a bogus 404 --
# only real GETs work, so do not "fix" a link on the strength of a failed HEAD.
#
# Run directly on the login/build node (NOT inside sbatch/docker, NOT over the
# SMB mount from a laptop -- an earlier symlink pass from a Mac died with
# ETIMEDOUT partway through, and this transfer is ~40x larger).
#
# Usage: bash sbatch/fetch_nih_cxr14.sh
# Safe to re-run -- skips any archive already downloaded and extracted.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"   # .../Task2
NIH_DIR="$PROJECT_ROOT/Data/external/nih_cxr14"
IMG_DIR="$NIH_DIR/images"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

mkdir -p "$NIH_DIR" "$IMG_DIR"

# Official NIH Box direct links (12 volumes, images_001 .. images_012).
LINKS=(
  "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
  "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz"
  "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz"
  "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz"
  "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz"
  "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz"
  "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz"
  "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz"
  "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz"
  "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz"
  "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz"
  "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz"
)

echo "=== labels: Data_Entry_2017_v2020.csv ==="
if [ -f "$NIH_DIR/Data_Entry_2017_v2020.csv" ]; then
    echo "already present"
else
    # The labels CSV is NOT under nihcc.box.com/shared/static/ like the image
    # volumes are -- it is only reachable through Box's vanity-download
    # endpoint with this file_id. Verified 2026-08-06: 112,120 rows + header,
    # 9,003,496 bytes.
    curl -fL --retry 5 --retry-delay 15 \
        "https://nihcc.app.box.com/index.php?rm=box_download_shared_file&vanity_name=ChestXray-NIHCC&file_id=f_219760887468" \
        -o "$NIH_DIR/Data_Entry_2017_v2020.csv"
fi

# Guard against silently saving Box's HTML error page as a "CSV" -- that failure
# only surfaces much later, as a confusing parse error during CSV building.
if ! head -1 "$NIH_DIR/Data_Entry_2017_v2020.csv" | grep -q "^Image Index,Finding Labels"; then
    echo "!! Data_Entry_2017_v2020.csv is not the expected CSV (Box may have" >&2
    echo "!! changed the file_id). Got: $(head -c 120 "$NIH_DIR/Data_Entry_2017_v2020.csv")" >&2
    rm -f "$NIH_DIR/Data_Entry_2017_v2020.csv"
    exit 1
fi

echo "=== images: 12 volumes, ~42 GB total ==="
for i in "${!LINKS[@]}"; do
    n=$(printf "%03d" $((i + 1)))
    tgz="$NIH_DIR/images_${n}.tar.gz"
    stamp="$NIH_DIR/.extracted_${n}"

    if [ -f "$stamp" ]; then
        echo "[$n/012] already extracted -- skipping"
        continue
    fi

    echo "[$n/012] downloading..."
    # -C - resumes a partial file rather than restarting a multi-GB transfer
    # from zero after a dropped connection.
    curl -fL --retry 5 --retry-delay 15 -C - "${LINKS[$i]}" -o "$tgz"

    # Verify before extracting: a truncated volume otherwise surfaces as a
    # confusing mid-extract error with half its images already written.
    if ! gzip -t "$tgz" 2>/dev/null; then
        echo "!! images_${n}.tar.gz failed its gzip integrity check -- removing" >&2
        echo "!! so a re-run redownloads it cleanly." >&2
        rm -f "$tgz"
        exit 1
    fi

    echo "[$n/012] extracting..."
    # The archives contain images/<name>.png; --strip-components=1 drops that
    # leading dir so everything lands flat in $IMG_DIR.
    tar -xzf "$tgz" -C "$IMG_DIR" --strip-components=1
    touch "$stamp"
    rm -f "$tgz"
done

N_IMG=$(find "$IMG_DIR" -name '*.png' | wc -l | tr -d ' ')
echo
echo "=== done: $N_IMG images under $IMG_DIR ==="
echo "(expected 112120)"
echo
echo "=== NEXT STEPS ==="
echo "1. Lung-crop preprocess (multi-hour, needs GPU for the PSPNet segmenter):"
echo "     python3 Code/preprocess.py --in-dir $IMG_DIR \\"
echo "         --out-dir $NIH_DIR/preprocessed --lung-crop --mask-channel"
echo "2. Extend Code/build_hardneg_train_csv.py to pull abnormal (>=1 finding,"
echo "   i.e. 'Finding Labels' != 'No Finding') NIH images as additional is_aux=1"
echo "   Normal rows, then retrain with sbatch/train_task2_rad_dino_hardneg.sbatch."
echo "   Consider subsampling -- 112k NIH images would swamp 7.7k challenge rows;"
echo "   something in the 3-6k range keeps the challenge signal dominant."
