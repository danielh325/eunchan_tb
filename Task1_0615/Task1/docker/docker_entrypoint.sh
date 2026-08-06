#!/bin/bash
# ============================================================================
# Task 1 submission entrypoint. Reads raw CXR images from $INPUT_DIR (default
# /input), classifies cavity presence with a 4-member ensemble (densenet201,
# tf_efficientnetv2_s, resnet50d, OrdFused/eva_x_base_notab), then -- only for
# classifier-positive cases -- segments the cavity with a 3-member foundation-
# decoder ensemble (RAD-DINO, CheXFound-ViT-L/16, EVA-X-base). Writes:
#   $OUTPUT_DIR/submission.csv   (our_id, prob_cavity, pred_cavity)
#   $OUTPUT_DIR/masks/*.png      (predicted cavity mask, one per positive case)
#
# NOTE: the exact I/O contract (input/output paths, expected filenames, and
# whether cavity-negative cases need an explicit empty mask file rather than
# no file at all) the organizers' harness uses was not confirmed when this
# was written -- adjust INPUT_DIR/OUTPUT_DIR below (or override via env vars
# at `docker run` time) to match their actual convention once known. Same
# caveat Task2/docker_entrypoint.sh carries for this project.
#
# No network access needed at runtime: all model weights (checkpoints/*, and
# rad_dino's architecture config.json) are baked into this image at build
# time.
# ============================================================================
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

mkdir -p "$OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "!! INPUT_DIR=$INPUT_DIR does not exist" >&2
    exit 1
fi

echo "=== scoring $INPUT_DIR -> $OUTPUT_DIR ==="
cd /workspace/Code
python3 predict_task1_submission.py \
    --image-dir "$INPUT_DIR" \
    --ckpt-dir /workspace/Code/runs \
    --seg-ckpt-dir /workspace/Code/runs_seg \
    --out-dir "$OUTPUT_DIR"

echo "=== done -> $OUTPUT_DIR/submission.csv (+ $OUTPUT_DIR/masks/) ==="
