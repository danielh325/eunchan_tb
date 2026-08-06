#!/bin/bash
# Runs INSIDE the container, on a single GPU. Trains OrdFused-CXR (CORN ordinal
# head + gated tabular fusion) over each encoder passed as an argument, runs
# ensemble inference for each, trains the image-only ablation on the primary
# (first) encoder, and finally ensembles all tabular variants into one
# submission_ordfused.csv — the competition entry.
#
# Usage: run_ordfused.sh tf_efficientnetv2_s densenet201
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ "$#" -eq 0 ]; then
    set -- tf_efficientnetv2_s densenet201   # default: the two strongest CNN baselines here
fi
PRIMARY="$1"

fail=0
run_dirs=()
for ENC in "$@"; do
    echo "=== [ordfused/$ENC] training 5 folds (CORN + gated tabular fusion) ==="
    python train_ordfused.py --encoder "$ENC" --fold all 2>&1 | tee "train_ordfused_${ENC}.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! training FAILED for ordfused/$ENC" >&2; fail=1; continue
    fi
    echo "=== [ordfused/$ENC] ensemble inference ==="
    python predict_ordfused.py --encoder "$ENC" 2>&1 | tee "predict_ordfused_${ENC}.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "!! inference FAILED for ordfused/$ENC" >&2; fail=1; }
    run_dirs+=("ordfused_${ENC}")
done

# Image-only ablation on the primary encoder (isolates the fusion contribution
# for the paper's ablation table).
echo "=== [ordfused/$PRIMARY] image-only ablation (--no-tabular) ==="
python train_ordfused.py --encoder "$PRIMARY" --no-tabular --fold all 2>&1 | tee "train_ordfused_${PRIMARY}_notab.log"
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    python predict_ordfused.py --encoder "$PRIMARY" --no-tabular 2>&1 | tee "predict_ordfused_${PRIMARY}_notab.log"
fi

# Final competition entry: ensemble every tabular OrdFused variant.
if [ "${#run_dirs[@]}" -gt 0 ]; then
    echo "=== FINAL: ensembling ${run_dirs[*]} -> submission_ordfused.csv ==="
    python predict_ordfused.py --ensemble "${run_dirs[@]}" --name ordfused 2>&1 | tee "predict_ordfused_ensemble.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "one or more ordfused steps failed — see logs above." >&2
    exit 1
fi
echo "=== DONE: ordfused -> Task1/submissions/submission_ordfused.csv ==="
