#!/bin/bash
# Runs INSIDE the container, on a single GPU. Trains every model passed as an
# argument through all 5 CV folds (sequentially, one model at a time), then
# runs ensemble inference for each. Called by the sbatch scripts with one GPU
# already selected via CUDA_VISIBLE_DEVICES.
#
# Usage: run_models.sh convnext_tiny
#        run_models.sh resnet50d densenet201 tf_efficientnetv2_s
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <model_name> [<model_name> ...]" >&2
    exit 1
fi

fail=0
for MODEL in "$@"; do
    echo "=== [$MODEL] training 5 folds on GPU ${CUDA_VISIBLE_DEVICES:-default} ==="
    python train.py --model "$MODEL" --fold all 2>&1 | tee "train_${MODEL}.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! training FAILED for $MODEL — skipping its inference" >&2
        fail=1
        continue
    fi

    echo "=== [$MODEL] 5-fold ensemble inference ==="
    python predict.py --model "$MODEL" 2>&1 | tee "predict_${MODEL}.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! inference FAILED for $MODEL" >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "one or more models failed — see logs above." >&2
    exit 1
fi
echo "=== DONE: $* -> Task1/submissions/submission_<model>.csv ==="
