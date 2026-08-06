#!/bin/bash
# ============================================================================
# Task 2 -- FULL chexfound_vitl16 run under the CORRECTED LoRA config
# (encoders.py, fixed 2026-08-05), then scored on every available dataset.
#
# WHY THIS EXISTS: every chexfound_vitl16 checkpoint trained before
# 2026-08-05 used a broken lora_targets list ["qkv","proj","fc1","fc2"]:
#   1. "fc1"/"fc2" matched NOTHING -- this backbone uses SwiGLUFFNFused
#      (w12/w3), so the ENTIRE feed-forward stack got zero LoRA adaptation.
#   2. "proj" ALSO matched patch_embed.proj (a Conv2d on raw pixels),
#      because a plain name list matches by path suffix.
# Confirmed by diagnostic job 121285 + exact parameter arithmetic
# (2,387,968 trainable = 24x(qkv+attn.proj) + patch_embed, no FFN term).
# Post-fix the adapter is 6,297,600 trainable across 96 target modules.
# So every existing chexfound number in CSV/master_results_all_models.csv
# was produced by an under-adapted model and is NOT comparable to this run.
#
# LEARNING RATE: uses the project default 1e-4 unless you override with
# LR=<value> in the environment. Be aware this LR was never tuned for
# chexfound (no sbatch script in this project has ever passed --lr), and
# the corrected adapter is 2.64x larger than what any previous chexfound
# run trained -- so 1e-4 is a REASONABLE STARTING POINT, not a known
# optimum. sweep_chexfound_lr.sbatch answers that question properly; this
# script deliberately does not pretend to.
#
# Writes to a NEW checkpoint dir so the old (invalid) baseline checkpoints
# under checkpoints/ are left untouched for reference.
#
# Usage:
#   sbatch train_eval_chexfound_lorafix.sbatch
#   LR=3e-5 sbatch train_eval_chexfound_lorafix.sbatch
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-chexfound-lorafix-full
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_chexfound_lorafix_${SLURM_JOB_ID}"

LR="${LR:-1e-4}"
OUT_DIR="checkpoints/checkpoints_chexfound_lorafix"
TAG="chexfound_lorafix"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST   LR=$LR   out=$OUT_DIR"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

echo "############################################################"
echo "### STAGE 0: verify the corrected LoRA config BEFORE training ###"
echo "############################################################"
# Multi-day job -- refuse to start if the adapter isn't what we think it is.
# Asserts all three post-fix expectations at once: no patch_embed adapter,
# SwiGLU FFN (w12/w3) actually targeted, and the exact trainable count.
"${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_gate" "$IMAGE" \
  bash -c "cd /workspace/Code && python -c \"
import sys
from encoders import build_encoder, apply_lora

backbone, _, _ = build_encoder('chexfound_vitl16', pretrained=True)
m = apply_lora(backbone, 'chexfound_vitl16', r=16, alpha=16)
names = [n for n, _ in m.named_modules() if 'lora' in n.lower()]

patch = [n for n in names if 'patch_embed' in n]
ffn   = sorted({t for n in names for t in ('w12','w3') if f'.{t}.' in n})
train = sum(p.numel() for p in m.parameters() if p.requires_grad)
ntgt  = len(names)//8
print(f'patch_embed adapters : {len(patch)}  (want 0)')
print(f'FFN adapters present : {ffn}  (want [w12, w3])')
print(f'target modules       : {ntgt}  (want 96)')
print(f'trainable params     : {train}  (want 6297600)')

bad = []
if patch:              bad.append('patch_embed still adapted')
if ffn != ['w12','w3']: bad.append('SwiGLU FFN not targeted')
if train != 6297600:   bad.append(f'unexpected trainable count {train}')
if bad:
    print('GATE FAILED: ' + '; '.join(bad)); sys.exit(1)
print('GATE PASSED')
\""

echo "############################################################"
echo "### STAGE 1: full LOMO training (15 epochs, lr=$LR) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_train" "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py \
      --encoder chexfound_vitl16 \
      --csv /workspace/Data/train.csv \
      --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --out-dir /workspace/$OUT_DIR \
      --fold-mode leave_one_modality_out \
      --epochs 15 \
      --lr $LR \
      --skip-existing-folds"

echo "############################################################"
echo "### STAGE 2: score on all 5 datasets ###"
echo "############################################################"
DS_NAMES=(test shenzhen montgomery tbx11k_full mendeley_tb)
DS_CSVS=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv Data/external/mendeley_tb/mendeley_tb.csv)
DS_IMGS=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0 Data/external/mendeley_tb/preprocessed/ch0)

SUMMARY_CSV="$PROJECT_ROOT/chexfound_lorafix_summary.csv"
echo "model,dataset,accuracy@0.5,auc,f1@0.5" > "$SUMMARY_CSV"

for i in "${!DS_NAMES[@]}"; do
  DS="${DS_NAMES[$i]}"
  echo "--- $TAG / $DS ---"
  RAW_LOG="$PROJECT_ROOT/.chexfound_lorafix_${DS}.raw.log"
  "${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_eval_${DS}" "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/$OUT_DIR \
        --csv /workspace/${DS_CSVS[$i]} \
        --image-dir /workspace/${DS_IMGS[$i]} \
        --encoders chexfound_vitl16 --score \
        --out /workspace/submission_${TAG}_${DS}.csv" 2>&1 | tee "$RAW_LOG"

  LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
  ACC="$(echo "$LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
  AUC="$(echo "$LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
  F1="$(echo "$LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
  echo "${TAG},${DS},${ACC:-NA},${AUC:-NA},${F1:-NA}" >> "$SUMMARY_CSV"
  rm -f "$RAW_LOG"
done

echo "############################################################"
echo "### STAGE 3: readout, incl. class-balanced TBX11K ###"
echo "############################################################"
# TBX11K raw is 9.5% TB (800/8400), so its F1@0.5 is dominated by
# prevalence, not discrimination -- balance it before ranking anything
# (see memory: task2_tbx11k_composition). seed=42 to match the balanced
# numbers computed for the 4fold-vs-3fold comparison.
"${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_readout" "$IMAGE" \
  bash -c "cd /workspace && python - <<'PY'
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

lab = pd.read_csv('Data/external/tbx11k/tbx11k_full.csv', encoding='utf-8-sig').set_index('new_id')
y = (lab['TB/Normal'] == 'TB')
tb = y[y].index
rng = np.random.RandomState(42)
norm = rng.choice(y[~y].index, size=len(tb), replace=False)
ids = list(tb) + list(norm)

p = pd.read_csv('submission_${TAG}_tbx11k_full.csv').set_index('new_id')['prob_tb']
d = pd.DataFrame({'y': y, 'p': p}).loc[ids].dropna()
yh = d.p >= 0.5
tp = (yh & d.y).sum(); fp = (yh & ~d.y).sum(); fn = (~yh & d.y).sum()
prec = tp/(tp+fp) if tp+fp else 0.0
rec  = tp/(tp+fn) if tp+fn else 0.0
f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
print(f'tbx11k_balanced (n={len(d)}): acc={(yh==d.y).mean():.4f} '
      f'auc={roc_auc_score(d.y,d.p):.4f} f1={f1:.4f}')
PY"

echo ""
echo "=== CHEXFOUND (LoRA-FIXED) FULL RUN DONE -- lr=$LR ==="
column -s, -t "$SUMMARY_CSV"
echo ""
echo "REFERENCE -- rad_dino baseline F1@0.5 (CSV/master_results_all_models.csv):"
echo "  test 0.9890 | shenzhen 0.8954 | montgomery 0.8780 | tbx11k_balanced 0.8054"
echo "PRE-FIX chexfound F1@0.5 (under-adapted, for before/after only):"
echo "  test 0.9912 | shenzhen 0.7733 | montgomery 0.7160 | tbx11k_balanced 0.7785"
echo ""
echo "Shenzhen/Montgomery are the honest cross-institution generalization"
echo "signal (see memory: task2_submission_vs_recommended_model). If chexfound"
echo "is still behind rad_dino there after this fix, that is a real result --"
echo "report it, do not keep tuning until it wins."
