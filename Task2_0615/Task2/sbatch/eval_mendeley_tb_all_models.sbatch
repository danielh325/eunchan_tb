#!/bin/bash
# ============================================================================
# Full external eval (accuracy/AUC/F1, NOT recall-only) on the Mendeley
# "Dataset of Tuberculosis Chest X-rays Images" (Pakistan hospital
# collection; 2494 TB + 514 Normal), moved into this project 2026-08-04 at
# Data/external/mendeley_tb/{normal,tb}/. Built via Code/build_mendeley_tb_csv.py.
#
# UNLIKE the Belarus cohort (eval_qu_tb_*.sbatch), this dataset has BOTH
# classes, so it supports the same accuracy/AUC/F1@0.5 metrics as Shenzhen/
# Montgomery/TBX11K. Provenance RESOLVED 2026-08-05 -- see
# Code/build_mendeley_tb_csv.py's docstring: Mendeley Data DOI
# 10.17632/8j2g3csprk.2 (Kiran & Jabeen, 2024), counts confirmed matching.
# No peer-reviewed paper is linked to it (weaker citation than Belarus's
# arXiv:2007.14895), so still flag it as such in any writeup, but it's a
# citable, DOI'd primary source now, not an untraceable pull.
#
# Mirrors eval_qu_tb_all_models.sbatch's structure and MODEL_TABLE (kept in
# sync manually -- if you add/remove a model there, mirror it here).
#
# PREREQUISITE:
#   python Code/build_mendeley_tb_csv.py \
#       --normal-dir Data/external/mendeley_tb/normal \
#       --tb-dir Data/external/mendeley_tb/tb \
#       --out-csv Data/external/mendeley_tb/mendeley_tb.csv \
#       --out-image-dir Data/external/mendeley_tb/raw
#   (run with --dry-run FIRST and check the dedup breakdown)
#
# Usage:  sbatch eval_mendeley_tb_all_models.sbatch
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-mendeley-tb-all-models
#SBATCH -w gpu18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_mendeleytb_${SLURM_JOB_ID}"

EXT_DIR="$PROJECT_ROOT/Data/external/mendeley_tb"
CSV="$EXT_DIR/mendeley_tb.csv"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"; export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"; export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"

if [ ! -f "$CSV" ] || [ ! -d "$EXT_DIR/raw" ]; then
  echo "!! $CSV or $EXT_DIR/raw missing -- run Code/build_mendeley_tb_csv.py first" >&2
  exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights
  -e PYTHONUNBUFFERED=1)

echo "############################################################"
echo "### STAGE 1: preprocessing (lung crop + mask channel) ###"
echo "############################################################"
if [ -d "$EXT_DIR/preprocessed/ch0" ] && [ -d "$EXT_DIR/preprocessed/ch2" ] \
   && [ "$(ls "$EXT_DIR/preprocessed/ch0" | wc -l)" -ge "$(ls "$EXT_DIR/raw" | wc -l)" ]; then
  echo "skip: $EXT_DIR/preprocessed already complete"
else
  rm -rf "$EXT_DIR/preprocessed"   # in case of a stale PARTIAL dir from a prior interrupted run
  "${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_pre" "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/Data/external/mendeley_tb/raw \
        --out-dir /workspace/Data/external/mendeley_tb/preprocessed \
        --lung-crop --mask-channel"
fi

echo "############################################################"
echo "### STAGE 2: score every model in the master table ###"
echo "############################################################"
# tag : ckpt-dir (relative to PROJECT_ROOT) : encoders (space-sep) : needs_mask (0/1)
MODEL_TABLE=(
  "baseline_rad_dino:checkpoints:rad_dino:0"
  "baseline_chexfound_vitl16:checkpoints:chexfound_vitl16:0"
  "baseline_densenet121:checkpoints:densenet121:0"
  "baseline_swin_tiny:checkpoints:swin_tiny:0"
  "baseline_convnext_tiny:checkpoints:convnext_tiny:0"
  "maskch_rad_dino:checkpoints/checkpoints_mask_ablation:rad_dino:1"
  "maskch_chexfound_vitl16:checkpoints/checkpoints_mask_ablation:chexfound_vitl16:1"
  "maskch_densenet121:checkpoints/checkpoints_mask_ablation:densenet121:1"
  "maskch_swin_tiny:checkpoints/checkpoints_mask_ablation:swin_tiny:1"
  "maskch_convnext_tiny:checkpoints/checkpoints_mask_ablation:convnext_tiny:1"
  "dgablation_strong_aug_no_mixup:checkpoints/checkpoints_dgablation_strong_aug_no_mixup:rad_dino:0"
  "dgablation_no_dg:checkpoints/checkpoints_dgablation_no_dg:rad_dino:0"
  "dgablation_stratified_fold:checkpoints/checkpoints_dgablation_stratified_fold:rad_dino:0"
  "dgablation_light_aug_no_mixup:checkpoints/checkpoints_dgablation_light_aug_no_mixup:rad_dino:0"
  "dgablation_mixup_only:checkpoints/checkpoints_dgablation_mixup_only:rad_dino:0"
  "dgablation_no_crop:checkpoints/checkpoints_dgablation_no_crop:rad_dino:0"
  "chexfound_glori:checkpoints/checkpoints_chexfound_glori:chexfound_vitl16:0"
)
# dgablation_no_crop trained on RAW (uncropped) images -- score against raw,
# not ch0, same rule as eval_qu_tb_all_models.sbatch's own no_crop handling.
NO_CROP_RAW_DIR="/workspace/Data/external/mendeley_tb/raw"

for entry in "${MODEL_TABLE[@]}"; do
  IFS=':' read -r TAG CKPT ENC NEEDS_MASK <<< "$entry"
  OUT="submission_mendeley_tb_${TAG}.csv"
  if [ -f "$PROJECT_ROOT/$OUT" ]; then
    echo "--- $TAG: skip (already scored) ---"
    continue
  fi
  if [ ! -d "$PROJECT_ROOT/$CKPT" ]; then
    echo "!! $TAG: $CKPT missing -- skipping" >&2
    continue
  fi

  IMG_DIR="/workspace/Data/external/mendeley_tb/preprocessed/ch0"
  if [ "$TAG" = "dgablation_no_crop" ]; then
    IMG_DIR="$NO_CROP_RAW_DIR"
  fi

  MASK_ARG=""
  if [ "$NEEDS_MASK" = "1" ]; then
    MASK_ARG="--mask-dir /workspace/Data/external/mendeley_tb/preprocessed/ch2"
  fi

  echo "--- $TAG (ckpt=$CKPT encoders=$ENC mask=$NEEDS_MASK) ---"
  RAW_LOG="$PROJECT_ROOT/.eval_mendeleytb_${TAG}.raw.log"
  if "${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_${TAG}" "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$CKPT \
          --csv /workspace/Data/external/mendeley_tb/mendeley_tb.csv \
          --image-dir $IMG_DIR \
          --encoders $ENC $MASK_ARG --score \
          --out /workspace/$OUT" > "$RAW_LOG" 2>&1; then
    tail -6 "$RAW_LOG"
    rm -f "$RAW_LOG"
  else
    echo "!! $TAG FAILED -- output follows" >&2
    cat "$RAW_LOG" >&2
    rm -f "$RAW_LOG"
  fi
done

echo "############################################################"
echo "### STAGE 3: consolidated readout, all models ###"
echo "############################################################"
"${RUN_DOCKER[@]}" --name "${CONTAINER_NAME}_readout" "$IMAGE" \
  bash -c "cd /workspace && python - <<'PY'
import glob, pandas as pd
from sklearn.metrics import roc_auc_score

lab = pd.read_csv('Data/external/mendeley_tb/mendeley_tb.csv')
y_all = (lab.set_index('new_id')['TB/Normal'] == 'TB')

ENS = {
    'ens_rad_dino+chexfound': ['baseline_rad_dino', 'baseline_chexfound_vitl16'],
    'ens_5encoder': ['baseline_rad_dino', 'baseline_chexfound_vitl16',
                      'baseline_densenet121', 'baseline_swin_tiny', 'baseline_convnext_tiny'],
}

preds = {}
for f in glob.glob('submission_mendeley_tb_*.csv'):
    tag = f[len('submission_mendeley_tb_'):-len('.csv')]
    preds[tag] = pd.read_csv(f).set_index('new_id')['prob_tb']

for name, members in ENS.items():
    parts = [preds[m] for m in members if m in preds]
    if len(parts) == len(members):
        preds[name] = sum(parts) / len(parts)

rows = []
for tag, s in preds.items():
    d = pd.DataFrame({'y': y_all, 'p': s}).dropna()
    if d.empty:
        continue
    yh = d.p >= 0.5
    tp = (yh & d.y).sum(); fp = (yh & ~d.y).sum(); fn = (~yh & d.y).sum()
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    rows.append(dict(model=tag, n=len(d), accuracy=(yh==d.y).mean(),
                      auc=roc_auc_score(d.y, d.p), precision=prec, recall=rec, f1=f1))

out = pd.DataFrame(rows).sort_values('f1', ascending=False)
out.to_csv('CSV/mendeley_tb_eval_all_models.csv', index=False, float_format='%.4f')
print(out.round(4).to_string(index=False))
print()
print(f'-> CSV/mendeley_tb_eval_all_models.csv  ({len(out)} models, n={len(lab)} images)')
print()
print('SOURCE: Mendeley Data DOI 10.17632/8j2g3csprk.2, Dataset of')
print('Tuberculosis Chest X-rays Images, Kiran and Jabeen, 2024 -- see')
print('Code/build_mendeley_tb_csv.py docstring. No peer-reviewed paper is')
print('linked to it (weaker citation than Shenzhen/Montgomery/TBX11K/')
print('Belarus), so flag it as such in any writeup, but provenance is')
print('resolved -- cite the DOI, not unverified.')
PY"

echo "=== MENDELEY TB ALL-MODELS EVAL DONE ==="
