#!/bin/bash
# ============================================================================
# Task 2 -- train chexfound_vitl16 with CheXFound's OWN validated downstream-
# adaptation recipe (GLoRI-lite): backbone FULLY FROZEN (no LoRA at all --
# supersedes the earlier, now-reverted LoRA-rank-32 attempt, see encoders.py's
# ENCODER_FAMILIES comment on chexfound_vitl16 for why that was wrong), head
# reads concatenated CLS tokens from the last 4 transformer blocks plus one
# learned-query attention-pooled "local" feature over the last layer's patch
# tokens -- see model.py's TBClassifier._forward_glori.
#
# WHY: research (in-session, 2026-07-31) found CheXFound's own paper/repo
# doesn't fine-tune the backbone at all for downstream tasks -- it freezes
# it and reads multi-layer features via the GLoRI head. Our existing
# checkpoints/chexfound_vitl16 (LoRA-adapted, single-pooled-output readout)
# never matched that recipe. This run tests whether using CheXFound the way
# it was actually designed/validated closes any of its gap to
# baseline_rad_dino on real external data (see memory:
# task2_submission_vs_recommended_model).
#
# EPOCHS was 30 (vs 15 everywhere else) on the theory that a frozen backbone's
# small head needs longer to converge. Two completed folds say otherwise, so
# it is back to 15 + early stopping (PATIENCE=8): CR peaked at epoch 11/30 and
# then drifted flat-to-down for 19 more epochs, and DX peaked at epoch 1-2
# (val_acc 0.9948) and COLLAPSED monotonically to 0.7120 / val_auc 0.7036 by
# epoch 29 while train_loss kept falling -- textbook overfitting of the head.
# Best-val_acc selection means the saved weights were never the collapsed
# ones, so this costs accuracy nothing; it just stops burning ~4h/fold past
# the useful point. The original reasoning for a longer schedule is preserved
# below because it is still correct about the COST per epoch:
#
# with the backbone fully
# frozen, only the small GLoRI-lite head (~a few million params: n_layers x
# feat_dim CLS concat + 1 attention query + a linear head) trains, which
# typically needs more epochs to converge than joint encoder+LoRA
# fine-tuning does. It's also genuinely cheaper per step than the LoRA runs,
# confirmed locally (standard PyTorch semantics, not just an assumption):
# when every backbone parameter AND the input image both have
# requires_grad=False, the backbone's output itself gets requires_grad=False
# automatically, so autograd builds NO graph at all for the backbone's
# forward pass (verified with a minimal nn.Linear reproduction: output.
# grad_fn is None) -- no explicit torch.no_grad() needed, and no backward-
# pass memory/compute spent on the backbone at any point.
#
# STATUS: job 119810 (2026-07-31 12:34) got 12h in and was scancel'd at
# 08-01 00:36 -- NOT a time limit (--time is 6 days), the GPU was freed for
# the DG-ablation queue. CR and DX were saved (~6h/fold); XA and XC never ran.
#
# Job 120352 (resubmit, 08-02) was scancel'd the same way at 08-03 07:11 and
# produced NOTHING, for two reasons now fixed:
#   1. This script's header claimed "STAGE 1 below therefore passes
#      --skip-existing-folds" but the invocation never actually passed it, so
#      120352 retrained CR from scratch for ~6h to land on the byte-identical
#      result (best val_acc=0.9401, same seed) before even starting DX. The
#      flag is passed for real now, gated on RESUME (default 1).
#   2. train_task2.py only wrote a fold's checkpoint AFTER its last epoch, so
#      the 29 DX epochs 120352 did complete were thrown away on scancel even
#      though the best one had been reached in the first 20 minutes. It now
#      flushes each new best to disk as it happens (atomic tmp+rename).
# Together: a resubmit reuses CR + DX and trains only XA + XC (~8h with the
# shorter schedule), and another scancel costs at most the epochs since the
# last improvement rather than a whole fold.
#
# train_task2.py hard-errors if a reused checkpoint's stored config (glori
# head, n_layers, channel mode, encoder) doesn't match this invocation, so the
# reuse can't silently mix configs across folds. It does NOT check --epochs:
# the reused CR/DX were trained on the old 30-epoch schedule and XA/XC will
# get 15+patience. That is a real (small) inconsistency, but both schedules
# select on best val_acc and both reused folds peaked at epoch <= 11, i.e.
# inside the new budget, so neither would have been saved differently.
# RESUME=0 forces a clean, uniform 4-fold run if you want it airtight.
#
# STAGE 0 smoke-tests the actual forward+backward path (frozen backbone +
# GLoRI-lite head) before the full job. STAGE 2 scores the finished
# checkpoints on test.csv + all 3 external datasets (mirroring
# train_task2_rad_dino_dg_ablation.sbatch), which is what the footer of this
# script used to ask you to do by hand -- it was described in this header
# before it existed; it exists now.
#
# PREREQUISITE: Data/Preprocessed/{train,test}_images/ch0 must exist.
# STAGE 2 additionally needs the preprocessed external dirs.
#
# Usage:
#   sbatch train_task2_chexfound_glori.sbatch
#   EPOCHS=50 GLORI_N_LAYERS=6 sbatch train_task2_chexfound_glori.sbatch
#   SKIP_EVAL=1 sbatch train_task2_chexfound_glori.sbatch     # train only
#   RESUME=0 sbatch train_task2_chexfound_glori.sbatch        # retrain all 4 folds
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-chexfound-glori
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # chexfound weights live here
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_chexfound_glori_${SLURM_JOB_ID}"

ENC="chexfound_vitl16"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-8}"
GLORI_N_LAYERS="${GLORI_N_LAYERS:-4}"

# RESUME=1 (default) reuses any fold checkpoint already in OUT_DIR. Kept as a
# plain string, NOT a bash array: this gets interpolated into the `docker run
# ... bash -c "..."` string below, where an array expansion silently drops the
# flag instead of failing loudly.
RESUME_FLAG=""
if [ "${RESUME:-1}" = "1" ]; then
  RESUME_FLAG="--skip-existing-folds"
fi

# Separate from checkpoints/ (original LoRA r=16 chexfound) -- never
# overwrite, both stay on disk for a direct comparison.
OUT_DIR="${OUT_DIR:-checkpoints_chexfound_glori}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "encoder: $ENC (GLoRI-lite, frozen backbone)   epochs=$EPOCHS   patience=$PATIENCE   glori_n_layers=$GLORI_N_LAYERS"
echo "resume: ${RESUME:-1} (${RESUME_FLAG:-retrain all folds})"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_images/ch0" ] || [ ! -d "$PROJECT_ROOT/Data/Preprocessed/test_images/ch0" ]; then
    echo "!! Data/Preprocessed/{train,test}_images/ch0 missing -- run" >&2
    echo "!! train_task2.sbatch's Stage 1 first." >&2
    exit 1
fi

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
echo "### STAGE 0: sanity-check the frozen-backbone GLoRI-lite head (no training) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_smoke" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python3 -c \"
import torch
from model import TBClassifier

enc = 'chexfound_vitl16'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TBClassifier(enc, pretrained=True, use_glori_head=True, glori_n_layers=$GLORI_N_LAYERS).to(device)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
n_backbone_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
print(f'{enc}: {n_trainable}/{n_total} trainable params overall, {n_backbone_trainable} of them in the backbone')
assert n_trainable > 0, 'nothing trainable at all'
assert n_backbone_trainable == 0, f'backbone should be FULLY frozen, found {n_backbone_trainable} trainable backbone params'

x = torch.randn(2, 3, 512, 512, device=device)
logit = model(x)
print(f'{enc}: logit shape={tuple(logit.shape)}')
assert logit.shape == (2,), logit.shape

logit.sum().backward()
grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f'{enc}: backward OK, total abs grad sum={grad_norm:.4f}')
assert grad_norm > 0, 'zero gradient -- head/query not receiving signal'
print(f'{enc}: SMOKE TEST PASSED')
\""

echo "############################################################"
echo "### full leave-one-modality-out run, chexfound_vitl16 GLoRI-lite (frozen backbone) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
      --csv /workspace/Data/train.csv \
      --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
      --use-glori-head --glori-n-layers $GLORI_N_LAYERS \
      --out-dir /workspace/$OUT_DIR \
      $RESUME_FLAG \
      --fold-mode leave_one_modality_out --epochs $EPOCHS --patience $PATIENCE"

echo "=== chexfound_vitl16 GLoRI-lite TRAINING DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="

# ============================================================================
# STAGE 2 -- score these checkpoints on test.csv + all 3 external datasets,
# same predict_task2.py invocation the DG-ablation script uses, so the numbers
# are directly comparable to baseline_rad_dino's. GLoRI trained on lung-cropped
# ch0, so it is scored against ch0 dirs throughout (no raw-image switch of the
# kind the no_crop ablation needs).
# ============================================================================
if [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "SKIP_EVAL=1 -- stopping after training."
  exit 0
fi

DATASETS_NAME=(test shenzhen montgomery tbx11k_full)
DATASETS_CSV=(Data/test.csv Data/external/shenzhen.csv Data/external/montgomery.csv Data/external/tbx11k/tbx11k_full.csv)
DATASETS_IMG=(Data/Preprocessed/test_images/ch0 Data/external/shenzhen/preprocessed/ch0 Data/external/montgomery/preprocessed/ch0 Data/external/tbx11k/preprocessed_full/ch0)

EVAL_CSV="$PROJECT_ROOT/chexfound_glori_eval_summary.csv"
echo "config,dataset,accuracy@0.5,auc,f1@0.5,best_tau,accuracy@best_tau" > "$EVAL_CSV"

echo "############################################################"
echo "### STAGE 2: external validation, chexfound_vitl16 GLoRI-lite ###"
echo "############################################################"

for d in "${!DATASETS_NAME[@]}"; do
  DS="${DATASETS_NAME[$d]}"

  if [ ! -f "$PROJECT_ROOT/${DATASETS_CSV[$d]}" ] || [ ! -d "$PROJECT_ROOT/${DATASETS_IMG[$d]}" ]; then
    echo "!! skipping $DS -- ${DATASETS_CSV[$d]} or ${DATASETS_IMG[$d]} missing" >&2
    echo "chexfound_glori,${DS},NA,NA,NA,NA,NA" >> "$EVAL_CSV"
    continue
  fi

  echo "--- chexfound_glori / $DS ---"
  RAW_LOG="$PROJECT_ROOT/.eval_chexfound_glori_${DS}.raw.log"

  # Training is already banked by now, so one failed dataset must not take the
  # job down with it: record NA and carry on to the next.
  if "${RUN_DOCKER[@]}" \
      --name "${CONTAINER_NAME}_eval_${DS}" \
      "$IMAGE" \
      bash -c "cd /workspace/Code && python predict_task2.py \
          --ckpt-dir /workspace/$OUT_DIR \
          --csv /workspace/${DATASETS_CSV[$d]} \
          --image-dir /workspace/${DATASETS_IMG[$d]} \
          --encoders $ENC \
          --out /workspace/submission_chexfound_glori_${DS}.csv --score" > "$RAW_LOG" 2>&1; then
    cat "$RAW_LOG"
    SCORE_LINE="$(grep -m1 '^accuracy@0.5=' "$RAW_LOG" || true)"
    TAU_LINE="$(grep -m1 '^best-tau=' "$RAW_LOG" || true)"
    ACC="$(echo "$SCORE_LINE" | sed -n 's/.*accuracy@0\.5=\([0-9.]*\).*/\1/p')"
    AUC="$(echo "$SCORE_LINE" | sed -n 's/.*AUC=\([0-9.]*\).*/\1/p')"
    F1="$(echo "$SCORE_LINE" | sed -n 's/.*F1@0\.5=\([0-9.]*\).*/\1/p')"
    BEST_TAU="$(echo "$TAU_LINE" | sed -n 's/best-tau=\([0-9.]*\).*/\1/p')"
    ACC_TAU="$(echo "$TAU_LINE" | sed -n 's/.*accuracy@best-tau=\([0-9.]*\).*/\1/p')"
    echo "chexfound_glori,${DS},${ACC:-NA},${AUC:-NA},${F1:-NA},${BEST_TAU:-NA},${ACC_TAU:-NA}" >> "$EVAL_CSV"
  else
    echo "!! scoring FAILED on $DS -- output follows" >&2
    cat "$RAW_LOG" >&2
    echo "chexfound_glori,${DS},NA,NA,NA,NA,NA" >> "$EVAL_CSV"
  fi
  rm -f "$RAW_LOG"
done

echo "=== chexfound_vitl16 GLoRI-lite EVAL DONE -> $EVAL_CSV ==="
column -s, -t "$EVAL_CSV"
echo ""
echo "THE question this run exists to answer: does CheXFound's own validated"
echo "recipe (frozen backbone + GLoRI head) close any of its gap to rad_dino on"
echo "REAL external data? Compare the shenzhen/montgomery F1@0.5 rows against"
echo "baseline_rad_dino:  shenzhen=0.8954  montgomery=0.8780  tbx11k_full=0.3225"
echo "Ignore the test row -- it is same-distribution and saturates at ~0.99 for"
echo "every variant. Also compare the original LoRA chexfound checkpoints:"
echo "  python Code/evaluate.py --ckpt-dir $OUT_DIR --encoder chexfound_vitl16"
echo "  python Code/evaluate.py --ckpt-dir checkpoints --encoder chexfound_vitl16"
