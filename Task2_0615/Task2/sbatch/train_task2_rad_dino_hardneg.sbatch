#!/bin/bash
# ============================================================================
# Task 2 -- rad_dino strong_aug_no_mixup RETRAINED WITH HARD NEGATIVES.
#
# WHY THIS EXISTS
# ---------------
# The DG ablation (see task2_dg_ablation_results) settled the DG *machinery*
# question: strong augmentation is the only load-bearing piece, mixup hurts,
# and strong_aug_no_mixup is the best config found. This script does not
# revisit any of that -- it keeps that exact config and changes ONE thing:
# what the negatives are.
#
# The threshold sweep (Code/../scratchpad/threshold_sweep.py, 2026-08-06)
# controlled BOTH prevalence and threshold and still found:
#
#     shenzhen         prevalence 0.508   F1@0.5 = 0.8966
#     montgomery       prevalence 0.420   F1@0.5 = 0.8976
#     tbx11k_balanced  prevalence 0.500   F1@0.5 = 0.7998
#
# Same prevalence, same threshold, ~0.10 F1 gap. Calibration and prevalence
# are both controlled out, so the only remaining variable is the composition
# of the NEGATIVE pool: Shenzhen/Montgomery negatives are largely healthy,
# TBX11K's include sick-but-not-TB patients. The model has partly learned
# "abnormal" rather than "TB". A worldwide multi-hospital external test set
# will resemble TBX11K's negative pool far more than Shenzhen's, so this gap
# is the single biggest addressable risk for the external phase -- roughly
# 10x anything threshold tuning recovers (which was ~+0.01, i.e. noise).
#
# WHAT CHANGES
# ------------
# --csv points at Data/train_hardneg.csv and --image-dir at the merged
# symlink dir, both produced by Code/build_hardneg_train_csv.py:
#   challenge rows  7757  (3681 TB / 4076 Normal)
#   + aux hard negs 3000  (all Normal -- TBX11K OFFICIAL train-split sick only)
#   = combined     10757  (3681 TB / 7076 Normal), 42.4% of negatives abnormal
#
# Everything else is byte-identical to the strong_aug_no_mixup arm of
# train_task2_rad_dino_dg_ablation.sbatch, so a difference in the external
# numbers is attributable to the hard negatives and nothing else.
#
# FOLD / LEAKAGE DISCIPLINE
# -------------------------
# The aux rows carry is_aux=1; train_task2.py computes leave-one-modality-out
# folds on the challenge rows ALONE and then pins aux rows to fold=-1, so they
# join EVERY fold's training set and NO fold's validation set. Per-fold val_acc
# therefore stays comparable to every earlier rad_dino run.
#
# Only TBX11K's official train split (lists/TBX11K_train.txt) is used, so
# TBX11K_val (800 health / 800 sick / 200 tb) stays a genuine held-out set.
# NOTE: TBX11K_val is same-source as the new training negatives, so after this
# run it is an in-domain check, NOT an external-domain one. SHENZHEN AND
# MONTGOMERY ARE THE TRUE EXTERNAL SETS HERE -- they are never trained on, in
# this run or any previous one. Judge generalization on those two.
# (TBX11K's own `test` split is unusable as a benchmark: its 3302 images are
# unlabeled, held on the authors' server.)
#
# EXPECT INTERNAL test.csv TO DROP. Its negatives are healthy-dominated, so
# adding hard negatives makes it a harder, less representative target. That is
# the point -- test.csv has been the misleading metric all along (0.9913 there
# vs 0.80 on balanced TBX11K). Do not treat a test.csv regression as failure.
#
# Usage:
#   sbatch sbatch/train_task2_rad_dino_hardneg.sbatch
#   SKIP_EVAL=1 sbatch sbatch/train_task2_rad_dino_hardneg.sbatch   # train only
#
# Prerequisite (run once, on the login node -- no GPU needed):
#   cd Code && python3 build_hardneg_train_csv.py
#
# TODO -- confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# ============================================================================
#SBATCH --job-name=task2-rad-dino-hardneg
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
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"
IMAGE="lisa-task2:latest"

OUT_DIR="checkpoints_hardneg_strong_aug_no_mixup"
CONTAINER_NAME="task2_hardneg_${SLURM_JOB_ID}"
TRAIN_CSV="/workspace/Data/train_hardneg.csv"
IMAGE_DIR="/workspace/Data/Preprocessed/train_hardneg/ch0"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "config: strong_aug_no_mixup + TBX11K sick hard negatives"
echo "out dir: $PROJECT_ROOT/$OUT_DIR"

if [ ! -f "$PROJECT_ROOT/Data/train_hardneg.csv" ]; then
    echo "!! Data/train_hardneg.csv missing -- run first (login node, no GPU):" >&2
    echo "!!   cd Code && python3 build_hardneg_train_csv.py" >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/Data/Preprocessed/train_hardneg/ch0" ]; then
    echo "!! Data/Preprocessed/train_hardneg/ch0 missing -- run build_hardneg_train_csv.py" >&2
    exit 1
fi
# The merged dir is symlinks; a broken link surfaces here as a training-time
# FileNotFoundError thousands of images in, so check one up front instead.
PROBE="$(find "$PROJECT_ROOT/Data/Preprocessed/train_hardneg/ch0" -name 's*.png' | head -1)"
if [ -n "$PROBE" ] && [ ! -e "$PROBE" ]; then
    echo "!! symlinks in train_hardneg/ch0 are broken (dangling: $PROBE)" >&2
    echo "!! rebuild with: cd Code && python3 build_hardneg_train_csv.py" >&2
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
echo "### STAGE 1 -- train rad_dino (strong aug, no mixup, + hard negatives) ###"
echo "############################################################"
"${RUN_DOCKER[@]}" \
  --name "${CONTAINER_NAME}_full" \
  "$IMAGE" \
  bash -c "cd /workspace/Code && python train_task2.py --encoder rad_dino \
      --csv $TRAIN_CSV \
      --image-dir $IMAGE_DIR \
      --fold-mode leave_one_modality_out \
      --aug-strength strong \
      --domain-mixup-alpha 0.0 \
      --out-dir /workspace/$OUT_DIR \
      --epochs 15"

echo "=== TRAINING DONE -> $PROJECT_ROOT/$OUT_DIR/ ==="

# ============================================================================
# STAGE 2 -- score on internal test + all three external sets.
# ============================================================================
if [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "SKIP_EVAL=1 -- stopping after training."
  exit 0
fi

echo "############################################################"
echo "### STAGE 2 -- scoring ###"
echo "############################################################"

# dataset | csv | preprocessed lung-cropped image dir
#
# Data/test.csv is DELIBERATELY ABSENT. Its 1940 images are now training data
# (build_hardneg_train_csv.py folds them in as test_* rows, since the challenge
# has moved to the external phase and they are no longer a held-out set), so any
# F1 reported on it would be train-on-test. Do not add it back without also
# passing --no-test to the builder.
#
# The TBX11K row scores the full 8400 (train+val), of which the 3000 train sick
# are now also training data -- only the 1800 official val rows are a clean
# number. scratchpad/threshold_sweep.py can split the per-image output by the
# official list if you want val-only.
EVAL_TABLE=(
  "shenzhen|/workspace/Data/external/shenzhen.csv|/workspace/Data/external/shenzhen/preprocessed/ch0"
  "montgomery|/workspace/Data/external/montgomery.csv|/workspace/Data/external/montgomery/preprocessed/ch0"
  "tbx11k_full|/workspace/Data/external/tbx11k/tbx11k_full.csv|/workspace/Data/external/tbx11k/preprocessed_full/ch0"
)

for row in "${EVAL_TABLE[@]}"; do
  IFS="|" read -r NAME CSV IMGDIR <<< "$row"
  echo "--- scoring $NAME ---"
  # NOTE: flags are passed as a plain string inside `bash -c`, never as a bash
  # array -- an array expanded into a `bash -c` string silently drops its
  # elements (see task2_docker_bash_array_bug).
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_eval_${NAME}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python predict_task2.py \
        --ckpt-dir /workspace/$OUT_DIR \
        --csv $CSV \
        --image-dir $IMGDIR \
        --encoders rad_dino \
        --score \
        --out /workspace/submissions/submission_hardneg_${NAME}.csv"
done

echo "=== DONE ==="
echo "Compare against the no-hard-negative baseline:"
echo "  CSV/rad_dino_dgablation_strong_aug_no_mixup_eval_summary.csv"
echo "    shenzhen 0.8966 | montgomery 0.8976 | tbx11k_full 0.3172"
echo "JUDGE ON SHENZHEN AND MONTGOMERY -- the only two sets here that are still"
echo "genuinely never-trained-on. TBX11K is now partly training data (its train"
echo "split's sick images), and Data/test.csv is fully training data."
