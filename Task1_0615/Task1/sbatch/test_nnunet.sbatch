#!/bin/bash
# ============================================================================
# Task 1 — nnU-Net independent test run.
#
# Re-runs INFERENCE (not training) with the already-trained fold_0 checkpoint
# against the 111 test-case images, using the plain pip `nnunetv2` package —
# there is no custom nnUNet source/Dockerfile in this project despite what
# info.txt describes (checked: Code/baseline/nnUNetv2/nnUNet/ doesn't exist;
# the workspace/ dir there is unrelated leftover files from a different
# project entirely). This independently reproduces (or refutes) the 0.306
# Dice number from nnUNet_results/.../fold_0/validation/summary.json, and
# produces fresh predicted masks you can feed into evaluate.py.
#
# One GPU, ~minutes (inference only, not the 1000-epoch training run).
#
# TODO — confirm these two against your cluster (run: sinfo -o "%P %G"):
#   --partition / --gres
# TODO — only if your cluster requires it: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task1-test-nnunet
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"          # .../Task1
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"            # .../Task1_0615 (has Dockerfile_nnunet)
NNUNET_DATA_ROOT="$(cd "$PROJECT_ROOT/../../Code/baseline/nnUNetv2/nnUNet_data" && pwd)"
IMAGE="lisa-task1:nnunet"
CONTAINER_NAME="task1_test_nnunet_${SLURM_JOB_ID}"

DATASET="Dataset001_Task1"
CONFIG="2d"
FOLD=0
CHECKPOINT="checkpoint_best.pth"   # or checkpoint_final.pth
# MUST match the actual trained model's -tr/-p -- nnUNetv2_predict silently
# falls back to nnU-Net's defaults (nnUNetTrainer/nnUNetPlans, the ORIGINAL
# 0.306-Dice baseline's config) if these aren't passed explicitly, which
# would score the wrong model entirely and look like a false confirmation
# of the 0.306 baseline rather than a real test of the ResEnc+FocalTversky
# retrain. Keep in sync with retrain_nnunet.sbatch's TRAINER/RESENC_PLANS_ID.
TRAINER="nnUNetTrainerFocalTversky_250epochs"
PLANS="nnUNetResEncUNetMPlans"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "nnUNet data root: $NNUNET_DATA_ROOT"

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building it now (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_nnunet" "$DOCKER_CONTEXT"
fi

docker run --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --shm-size=8gb \
  -v "$PROJECT_ROOT":/workspace \
  -v "$NNUNET_DATA_ROOT":/nnunet_data \
  -e nnUNet_raw=/nnunet_data/nnUNet_raw \
  -e nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed \
  -e nnUNet_results=/nnunet_data/nnUNet_results \
  "$IMAGE" \
  bash -c "
    set -euo pipefail
    cd /workspace/Code

    echo '=== 0/3: install custom trainer into nnunetv2 package ==='
    # This is a FRESH container -- unlike retrain_nnunet.sbatch's own run,
    # nothing here has copied nnUNetTrainerFocalTversky.py into the installed
    # nnunetv2 package yet. Without this, nnUNetv2_predict fails with
    # 'Unable to locate trainer class ... in nnunetv2.training.nnUNetTrainer'
    # (exactly what happened in job 117666) -- same fix as
    # retrain_nnunet.sbatch's Stage 1, including the __init__.py (pkgutil-based
    # trainer discovery needs a proper package, not just a loose .py file).
    NNUNET_TRAINER_DIR=\"\$(python -c \"import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer', 'variants', 'custom'))\")\"
    mkdir -p \"\$NNUNET_TRAINER_DIR\"
    cp /workspace/Code/nnunet_custom/nnUNetTrainerFocalTversky.py \"\$NNUNET_TRAINER_DIR/\"
    touch \"\$NNUNET_TRAINER_DIR/__init__.py\"
    echo \"installed to \$NNUNET_TRAINER_DIR\"

    echo '=== 1/3: filter test-case input images ==='
    python filter_test_images.py ../Data/test.csv \
        /nnunet_data/nnUNet_raw/${DATASET}/imagesTr \
        /tmp/test_images

    echo '=== 2/3: nnUNetv2_predict (fold ${FOLD}, trainer=${TRAINER}, plans=${PLANS}, ${CHECKPOINT}) ==='
    mkdir -p /tmp/test_pred_masks
    nnUNetv2_predict \
        -i /tmp/test_images \
        -o /tmp/test_pred_masks \
        -d ${DATASET} -c ${CONFIG} -f ${FOLD} \
        -tr ${TRAINER} -p ${PLANS} \
        -chk ${CHECKPOINT}

    echo '=== 3/3: score against ground truth (real Dice, this run) ==='
    python filter_gt_masks.py ../Data/test.csv \
        /nnunet_data/nnUNet_raw/${DATASET}/labelsTr \
        /tmp/gt_masks_test

    python - <<'PYEOF'
import csv, glob, os
import numpy as np
import SimpleITK as sitk

def load_binary_mask(path):
    m = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return (np.squeeze(m) > 0).astype(np.uint8)

def dice_score(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return np.nan
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * np.logical_and(pred, gt).sum() / (ps + gs)

gt_paths = sorted(glob.glob('/tmp/gt_masks_test/*.nii.gz'))
dices = []
for gp in gt_paths:
    case_id = os.path.basename(gp).replace('.nii.gz', '')
    pp = f'/tmp/test_pred_masks/{case_id}.nii.gz'
    if not os.path.exists(pp):
        print(f'!! missing prediction for {case_id}'); continue
    dices.append(dice_score(load_binary_mask(pp), load_binary_mask(gp)))
print(f'n cases scored: {len(dices)}')
print(f'Mean Dice (this inference run): {np.nanmean(dices):.4f}')
print('(compare against nnUNet_results/.../fold_0/validation/summary.json foreground_mean.Dice = 0.3062)')
PYEOF

    echo '--- copying predicted masks out to the project tree for reuse ---'
    mkdir -p /workspace/Code/nnunet_test_masks
    cp /tmp/test_pred_masks/*.nii.gz /workspace/Code/nnunet_test_masks/
  "

echo "=== DONE. Predicted masks saved to $PROJECT_ROOT/Code/nnunet_test_masks/ ==="
echo "Use that as --pred-mask-dir with Code/evaluate.py to combine with a classifier CSV."
