#!/bin/bash
# ============================================================================
# Task 2 — domain-robust TB/Normal classifier: RAD-DINO + CheXFound (frozen
# + LoRA) + DenseNet-121 (full fine-tune), MixStyle + multi-level
# augmentation (per the user's own SSRN paper) applied throughout, evaluated
# by leave-one-modality-out CV (the real proxy for external-domain
# generalization, since Modality_DICOM is a genuine domain signal already in
# the data -- a plain stratified split is actively misleading here, see the
# baseline notebook's own 0.9923-val-acc-but-shortcut-learning diagnosis).
#
# Staged: (1) preprocessing (lung-crop, GPU segmenter pass) if not already
# done, (2) 1-epoch/1-fold smoke test across all 3 backbones, (3) full
# leave-one-modality-out training. Sequential on one GPU so nothing runs
# concurrently.
#
# PREREQUISITE: run AFTER whatever Task1 job currently holds the GPU
# finishes -- CheXFound (ViT-L @ 512x512) is the heaviest component here,
# same as it was in Task1's own foundation-model build.
#
# TODO — confirm --partition/--gres against your cluster (sinfo -o "%P %G").
# TODO — only if required: uncomment --account below.
# ============================================================================
#SBATCH --job-name=task2-train
#SBATCH --partition=V100
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.log
# #SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# NOTE: sbatch copies this script elsewhere before running it, so derive
# paths from $SLURM_SUBMIT_DIR (where you ran sbatch), not BASH_SOURCE.
PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"            # .../Task2
DOCKER_CONTEXT="$(cd "$PROJECT_ROOT/.." && pwd)"              # .../Task2_0615 (has Dockerfile_final)
TASK1_WEIGHTS="$(cd "$PROJECT_ROOT/../../Task1_0615/Task1/weights" && pwd)"  # reuse Task1's already-downloaded rad_dino/chexfound checkpoints
IMAGE="lisa-task2:latest"
CONTAINER_NAME="task2_train_${SLURM_JOB_ID}"

# swin_tiny/convnext_tiny added for architectural diversity beyond the
# original 3 (RAD-DINO+CheXFound are both plain ViT, DenseNet-121 is a
# classic CNN) -- both are cheap (224px, same compute class as
# densenet121), so adding them doesn't meaningfully change the job's total
# runtime, which is dominated by CheXFound (ViT-L @ 512px).
ENCODERS="densenet121 swin_tiny convnext_tiny rad_dino chexfound_vitl16"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.mi2rl.co:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.mi2rl.co:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.mi2rl.co}"
export no_proxy="${NO_PROXY}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "project root: $PROJECT_ROOT"
echo "task1 weights (reused, read-only): $TASK1_WEIGHTS"

if [ ! -f "$TASK1_WEIGHTS/rad_dino/config.json" ] || [ ! -f "$TASK1_WEIGHTS/chexfound/teacher_checkpoint.pth" ]; then
    echo "!! rad_dino/chexfound weights missing under $TASK1_WEIGHTS -- these" >&2
    echo "!! should already exist from Task1's foundation-model job; if not," >&2
    echo "!! run Task1_0615/Task1/sbatch/fetch_foundation_weights.sh first." >&2
    exit 1
fi
if [ ! -d "$PROJECT_ROOT/Data/train" ]; then
    echo "!! $PROJECT_ROOT/Data/train missing -- extract train.zip/test.zip on" >&2
    echo "!! the host first: unzip Data/train.zip -d Data && unzip Data/test.zip -d Data" >&2
    exit 1
fi

trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null || true' EXIT

# NOTE (lesson learned in Task1's own build): this only rebuilds when the
# "$IMAGE" tag is completely missing. If Dockerfile_final gets new layers
# added later (e.g. another pip package) after this tag already exists
# locally, this will silently reuse the STALE image instead of picking up
# the change -- Task1 hit exactly this and fixed it by bumping the tag
# (lisa-task1:foundation vs :latest) whenever Dockerfile_final changes.
# Do the same here if Dockerfile_final is edited after the first build.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found locally -- building (proxy=$HTTPS_PROXY)..."
    docker build -t "$IMAGE" -f "$DOCKER_CONTEXT/Dockerfile_final" "$DOCKER_CONTEXT"
fi

# NOTE: docker requires `docker run [OPTIONS] IMAGE [COMMAND]` -- IMAGE must
# come immediately before the command, AFTER every option including --name.
# This array is a pure OPTIONS prefix (no image, no --name) precisely so
# each call site below can insert its own --name and "$IMAGE" in the right
# position: "${RUN_DOCKER[@]}" --name "..." "$IMAGE" bash -c "...". Putting
# "$IMAGE" inside this array (as an earlier version of this script did)
# silently breaks every call site that appends --name afterward -- docker
# then treats "--name ..." as the container's COMMAND instead of an option,
# and the container's entrypoint fails with "exec: --: invalid option".
RUN_DOCKER=(docker run --rm --device nvidia.com/gpu=all --shm-size=16gb
  -v "$PROJECT_ROOT":/workspace
  -v "$TASK1_WEIGHTS":/weights:ro
  -e TASK2_WEIGHTS_DIR=/weights)

echo "############################################################"
echo "### STAGE 1/3: preprocessing (lung-crop; mask generated but unused by default) ###"
echo "############################################################"
# CLAHE/bone-suppression are NOT precomputed here -- they're cheap CPU
# filters applied on the fly by common.preprocess_image every __getitem__
# (Cfg.use_clahe/use_bone_suppression), same as Task1's pipeline. Only the
# lung segmenter's GPU forward pass is worth caching to disk. --mask-channel
# is included below since it's the same segmenter pass as --lung-crop (no
# extra cost) -- the resulting ch2 mask is available for the --mask-dir
# ablation without a second preprocessing pass, even though the default
# training run below doesn't pass --mask-dir (per the plan: ablate first,
# don't bake it in as mandatory).

for SPLIT in train test; do
  if [ -d "$PROJECT_ROOT/Data/Preprocessed/${SPLIT}_images/ch0" ]; then
    echo "skip: Data/Preprocessed/${SPLIT}_images already exists"
    continue
  fi
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_preprocess_${SPLIT}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python preprocess.py \
        --in-dir /workspace/Data/${SPLIT} \
        --out-dir /workspace/Data/Preprocessed/${SPLIT}_images \
        --lung-crop --mask-channel"
done

echo "############################################################"
echo "### STAGE 2/3: smoke test (1 epoch, stratified, all 5 backbones) ###"
echo "############################################################"

for ENC in $ENCODERS; do
  echo "--- smoke: $ENC ---"
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_smoke_${ENC}" \
    "$IMAGE" \
    bash -c "
      set -euo pipefail
      cd /workspace/Code
      python train_task2.py --encoder $ENC \
          --csv /workspace/Data/train.csv \
          --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
          --out-dir /workspace/checkpoints_smoke \
          --fold-mode stratified --n-folds 5 --epochs 1 --num-workers 2
      echo '=== SMOKE OK: $ENC ==='
    "
done
rm -rf "$PROJECT_ROOT/checkpoints_smoke"

echo "############################################################"
echo "### STAGE 3/3: full leave-one-modality-out run ($ENCODERS) ###"
echo "############################################################"

for ENC in $ENCODERS; do
  "${RUN_DOCKER[@]}" \
    --name "${CONTAINER_NAME}_full_${ENC}" \
    "$IMAGE" \
    bash -c "cd /workspace/Code && python train_task2.py --encoder $ENC \
        --csv /workspace/Data/train.csv \
        --image-dir /workspace/Data/Preprocessed/train_images/ch0 \
        --out-dir /workspace/checkpoints \
        --fold-mode leave_one_modality_out --epochs 15"
done

echo "=== TASK2 TRAIN DONE -> $PROJECT_ROOT/checkpoints/ ==="
echo "Run evaluate.py to aggregate leave-one-modality-out results, then"
echo "predict_task2.py for the ensembled submission."
