# Kaggle runbook (2× T4)

Everything here is self-contained: none of these scripts import `Task1/Code/common.py`,
`ordfused.py`, `segmentation_model.py`, or anything nnU-Net. That is deliberate — the
existing code hard-codes cluster paths (`/nnunet_data`, `proxy.mi2rl.co`, sbatch,
docker) and a Kaggle notebook is the wrong place to fight those. The existing code is
not deleted or modified; this is a parallel path.

## Order of operations

| # | Where | GPU? | Wall time | Script |
|---|-------|------|-----------|--------|
| 0 | your Mac | no | ~2 min (Task1), ~25 min (+Task2) | `00_pack_local.py` |
| 1 | Kaggle | T4×2 | ~10 min | `10_task1_seg_kaggle.py --stage lungmask` |
| 2 | Kaggle | T4×2 | ~1.5 h total | `10_task1_seg_kaggle.py --stage train` × 3 encoders |
| 3 | Kaggle | T4×2 | ~2 h | `20_task1_cls_kaggle.py --stage pre` then `--stage ft` |
| 4 | anywhere | no | ~3 min | `30_task1_fuse_score.py` |
| 5 | anywhere | no | ~1 min | `40_task2_threshold_analysis.py` |

Steps 4 and 5 need no GPU at all. **Step 5 needs nothing but what is already committed
in this repo** — run it first, it costs a minute and it changes what the Task 2 paper
should claim.

---

## Step 0 — pack the data (on the Mac)

```bash
cd ~/MMTB_2026
python3 eunchan_tb/kaggle/00_pack_local.py \
    --task1-data Task1_0615/Task1/Data \
    --task2-data Task2_0615/Task2/Data \
    --out ~/mmtb_kaggle_pack
```

Turns 5.0 GB of DICOM/PNG into ~840 MB (241 MB if you pass `--skip-task2`). Verified:
555/555 Task1 cases pack cleanly in 90 s.

Then upload `~/mmtb_kaggle_pack` as a **private** Kaggle Dataset. Name it
`mmtb-2026-pack` or set `MMTB_PACK` to wherever it lands.

> This is patient imaging data. Kaggle Datasets default to private; keep it that way,
> and check that the challenge's data agreement permits third-party cloud storage
> before uploading. If it does not, none of the rest of this applies and you need a
> different compute source.

## Step 1–2 — Task 1 segmentation

Notebook settings: **Accelerator = GPU T4 ×2**, **Internet = ON**.

```python
!pip -q install segmentation-models-pytorch torchxrayvision
!cp /kaggle/input/<your-code-dataset>/*.py .

!python 10_task1_seg_kaggle.py --stage lungmask
```

Check the line it prints at the end:

```
[lungmask] GT retained by crop: mean=... min=... frac<0.95: ...
```

If `frac<0.95` is more than a few percent, re-run with `--pad 0.08`. A crop that clips
cavities is worse than no crop.

```python
# two encoders at once, one per T4
!python 10_task1_seg_kaggle.py --stage train --encoder resnet34 --gpu 0 &
!python 10_task1_seg_kaggle.py --stage train --encoder timm-efficientnet-b0 --gpu 1
!python 10_task1_seg_kaggle.py --stage train --encoder se_resnext50_32x4d --gpu 0
```

Two single-GPU processes, not DDP. At 256 px with batch 16 the model is far too small
for DDP to pay for its own synchronisation; running two independent encoders keeps both
cards busy at 100 % and needs no distributed code at all.

Outputs: `segprob_<encoder>.npz` (OOF + test probability maps) in `/kaggle/working`.
**Download these** — they are small and they are what step 4 consumes.

## Step 3 — Task 1 classifier

```python
!pip -q install timm
# stage `pre`: TB/Normal pretraining on Task2's 7757 images
!python 20_task1_cls_kaggle.py --stage pre --backbone convnext_small --gpu 0 &
!python 20_task1_cls_kaggle.py --stage pre --backbone densenet201    --gpu 1

# stage `ft`: fine-tune on the 444 cavity labels
!python 20_task1_cls_kaggle.py --stage ft --backbone convnext_small --gpu 0 &
!python 20_task1_cls_kaggle.py --stage ft --backbone densenet201    --gpu 1

# the ablation that decides whether pretraining earned its keep
!python 20_task1_cls_kaggle.py --stage ft --backbone convnext_small --no-pretrain --gpu 0
```

The number to watch is OOF AUC. Today it is **0.90** across every member and every
ensemble of the six saved OOF files in `Task1/Code/oof/` — and at AUC 0.90 with 44 %
prevalence there is no accuracy left to extract by rethresholding. If Task2 pretraining
does not move OOF AUC above ~0.92, it did not work, and you should say so rather than
ship it.

## Step 4 — fuse and score under the real metric

```bash
python 30_task1_fuse_score.py \
    --seg segprob_resnet34.npz segprob_timm-efficientnet-b0.npz segprob_se_resnext50_32x4d.npz \
    --cls clsprob_convnext_small.npz clsprob_densenet201.npz \
    --write-masks out_masks/
```

Sweeps segmentation threshold × min-component-area × classifier τ × gate on/off, on the
**444 OOF cases only**, then applies the single selected config to the 111 test cases
exactly once. The 89-image dev slice the current pipeline tunes on is too small to pick
between configurations — the paper says so itself (a dev-optimal reweighting that moved
dev Dice 0.348→0.366 moved test Dice 0.2121→0.2119). 444 OOF cases is 5× the sample.

## Step 5 — Task 2 reanalysis (no GPU, no retraining)

```bash
cd Task2_0615/Task2
python3 ../../kaggle/40_task2_threshold_analysis.py --sub-dir submissions
```

Reads `prob_tb` out of the submission CSVs already committed here.

---

## Kaggle constraints worth knowing before you start

- **12 h per session, 30 h/week of GPU quota.** Everything above fits in one session with
  room to spare. Do not try to port the 1000-epoch nnU-Net run (~22 h on a V100, and a T4
  is roughly 2.5–3× slower than a V100 for this) — it cannot finish, and per
  `retrain_nnunet_1000epochs.sbatch` the 250-epoch version scored 0.2912 vs a 0.306
  baseline anyway.
- **T4 is Turing (sm_75): fp16 only, no bf16.** All four scripts use
  `torch.autocast(dtype=torch.float16)` + `GradScaler`. If you copy any of the existing
  `train_*.py` over, check its autocast dtype — a bf16 autocast silently falls back or
  errors here.
- `/kaggle/working` is 20 GB and persists as notebook output; `/kaggle/temp` is bigger
  but is discarded. Write checkpoints to `/kaggle/working`.
- Internet is per-notebook and off by default. `timm`, `smp`, and `torchxrayvision` all
  download weights on first use.
- Sessions die on browser disconnect unless you "Save & Run All (Commit)". For the
  multi-hour steps, commit rather than running interactively.
