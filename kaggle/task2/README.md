# Task 2 on Kaggle — what to run next, and why

Task 1 is out of scope. This directory is Task 2 only.

## The short version

The paper is in good shape and most of its conclusions hold up. Re-deriving
everything from the saved per-image probabilities turns up three things:

1. **One headline claim doesn't survive a bootstrap.** "Mixup does not earn its
   place" is +0.0012 F1 on Shenzhen with a 95% CI of [−0.011, +0.013]. Neutral,
   not harmful. Reword before submission — a reviewer can run this check in a minute.
2. **The stated next direction is a dead end.** The conclusion points at the fixed
   0.5 threshold as "the most promising direction" for closing the 0.09 gap.
   Measured: every label-free recalibration method makes it *worse*, and even an
   oracle threshold is worth only +0.009 (Shenzhen) / +0.024 (Montgomery).
3. **The highest-value experiment is already designed and has never been run.**
   `sbatch/train_task2_rad_dino_hardneg.sbatch` — zero result files anywhere.
   It targets the one deficit that is large, measured, and not a threshold artifact.

## Why hard negatives, specifically

Decomposing TBX11K by negative type, for the paper's final model:

| | |
|---|---|
| FP rate on healthy negatives | 10.3% |
| FP rate on sick-but-not-TB | **76.0%** |
| AUC, TB vs healthy | 0.9807 |
| AUC, TB vs sick-but-not-TB | **0.8364** |
| AUC, healthy vs sick — using P(TB) as the score | 0.8999 |
| **max F1 on TB-vs-sick at any threshold** | **0.6012** |

That last row is the argument. Even a perfect threshold caps TB-vs-sick at 0.60,
so this is a discrimination deficit, not a calibration one. And P(TB) separates
healthy from sick-non-TB at AUC 0.90 — the model is substantially an *abnormality*
detector. The training pool is TB-vs-Normal from a single cohort with no
abnormal-but-not-TB class, so this is exactly the behaviour you'd predict.

`build_hardneg_train_csv.py` already implements the fix and its split discipline
is already correct (train-split sick images only, val untouched).

## Priority order

| | Experiment | Status | Expected value |
|---|---|---|---|
| **P0** | Hard negatives (TBX11K train-split `sick`) | designed, never run | **High** — targets a measured 0.84 AUC deficit |
| P1 | Add abnormal-non-TB negatives from *other* institutions (NIH ChestX-ray14 — `fetch_nih_cxr14.sh` already exists) | not started | High — hits the task gap and the domain gap at once |
| P2 | Bootstrap CIs on every external number | — | High, free, CPU-only |
| **P3** | IRM (`train_task2_irm_rad_dino.sbatch`) | designed, never run | **Low — recommend skipping** |
| P4 | Threshold/calibration work | — | Near zero, measured |

**On P3.** [DomainBed](https://arxiv.org/pdf/2007.01434) found no DG algorithm beats
ERM by more than one point once model selection is done fairly, and a 2026 study of
[single-domain generalization in real deployments](https://arxiv.org/html/2601.16359v1)
concludes SOTA SDG methods are "nowhere near clinically relevant performance." Your
own ablation already found the one batch-level DG method you tried was neutral at best.
IRM is the same family. It costs a full 3-fold run; spend that on P1 instead.

**On P1.** The production TB systems that generalize are trained on ~250k images
pooled from CheXpert, PadChest, NIH ChestX-ray8, Tuberculosis Portals, Shenzhen and
Montgomery. Yours is 7,757 from one cohort. Every ablation in the paper is a
regularizer fighting a data-diversity problem — which is consistent with the
paper's own finding that adding capacity or supervision never helped.

## Run order

Prereqs: upload `Task2/Data` (raw, ~1.0 GB, no packing needed) and `Task2/Code` as
Kaggle Datasets. Accelerator **GPU T4 ×2**, Internet **ON**.

```python
!python 01_setup_kaggle.py --stage deps
!python 01_setup_kaggle.py --stage weights       # RAD-DINO from HF
!python 01_setup_kaggle.py --stage preprocess    # lung crop -> ch0
!python 02_launch_train.py --run baseline        # control, ~2-4 h
```

Run the baseline even though its result is known. Without it retrained in this
environment, a hardneg number can't be attributed to hard negatives rather than to
Kaggle-vs-cluster differences.

```python
!python 01_setup_kaggle.py --stage tbx11k
!python Code/build_hardneg_train_csv.py --no-test --train-csv ... --tbx-root ...
!python 02_launch_train.py --run hardneg
```

**Use `--no-test`.** The script defaults to folding `Data/test.csv` into training,
which permanently destroys the internal benchmark and makes baseline-vs-hardneg
non-comparable. Add the test data later as a separate run if you want it, once the
hard-negative variable has been isolated.

Then score everything with intervals:

```bash
python 03_eval_external.py --pred-dir <preds> --model hardneg --baseline baseline
```

**The number to watch is `AUC TB-vs-sick`, currently 0.8364.** If hard negatives
don't move it well above that, the experiment failed and you should say so —
that would itself be a publishable result given how confidently the mechanism
predicts otherwise.

## Compute notes

- **3 folds, not 4.** The XA modality has 11 training images; a leave-one-modality-out
  fold holding out 11 images is not a validation fold. Already measured in
  `CSV/rad_dino_4fold_vs_3fold_summary.csv` — 3fold_no_xa is indistinguishable from
  4fold_full (Montgomery slightly better). Free 25% of the budget. `02_launch_train.py`
  drops XA automatically.
- **`02_launch_train.py` applies exactly one patch** to `train_task2.py`: an
  `--only-fold` selector so folds can run one-per-GPU. It changes no training
  behaviour. Everything else in `Code/` runs unmodified, so the new numbers stay
  comparable to the cluster ones.
- **Time per fold is unmeasured on a T4.** RAD-DINO is ViT-B/14 at 518px = 1369
  tokens, which is heavy, and a T4 is roughly 2.5–3× slower than the V100 these
  were trained on. Watch `train_baseline_fold0.log` for the first epoch time and
  extrapolate before committing to a 12 h session. If 3 folds × 15 epochs won't
  fit: drop to 10 epochs, or lower `img_size` to 392 in `encoders.py` (1.75× fewer
  tokens) — but if you change resolution, the baseline control must be retrained
  at the same resolution or the comparison is void.
- T4 is sm_75: fp16 only, no bf16. `train_task2.py` already uses fp16 + GradScaler.

## One caution

`build_hardneg_train_csv.py` copies TBX11K images into the training pool. Once you
train on TBX11K's train split, **TBX11K is no longer a fully external cohort** —
different split, same source. Shenzhen and Montgomery remain genuinely untouched
and become your only true external benchmarks. Say this explicitly in the paper.
