# Running this on Kaggle — exact steps

You have the git repo mounted. That gives you `Code/` and `kaggle/task2/`.
It does **not** give you the two things that are gitignored:

```
Task2_0615/Task2/Data/          # 9,697 PNGs, ~1.0 GB   <- you must upload
Task2_0615/Task2/checkpoints/   # trained weights       <- gone, retraining anyway
Task2_0615/Task2/weights/       # RAD-DINO              <- fetched from HF below
```

So there is exactly one upload to do before anything runs.

---

## Before you open Kaggle

Upload `Task2_0615/Task2/Data` as a **private** Kaggle Dataset. Raw, no packing —
it's only ~1.0 GB. Call it `mmtb-task2-data`.

> Patient imaging. Keep it private, and confirm the challenge data agreement
> permits third-party cloud storage before uploading.

## Notebook settings

- **Accelerator: GPU T4 ×2**
- **Internet: ON** (needed for pip and the RAD-DINO weights)
- Attach: your repo (however you have it mounted) + `mmtb-task2-data`

---

## Cell 1 — paths

Set these two to match what you actually see in the file browser, then run.
Everything below depends only on these.

```python
import os, subprocess, sys

# wherever the repo is mounted -- adjust to what you see under /kaggle/input
REPO = "/kaggle/input/mmtb-repo/eunchan_tb"
DATA = "/kaggle/input/mmtb-task2-data/Data"

WORK = "/kaggle/working"
KG   = f"{REPO}/kaggle/task2"
os.environ["TASK2_WEIGHTS_DIR"] = f"{WORK}/weights"
os.environ["TASK2_DATA"] = DATA
os.environ["TASK2_CODE"] = f"{REPO}/Task2_0615/Task2/Code"

assert os.path.isdir(f"{REPO}/Task2_0615/Task2/Code"), "REPO path wrong"
assert os.path.isdir(DATA), "DATA path wrong"
print("ok")
```

## Cell 2 — deps and RAD-DINO weights (~4 min)

```python
!pip -q install torchxrayvision transformers huggingface_hub peft
!python {KG}/01_setup_kaggle.py --stage deps
!python {KG}/01_setup_kaggle.py --stage weights
```

## Cell 3 — lung crop (~15 min, GPU)

The crop is worth +0.044 F1 on Shenzhen with a bootstrap CI excluding zero.
Not optional.

```python
!python {KG}/01_setup_kaggle.py --stage preprocess
```

## Cell 4 — patch a working copy of the code

`/kaggle/input` is read-only, so this copies `Code/` into `/kaggle/working`
and applies both patches (`--only-fold` for multi-GPU, then SWAD + ATFS).

```python
!python {KG}/02_launch_train.py --run baseline --help >/dev/null 2>&1  # copies+patches Code/
!python {KG}/05_patch_dg.py --code {WORK}/Code
```

## Cell 5 — SMOKE TEST (~5 min) — do not skip this

Everything in `dg_layers.py` was verified against synthetic ViTs. This is the
first time it meets real RAD-DINO weights on a real GPU. It answers: do the
hooks find the real blocks, does fp16 survive, how much extra memory, and how
long is a fold.

```python
!python {KG}/06_smoke_gpu.py --code {WORK}/Code --batch-size 8
```

Read the last three lines. **If the 2-GPU fold estimate exceeds ~10 h, drop
`--epochs` to 10 before committing** — Kaggle kills the notebook at 12 h.
If it prints anything other than `PASS`, stop and send me the output.

## Cell 6 — the control run

```python
!python {KG}/02_launch_train.py --run baseline --epochs 15
```

Runs 3 LOMO folds (XA dropped — 11 images is not a validation fold, and your
own 4-vs-3-fold table shows it costs nothing), two at a time across the T4s.

## Cell 7 — the DG run

```python
import subprocess, os
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
subprocess.run(f"python {WORK}/Code/train_task2.py --encoder rad_dino "
               f"--csv {WORK}/train_baseline.csv --image-dir {WORK}/preprocessed/train/ch0 "
               f"--out-dir {WORK}/checkpoints_dg --fold-mode leave_one_modality_out "
               f"--aug-strength strong --domain-mixup-alpha 0.0 --epochs 15 "
               f"--swad --tfs-vit --only-fold 0", shell=True, env=env, check=True)
```

Repeat for `--only-fold 1` and `2` (fold 1 on `CUDA_VISIBLE_DEVICES=1`, in
parallel). Or just run `02_launch_train.py` after editing its `base` string to
add `--swad --tfs-vit`.

## Cell 8 — score, with intervals

```python
!python {KG}/03_eval_external.py --pred-dir <your preds> --model dg --baseline baseline
```

**Ship the DG model only if it beats the control on Shenzhen AND Montgomery.**
Ahead on one of two is a coin flip on a country you can't inspect.

## Cell 9 — the threshold (no GPU, biggest verified win)

```python
!python {KG}/04_pick_threshold.py --pred-dir <your preds> --model dg \
    --prevalence 0.35 0.5 0.65 --sick-frac 0.2 0.5
```

Put the returned τ into `submissions/submission_predict_task2.py` before you
build the container. This is the only item in the whole plan that is verified
against your real cohorts.

---

## If you only have time for some of it

In this order:

1. **Cell 9** — threshold. Zero GPU, +0.02 expected / +0.05 worst-case F1, verified on real data.
2. **The container.** `SUBMISSION_CHECKLIST.md` still lists the build as open, and Kaggle *cannot* build Docker — you need the Mac. This is the most likely way to lose entirely.
3. **Cells 1–6** — hard negatives / control.
4. **Cell 7** — SWAD + ATFS.

SWAD and ATFS are the least-proven items here. They're well-founded and the code
is verified as far as simulation can take it, but their published gains are on
DomainBed natural images, not chest X-rays with a frozen medical backbone — and
your own record is four architectural additions that each won internally and
lost externally. Treat this as an experiment with a control, not as the plan.
