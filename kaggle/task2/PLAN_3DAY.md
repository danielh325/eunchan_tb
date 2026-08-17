# 3 days to a 14-country private test set

## Read this first

**≥90% F1 on all 14 countries is very unlikely, and the reason is arithmetic, not effort.**

Your model's discrimination between TB and *sick-but-not-TB* chest films is
AUC **0.8364**. On that population, the best F1 achievable at *any* threshold is
**0.6012**. If even one of the 14 countries contributes a realistic clinic
population — negatives with pneumonia, COPD, old TB scarring — no threshold and
no 3-day training run reaches 0.90 there.

Where you actually stand, by scenario:

| Country profile | Best F1 available today |
|---|---|
| ~50% prevalence, healthy negatives | **0.93** ✓ |
| ~50% prevalence, mixed negatives | 0.82 |
| ~30% prevalence, healthy negatives | **0.91** ✓ |
| ~30% prevalence, mixed negatives | 0.72 |
| ~10% prevalence, healthy negatives | 0.85 |
| ~10% prevalence, mixed negatives | 0.48 |

So the goal for the next 3 days is **maximise the realistic score and protect
against the bad scenarios**, not hit 0.90 everywhere. Plan accordingly, and if
the competition report asks for expected performance, give a range by scenario
rather than a single number you can't defend.

---

## Day 1 — the two free wins, and start the one retrain

### 1a. Fix the threshold (30 minutes, no GPU) — do this first

The container writes a **binary** `TB/Normal` column
(`submissions/submission_predict_task2.py:80`), so τ is baked in and graded.
It is currently 0.5, a value that was inherited rather than chosen.

```bash
python 04_pick_threshold.py --pred-dir submissions \
    --prevalence 0.35 0.5 0.65 --sick-frac 0.2 0.5
```

Under a plausible curated-competition profile this returns **τ ≈ 0.75–0.90**:

| | τ=0.50 | τ=0.80 |
|---|---|---|
| expected F1 across scenarios | 0.8286 | **0.8459** |
| worst-case F1 across scenarios | 0.6952 | **0.7416** |
| Montgomery (real) | 0.8976 | **0.9204** |
| Shenzhen (real) | 0.8966 | 0.8762 |

+0.02 expected and +0.05 worst-case for zero compute. That is comparable to
anything you could train in the time available. Adjust `--prevalence` and
`--sick-frac` to match whatever you know about the 14 countries — if you know
nothing, keep the spread wide and take the scenario-weighted τ, not the minimax
one (minimax drags τ to 0.95 and costs you the balanced case).

### 1b. Ask the organizers one question (10 minutes)

Whether scoring is F1 on the binary column or AUC on a score column. If there is
*any* route to being scored on ranking rather than a hard label, take it — your
AUC is 0.96–0.98 on every cohort with healthy negatives, and the entire
threshold problem disappears. This is the single highest-value email you can send.

### 1c. Launch the hard-negative retrain (starts now, runs overnight)

This is the only intervention that moves the 0.8364 ceiling, because it is the
only one that changes what the model is trained to discriminate.

```bash
python 01_setup_kaggle.py --stage deps
python 01_setup_kaggle.py --stage weights
python 01_setup_kaggle.py --stage preprocess
python 01_setup_kaggle.py --stage tbx11k

python Code/build_hardneg_train_csv.py --no-test \
    --train-csv <DATA>/train.csv --tbx-root <TBX11K_ROOT> \
    --main-image-dir <WORK>/preprocessed/train/ch0 \
    --out-csv <WORK>/train_hardneg.csv --out-image-dir <WORK>/hardneg_images

python 02_launch_train.py --run hardneg
```

**Drop `--no-test` this once.** Normally I would insist on it to keep the
internal benchmark clean — but you are 3 days out, Shenzhen and Montgomery are
your real benchmarks anyway, and `Data/test.csv` is 1,940 labelled
challenge-distribution images that are the single best source of extra TB signal
available. Take the data. Just never quote an internal F1 again afterwards.

Skip the baseline control run. It is the right experiment and you don't have
time for it; compare against the existing cluster numbers and note the caveat.

---

## Day 2 — evaluate, and test the one free inference-time lever

### 2a. Score the hardneg model with intervals

```bash
python 03_eval_external.py --pred-dir <preds> --model hardneg \
    --baseline dgablation_strong_aug_no_mixup
```

**The number that decides everything is `AUC TB-vs-sick`, currently 0.8364.**

- Moves to >0.90 → ship it, re-run `04_pick_threshold.py` on the new
  probabilities (the optimal τ will shift), rebuild the container.
- Barely moves → the hard-negative hypothesis failed. Ship the old model with
  the new threshold. Do not burn Day 3 on a second training idea.

### 2b. Test source-free test-time adaptation (inference only, ~1 h)

`Code/tta.py` implements source-free adaptation and `predict_task2.py --tta`
wires it in. **It has never been evaluated** — no results in any summary CSV.

This is unusually well-suited to your situation: the organizers run your
container *on their data*, so the container can adapt to the target distribution
at inference with no labels. For 14 unseen countries that is exactly the setting
it exists for.

```bash
python Code/predict_task2.py --tta --tta-steps 5 ...   # on shenzhen, montgomery, tbx11k
```

Rule: ship it only if it helps on **all three** cohorts. Entropy-minimisation TTA
can collapse to a single class, and on an unseen country you would never find
out. If any cohort regresses, drop it — a variance-increasing change is the last
thing you want on a blind test set.

---

## Day 3 — the container (start it in the morning, not the evening)

**This is the most likely way to lose entirely, and it has nothing to do with modelling.**

`SUBMISSION_CHECKLIST.md` still lists the image build as open:

> NEEDS RESUBMISSION now that the Dockerfile/entrypoint contract changed
> `sbatch/build_submission_image.sbatch`

That build has never been re-run against the corrected entrypoint. And **Kaggle
cannot build or test Docker images** — no daemon, no privileged containers. You
need Docker Desktop on the Mac or another machine.

Order:
1. Set the chosen τ in `submissions/submission_predict_task2.py`.
2. Copy the new checkpoints in.
3. Build locally, then smoke-test the real contract:
   `docker run -v $PWD/in:/input -v $PWD/out:/output <img>` and confirm
   `/output/prediction.csv` has columns `filename,TB/Normal` with the
   *original* filenames.
4. `docker save | gzip`, upload, send the link.

Budget the whole day. A 4 GB image save and upload on a domestic connection is
not a five-minute operation, and the smoke test has caught two real bugs before
(the `find | head` SIGPIPE and the wrong output filename).

---

## What not to do

| | Why |
|---|---|
| IRM (`train_task2_irm_rad_dino.sbatch`) | [DomainBed](https://arxiv.org/pdf/2007.01434): no DG algorithm beats ERM by >1 point under fair selection. Costs a full run you don't have. |
| Calibration / prior-correction methods | Measured: Saerens EM and rate-matching both make F1 *worse* on all three cohorts. |
| Adding a second encoder back | Bootstrap-confirmed loss: Montgomery −0.060, Shenzhen −0.018. |
| Any new architecture | Four attempts already failed, all improving internal and losing external. Three days is not enough to break that pattern. |
| Chasing the internal number | It is 0.9913 and saturated, and your own ablation shows it is anti-correlated with external. |
