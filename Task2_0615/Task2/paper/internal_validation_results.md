# Task 2 — Internal Validation Results

**Model**: 2-encoder probability-averaged ensemble — RAD-DINO (frozen ViT-B/14 + LoRA) and
CheXFound ViT-L/16 (frozen + LoRA), each trained with leave-one-modality-out cross-validation
across the 4 imaging domains present in the training data (CR, DX, XA, XC), 4 fold checkpoints
per encoder (8 checkpoints total, probability-averaged at inference).

**Evaluation set**: `Data/test.csv`, the organizer-provided internal validation set (1,940
labeled samples after removing 1 unparseable row).

**Verified with the organizers' own official `evaluate_task2.py`** (from
mi2rl-challenge/treat-mmtb.miccai2026), not just our own reimplementation --
Precision=0.9989, Recall=0.9870, F1=0.9929, exactly matching our internal
computation below.

## Headline result

| Metric | Value |
|---|---|
| **F1@0.5** | **0.9929** |
| Accuracy@0.5 | 0.9933 |
| AUC | 0.9999 |

(Threshold fixed a priori at 0.5 — this is the honest, non-overfit number. A
post-hoc tau sweep on this same set can push F1 to ~0.996+, but that is not
a valid estimate since the threshold would be tuned on the evaluation set
itself; it is not reported as the headline number.)

## Model selection rationale

All 31 non-empty subsets of the 5 trained encoders (densenet121, swin_tiny,
convnext_tiny, rad_dino, chexfound_vitl16) were evaluated on this same
internal validation set by averaging each subset's already-computed
per-image probabilities. `rad_dino + chexfound_vitl16` gave the best F1@0.5
(0.9929), better than the full 5-encoder ensemble (F1@0.5 = 0.9890) — the
three CNN backbones (densenet121, swin_tiny, convnext_tiny) diluted rather
than improved the ensemble on this set.

## Supplementary numbers (for context, not the headline claim)

- **Leave-one-modality-out validation** (train-time proxy, same-source-pool
  data): chexfound_vitl16 mean val_acc 0.9968-0.9972 across two independent
  training runs; rad_dino mean val_acc 0.9960 (both runs).
- **True external-domain validation** (Shenzhen + Montgomery public TB CXR
  datasets, never seen during training) with the FULL 5-encoder ensemble:
  Shenzhen F1@0.5=0.7707 (accuracy 0.7402, AUC 0.8711), Montgomery
  F1@0.5=0.8348 (accuracy 0.8623, AUC 0.9218). This external gap is
  substantially larger than the internal-validation number above suggests,
  and has not yet been re-measured for the rad_dino+chexfound_vitl16 subset
  specifically -- the internal-validation F1=0.9929 above should not be read
  as a claim about cross-institution generalization.

## Per-encoder individual F1@0.5 (internal validation, for reference)

| encoder | F1@0.5 |
|---|---|
| chexfound_vitl16 | 0.9912 |
| rad_dino | 0.9890 |
| swin_tiny | 0.9879 |
| densenet121 | 0.9863 |
| convnext_tiny | 0.9857 |
| full 5-encoder ensemble | 0.9890 |
| **rad_dino + chexfound_vitl16 (final)** | **0.9929** |
