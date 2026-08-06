"""Custom nnU-Net v2 trainer: Focal Tversky + weighted CE loss, in place of
the default Dice+CE, per the plan's section 12.2.

Why: the default Dice+CE loss (which produced this dataset's 0.306 Dice,
fully converged over 1000 epochs) is known to under-segment small lesions --
a small cavity contributes almost nothing to a Dice loss averaged over a
mostly-background image, so gradient signal for the lesion class is weak.
Focal Tversky generalizes Dice with asymmetric false-positive/false-negative
weighting (alpha/beta) plus a focusing exponent (gamma) that sharpens the
loss on hard-to-segment pixels -- both levers specifically counteract the
small-lesion-underweighting problem (see plan section 12 sources: Abraham &
Khan 2018 "A Novel Focal Tversky Loss Function with Improved Attention
U-Net for Lesion Segmentation").

Deployment note (nnU-Net v2 custom trainer discovery): nnU-Net finds trainer
classes by recursively searching within the INSTALLED nnunetv2 package's
`nnunetv2/training/nnUNetTrainer/` directory for a class matching the name
passed via `-tr`. This file must be copied there at container start (see
sbatch/retrain_nnunet.sbatch) -- it is not picked up from an arbitrary
PYTHONPATH location. Also note: nnU-Net's internal loss-building API
(`_build_loss`, `DeepSupervisionWrapper`, `MemoryEfficientSoftDiceLoss`
import paths) has shifted slightly across nnunetv2 minor versions -- this is
written against the nnunetv2==2.5.1 API (pinned in Dockerfile_nnunet) but
was not verified by actually running it (no local nnU-Net environment
available this session). The retrain sbatch's sanity-check stage
(instantiate the trainer + build its loss on a dummy batch, before
committing to a real fold-0 training run) is what actually confirms this
works -- treat this file as "best-effort against documented conventions,
verify before trusting."
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalTverskyLoss(nn.Module):
    """Tversky index generalizes Dice with independent FP/FN weighting
    (alpha penalizes false positives, beta penalizes false negatives --
    beta > alpha biases the loss toward recall, appropriate when missing a
    cavity is worse than over-calling one); the focal exponent gamma
    sharpens the loss on pixels the model is already getting wrong, which is
    exactly the "score is dominated by easy background pixels" failure mode
    of small-lesion segmentation.

    Interface matches nnU-Net's SoftDiceLoss family: forward(net_output,
    target) where net_output is raw logits (B, C, H, W) and target is either
    (B, 1, H, W) integer labels or already one-hot (B, C, H, W).
    """

    def __init__(self, apply_nonlin=softmax_helper_dim1, batch_dice: bool = True,
                 do_bg: bool = False, smooth: float = 1e-5,
                 alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        shp_x, shp_y = x.shape, y.shape
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        if x.shape == y.shape:
            y_onehot = y
        else:
            y = y.long()
            y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
            y_onehot.scatter_(1, y, 1)

        axes = [0] + list(range(2, len(shp_x))) if self.batch_dice else list(range(2, len(shp_x)))

        tp = (x * y_onehot).sum(dim=axes)
        fp = (x * (~y_onehot)).sum(dim=axes)
        fn = ((1 - x) * y_onehot).sum(dim=axes)

        if not self.do_bg:
            # batch_dice=True: axes included dim 0, so tp/fp/fn are (C,) -- drop channel 0.
            # batch_dice=False: tp/fp/fn are (B, C) -- drop channel 0 along the last dim.
            if self.batch_dice:
                tp, fp, fn = tp[1:], fp[1:], fn[1:]
            else:
                tp, fp, fn = tp[:, 1:], fp[:, 1:], fn[:, 1:]

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky).clamp(min=1e-6) ** self.gamma
        return focal_tversky.mean()


class FocalTversky_and_CE_loss(nn.Module):
    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce: float = 1.0, weight_tversky: float = 1.0,
                 ignore_label=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.tversky = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            mask = target != self.ignore_label
            target_ce = torch.clone(target)
            target_ce[~mask] = 0
        else:
            mask = None
            target_ce = target

        tversky_loss = self.tversky(net_output, target)
        ce_loss = self.ce(net_output, target_ce[:, 0].long()) if mask is None \
            else (self.ce(net_output, target_ce[:, 0].long()) * mask[:, 0]).sum() / torch.clip(mask.sum(), min=1e-8)

        return self.weight_ce * ce_loss + self.weight_tversky * tversky_loss


class nnUNetTrainerFocalTversky(nnUNetTrainer):
    """Drop-in trainer swap: `-tr nnUNetTrainerFocalTversky`. Everything else
    (data pipeline, architecture from whatever `-p` plans file is passed,
    optimizer, LR schedule, number of epochs) is inherited unchanged from
    nnUNetTrainer, except oversample_foreground_percent (see __init__) --
    only the loss function and the foreground-sampling rate differ.

    Why bump oversample_foreground_percent: the baseline run's own
    fold_0/validation/summary.json shows 69/111 cases predicted a completely
    EMPTY mask, and of those, at least 16 have a real cavity (n_ref>0) --
    total misses, not partial-overlap failures. Mean FN (1058.7) is ~2x mean
    TP (531.4) even on cases where something WAS predicted. nnU-Net's
    default (0.33) guarantees a foreground-containing patch in only 1/3 of
    training patches -- given how sparse cavity pixels are per image (mean
    n_ref ~1590px out of ~230k, under 1%), that's likely starving gradient
    signal for the minority class. Raising it directly increases how often
    the model sees a positive example per training step, layered on top of
    (not a replacement for) the Focal Tversky loss's own recall bias."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.66

    def _build_loss(self):
        loss = FocalTversky_and_CE_loss(
            tversky_kwargs={"batch_dice": self.configuration_manager.batch_dice,
                             "smooth": 1e-5, "do_bg": False,
                             "alpha": 0.3, "beta": 0.7, "gamma": 1.33},
            ce_kwargs={},
            weight_ce=1.0, weight_tversky=1.0,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalTversky_250epochs(nnUNetTrainerFocalTversky):
    """Same loss as nnUNetTrainerFocalTversky, but a 250-epoch schedule
    instead of the 1000-epoch default -- matches nnU-Net's own convention
    for its built-in nnUNetTrainer_XXXepochs variants (rescale self.num_epochs
    in __init__; the polynomial LR decay in PolyLRScheduler reads
    self.num_epochs directly, so this properly anneals the LR to zero by
    epoch 250 instead of getting cut off mid-schedule at whatever epoch
    --time runs out).

    Why this is needed, not just a nice-to-have: observed steady-state
    training speed on this ResEnc-M configuration is ~23 min/epoch (see the
    117570 log) -- the 1000-epoch default would take ~16 days, but
    retrain_nnunet.sbatch only requests --time=3-00:00:00 (3 days), which
    covers only ~185 epochs. Since PolyLRScheduler's decay is tied to
    self.num_epochs=1000, getting SLURM-killed at epoch 185 would leave the
    LR at ~85% of its initial value -- the run would be stopped while still
    in its unstable high-LR phase, never annealing, likely landing WORSE
    than the fully-converged 0.306 baseline. A properly-scoped 250-epoch
    schedule (~96 hours at the observed rate) actually completes and
    anneals within a feasible --time budget instead.
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
