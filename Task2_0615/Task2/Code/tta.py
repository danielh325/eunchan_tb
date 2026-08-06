"""Source-free test-time adaptation (Kim et al. 2026-style: ranking-based
pseudo-labeling + selective fine-tuning of normalization/head layers only),
applied at inference time using ONLY the already-trained model + the
unlabeled target batch being scored -- no source training data, no external
data at all, so this has zero competition-rules exposure and layers on top
of any single-image checkpoint (TBClassifier, densenet121/swin_tiny/
convnext_tiny/rad_dino/chexfound_vitl16 alike).

Two techniques combined:
  1. Adapt ONLY normalization-layer affine parameters (BatchNorm/LayerNorm
     gamma/beta) + the final classification head -- freezes the rest of the
     backbone, matching Tent/SHOT-style test-time-adaptation safety (avoids
     catastrophic forgetting / overfitting to a handful of pseudo-labeled
     target samples, which a naive full-model fine-tune risks badly at
     small target-batch sizes).
  2. Rank-based pseudo-labeling: only the most confident predictions
     (top --confidence-frac by |p-0.5| distance) are used as adaptation
     targets each step -- low-confidence, more-likely-wrong predictions are
     excluded from the pseudo-label pool rather than trusted.

Not yet wired for GlobalLocalTBClassifier (dual-stream) -- its forward()
takes two tensors, and this module's adaptation loop assumes a single-input
forward. Flagged as a follow-up, not silently half-supported.
"""
import copy

import torch
import torch.nn as nn


def _adaptable_params(model):
    """Only normalization-layer affine params + the final head."""
    params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for p in module.parameters(recurse=False):
                p.requires_grad_(True)
                params.append(p)
    for p in model.head.parameters():
        p.requires_grad_(True)
        params.append(p)
    return params


def source_free_adapt(model, target_loader, device, steps: int = 5, lr: float = 1e-4,
                       confidence_frac: float = 0.3, min_batch: int = 8):
    """Returns a NEW model (deep copy of `model`) adapted over `steps` passes
    through `target_loader`'s images. `model` itself is left untouched --
    callers keep the original for a with/without-TTA comparison. Every batch
    from `target_loader` is treated as unlabeled: whatever label TBDataset
    returns alongside it is ignored (present only because TBDataset always
    returns a 3-tuple), so this is safe to run against a genuinely
    unlabeled external file, not just a labeled one used for ablation.
    """
    adapted = copy.deepcopy(model).to(device)
    for p in adapted.parameters():
        p.requires_grad_(False)
    params = _adaptable_params(adapted)
    if not params:
        return adapted  # nothing adaptable for this architecture -- no-op, not an error
    optimizer = torch.optim.Adam(params, lr=lr)

    # eval() keeps BatchNorm's running mean/var fixed (computed during real
    # training on real labeled data) -- only the affine gamma/beta adapt,
    # not the statistics themselves, which is the safer of the two common
    # TTA variants (the other, adapting running stats too via train() mode,
    # is more aggressive and more prone to collapse on small target batches).
    adapted.eval()

    for _step in range(steps):
        for x, _y, _dom in target_loader:
            x = x.to(device, non_blocking=True)
            if x.size(0) < min_batch:
                continue
            with torch.no_grad():
                probs = torch.sigmoid(adapted(x))
            confidence = (probs - 0.5).abs()
            k = max(1, int(len(probs) * confidence_frac))
            keep = torch.topk(confidence, k).indices
            pseudo_y = (probs[keep] >= 0.5).float()

            logits = adapted(x[keep])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, pseudo_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return adapted
