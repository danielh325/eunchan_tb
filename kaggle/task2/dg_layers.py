"""Token-level feature stylization (TFS-ViT / ATFS-ViT) for a frozen ViT + LoRA.

Closes the gap model.py documents and defers:

    "MixStyle (feature-level DG) only wired for the CNN path ... The two ViT
     backbones are wrapped black-box modules whose internal block outputs
     aren't cleanly hookable through this file without invasive changes to
     encoders.py -- deferred."

They are hookable. `transformers.Dinov2Model` exposes `encoder.layer` as a
plain `nn.ModuleList`, and peft's wrapper preserves it under `named_modules()`,
so a forward hook on each block is enough -- no change to encoders.py at all.

WHY THIS AND NOT THE MIXUP THAT FAILED
--------------------------------------
The paper diagnosed domain-aware mixup's failure precisely:

    "the blended images themselves are anatomically impossible superpositions
     of two ribcages carrying a soft label with no test-time counterpart"

and separately noted the four Modality_DICOM values "are acquisition tags on
the same imaging type rather than genuinely distinct modalities, so the
invariance mixup enforces is nearly orthogonal to the inter-hospital shift."

Token stylization avoids both failure modes by construction:

  1. It mixes only the *normalization statistics* (per-token mean and std --
     the "style") of intermediate features, and re-applies them to the
     original normalized content. The anatomy is untouched: no superimposed
     ribcages, and no soft label, because the label is not mixed at all.
  2. It draws its mixing partner **at random from the batch**, needing no
     domain labels. So it is not betting on Modality_DICOM being a real
     domain axis -- the bet that lost last time.

Reference: Noori et al., "TFS-ViT: Token-Level Feature Stylization for Domain
Generalization", Pattern Recognition 2024 (arXiv:2303.15698). Reported +2.37pp
(TFS) / +2.64pp (attention-aware ATFS) over ERM-ViT on PACS with DeiT-Small,
and beats SDViT by 1.24pp there, at negligible compute.

Cost here: one extra mean/std per hooked block on the training path only.
Hooks are inert in eval mode, so **inference is bit-identical and free.**
"""
import re

import torch
import torch.nn as nn

# Block naming conventions. Dinov2 (RAD-DINO) uses encoder.layer.N; timm-style
# and the vendored CheXFound ViT use blocks.N. Anchored to the end of the name
# so intermediate submodules (…layer.3.attention…) never match.
_BLOCK_PATTERNS = (
    re.compile(r"encoder\.layer\.\d+$"),
    re.compile(r"(^|\.)blocks\.\d+$"),
)


def find_transformer_blocks(model):
    """Every transformer block in `model`, in depth order, robust to how many
    wrappers (peft PeftModel, _RadDinoWrapper, ...) sit on top.

    Two subtleties, both of which bit during testing:

      * CheXFound's ViT is built with `block_chunks=4`, so its module tree is
        `blocks.{chunk}.blocks.{i}` -- the naive pattern matches BOTH the outer
        chunk and the inner block, hooking the same computation twice. Only the
        innermost matches are kept, by dropping any hit that is a prefix of
        another hit.
      * Ordering by "the last number in the name" then puts `blocks.0.blocks.1`
        before `blocks.1.blocks.0`. Sorting by the tuple of ALL indices in the
        name fixes it and is correct for the flat Dinov2 case too.
    """
    hits = [(n, m) for n, m in model.named_modules()
            if any(p.search(n) for p in _BLOCK_PATTERNS)]
    if not hits:
        raise RuntimeError(
            "no transformer blocks found. Print [n for n,_ in model.named_modules()] "
            "and add the right pattern to _BLOCK_PATTERNS rather than guessing.")

    names = {n for n, _ in hits}
    hits = [(n, m) for n, m in hits
            if not any(o != n and o.startswith(n + ".") for o in names)]

    def path_index(name):
        return tuple(int(t) for t in re.findall(r"\d+", name))

    hits.sort(key=lambda kv: path_index(kv[0]))
    return hits


class TokenStylizer:
    """Stylizes token features inside a ViT during training.

    Per forward pass it samples `n_layers` of the hooked blocks at random (the
    paper's ablation: random layers beat always-early ones), and in each,
    replaces a fraction `token_frac` of tokens with style-mixed versions.

    Args:
        model:       the (possibly peft-wrapped) backbone
        n_layers:    how many blocks to stylize per forward pass
        p:           probability of stylizing at all on a given forward pass
        alpha:       Beta(alpha, alpha) for the mixing coefficient (paper: 0.1,
                     which concentrates mass near 0 and 1 -- mostly *swapping*
                     style rather than blending it)
        token_frac:  fraction of tokens replaced (paper grid: 0.1/0.3/0.5/0.8)
        attention_aware:  ATFS variant -- weight the statistics by CLS attention
                     so the style comes from where the model is actually looking
        cls_tokens:  leading non-patch tokens to protect (Dinov2 = 1 CLS;
                     CheXFound adds 4 register tokens -> 5)
        seed:        controls layer choice, token choice and partner shuffling.
                     The Beta mixing coefficient uses the GLOBAL torch RNG
                     (torch.distributions takes no generator), so for a fully
                     reproducible A/B rely on train_task2.py's own set_seed()
                     rather than this argument alone.

    Note: with `use_glori_head` the backbone is frozen with no LoRA, so nothing
    inside the encoder can adapt to the stylization and this becomes an
    expensive no-op. Only use it on the LoRA path.
    """

    def __init__(self, model, n_layers=3, p=0.5, alpha=0.1, token_frac=0.5,
                 attention_aware=True, cls_tokens=1, seed=None):
        self.blocks = find_transformer_blocks(model)
        self.n_layers = min(n_layers, len(self.blocks))
        self.p = p
        self.alpha = alpha
        self.token_frac = token_frac
        self.attention_aware = attention_aware
        self.cls_tokens = cls_tokens
        self.enabled = False          # off until the training loop turns it on
        self._active = set()
        self._handles = [m.register_forward_hook(self._make_hook(i))
                         for i, (_, m) in enumerate(self.blocks)]
        self._gen = torch.Generator().manual_seed(seed) if seed is not None else None
        print(f"[tfs-vit] hooked {len(self.blocks)} blocks "
              f"({self.blocks[0][0]} ... {self.blocks[-1][0]}), "
              f"n_layers={self.n_layers} p={p} alpha={alpha} "
              f"token_frac={token_frac} attention_aware={attention_aware}")

    # -- lifecycle ---------------------------------------------------------
    def train(self):
        self.enabled = True

    def eval(self):
        self.enabled = False

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def resample_layers(self):
        """Pick which blocks stylize on this forward pass. Call once per batch,
        before the forward -- the hooks themselves must stay stateless so they
        behave identically regardless of execution order."""
        if not self.enabled or torch.rand(1, generator=self._gen).item() > self.p:
            self._active = set()
            return
        idx = torch.randperm(len(self.blocks), generator=self._gen)[:self.n_layers]
        self._active = set(idx.tolist())

    # -- the hook ----------------------------------------------------------
    def _make_hook(self, layer_idx):
        def hook(_module, _inp, out):
            if not self.enabled or layer_idx not in self._active:
                return out
            is_tuple = isinstance(out, tuple)
            x = out[0] if is_tuple else out
            if x.dim() != 3 or x.size(0) < 2:
                return out          # need (B,N,C) and a partner to mix with
            y = self._stylize(x)
            return (y,) + out[1:] if is_tuple else y
        return hook

    def _stylize(self, x):
        """x: (B, N, C).

        Statistics are per-CHANNEL, aggregated over the token axis -- i.e.
        mu_c = (1/S) sum_s x_{c,s}, matching the paper's formulation and
        MixStyle's original design, where "style" lives in channel statistics
        and "content" in how tokens deviate from them. (Taking them per-token
        over channels instead is the natural-looking mistake: it produces a
        perturbation the residual stream almost entirely absorbs -- measured
        at ~0.07% change in the output logit, i.e. a no-op.)
        """
        B, N, C = x.shape
        k = self.cls_tokens
        orig_dtype = x.dtype
        # Statistics in fp32 regardless of the autocast dtype. Two reasons, one
        # fatal: (a) a variance over ~1400 tokens of fp16 activations can
        # overflow fp16's 65504 ceiling and silently become inf, and (b) the
        # Beta sample below is fp32, so the mixed result promotes to fp32 and
        # the masked write-back `patches[:, sel] = restyled[:, sel]` raises
        # "Index put requires the source and destination dtypes match".
        x = x.float()
        patches = x[:, k:, :]                                   # (B, P, C)
        P = patches.size(1)
        if P < 2:
            return x.to(orig_dtype)

        if self.attention_aware:
            # ATFS: weight each token's contribution to the style statistics by
            # how strongly the CLS token attends to it, so the style is taken
            # from the regions the model actually reads. Approximated by
            # CLS-to-token similarity -- which is what that attention is
            # computed from -- so this stays a pure forward hook and needs no
            # access to the block's attention internals.
            cls = x[:, 0:1, :]                                   # (B,1,C)
            w = torch.softmax((patches * cls).sum(-1) / (C ** 0.5),
                              dim=-1).unsqueeze(-1)              # (B,P,1)
            mu = (patches * w).sum(1, keepdim=True)              # (B,1,C)
            var = ((patches - mu).pow(2) * w).sum(1, keepdim=True)
        else:
            mu = patches.mean(dim=1, keepdim=True)               # (B,1,C)
            var = patches.var(dim=1, unbiased=False, keepdim=True)
        sd = var.clamp_min(0).sqrt() + 1e-6                      # (B,1,C)

        perm = torch.randperm(B, generator=self._gen).to(x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)

        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sd_mix = lam * sd + (1 - lam) * sd[perm]

        normed = (patches - mu) / sd
        restyled = normed * sd_mix + mu_mix

        # Replace only a random subset of tokens: keeps some of the original
        # style present so the block never sees a wholly synthetic input.
        n_rep = max(1, int(P * self.token_frac))
        sel = torch.randperm(P, generator=self._gen)[:n_rep].to(x.device)
        patches = patches.clone()
        patches[:, sel, :] = restyled[:, sel, :]

        return torch.cat([x[:, :k, :], patches], dim=1).to(orig_dtype)
