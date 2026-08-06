"""Dino-U-Net-style segmentation model: a lightweight DPT-style dense-
prediction decoder on top of FROZEN dense patch features from one of our
already-integrated CXR foundation encoders (RAD-DINO or CheXFound).

Why this over training a CNN segmentation model from scratch (what nnU-Net
does): both RAD-DINO and CheXFound were pretrained via self-supervision on
hundreds of thousands of real chest X-rays (RAD-DINO's own corpus, CheXFound
on ~987K). Their frozen dense (patch-level, not just pooled) features likely
encode far richer CXR-specific texture/anatomy information than a UNet
trained from scratch on only 444 images — nnU-Net's real bottleneck may be
data volume for a from-scratch encoder, not architecture or loss function
alone. This reuses the same underlying pretrained backbones and freeze+LoRA
recipe already built for classification (`ordfused.py`), via dedicated
dense-output wrapper classes here, since `ordfused.py`'s own wrappers reduce
straight to a pooled embedding and can't expose patch-level tokens.

Reference: "Dino U-Net" (arXiv 2508.20909), "MedDINOv3" (arXiv 2509.02379) —
both attach a DPT-style decoder to frozen DINOv2-family features for medical
segmentation instead of training a segmentation CNN end-to-end.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordfused import ENCODER_FAMILIES, WEIGHTS_DIR, _build_eva_x, apply_lora, build_chexfound_backbone

# All three foundation encoders in weights/ (rad_dino, chexfound, eva_x) now
# have a dense-feature path. EVA-X needed its own wrapper (see
# _EvaXDenseWrapper) since, unlike the two DINOv2-family encoders, it has no
# built-in intermediate-layer readout API and threads a rotary position
# embedding through every block call.
SEG_ENCODER_FAMILIES = {"rad_dino", "chexfound", "eva_x"}

# Number of intermediate transformer layers read out for the multi-scale DPT
# decoder (see DPTDecoder below). 4 matches the original DPT paper's readout
# count. Evenly spaced across depth so we get an early/fine, two mid, and a
# final/semantic layer, not just the last one (the earlier "-lite" decoder's
# simplification, kept only as a documented limitation, not the behavior now).
N_READOUT_LAYERS = 4


def _square_grid(tokens):
    """(B, N, C) patch tokens -> (B, C, H, W), asserting a square grid."""
    n = tokens.shape[1]
    h = w = int(n ** 0.5)
    assert h * w == n, f"non-square patch grid ({n} tokens)"
    return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)


def _readout_layer_indices(depth: int, n: int = N_READOUT_LAYERS):
    """n evenly-spaced 0-indexed block indices out of `depth` total blocks,
    e.g. depth=12,n=4 -> [2,5,8,11]; depth=24,n=4 -> [5,11,17,23]. Always
    includes the final block (index depth-1) so the deepest/most-semantic
    features are part of the readout."""
    return [max(0, (depth * (i + 1)) // n - 1) for i in range(n)]


class _RadDinoDenseWrapper(nn.Module):
    """Same underlying transformers.Dinov2Model as ordfused.py's
    _RadDinoWrapper, but forward() returns a list of dense patch-token grids
    from N_READOUT_LAYERS intermediate layers (shallow -> deep), not the
    pooled CLS embedding -- _RadDinoWrapper's forward can't be reused here
    since it already reduces to a single pooled vector, and the old
    single-layer dense wrapper only exposed the final layer."""

    def __init__(self, pretrained: bool):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        local_dir = os.path.join(WEIGHTS_DIR, "rad_dino")
        self.backbone = Dinov2Model.from_pretrained(local_dir) if pretrained \
            else Dinov2Model(Dinov2Config.from_pretrained(local_dir))
        self.feat_dim = self.backbone.config.hidden_size
        depth = self.backbone.config.num_hidden_layers
        self.layer_indices = _readout_layer_indices(depth)

    def forward(self, x):
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        # hidden_states[0] is the pre-block embedding output; hidden_states[i+1]
        # is the output after block i -- so block index `idx` lives at [idx+1].
        hs = out.hidden_states
        return [_square_grid(hs[idx + 1][:, 1:]) for idx in self.layer_indices]  # drop CLS token


class _CheXFoundDenseWrapper(nn.Module):
    """Same underlying CheXFound ViT-L backbone as ordfused.py's
    _CheXFoundWrapper (shares build_chexfound_backbone so checkpoint-loading
    logic isn't duplicated), but forward() returns a list of dense patch-token
    grids from N_READOUT_LAYERS intermediate layers (shallow -> deep) via the
    backbone's own get_intermediate_layers -- which already correctly handles
    the block_chunks=4 chunked-block structure and strips CLS/register
    tokens, so we don't have to reimplement that here."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.backbone = build_chexfound_backbone(pretrained)
        self.feat_dim = self.backbone.embed_dim
        self.layer_indices = _readout_layer_indices(self.backbone.n_blocks)

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(
            x, n=self.layer_indices, reshape=True, norm=True)
        return list(feats)  # each (B, C, Hp, Wp), already in shallow->deep order


class _EvaXDenseWrapper(nn.Module):
    """vendor/eva_x.py's EVA_X (a timm.models.eva.Eva subclass) has no
    get_intermediate_layers-style helper and threads a rotary position
    embedding through every block call (blk(x, rope=rot_pos_embed)), so
    intermediate layers can't be read out via a generic hook the way
    RAD-DINO's output_hidden_states or CheXFound's own helper allow --
    forward() manually replicates EVA_X.forward_features()'s loop, just
    keeping the outputs at N_READOUT_LAYERS points instead of only the last."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.backbone = _build_eva_x(pretrained)
        self.feat_dim = self.backbone.feat_dim  # set by _build_eva_x (aliases num_features)
        depth = len(self.backbone.blocks)
        self.layer_indices = _readout_layer_indices(depth)
        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)

    def forward(self, x):
        b = self.backbone
        x = b.patch_embed(x)
        x, rot_pos_embed = b._pos_embed(x)
        wanted = set(self.layer_indices)
        outputs = []
        for i, blk in enumerate(b.blocks):
            x = blk(x, rope=rot_pos_embed)
            if i in wanted:
                outputs.append(_square_grid(x[:, self.num_prefix_tokens:]))
        return outputs  # shallow -> deep, since self.blocks is iterated in order


def build_dense_encoder(encoder_name: str, pretrained: bool):
    cfg = ENCODER_FAMILIES[encoder_name]
    family = cfg["family"]
    if family == "rad_dino":
        m = _RadDinoDenseWrapper(pretrained)
    elif family == "chexfound":
        m = _CheXFoundDenseWrapper(pretrained)
    elif family == "eva_x":
        m = _EvaXDenseWrapper(pretrained)
    else:
        raise ValueError(f"{encoder_name} (family={family}) has no dense-feature path -- "
                          f"use rad_dino, chexfound_vitl16, or eva_x_base")
    return m, m.feat_dim, family


class _ReassembleBlock(nn.Module):
    """DPT's "Reassemble" step for one readout layer: project to a common
    hidden dim, then resample to this layer's pyramid scale. Shallow layers
    (fine detail, less semantic) get upsampled so they dominate high
    resolution; deep layers (coarse, most semantic) get downsampled -- the
    resulting set of feature maps forms a genuine multi-resolution pyramid,
    not just N copies of the same-size grid."""

    def __init__(self, in_channels, hidden, scale_factor):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.proj(x)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x


class _RefinementBlock(nn.Module):
    """RefineNet-style top-down fusion unit: takes the running (coarser,
    already-fused) feature map and one finer pyramid level, resizes the
    running map to match, adds them, and applies a residual conv block."""

    def __init__(self, channels):
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.GELU(),
        )

    def forward(self, running, skip):
        skip = self.skip_conv(skip)
        running = F.interpolate(running, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_conv(running + skip)


class DPTDecoder(nn.Module):
    """Multi-scale DPT-style decoder: reassembles N intermediate ViT layers'
    patch-token grids into a feature pyramid (shallow->fine resolution,
    deep->coarse), then fuses top-down (RefineNet-style, coarsest first) back
    to full image resolution. This is the real multi-scale DPT recipe used by
    Dino U-Net / MedDINOv3, replacing the single-scale "DPT-lite" first cut
    (which only ever read the last transformer layer)."""

    def __init__(self, in_channels, n_layers, out_size, hidden=256):
        super().__init__()
        self.out_size = out_size
        # DPT's own reassemble schedule for a 4-layer readout: shallowest
        # layer upsampled 4x, then 2x, then unchanged, then deepest
        # downsampled 2x. Interpolated in log-space for any other n_layers.
        scale_schedule = [4.0, 2.0, 1.0, 0.5] if n_layers == 4 \
            else list(np.geomspace(4.0, 0.5, num=n_layers))
        self.reassemble = nn.ModuleList([
            _ReassembleBlock(in_channels, hidden, s) for s in scale_schedule])
        self.refine = nn.ModuleList([_RefinementBlock(hidden) for _ in range(n_layers - 1)])
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GroupNorm(8, hidden // 2), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden // 2, 1, kernel_size=1)

    def forward(self, feats):
        """feats: list of (B, C, Hp, Wp), shallow -> deep, len == n_layers
        (same order/length as the encoder wrappers' layer_indices)."""
        pyramid = [r(f) for r, f in zip(self.reassemble, feats)]
        running = pyramid[-1]  # start from the coarsest (deepest) level
        for skip, refine_block in zip(reversed(pyramid[:-1]), self.refine):
            running = refine_block(running, skip)
        x = self.final_upsample(running)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.head(x)  # (B, 1, H, W) logits


class FoundationSegModel(nn.Module):
    """encoder_name must be one of ENCODER_FAMILIES' rad_dino/chexfound_vitl16
    entries. Backbone is frozen + LoRA (reuses ordfused.py's exact recipe);
    only the LoRA adapters + decoder train."""

    def __init__(self, encoder_name, pretrained=True, lora_r=16, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        cfg = ENCODER_FAMILIES[encoder_name]
        self.img_size = cfg["img_size"]

        backbone, feat_dim, family = build_dense_encoder(encoder_name, pretrained)
        n_layers = len(backbone.layer_indices)
        self.encoder = apply_lora(backbone, encoder_name, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.decoder = DPTDecoder(in_channels=feat_dim, n_layers=n_layers, out_size=self.img_size)

    def forward(self, x):
        feats = self.encoder(x)  # list of (B, C, Hp, Wp) -- peft wraps forward transparently
        return self.decoder(feats)  # (B, 1, H, W) logits
