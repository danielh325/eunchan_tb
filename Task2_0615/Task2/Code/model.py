"""TBClassifier -- plain binary classifier on top of encoders.build_encoder().

Much simpler than Task1's OrdFusedModel: no CORN ordinal head (Task2 has no
grade labels, just TB/Normal), no tabular gate (per the plan's metadata
decision -- Modality_DICOM is the shortcut the model must NOT key on, so
metadata never enters the model; age/gender fusion is a stretch, not v1).
"""
import torch
import torch.nn as nn

from domain_gen import MixStyle
from encoders import ENCODER_FAMILIES, FOUNDATION_ENCODER_NAMES, apply_lora, build_encoder


class TBClassifier(nn.Module):
    def __init__(self, encoder_name, pretrained=True, dropout=0.2,
                 lora_r=None, lora_alpha=None, lora_dropout=0.05,
                 use_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.1,
                 use_mask_attention=False,
                 use_glori_head=False, glori_n_layers=4):
        super().__init__()
        self.use_glori_head = use_glori_head
        self.glori_n_layers = glori_n_layers
        if use_glori_head and encoder_name != "chexfound_vitl16":
            raise ValueError("use_glori_head needs encoder_name='chexfound_vitl16' -- "
                              "it reads _CheXFoundWrapper.get_last_n_layers, which only "
                              "rad_dino/other encoders don't implement.")

        backbone, feat, family = build_encoder(encoder_name, pretrained)
        if use_glori_head:
            # CheXFound's OWN validated downstream-adaptation recipe (the
            # GLoRI paper): backbone stays FULLY frozen, no LoRA at all --
            # only a small multi-layer readout head trains. This is a
            # DIFFERENT axis from the LoRA rank tuning tried earlier (which
            # turned out to be based on a flawed capacity comparison, see
            # encoders.py's ENCODER_FAMILIES comment on chexfound_vitl16)
            # -- not "more adaptation everywhere", but "use the backbone's
            # own multi-depth features properly, adapt nothing".
            for p in backbone.parameters():
                p.requires_grad_(False)
        elif encoder_name in FOUNDATION_ENCODER_NAMES:  # frozen backbone + LoRA
            # None (the default) defers to this encoder's own ENCODER_FAMILIES
            # rank/alpha (currently a flat 16/16 for both rad_dino and
            # chexfound_vitl16 -- see encoders.py's comment on why a
            # size-scaled rank was tried and reverted) rather than a value
            # hardcoded here regardless of which encoder is passed in. An
            # explicit lora_r/lora_alpha argument still overrides per-call,
            # e.g. for a rank ablation sweep.
            fam_cfg = ENCODER_FAMILIES.get(encoder_name, {})
            r = lora_r if lora_r is not None else fam_cfg.get("lora_r", 16)
            alpha = lora_alpha if lora_alpha is not None else fam_cfg.get("lora_alpha", 16)
            backbone = apply_lora(backbone, encoder_name, r=r, alpha=alpha,
                                   dropout=lora_dropout)
        self.encoder = backbone
        self.encoder_family = family  # None for plain-timm (densenet121): full fine-tune
        self.feat_dim = feat

        if use_glori_head:
            # GLoRI-lite readout: concatenate the CLS token from each of the
            # last glori_n_layers blocks (a multi-depth "global" summary --
            # CheXFound's own paper found combining features across depth
            # this way outperforms reading only the final layer) plus one
            # attention-pooled "local" feature over the LAST layer's patch
            # tokens (a single learned query attending over all patches --
            # a simplified, single-target stand-in for GLoRI's disease-
            # specific multi-query cross-attention, appropriately scoped
            # down from GLoRI's original 40-disease multi-label setting to
            # our single TB/Normal binary label -- there's no obvious need
            # for multiple distinct queries when there's only one target).
            self.glori_query = nn.Parameter(torch.randn(1, 1, feat) * 0.02)
            glori_fused_dim = feat * glori_n_layers + feat
            self.head = nn.Sequential(nn.LayerNorm(glori_fused_dim), nn.Dropout(dropout),
                                       nn.Linear(glori_fused_dim, 1))
            # forward()'s standard/mask-guided branches never run for this
            # path (see forward()'s use_glori_head check at the top), but
            # these still need to exist as plain attributes -- other code
            # (e.g. is_vit_family checks) may introspect them.
            self.use_mixstyle = False
            self.use_mask_attention = False
            self.is_vit_family = False
            return  # skip the rest of __init__ below (mixstyle/mask-attn setup, standard head)

        # MixStyle (feature-level DG) only wired for the CNN path: densenet121
        # (timm) exposes real intermediate submodules to hook, matching
        # MixStyle's own design (mix *early*, low-level style statistics --
        # not late/semantic features). The two ViT backbones are wrapped
        # black-box modules (transformers' Dinov2Model / vendored CheXFound
        # ViT) whose internal block outputs aren't cleanly hookable through
        # this file without invasive changes to encoders.py -- deferred.
        # Image-level multi-level augmentation (domain_gen.py's
        # build_augmentation, wired in dataset.py) already applies to all
        # three backbones, so RAD-DINO/CheXFound aren't DG-technique-free.
        self.use_mixstyle = use_mixstyle and encoder_name == "densenet121"
        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
            # timm's DenseNet mirrors torchvision's submodule naming
            # (features.denseblock1/transition1/.../denseblock4/norm5) --
            # hook right after the first dense block, an early/low-level
            # feature map, per MixStyle's original placement.
            self.encoder.features.denseblock1.register_forward_hook(self._mixstyle_hook)

        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(dropout), nn.Linear(feat, 1))

        # Mask-guided attention: a gate on the LAST feature map (pre-pool),
        # sigmoid'd to [0,1] and multiplied elementwise into the features
        # before pooling -- trained with an explicit Dice loss against the
        # real PSPNet lung mask (preprocess.py's ch2), so the model is
        # *penalized* for attending outside the lung silhouette during
        # training, not just handed the mask as an extra input channel and
        # left to ignore it (that's the separate, already-built Cfg.mask_dir
        # path -- passive vs. this active supervision).
        #
        # Two gate shapes depending on encoder family:
        #   - CNN (densenet121/convnext_tiny/swin_tiny, family=None): a 1x1
        #     conv over the (B,C,H,W)/(B,H,W,C) spatial feature map -- one
        #     gate value per spatial location.
        #   - ViT (rad_dino/chexfound, family="rad_dino"/"chexfound"): a
        #     per-token nn.Linear(feat,1) over the (B,N,C) patch-token
        #     sequence -- mathematically the same "1x1 conv" idea, just
        #     applied to tokens-as-pixels instead of a spatial grid, since
        #     ViT patch tokens ARE already flattened. See encoders.py's
        #     _RadDinoWrapper.forward_features / _CheXFoundWrapper.forward_
        #     features (added alongside this) for the (B,N,C) token source,
        #     and gradcam_task2.gradcam_rad_dino for the same NxN grid
        #     reshape convention used below (N=1369->37x37 for rad_dino,
        #     N=1024->32x32 for chexfound).
        #
        # NOT verified by an actual forward pass in this session (no local
        # GPU/rad_dino weights) -- calling .forward_features() through the
        # peft-wrapped LoRA model relies on PeftModel's attribute-delegation
        # falling through to the underlying wrapper's custom method, which
        # matches how peft is documented to behave but hasn't been smoke-
        # tested here the way mask_guided_demo.py smoke-tested the CNN path
        # before committing to a real run -- do the same before trusting
        # this on a full training job.
        self.use_mask_attention = use_mask_attention
        self.is_vit_family = family in ("rad_dino", "chexfound")
        if self.use_mask_attention:
            if self.is_vit_family:
                self.mask_attn = nn.Linear(feat, 1)
            else:
                self.mask_attn = nn.Conv2d(feat, 1, kernel_size=1)

    def _mixstyle_hook(self, module, inp, out):
        return self.mixstyle(out) if self.training else out

    def forward(self, x, return_attn=False):
        if self.use_glori_head:
            return self._forward_glori(x)

        if not self.use_mask_attention:
            z = self.encoder(x)
            return self.head(z).squeeze(1)

        if self.is_vit_family:
            return self._forward_vit_gated(x, return_attn)
        return self._forward_cnn_gated(x, return_attn)

    def _forward_glori(self, x):
        # layers: tuple of glori_n_layers (patch_tokens, cls_token) pairs,
        # one per of the LAST glori_n_layers transformer blocks (frozen
        # backbone -- see encoders.py's _CheXFoundWrapper.get_last_n_layers).
        layers = self.encoder.get_last_n_layers(x, n=self.glori_n_layers)
        cls_tokens = torch.cat([cls for (_patch, cls) in layers], dim=1)  # (B, n*feat)

        last_patch_tokens = layers[-1][0]  # (B, N, feat) -- only the final layer's patches
        query = self.glori_query.expand(last_patch_tokens.shape[0], -1, -1)  # (B,1,feat)
        attn_scores = torch.bmm(query, last_patch_tokens.transpose(1, 2)) / (self.feat_dim ** 0.5)  # (B,1,N)
        attn_weights = attn_scores.softmax(dim=-1)
        local_feat = torch.bmm(attn_weights, last_patch_tokens).squeeze(1)  # (B, feat)

        fused = torch.cat([cls_tokens, local_feat], dim=1)  # (B, n*feat + feat)
        return self.head(fused).squeeze(1)

    def _forward_cnn_gated(self, x, return_attn):
        feat = self.encoder.forward_features(x)
        channels_last = feat.dim() == 4 and feat.shape[-1] > feat.shape[1]  # swin-style (B,H,W,C)
        feat_chw = feat.permute(0, 3, 1, 2) if channels_last else feat

        attn = torch.sigmoid(self.mask_attn(feat_chw))          # (B,1,H,W)
        gated = feat_chw * attn
        gated = gated.permute(0, 2, 3, 1) if channels_last else gated

        if hasattr(self.encoder, "forward_head"):
            pooled = self.encoder.forward_head(gated)
        else:
            pooled = self.encoder.global_pool(gated)
        logit = self.head(pooled).squeeze(1)

        if return_attn:
            return logit, attn.squeeze(1)  # (B,H,W)
        return logit

    def _forward_vit_gated(self, x, return_attn):
        tokens = self.encoder.forward_features(x)                # (B,N,C)
        attn = torch.sigmoid(self.mask_attn(tokens)).squeeze(-1)  # (B,N)
        gated = tokens * attn.unsqueeze(-1)                       # (B,N,C)
        pooled = gated.mean(dim=1)                                # (B,C) -- replaces CLS-token pooling
        logit = self.head(pooled).squeeze(1)

        if return_attn:
            n = attn.shape[-1]
            grid = int(round(n ** 0.5))
            attn_grid = attn.reshape(attn.shape[0], grid, grid)   # (B,grid,grid)
            return logit, attn_grid
        return logit


class GlobalLocalTBClassifier(nn.Module):
    """Dual-stream fusion classifier -- CheXFound's own "Global and Local
    Representations Integration" design, adapted to a binary head: one
    encoder branch sees the full, uncropped image (global context -- body
    habitus, mediastinum, overall positioning); a second, independent
    encoder branch sees the lung-cropped image (local anatomical detail,
    the same crop TBDataset/preprocess.py's --lung-crop already produces).
    Pooled features from both are concatenated and fused through a small
    MLP before the binary head.

    The two branches are separate, independently-weighted encoder
    instances (not a single shared-weight encoder run twice) -- global vs.
    local content differs enough in scale/framing that separate learned
    weights are the more faithful match to the source design, at the cost
    of roughly 2x the single-stream model's compute. Each branch can use a
    different encoder_name/family (LoRA applied independently per branch if
    either is a foundation model); no MixStyle wired here (deferred -- would
    need per-branch hooking, kept out for a first version).
    """

    def __init__(self, global_encoder_name, local_encoder_name, pretrained=True, dropout=0.2,
                 lora_r=16, lora_alpha=16, lora_dropout=0.05, fusion_hidden=256):
        super().__init__()
        self.global_encoder_name = global_encoder_name
        self.local_encoder_name = local_encoder_name

        self.global_encoder, global_feat, _ = build_encoder(global_encoder_name, pretrained)
        if global_encoder_name in FOUNDATION_ENCODER_NAMES:
            self.global_encoder = apply_lora(self.global_encoder, global_encoder_name,
                                              r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        self.local_encoder, local_feat, _ = build_encoder(local_encoder_name, pretrained)
        if local_encoder_name in FOUNDATION_ENCODER_NAMES:
            self.local_encoder = apply_lora(self.local_encoder, local_encoder_name,
                                             r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        fused_dim = global_feat + local_feat
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_hidden, 1)

    def forward(self, x_global, x_local):
        z_global = self.global_encoder(x_global)
        z_local = self.local_encoder(x_local)
        z = torch.cat([z_global, z_local], dim=1)
        z = self.fusion(z)
        return self.head(z).squeeze(1)
