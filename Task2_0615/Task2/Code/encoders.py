"""Encoder registry for Task 2 (TB/Normal binary classification).

Lifted from Task1_0615/Task1/Code/ordfused.py's ENCODER_FAMILIES/build_encoder/
apply_lora, which are already task-agnostic (they return a pooled (B, feat_dim)
embedding with no Task1-specific ordinal/CORN/tabular-gate logic attached).
Dropped `eva_x_base` (not part of Task2's original 3-backbone ensemble:
RAD-DINO + CheXFound + DenseNet-121); added `densenet121` as a plain-timm
family so all three backbones share one build_encoder() dispatch.

Extended with `swin_tiny` and `convnext_tiny` for architectural diversity
beyond the original 3 (RAD-DINO + CheXFound are both plain ViT; DenseNet-121
is a classic CNN) -- Swin's hierarchical windowed attention and ConvNeXt's
modernized-CNN design are each a genuinely different inductive bias, per a
second research pass finding Swin/ConvNeXt beating plain ViT/older-CNN
baselines in unrelated CXR-adjacent domain-shift comparisons. Both route
through timm's plain-model path (family="timm") -- same full-fine-tune, no
freeze/LoRA behavior as densenet121, just a different architecture name.
"""
import os
import sys

import timm
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_HERE, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("XFORMERS_DISABLED", "1")  # chexfound falls back to plain attention


def _stub_broken_torchaudio():
    """Outside the training Docker image (e.g. a bare login-node Python), an
    installed-but-ABI-mismatched torchaudio .so can crash the whole `import
    transformers` -- newer transformers versions unconditionally import
    torchaudio at module load time (models/dinov2 -> modeling_layers ->
    processing_utils -> modeling_utils -> loss/loss_utils -> loss_rnnt ->
    torchaudio, for an audio loss neither rad_dino nor Task2 ever touches).
    rad_dino is a plain image ViT with zero audio dependency, so if the real
    torchaudio fails to import cleanly, swap in an empty stub module instead
    of letting an unrelated, unused dependency block loading the CXR model.
    No-op (real torchaudio used as-is) when it imports fine, e.g. inside the
    training Docker image."""
    if "torchaudio" in sys.modules:
        return
    try:
        import torchaudio  # noqa: F401
    except Exception:
        import types
        sys.modules["torchaudio"] = types.ModuleType("torchaudio")

WEIGHTS_DIR = os.environ.get("TASK2_WEIGHTS_DIR", os.path.join(_HERE, "..", "weights"))

# encoder name -> family config. Any encoder name NOT in this dict falls back to
# plain timm.create_model (full fine-tune, no freeze/LoRA) -- densenet121 is
# listed explicitly anyway so its img_size/norm stats are looked up the same
# way as the two foundation models (see dataset.py).
ENCODER_FAMILIES = {
    "densenet121": dict(
        family="timm", img_size=224,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        batch_size=48, accum_steps=1,
    ),
    "swin_tiny": dict(
        family="timm", timm_name="swin_tiny_patch4_window7_224", img_size=224,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        batch_size=32, accum_steps=1,
    ),
    "convnext_tiny": dict(
        family="timm", timm_name="convnext_tiny", img_size=224,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        batch_size=32, accum_steps=1,
    ),
    "rad_dino": dict(
        family="rad_dino", img_size=518,
        mean=(0.5307, 0.5307, 0.5307), std=(0.2583, 0.2583, 0.2583),
        lora_targets=["query", "key", "value", "dense", "fc1", "fc2"],
        batch_size=4, accum_steps=4,
        # ViT-B/14 (12 layers, hidden=768, ~86M params) -- r=16 alongside
        # 6 target-module types/block is our original, unexamined default.
        lora_r=16, lora_alpha=16,
    ),
    "chexfound_vitl16": dict(
        family="chexfound", img_size=512,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        # FIXED 2026-08-05. The old list ["qkv","proj","fc1","fc2"] had TWO
        # defects, both confirmed by a diagnostic dump of peft's actually-
        # wrapped module names (job 121285) plus exact parameter-count
        # arithmetic reproducing that job's totals to the digit:
        #
        #  1. "fc1"/"fc2" matched NOTHING. This backbone is built with
        #     ffn_layer="swiglufused" (build_chexfound_backbone), i.e.
        #     SwiGLUFFNFused, whose Linears are named `w12`/`w3`
        #     (vendor/chexfound/layers/swiglu_ffn.py), not fc1/fc2. So the
        #     ENTIRE feed-forward stack -- the majority of a ViT-L's
        #     parameters -- received zero LoRA adaptation, in every
        #     chexfound checkpoint ever trained here. Proof: measured
        #     trainable = 2,387,968 = 24 layers x (qkv 65536 + attn.proj
        #     32768) + patch_embed 28672, exactly, with no FFN term.
        #  2. "proj" ALSO matched patch_embed.proj (a Conv2d reading raw
        #     pixels) because a plain name list matches by path SUFFIX --
        #     8 real lora_A/lora_B/... entries under
        #     `backbone.patch_embed.proj`. rad_dino's list never collides
        #     this way (HF names its patch conv "projection", and the list
        #     targets "dense").
        #
        # Net effect: chexfound was UNDER-adapted (2,387,968 trainable) vs
        # rad_dino (2,654,208) despite being a 3.6x larger model, while
        # simultaneously adapting the one layer we least wanted touched.
        # This also invalidates the 2026-07-31 note below: its "~2.37x MORE
        # LoRA parameters" figure was a theoretical count that ASSUMED
        # fc1/fc2 matched (24 layers of w12+w3 would indeed have added
        # 3,938,304, landing at ~2.37x rad_dino) -- it was never checked
        # against the wrapped modules, so the r=32 revert rested on a
        # premise that was false. Treat that note as historical.
        #
        # A regex STRING (not a list) makes peft match via re.fullmatch on
        # the full dotted path, so `.attn.proj` matches while
        # `.patch_embed.proj` does not, and the SwiGLU FFN is targeted by
        # its real names.
        lora_targets=r".*\.attn\.(qkv|proj)$|.*\.mlp\.(w12|w3)$",
        batch_size=2, accum_steps=8,
        # REVERTED 2026-07-31: r=32 was proposed on the theory that chexfound
        # (ViT-L/16, 24 layers, hidden=1024) was under-adapted relative to
        # rad_dino (ViT-B/14, 12 layers, hidden=768) at the same nominal
        # rank. Actually computing LoRA parameter counts disproved this --
        # chexfound's fused qkv + 2x the layers already gives it ~2.37x MORE
        # total LoRA parameters than rad_dino at the SAME r=16 (393216 vs
        # 165888 params-per-unit-rank), so it was never under-resourced in
        # absolute terms; r=32 would have made that gap ~4.74x, the wrong
        # direction. Back to a flat 16/16 (matching rad_dino, and every
        # checkpoint ever trained before this whole detour). See instead
        # `use_glori_head` in model.py -- a frozen-backbone, multi-layer-
        # feature head matching CheXFound's OWN validated downstream recipe
        # (GLoRI), which doesn't use LoRA on this encoder at all.
        lora_r=16, lora_alpha=16,
    ),
}
FOUNDATION_ENCODER_NAMES = {n for n, cfg in ENCODER_FAMILIES.items() if cfg["family"] != "timm"}


class _RadDinoWrapper(nn.Module):
    """Wraps transformers.Dinov2Model to expose a plain (B, feat_dim) forward."""

    def __init__(self, pretrained: bool):
        super().__init__()
        _stub_broken_torchaudio()  # see helper docstring: transformers pulls this in unconditionally
        from transformers import Dinov2Model, Dinov2Config
        local_dir = os.path.join(WEIGHTS_DIR, "rad_dino")
        if pretrained:
            self.backbone = Dinov2Model.from_pretrained(local_dir)
        else:
            self.backbone = Dinov2Model(Dinov2Config.from_pretrained(local_dir))
        self.feat_dim = self.backbone.config.hidden_size

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return pooled

    def forward_features(self, x):
        """Raw patch tokens (B, N, C), CLS token dropped -- for TBClassifier's
        ViT mask-guided-attention path (model.py). N=1369 (37x37 grid) at the
        default 518px/patch14 config -- same tokens gradcam_task2.gradcam_rad_dino
        reshapes for visualization; this is the training-time counterpart."""
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 1:, :]


def build_chexfound_backbone(pretrained: bool):
    from chexfound.models.vision_transformer import vit_large
    backbone = vit_large(
        img_size=512, patch_size=16, ffn_layer="swiglufused",
        block_chunks=4, num_register_tokens=4, init_values=1.0,
    )
    if pretrained:
        ckpt_path = os.path.join(WEIGHTS_DIR, "chexfound", "teacher_checkpoint.pth")
        raw = torch.load(ckpt_path, map_location="cpu")
        sd = raw.get("teacher", raw)
        marker = "backbone."
        new_sd = {}
        for k, v in sd.items():
            idx = k.find(marker)
            if idx == -1:
                continue
            new_sd[k[idx + len(marker):]] = v
        msg = backbone.load_state_dict(new_sd, strict=False)
        print(f"[chexfound] load_state_dict: {msg}", flush=True)
    return backbone


class _CheXFoundWrapper(nn.Module):
    """Wraps the vendored CheXFound (DINOv2 ViT-L/16 @ 512) backbone -- pooled
    (CLS-token) output, for classification."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.backbone = build_chexfound_backbone(pretrained)
        self.feat_dim = self.backbone.embed_dim

    def forward(self, x):
        return self.backbone(x, is_training=False)

    def forward_features(self, x):
        """Raw patch tokens (B, N, C), register tokens + CLS dropped -- for
        TBClassifier's ViT mask-guided-attention path (model.py). N=1024
        (32x32 grid) at the default 512px/patch16 config."""
        out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"]

    def get_last_n_layers(self, x, n=4):
        """CheXFound's OWN validated downstream-adaptation recipe (the
        GLoRI paper) reads a frozen backbone's LAST n transformer layers,
        not just the final pooled output -- see TBClassifier's
        use_glori_head (model.py). Returns a tuple of n (patch_tokens,
        cls_token) pairs, one per of the last n blocks, via the vendored
        model's own get_intermediate_layers (already handles this
        checkpoint's block_chunks=4 chunked-block layout internally)."""
        return self.backbone.get_intermediate_layers(
            x, n=n, reshape=False, return_class_token=True, norm=True)


def build_encoder(encoder_name: str, pretrained: bool):
    """Returns (module, feat_dim, family_or_None). family is None for plain-timm
    encoders (no freeze/LoRA applied by the caller)."""
    cfg = ENCODER_FAMILIES.get(encoder_name)
    if cfg is None or cfg["family"] == "timm":
        timm_name = cfg.get("timm_name", encoder_name) if cfg else encoder_name
        m = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return m, m.num_features, None
    family = cfg["family"]
    if family == "rad_dino":
        m = _RadDinoWrapper(pretrained)
        return m, m.feat_dim, family
    if family == "chexfound":
        m = _CheXFoundWrapper(pretrained)
        return m, m.feat_dim, family
    raise ValueError(f"unknown encoder family: {family}")


def apply_lora(backbone: nn.Module, encoder_name: str, r: int = 16, alpha: int = 16, dropout: float = 0.05):
    """Freezes `backbone` and wraps it with a LoRA adapter (peft) targeting the
    attention/MLP linear layers appropriate to this encoder's family."""
    from peft import LoraConfig, get_peft_model
    for p in backbone.parameters():
        p.requires_grad_(False)
    targets = ENCODER_FAMILIES[encoder_name]["lora_targets"]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets)
    return get_peft_model(backbone, cfg)
