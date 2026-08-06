"""OrdFused-CXR — the actual competition model for Task 1 (TB cavity detection).

Design (grounded in this cohort's EDA, see Task1/EDA and the CSV analysis):

  CXR ──► timm encoder ─────────────► z_img (feat)
                                          │
  metadata ──► TabularMLP ──► z_tab ──────┤ confounder gate
                                          │   g = σ(w·[h_img, z_tab])  (scalar per sample)
                                          ▼
                    z = z_img + g · proj(z_tab)          (feat)
                                          │
                                          ▼
                               CORN ordinal head  (none<small<medium<large)
                                          │
                    inference: P(cavity) = P(y>0) = σ(logit_0)
                                          │
                          threshold τ*  (tuned on 0.7·Acc + 0.3·Dice)

Why this beats the plain-binary CNN baselines (resnet50d/densenet201/
tf_efficientnetv2_s, which all threw the grade labels away):

  1. CORN ordinal supervision. The first CORN conditional node is *exactly*
     the binary presence task P(y>0), so cavity detection is trained directly,
     while the small/medium/large ranking on positive cases adds auxiliary
     gradient to the shared encoder for free. More signal, same labels.

  2. Confounder-gated tabular fusion. has_HIV (OR=0.34, p=0.008) and
     type_of_resistance (chi2 p=7e-5) are real, inference-available drivers of
     whether a cavity exists; age/gender/outcome/imaging_date are flat or
     leakage and excluded. The scalar gate learns per-sample how much tabular
     signal to admit; when metadata is useless it drives g→0 and the model
     degrades gracefully to the image-only baseline (provably harmless fusion).

Reuses common.py's preprocessing / metric / fold code so results are directly
comparable to the four CNN baselines on one tested code path.
"""
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from common import (  # noqa: F401  (re-exported for train/predict convenience)
    GRADE2ORD,
    ID_COL,
    challenge_score,
    find_file,
    make_folds,
    preprocess_image,
    set_seed,
    sitk_read,
    sweep_tau,
)

# ---------------------------------------------------------------------------
# Foundation-model encoders (EVA-X, RAD-DINO, CheXFound) — in-domain CXR
# backbones per the method doc, frozen + LoRA fine-tuned. Weights are fetched
# once on the host (sbatch/fetch_foundation_weights.sh) into Task1/weights/,
# which is bind-mounted into the container at /workspace, so no network
# access is needed at container runtime.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_HERE, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("XFORMERS_DISABLED", "1")  # chexfound/eva_x fall back to plain attention

WEIGHTS_DIR = os.environ.get("ORDFUSED_WEIGHTS_DIR", os.path.join(_HERE, "..", "weights"))

# encoder name -> family config. Any encoder name NOT in this dict falls back to
# plain timm.create_model (today's behavior: full fine-tune, ImageNet 224 stats,
# no freeze/LoRA) so the already-trained CNN runs are completely unaffected.
ENCODER_FAMILIES = {
    # batch_size/accum_steps: RAD-DINO (518px, patch14 -> ~1369 tokens) and
    # CheXFound (512px ViT-L, patch16 -> ~1024 tokens) have far more attention
    # tokens than EVA-X's 224px (~196 tokens), so their activation memory is
    # much larger at fp32 -- batch 16 OOMs a 32GB V100 on RAD-DINO's attention
    # alone. Smaller physical batch + gradient accumulation keeps the
    # effective batch size at ~16 (matching the CNN/EVA-X recipe) without
    # changing training dynamics much.
    "eva_x_base": dict(
        family="eva_x", img_size=224,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        lora_targets=["q_proj", "k_proj", "v_proj", "proj", "fc1", "fc2"],
        batch_size=16, accum_steps=1,
    ),
    "rad_dino": dict(
        family="rad_dino", img_size=518,
        mean=(0.5307, 0.5307, 0.5307), std=(0.2583, 0.2583, 0.2583),
        lora_targets=["query", "key", "value", "dense", "fc1", "fc2"],
        batch_size=4, accum_steps=4,
    ),
    "chexfound_vitl16": dict(
        family="chexfound", img_size=512,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        lora_targets=["qkv", "proj", "fc1", "fc2"],
        batch_size=2, accum_steps=8,
    ),
}
FOUNDATION_ENCODER_NAMES = set(ENCODER_FAMILIES)


def _build_eva_x(pretrained: bool):
    from vendor.eva_x import eva_x_base_patch16
    ckpt = os.path.join(WEIGHTS_DIR, "eva_x", "eva_x_base_patch16_merged520k_mim.pt")
    model = eva_x_base_patch16(pretrained=ckpt if pretrained else False, num_classes=0)
    model.feat_dim = model.num_features
    return model


class _RadDinoWrapper(nn.Module):
    """Wraps transformers.Dinov2Model to expose a plain (B, feat_dim) forward,
    matching the convention of every other encoder in this file."""

    def __init__(self, pretrained: bool):
        super().__init__()
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


def build_chexfound_backbone(pretrained: bool):
    """Shared by _CheXFoundWrapper (pooled, for classification) and
    segmentation_model.py's dense-feature wrapper (for segmentation) — both
    need the identical backbone construction + checkpoint loading, just a
    different forward() over the same underlying module."""
    from chexfound.models.vision_transformer import vit_large
    backbone = vit_large(
        img_size=512, patch_size=16, ffn_layer="swiglufused",
        block_chunks=4, num_register_tokens=4, init_values=1.0,
    )
    if pretrained:
        ckpt_path = os.path.join(WEIGHTS_DIR, "chexfound", "teacher_checkpoint.pth")
        raw = torch.load(ckpt_path, map_location="cpu")
        sd = raw.get("teacher", raw)
        # Handles both "backbone.xxx" and "teacher.backbone.xxx" flat-key
        # checkpoints (exact prefix structure unconfirmed — this is the one
        # real unknown flagged in the plan; smoke_encoders.py's printed
        # missing/unexpected key report is what verifies this actually
        # worked before committing to full training).
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
    (CLS-token) output, for classification. For dense patch-token output
    (segmentation), see segmentation_model.py's wrapper, which shares
    build_chexfound_backbone() above but calls forward_features differently."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.backbone = build_chexfound_backbone(pretrained)
        self.feat_dim = self.backbone.embed_dim

    def forward(self, x):
        return self.backbone(x, is_training=False)


def build_encoder(encoder_name: str, pretrained: bool):
    """Returns (module, feat_dim, family_or_None). family is None for plain-timm
    encoders (today's behavior, no freeze/LoRA applied by the caller)."""
    cfg = ENCODER_FAMILIES.get(encoder_name)
    if cfg is None:
        m = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return m, m.num_features, None
    family = cfg["family"]
    if family == "eva_x":
        m = _build_eva_x(pretrained)
        return m, m.feat_dim, family
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

N_GRADES = len(GRADE2ORD)  # 4: none < small < medium < large  -> ranks 0..3

# ---------------------------------------------------------------------------
# Tabular feature engineering — only the columns the EDA showed carry real,
# inference-available signal for cavity *presence*. Everything else (age,
# gender, series_modality, outcome, imaging_date) is deliberately excluded:
# flat predictors add overfitting surface on 444 images, and outcome /
# imaging_date are post-hoc leakage. Works identically on train.csv and
# test.csv (both share the schema), no fitted scaler so there's nothing to
# leak across the split.
# ---------------------------------------------------------------------------

# Clinical drug-resistance severity ladder (increasing chronicity/resistance);
# matches the monotone rise in cavity rate across these categories in-cohort.
RESISTANCE_ORD = {
    "Sensitive": 0,
    "Mono DR": 1,
    "Poly DR": 2,
    "MDR non XDR": 3,
    "Pre-XDR": 4,
    "XDR": 5,
}
_RES_MAX = max(RESISTANCE_ORD.values())

TAB_FEATURES = [
    "has_HIV",          # strong: immunosuppression blunts cavitation (OR 0.34)
    "resistance_ord",   # strong: resistance severity tracks chronicity (p=7e-5)
    "como_unknown",     # missingness indicator (~46% blank/"Not specified")
    "has_diabetes",     # weak/directional; gate can suppress
    "has_covid",        # weak/directional; gate can suppress
]
TAB_DIM = len(TAB_FEATURES)


def build_tabular_features(df: pd.DataFrame) -> np.ndarray:
    """Return an (n, TAB_DIM) float32 matrix aligned row-for-row with `df`."""
    como = df.get("comorbidity", pd.Series([np.nan] * len(df))).astype("object")
    como_str = como.fillna("").astype(str)
    unknown = como.isna() | como_str.str.strip().str.lower().eq("not specified")

    res = df.get("type_of_resistance", pd.Series([""] * len(df))).astype(str)
    res_ord = res.map(RESISTANCE_ORD).fillna(0).to_numpy(dtype=np.float32) / _RES_MAX

    feat = np.stack([
        como_str.str.contains("HIV", case=False).to_numpy(dtype=np.float32),
        res_ord,
        unknown.to_numpy(dtype=np.float32),
        como_str.str.contains("Diabetes", case=False).to_numpy(dtype=np.float32),
        como_str.str.contains("COVID", case=False).to_numpy(dtype=np.float32),
    ], axis=1)
    return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset — image (via the shared preprocessing) + engineered tabular vector +
# both binary and ordinal labels.
# ---------------------------------------------------------------------------

class OrdFusedDataset(Dataset):
    def __init__(self, df, cfg, train=True):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.tab = build_tabular_features(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        _id = row[ID_COL]
        ip = find_file(self.cfg.image_dir, _id, self.cfg.image_ext)
        if ip is None:
            raise FileNotFoundError(f"no image for {ID_COL}={_id} under {self.cfg.image_dir}")
        x = preprocess_image(sitk_read(ip), self.cfg, augment=self.train)
        ordl = GRADE2ORD[row.cavity] if "cavity" in row else 0
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.tab[i]),
            torch.tensor(1 if ordl > 0 else 0, dtype=torch.float32),
            torch.tensor(ordl, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# CORN ordinal head + loss (Shi, Cao & Raschka 2021 — rank-consistent ordinal
# regression via conditional probabilities). K-1 logits; node j models the
# conditional P(y > j | y > j-1), trained only on the subset {y >= j}. The
# unconditional P(y > j) = prod_{i<=j} σ(logit_i) is monotone by construction.
# Node 0 uses the full batch, so σ(logit_0) is a direct estimate of the binary
# cavity-presence probability we ultimately threshold.
# ---------------------------------------------------------------------------

def corn_loss(logits, y_ord, node0_pos_weight=None):
    """logits: (B, K-1), y_ord: (B,) long in [0, K-1]."""
    B, km1 = logits.shape
    total = logits.new_zeros(())
    n_terms = 0
    for j in range(km1):
        subset = y_ord >= j                       # conditioned on y > j-1
        if subset.sum() == 0:
            continue
        target = (y_ord[subset] > j).float()
        lj = logits[subset, j]
        w = node0_pos_weight if (j == 0 and node0_pos_weight is not None) else None
        total = total + F.binary_cross_entropy_with_logits(lj, target, pos_weight=w)
        n_terms += 1
    return total / max(n_terms, 1)


def corn_cavity_prob(logits):
    """P(cavity) = P(y > 0) = σ(logit_0)."""
    return torch.sigmoid(logits[:, 0])


def corn_cumprobs(logits):
    """Unconditional P(y > j) for all j — monotone non-increasing in j."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)


# ---------------------------------------------------------------------------
# Tabular branch + confounder gate + full model
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, t):
        return self.net(t)


class OrdFusedModel(nn.Module):
    """timm encoder + gated tabular fusion + CORN ordinal head.

    use_tabular=False collapses to an image-only CORN model (the ablation
    baseline that isolates the fusion contribution).
    """

    def __init__(self, encoder_name, use_tabular=True, pretrained=True,
                 tab_dim=TAB_DIM, tab_out=64, gate_hidden=64, dropout=0.2,
                 lora_r=16, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.use_tabular = use_tabular
        backbone, feat, family = build_encoder(encoder_name, pretrained)
        if family is not None:  # foundation model: freeze backbone + LoRA adapters
            backbone = apply_lora(backbone, encoder_name, r=lora_r, alpha=lora_alpha,
                                   dropout=lora_dropout)
        self.encoder = backbone
        self.encoder_family = family  # None for plain-timm CNNs
        self.feat_dim = feat

        if use_tabular:
            self.tab_mlp = TabularMLP(tab_dim, hidden=gate_hidden, out_dim=tab_out)
            # gate sees a cheap reduction of the image embedding + the tabular
            # embedding, and emits a single scalar in (0,1) per sample.
            self.img_reduce = nn.Linear(feat, gate_hidden)
            self.gate = nn.Linear(gate_hidden + tab_out, 1)
            self.tab_proj = nn.Linear(tab_out, feat)  # lift z_tab into image space

        self.head = nn.Sequential(
            nn.LayerNorm(feat), nn.Dropout(dropout), nn.Linear(feat, N_GRADES - 1))

    def forward(self, x, tab=None, return_gate=False):
        z_img = self.encoder(x)
        if self.use_tabular and tab is not None:
            z_tab = self.tab_mlp(tab)
            h_img = torch.relu(self.img_reduce(z_img))
            g = torch.sigmoid(self.gate(torch.cat([h_img, z_tab], dim=1)))  # (B,1)
            z = z_img + g * self.tab_proj(z_tab)
        else:
            g = z_img.new_zeros(z_img.size(0), 1)
            z = z_img
        logits = self.head(z)
        if return_gate:
            return logits, g.squeeze(1)
        return logits
