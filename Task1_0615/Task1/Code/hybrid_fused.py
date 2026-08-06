"""HybridFused-CXR — joint CNN+ViT feature-fusion model for Task 1 (TB cavity).

Motivated by Alhafdhi et al. 2026 (Diagnostics 16(7):999, "Syncretic Grad-CAM
Integrated ViT-CNN Hybrids ... for Early Thyroid Cancer Diagnosis"): the paper's
central architectural claim is that fusing a CNN's local/textural stream with a
ViT's global/relational stream *during* training (shared gradient flow, one
joint head) beats late/logit-level ensembling of independently trained models.

Every model in ordfused.py is single-stream (one encoder, either a plain CNN
*or* one foundation ViT). Task1's current best submission
(submission_valid_ensemble_v1.csv, see memory) is late fusion: separate
single-stream checkpoints, logits averaged post-hoc. This file adds the
missing piece: one model with a CNN branch and a ViT branch trained jointly,
mirroring the paper's Stream A (local CNN) / Stream B (global ViT) / learned
fusion / shared head design, adapted to this task's CORN ordinal head instead
of softmax (cavity severity is ordinal, not the paper's binary softmax).

Default pairing: densenet201 (CNN, 224px, ImageNet stats, full fine-tune) +
eva_x_base (ViT, 224px, foundation stats, frozen+LoRA) -- both 224px so a
single augmented base image feeds both branches (just re-normalized per
branch), keeping this affordable on a single V100. rad_dino/chexfound_vitl16
also work as --vit-encoder but cost much more memory (see ENCODER_FAMILIES).
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from common import (
    GRADE2ORD,
    ID_COL,
    find_file,
    preprocess_image_dual,
    sitk_read,
)
from ordfused import (
    ENCODER_FAMILIES,
    N_GRADES,
    TAB_DIM,
    TabularMLP,
    apply_lora,
    build_encoder,
    build_tabular_features,
)

DEFAULT_CNN_ENCODER = "densenet201"
DEFAULT_VIT_ENCODER = "eva_x_base"


class FusionBlock(nn.Module):
    """Learned, adaptive projection + fusion of two feature streams (paper's
    Eq. 9: f_fused = phi(W_a f_a + W_b f_b + b)), rather than a fixed
    concatenation -- lets the model calibrate how much weight each stream
    gets per-sample instead of a static 50/50 split."""

    def __init__(self, dim_a, dim_b, fuse_dim, dropout=0.2):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, fuse_dim)
        self.proj_b = nn.Linear(dim_b, fuse_dim)
        self.gate = nn.Linear(2 * fuse_dim, fuse_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(fuse_dim), nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim), nn.ReLU(inplace=True),
        )

    def forward(self, z_a, z_b):
        h_a = self.proj_a(z_a)
        h_b = self.proj_b(z_b)
        g = torch.sigmoid(self.gate(torch.cat([h_a, h_b], dim=1)))
        fused = g * h_a + (1 - g) * h_b
        return self.out(fused), g


class HybridFusedModel(nn.Module):
    """CNN branch + ViT branch, fused feature-level, one shared CORN head.

    Optionally also accepts the same confounder-gated tabular branch as
    OrdFusedModel, fused on top of the CNN+ViT representation.
    """

    def __init__(self, cnn_encoder=DEFAULT_CNN_ENCODER, vit_encoder=DEFAULT_VIT_ENCODER,
                 use_tabular=True, pretrained=True, fuse_dim=512, tab_dim=TAB_DIM,
                 tab_out=64, gate_hidden=64, dropout=0.2,
                 lora_r=16, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.use_tabular = use_tabular

        cnn_backbone, cnn_feat, cnn_family = build_encoder(cnn_encoder, pretrained)
        if cnn_family is not None:  # allow a foundation model as the "CNN" slot too
            cnn_backbone = apply_lora(cnn_backbone, cnn_encoder, r=lora_r, alpha=lora_alpha,
                                       dropout=lora_dropout)
        self.cnn_encoder = cnn_backbone
        self.cnn_feat_dim = cnn_feat

        vit_backbone, vit_feat, vit_family = build_encoder(vit_encoder, pretrained)
        if vit_family is None:
            raise ValueError(f"{vit_encoder!r} is not a registered foundation/ViT encoder "
                              f"(expected one of {sorted(ENCODER_FAMILIES)})")
        vit_backbone = apply_lora(vit_backbone, vit_encoder, r=lora_r, alpha=lora_alpha,
                                   dropout=lora_dropout)
        self.vit_encoder = vit_backbone
        self.vit_feat_dim = vit_feat

        self.fusion = FusionBlock(cnn_feat, vit_feat, fuse_dim, dropout=dropout)
        feat = fuse_dim

        if use_tabular:
            self.tab_mlp = TabularMLP(tab_dim, hidden=gate_hidden, out_dim=tab_out)
            self.img_reduce = nn.Linear(feat, gate_hidden)
            self.tab_gate = nn.Linear(gate_hidden + tab_out, 1)
            self.tab_proj = nn.Linear(tab_out, feat)

        self.head = nn.Sequential(
            nn.LayerNorm(feat), nn.Dropout(dropout), nn.Linear(feat, N_GRADES - 1))

    def forward(self, x_cnn, x_vit, tab=None, return_gate=False):
        z_cnn = self.cnn_encoder(x_cnn)
        z_vit = self.vit_encoder(x_vit)
        z, fusion_gate = self.fusion(z_cnn, z_vit)

        if self.use_tabular and tab is not None:
            z_tab = self.tab_mlp(tab)
            h_img = torch.relu(self.img_reduce(z))
            tg = torch.sigmoid(self.tab_gate(torch.cat([h_img, z_tab], dim=1)))
            z = z + tg * self.tab_proj(z_tab)
        else:
            tg = z.new_zeros(z.size(0), 1)
        logits = self.head(z)
        if return_gate:
            return logits, fusion_gate.mean(dim=1), tg.squeeze(1)
        return logits


class HybridFusedDataset(Dataset):
    """Yields (x_cnn, x_vit, tab, y_bin, y_ord) -- two co-augmented views of
    the same image (see preprocess_image_dual), the same engineered tabular
    vector as OrdFusedDataset, and both label encodings."""

    def __init__(self, df, cfg_cnn, cfg_vit, train=True):
        self.df = df.reset_index(drop=True)
        self.cfg_cnn = cfg_cnn
        self.cfg_vit = cfg_vit
        self.train = train
        self.tab = build_tabular_features(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        _id = row[ID_COL]
        ip = find_file(self.cfg_cnn.image_dir, _id, self.cfg_cnn.image_ext)
        if ip is None:
            raise FileNotFoundError(f"no image for {ID_COL}={_id} under {self.cfg_cnn.image_dir}")
        x_cnn, x_vit = preprocess_image_dual(
            sitk_read(ip), self.cfg_cnn, self.cfg_vit, augment=self.train)
        ordl = GRADE2ORD[row.cavity] if "cavity" in row else 0
        return (
            torch.from_numpy(x_cnn),
            torch.from_numpy(x_vit),
            torch.from_numpy(self.tab[i]),
            torch.tensor(1 if ordl > 0 else 0, dtype=torch.float32),
            torch.tensor(ordl, dtype=torch.long),
        )
