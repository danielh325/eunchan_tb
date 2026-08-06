#!/usr/bin/env python
"""Standalone sanity check for HybridFusedModel BEFORE committing to a
multi-fold training job -- mirrors smoke_encoders.py's role for the
single-stream foundation encoders.

Builds the model with real pretrained weights, runs one forward pass on
random CNN-view + ViT-view tensors (+ a tabular vector), checks output shape,
checks the CNN branch is fully trainable while the ViT branch's backbone is
frozen except its LoRA adapters, and reports fusion/tabular gate values are in
(0, 1).

Run inside the container (needs timm/transformers/peft installed):

    cd /workspace/Code && python smoke_hybrid_fused.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_fused import DEFAULT_CNN_ENCODER, DEFAULT_VIT_ENCODER, HybridFusedModel
from ordfused import ENCODER_FAMILIES, TAB_DIM


def main():
    cnn_encoder, vit_encoder = DEFAULT_CNN_ENCODER, DEFAULT_VIT_ENCODER
    cnn_img_size = 224
    vit_img_size = ENCODER_FAMILIES[vit_encoder]["img_size"]
    print(f"cnn={cnn_encoder} ({cnn_img_size}px)  vit={vit_encoder} ({vit_img_size}px)")

    model = HybridFusedModel(cnn_encoder, vit_encoder, use_tabular=True, pretrained=True)
    model.eval()

    n_cnn = sum(p.numel() for p in model.cnn_encoder.parameters())
    n_cnn_trainable = sum(p.numel() for p in model.cnn_encoder.parameters() if p.requires_grad)
    n_vit = sum(p.numel() for p in model.vit_encoder.parameters())
    n_vit_trainable = sum(p.numel() for p in model.vit_encoder.parameters() if p.requires_grad)
    print(f"CNN branch: {n_cnn_trainable:,}/{n_cnn:,} trainable (expect == total, full fine-tune)")
    print(f"ViT branch: {n_vit_trainable:,}/{n_vit:,} trainable (expect << total, LoRA-only)")
    assert n_cnn_trainable == n_cnn, "CNN branch should be fully trainable (no LoRA/freeze applied)"
    assert 0 < n_vit_trainable < n_vit, "ViT branch should be frozen backbone + LoRA adapters only"

    b = 2
    x_cnn = torch.randn(b, 3, cnn_img_size, cnn_img_size)
    x_vit = torch.randn(b, 3, vit_img_size, vit_img_size)
    tab = torch.randn(b, TAB_DIM)

    with torch.no_grad():
        logits, fg, tg = model(x_cnn, x_vit, tab, return_gate=True)
    print(f"logits shape: {tuple(logits.shape)} (expect ({b}, 3) -- CORN K-1 nodes)")
    assert logits.shape == (b, 3)
    assert fg.shape == (b,) and tg.shape == (b,)
    assert ((0 <= fg) & (fg <= 1)).all() and ((0 <= tg) & (tg <= 1)).all()
    print(f"fusion_gate={fg.tolist()}  tab_gate={tg.tolist()}")

    print("\nchecking --no-tabular path ...")
    model_notab = HybridFusedModel(cnn_encoder, vit_encoder, use_tabular=False, pretrained=False)
    model_notab.eval()
    with torch.no_grad():
        logits2 = model_notab(x_cnn, x_vit)
    assert logits2.shape == (b, 3)
    print("OK")

    print("\nAll checks complete.")


if __name__ == "__main__":
    main()
