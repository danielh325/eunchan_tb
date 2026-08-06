#!/usr/bin/env python
"""Standalone CPU-only sanity check for FoundationSegModel (segmentation_model.py)
BEFORE committing to a multi-day training job -- same philosophy as
smoke_encoders.py for the classification side, and retrain_nnunet.sbatch's
Stage 2 for the nnU-Net side.

For each encoder: instantiate the model (random-init by default, so this runs
without needing the real pretrained checkpoints present), run one forward
pass on a random tensor of the right input shape, and check the output shape
matches (B, 1, img_size, img_size) exactly. This is what would have caught,
before any GPU time was spent, bugs like: get_intermediate_layers returning
the wrong number of layers, a shape mismatch between the encoder's
layer_indices and the decoder's n_layers, or the multi-scale reassemble
schedule producing a spatial size the fusion blocks can't merge.

Run:
    python smoke_segmentation.py                    # random-init, no weights needed
    python smoke_segmentation.py --pretrained        # loads real checkpoints too
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordfused import ENCODER_FAMILIES
from segmentation_model import FoundationSegModel

# encoder names (ENCODER_FAMILIES dict keys), not SEG_ENCODER_FAMILIES' family
# names ("rad_dino"/"chexfound"/"eva_x") -- FoundationSegModel takes the former.
SEG_ENCODER_NAMES = ["rad_dino", "chexfound_vitl16", "eva_x_base"]


def check_one(encoder_name, pretrained):
    print(f"\n{'=' * 70}\n{encoder_name} (pretrained={pretrained})\n{'=' * 70}")
    cfg = ENCODER_FAMILIES[encoder_name]
    img_size = cfg["img_size"]

    print(f"building FoundationSegModel ... img_size={img_size}")
    model = FoundationSegModel(encoder_name, pretrained=pretrained)
    model.eval()

    n_layers = len(model.decoder.reassemble)
    print(f"encoder readout layers: {model.encoder.get_base_model().layer_indices} "
          f"(n_layers={n_layers})")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")
    if n_trainable == 0:
        print("!! WARNING: zero trainable params -- LoRA/decoder wrapping did not attach to anything")

    x = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        feats = model.encoder(x)
        assert isinstance(feats, list) and len(feats) == n_layers, \
            f"encoder returned {type(feats)} len={len(feats) if hasattr(feats, '__len__') else '?'}, " \
            f"expected a list of {n_layers} tensors"
        for i, f in enumerate(feats):
            print(f"  layer[{i}] features: {tuple(f.shape)}")
        out = model.decoder(feats)
        assert out.shape == (2, 1, img_size, img_size), \
            f"unexpected decoder output shape {tuple(out.shape)}, expected (2, 1, {img_size}, {img_size})"
        out_direct = model(x)
        assert torch.equal(out, out_direct), "model(x) should match decoder(encoder(x)) exactly"

    print(f"forward OK: input {tuple(x.shape)} -> output {tuple(out.shape)}")

    # A backward pass on a dummy loss catches any non-differentiable op or
    # shape mismatch that only shows up when gradients actually flow (the
    # forward-only check above wouldn't catch e.g. an in-place op breaking
    # autograd through the LoRA-wrapped encoder).
    model.train()
    x2 = torch.randn(1, 3, img_size, img_size, requires_grad=False)
    out2 = model(x2)
    loss = out2.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
    n_with_grad = len(grad_norms)
    n_trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"backward OK: {n_with_grad}/{n_trainable_params} trainable params received a gradient")
    if n_with_grad < n_trainable_params:
        print("!! WARNING: some trainable params got no gradient -- dead branch in the forward pass?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true",
                    help="load real checkpoints (needs weights/ populated); default is "
                         "random-init, which only checks shapes/wiring, not the checkpoint "
                         "loading path (smoke_encoders.py already covers that separately)")
    ap.add_argument("--encoders", nargs="+", default=SEG_ENCODER_NAMES,
                    choices=SEG_ENCODER_NAMES)
    args = ap.parse_args()

    for name in args.encoders:
        check_one(name, args.pretrained)
    print("\nAll segmentation smoke checks complete.")
