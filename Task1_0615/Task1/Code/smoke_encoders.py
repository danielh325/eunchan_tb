#!/usr/bin/env python
"""Standalone sanity check for the three new foundation-model encoders
(eva_x_base, rad_dino, chexfound_vitl16) BEFORE committing to a multi-day
training job.

For each encoder: instantiate it, load its checkpoint, run one forward pass
on a random tensor of the right input shape, print the output feature dim
and load_state_dict's missing/unexpected key report, apply LoRA and print
which named modules actually got wrapped (catches a target_modules regex
that silently matches zero layers). This is what resolves the one real
unknown in the plan — CheXFound's exact checkpoint key prefix — cheaply.

Run this inside the container (needs timm/transformers/peft installed), after
sbatch/fetch_foundation_weights.sh has populated Task1/weights/:

    cd /workspace/Code && python smoke_encoders.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordfused import ENCODER_FAMILIES, apply_lora, build_encoder

WEIGHTS_DIR = os.environ.get(
    "ORDFUSED_WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"),
)


def check_one(encoder_name):
    print(f"\n{'=' * 70}\n{encoder_name}\n{'=' * 70}")
    cfg = ENCODER_FAMILIES[encoder_name]
    img_size = cfg["img_size"]

    print(f"building (pretrained=True) ... img_size={img_size}")
    try:
        backbone, feat_dim, family = build_encoder(encoder_name, pretrained=True)
    except (FileNotFoundError, OSError) as e:
        print(f"!! SKIPPED — checkpoint not found: {e}")
        print(f"   run sbatch/fetch_foundation_weights.sh first (see WEIGHTS_DIR={WEIGHTS_DIR})")
        return
    print(f"feat_dim={feat_dim}  family={family}")

    x = torch.randn(2, 3, img_size, img_size)
    backbone.eval()
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (2, feat_dim), f"unexpected output shape {out.shape}, expected (2, {feat_dim})"
    print(f"forward OK: input {tuple(x.shape)} -> output {tuple(out.shape)}")

    print("named_modules() sample (first 40):")
    names = [n for n, _ in backbone.named_modules() if n]
    for n in names[:40]:
        print(f"  {n}")
    print(f"  ... ({len(names)} total)")

    targets = cfg["lora_targets"]
    matched = [n for n in names if any(n.endswith(t) or f".{t}" in n for t in targets)]
    print(f"LoRA target_modules={targets}")
    print(f"  -> {len(matched)} module(s) matched by substring check (sanity only; "
          f"peft does its own matching internally)")
    if not matched:
        print("  !! WARNING: zero modules matched — target_modules is almost certainly "
              "wrong for this architecture, fix before running LoRA training")

    print("applying LoRA ...")
    lora_model = apply_lora(backbone, encoder_name)
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    print(f"trainable params after LoRA: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    if trainable == 0:
        print("  !! WARNING: zero trainable params — LoRA wrapping did not attach to anything")

    with torch.no_grad():
        out2 = lora_model(x)
    assert out2.shape == (2, feat_dim)
    print(f"LoRA-wrapped forward OK: output {tuple(out2.shape)}")


if __name__ == "__main__":
    for name in ENCODER_FAMILIES:
        check_one(name)
    print("\nAll checks complete.")
