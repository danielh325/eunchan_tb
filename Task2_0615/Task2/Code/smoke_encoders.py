#!/usr/bin/env python
"""Standalone sanity check for every backbone in encoders.ENCODER_FAMILIES
(densenet121, swin_tiny, convnext_tiny, rad_dino, chexfound_vitl16) BEFORE
committing to a multi-day training job. Ported from
Task1_0615/Task1/Code/smoke_encoders.py -- same purpose (catch a
checkpoint-key or LoRA target_modules mismatch cheaply, on a dummy tensor,
instead of during a real multi-day run).

Run inside the container after TASK2_WEIGHTS_DIR points at rad_dino/chexfound
checkpoints (Task2 reuses Task1's already-downloaded weights -- see
train_task2.sbatch's -v .../Task1/weights:/weights:ro mount):

    cd /workspace/Code && python smoke_encoders.py
"""
import os

import torch

from encoders import ENCODER_FAMILIES, FOUNDATION_ENCODER_NAMES, apply_lora, build_encoder

WEIGHTS_DIR = os.environ.get(
    "TASK2_WEIGHTS_DIR",
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
        print(f"   check TASK2_WEIGHTS_DIR={WEIGHTS_DIR}")
        return
    print(f"feat_dim={feat_dim}  family={family}")

    x = torch.randn(2, 3, img_size, img_size)
    backbone.eval()
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (2, feat_dim), f"unexpected output shape {out.shape}, expected (2, {feat_dim})"
    print(f"forward OK: input {tuple(x.shape)} -> output {tuple(out.shape)}")

    if encoder_name not in FOUNDATION_ENCODER_NAMES:
        print("plain-timm family: no LoRA applied (full fine-tune), skipping LoRA check")
        return

    names = [n for n, _ in backbone.named_modules() if n]
    print(f"named_modules(): {len(names)} total")

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
