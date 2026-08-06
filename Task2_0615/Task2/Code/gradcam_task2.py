"""CPU-only Grad-CAM for Task 2's trained checkpoints, across the 4 backbones
that share a plain-timm or single-ViT-tower architecture: densenet121,
swin_tiny, convnext_tiny, rad_dino. (chexfound_vitl16 not included -- its
vendored ViT wraps attention differently and is by far the slowest at 512px,
not a good fit for a "fast, no GPU" check.)

Runs a single image through one checkpoint per encoder, backprops the TB
logit (not the sigmoid probability -- standard Grad-CAM convention) to the
last spatial feature map (timm backbones) or the last-layer patch tokens
(rad_dino, reshaped to their 37x37 grid), and overlays the resulting
heatmap on the exact `ch0` crop the model was actually trained on (loaded
via common.preprocess_image/Cfg, same code path as training/predict).

No GPU needed and no training loop -- one forward+backward pass per
encoder on a single image, seconds each on CPU.

Usage:
  cd Task2_0615/Task2/Code
  python gradcam_task2.py --image-id 1193 --fold-tag CR \
      --ckpt-dir ../checkpoints --image-dir ../Data/Preprocessed/train_images/ch0 \
      --csv ../Data/train.csv --out gradcam_1193.png

  # or point --ckpt-dir/--image-dir/--csv anywhere (e.g. external Shenzhen/
  # Montgomery data) as long as the image-id + a matching ch0 crop exist.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import find_file, sitk_read, preprocess_image
from dataset import Cfg
from model import TBClassifier

ENCODERS = ["densenet121", "swin_tiny", "convnext_tiny"]
ALL_ENCODERS = ENCODERS + ["rad_dino"]  # rad_dino also valid via --encoders, just not in the default run


def load_checkpoint(ckpt_dir, encoder_name, fold_tag):
    path = os.path.join(ckpt_dir, f"{encoder_name}_{fold_tag}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def build_model(ckpt):
    # lora_r/lora_alpha fallback of 16/16 and use_glori_head fallback of
    # False match predict_task2.py's build_model -- see that file's comment
    # (strict=False below tolerates missing/unexpected KEYS, not a shape
    # mismatch on a key that IS present, so an old checkpoint reconstructed
    # with the wrong rank/head would still fail here).
    model = TBClassifier(ckpt["encoder_name"], pretrained=False,
                          use_mask_attention=ckpt.get("use_mask_attention", False),
                          use_glori_head=ckpt.get("use_glori_head", False),
                          glori_n_layers=ckpt.get("glori_n_layers", 4),
                          lora_r=ckpt.get("lora_r", 16),
                          lora_alpha=ckpt.get("lora_alpha", 16))
    # strict=False: swin_tiny checkpoints trained under a different timm
    # version can miss relative_position_index/attn_mask keys -- these are
    # deterministic buffers recomputed from window size at __init__ time, not
    # learned weights, so a version-driven persistent/non-persistent mismatch
    # here is safe to ignore. Any OTHER missing/unexpected key (an actual
    # weight/bias) still gets printed so a real mismatch isn't silently eaten.
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not (k.endswith("relative_position_index") or k.endswith("attn_mask"))]
    if real_missing or unexpected:
        print(f"    !! load_state_dict mismatch -- missing={real_missing} unexpected={unexpected}")
    model.eval()
    return model


def load_input(image_dir, image_id, cfg, image_ext=".png"):
    path = find_file(image_dir, image_id, image_ext)
    if path is None:
        raise FileNotFoundError(f"no image for id={image_id} under {image_dir}")
    raw = sitk_read(path)
    x = preprocess_image(raw, cfg)  # (3, H, W), already normalized -- see common.py
    return torch.from_numpy(x).unsqueeze(0), raw  # raw kept only for the display crop


def gradcam_timm(model, x):
    """densenet121 / swin_tiny / convnext_tiny -- all timm models exposing
    forward_features (spatial map) + a pooling step before TBClassifier's
    own head. densenet121 has no forward_head (older timm API quirk), so it
    falls back to its .global_pool module directly; swin's spatial map is
    channels-LAST (B,H,W,C) unlike densenet/convnext's (B,C,H,W) -- handled
    by summing over the correct axis rather than forcing a permute."""
    x = x.clone().requires_grad_(True)
    enc = model.encoder
    feat = enc.forward_features(x)
    feat.retain_grad()
    if hasattr(enc, "forward_head"):
        pooled = enc.forward_head(feat)
    else:
        pooled = enc.global_pool(feat)
    logit = model.head(pooled).squeeze(1)
    model.zero_grad(set_to_none=True)
    logit.backward()
    grad = feat.grad

    channels_last = feat.dim() == 4 and feat.shape[-1] > feat.shape[1]  # swin: (B,H,W,C)
    if channels_last:
        weights = grad.mean(dim=(1, 2), keepdim=True)          # (B,1,1,C)
        cam = F.relu((weights * feat).sum(dim=-1))              # (B,H,W)
    else:
        weights = grad.mean(dim=(2, 3), keepdim=True)          # (B,C,1,1)
        cam = F.relu((weights * feat).sum(dim=1))               # (B,H,W)
    return cam[0].detach().numpy(), torch.sigmoid(logit).item()


def gradcam_rad_dino(model, x):
    """rad_dino -- true, class-discriminative Grad-CAM on hidden_states[-2]
    (the second-to-last transformer block's patch-token outputs), NOT the
    CLS-attention map an earlier version of this function used.

    Backprop from the logit through TBClassifier's head hits
    Dinov2Model's pooler_output, computed from last_hidden_state[:, 0]
    (the FINAL block's CLS token) only -- so grad at last_hidden_state[:, 1:]
    (hidden_states[-1]'s patch tokens) is exactly zero (verified empirically
    on a real checkpoint/image). An earlier version of this function hooked
    exactly that dead gradient, producing an all-zero heatmap for every
    image, then was replaced with a gradient-free CLS-attention map instead
    -- but that map is class-agnostic (no backward pass at all), so it looks
    the same regardless of whether the model predicts TB or Normal and can't
    support a "the model localizes TB lesions" claim.

    Fix: hook one block earlier. hidden_states[-2] (input to the final
    self-attention block) DOES receive nonzero gradient, because the final
    block's CLS output attends to those patch tokens directly -- the chain
    rule only breaks at the pooler's CLS-only read of the very last layer,
    not before it. Same weighted-sum-of-activations formula as gradcam_timm,
    just applied to ViT patch tokens (1, N, C) instead of a conv feature map.
    """
    backbone = model.encoder.backbone
    out = backbone(pixel_values=x, output_hidden_states=True)
    feat = out.hidden_states[-2]                       # (1, 1+N, C)
    feat.retain_grad()
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0]
    logit = model.head(pooled).squeeze(1)
    model.zero_grad(set_to_none=True)
    logit.backward()

    grad = feat.grad[:, 1:]                             # (1, N, C) -- drop CLS
    patch = feat[:, 1:].detach()
    weights = grad.mean(dim=1, keepdim=True)             # (1, 1, C) -- GAP over patches
    cam = F.relu((weights * patch).sum(dim=-1))          # (1, N)
    cam = cam[0].detach().numpy()
    n = cam.shape[-1]
    grid = int(round(n ** 0.5))
    cam = cam.reshape(grid, grid)

    # DINOv2-family ViTs (RAD-DINO included, despite having no dedicated
    # register tokens) route a handful of tokens -- usually at image
    # borders -- to carry activation norms an order of magnitude above
    # ordinary patch tokens (Darcet et al. 2023, "Vision Transformers Need
    # Registers"). This isn't specific to attention weights: it's the raw
    # hidden-state activations at those tokens that are huge, so
    # hidden_states[-2]'s outlier tokens dominate weights*patch above even
    # though the gradient itself is small and roughly uniform everywhere
    # -- observed as a SOLID edge column/row (id=1193/CR: right edge; id=3/
    # XC: top edge), not just a corner point or two.
    #
    # Two clipping attempts already failed on real checkpoints/images here:
    # (1) a flat 99th-percentile clip barely moved the outliers (this grid
    # is 37x37=1369 tokens, so one full edge row/column alone is 37/1369 ~=
    # 2.7% of tokens -- more than the top 1% a 99th-percentile clip
    # removes); (2) clipping to the 99th percentile of the INTERIOR
    # (cam[1:-1, 1:-1]) still left the same solid edge stripe, because the
    # outlier band is apparently more than one ring deep -- the "interior"
    # slice still contains contaminated tokens near the border, which
    # inflates its own 99th percentile right back up (reproduced with a
    # synthetic 2-ring-deep outlier: interior-clip barely helps, 40.97 vs a
    # true signal range of ~1.0).
    #
    # Fix: median + MAD (median absolute deviation) instead of a percentile.
    # The median has a 50% breakdown point -- it stays put no matter how
    # many border tokens are outliers or how many rings deep they go, as
    # long as outliers are under half the grid (true here by a wide
    # margin). Verified on the same synthetic 2-ring case: this clips the
    # edge down to ~2x the true interior max, vs ~40x for the interior-
    # percentile attempt.
    med = np.median(cam)
    mad = np.median(np.abs(cam - med)) + 1e-8
    clip_val = med + 6 * mad
    cam = np.clip(cam, None, clip_val)
    return cam, torch.sigmoid(logit).item()


def overlay(raw_img, cam, out_size):
    """raw_img: original grayscale array (any size). cam: small heatmap grid.
    Resizes both to out_size, blends cam (JET colormap) over the grayscale."""
    base = cv2.resize(raw_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    base_rgb = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam_resized = cv2.resize(cam, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    # INTER_CUBIC overshoots near sharp edges in the tiny native CAM grid
    # (7x7 for these backbones at 224px) -- resized values can land outside
    # [0,1] (observed range on a real sample: -0.12 to 1.05). Casting that
    # straight to uint8 wraps negative values around (e.g. -30 -> ~226), which
    # JET renders as a bright, spurious red blob at every ringing dip next to
    # a real hotspot -- clip before the cast, or the heatmap shows fake
    # localized "attention" that never came from the model.
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    blended = cv2.addWeighted(base_rgb, 0.55, heat, 0.45, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--fold-tag", default="CR",
                     help="which leave-one-modality-out checkpoint to load per encoder "
                          "(the fold whose held-out modality == this tag); must match a "
                          "file named <encoder>_<fold-tag>.pth under --ckpt-dir")
    ap.add_argument("--ckpt-dir", default="../checkpoints")
    ap.add_argument("--image-dir", default="../Data/Preprocessed/train_images/ch0")
    ap.add_argument("--encoders", nargs="*", default=ENCODERS, choices=ALL_ENCODERS,
                     help="default: densenet121/swin_tiny/convnext_tiny only -- rad_dino left "
                          "out of the default run (needs transformers+peft+a working torchaudio "
                          "import; pass --encoders rad_dino explicitly once that's sorted out)")
    ap.add_argument("--out", default="gradcam_out.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    panels = []
    for enc_name in args.encoders:
        print(f"--- {enc_name} ---")
        ckpt = load_checkpoint(args.ckpt_dir, enc_name, args.fold_tag)
        model = build_model(ckpt)
        cfg = Cfg(enc_name, image_dir=args.image_dir)
        x, raw = load_input(args.image_dir, args.image_id, cfg)

        if enc_name == "rad_dino":
            cam, prob_tb = gradcam_rad_dino(model, x)
        else:
            cam, prob_tb = gradcam_timm(model, x)

        vis = overlay(raw, cam, out_size=320)
        panels.append((enc_name, prob_tb, vis))
        print(f"    P(TB) = {prob_tb:.4f}")

    save_panel_grid(panels, args.image_id, args.fold_tag, args.out)
    print(f"wrote {args.out}")


def save_panel_grid(panels, image_id, fold_tag, out_path):
    """Pure-OpenCV panel grid (no matplotlib) -- avoids matplotlib entirely,
    since a stale system-wide install (compiled against NumPy 1.x) can crash
    at import time against a newer NumPy 2.x in the same environment, while
    cv2 (already used throughout for the heatmap overlay itself) doesn't hit
    that problem here."""
    header_h, title_h, pad = 40, 34, 6
    tiles = []
    for name, prob_tb, vis in panels:
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        w = vis_bgr.shape[1]
        header = np.full((header_h, w, 3), 255, dtype=np.uint8)
        cv2.putText(header, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(header, f"P(TB)={prob_tb:.3f}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tile = np.vstack([header, vis_bgr])
        tiles.append(tile)

    sep = np.full((tiles[0].shape[0], pad, 3), 255, dtype=np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, sep, t])

    title_bar = np.full((title_h, row.shape[1], 3), 255, dtype=np.uint8)
    title = f"Grad-CAM -- image {image_id}, held-out fold {fold_tag}"
    cv2.putText(title_bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    grid = np.vstack([title_bar, row])
    cv2.imwrite(out_path, grid)


if __name__ == "__main__":
    main()
