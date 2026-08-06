"""CPU-only demo of mask-guided attention (see model.py's TBClassifier
use_mask_attention / train_task2.py's --mask-guide-dir): warm-starts from an
already-trained densenet121 checkpoint, freezes everything except the new
1x1-conv attention gate, and fine-tunes JUST that gate for a short burst
against the real PSPNet lung mask (ch2) on a small set of real images --
enough to visibly demonstrate the mechanism pulling attention into the lung
silhouette, not a real accuracy-preserving retrain (that's train_task2.py's
job, on the cluster, over the full leave-one-modality-out protocol).

Shows, per held-out demo image: original crop | baseline Grad-CAM (no gate)
| learned attention gate (vs. the real mask) | Grad-CAM computed on the
gated model. The fit images and the shown demo images are disjoint, so this
is at least a weak held-out check of the gate, not just memorization.

Usage:
  cd Task2_0615/Task2/Code
  python mask_guided_demo.py --ckpt-dir ../checkpoints --fold-tag CR \
      --image-dir ../Data/Preprocessed/train_images/ch0 \
      --mask-dir ../Data/Preprocessed/train_images/ch2 \
      --csv ../Data/train.csv --out mask_guided_demo.png
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import find_file, sitk_read
from dataset import Cfg
from gradcam_task2 import gradcam_timm, load_checkpoint, load_input, overlay
from gradcam_grid_task2 import plain_image_tile, save_grid
from model import TBClassifier


def load_mask(mask_dir, image_id, img_size, ext=".png"):
    mp = find_file(mask_dir, image_id, ext)
    if mp is None:
        raise FileNotFoundError(f"no mask for id={image_id} under {mask_dir}")
    mask = (sitk_read(mp) > 127).astype("float32")
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask)


def dice_loss(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def gradcam_gated(model, x):
    """Grad-CAM computed on a use_mask_attention=True model -- same formula
    as gradcam_task2.gradcam_timm, just reading the model's own gated
    features (post sigmoid-gate multiply) instead of the raw encoder output,
    so the heatmap reflects what the gate actually lets through to the head."""
    x = x.clone().requires_grad_(True)
    enc = model.encoder
    feat = enc.forward_features(x)
    channels_last = feat.dim() == 4 and feat.shape[-1] > feat.shape[1]
    feat_chw = feat.permute(0, 3, 1, 2) if channels_last else feat
    attn = torch.sigmoid(model.mask_attn(feat_chw))
    gated = feat_chw * attn
    gated_bp = gated.permute(0, 2, 3, 1) if channels_last else gated
    gated_bp.retain_grad()

    pooled = enc.forward_head(gated_bp) if hasattr(enc, "forward_head") else enc.global_pool(gated_bp)
    logit = model.head(pooled).squeeze(1)
    model.zero_grad(set_to_none=True)
    logit.backward()
    grad = gated_bp.grad

    if channels_last:
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * gated_bp).sum(dim=-1))
    else:
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * gated_bp).sum(dim=1))
    return cam[0].detach().numpy(), attn[0, 0].detach().numpy(), torch.sigmoid(logit).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="densenet121", choices=["densenet121", "convnext_tiny"])
    ap.add_argument("--fold-tag", default="CR")
    ap.add_argument("--ckpt-dir", default="../checkpoints")
    ap.add_argument("--image-dir", default="../Data/Preprocessed/train_images/ch0")
    ap.add_argument("--mask-dir", default="../Data/Preprocessed/train_images/ch2")
    ap.add_argument("--csv", default="../Data/train.csv")
    ap.add_argument("--n-fit", type=int, default=12, help="images used to fine-tune the gate")
    ap.add_argument("--n-show", type=int, default=5, help="held-out images shown in the output")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--mask-guide-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile-size", type=int, default=200)
    ap.add_argument("--out", default="mask_guided_demo.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.csv)
    df["new_id"] = df["new_id"].astype(str)
    df["TB/Normal"] = df["TB/Normal"].astype(str).str.strip().str.lower()
    pool = df[(df["Modality_DICOM"] == args.fold_tag) & df["TB/Normal"].isin(["tb", "normal"])]
    picked = pool.sample(min(args.n_fit + args.n_show, len(pool)), random_state=args.seed)
    fit_ids = picked["new_id"].tolist()[: args.n_fit]
    show_ids = picked["new_id"].tolist()[args.n_fit: args.n_fit + args.n_show]
    print(f"fitting gate on {len(fit_ids)} images, showing {len(show_ids)} held-out images")

    ckpt = load_checkpoint(args.ckpt_dir, args.encoder, args.fold_tag)
    cfg = Cfg(args.encoder, image_dir=args.image_dir)

    # baseline (unmodified) model, for the "before" Grad-CAM column
    baseline = TBClassifier(ckpt["encoder_name"], pretrained=False)
    baseline.load_state_dict(ckpt["state_dict"])
    baseline.eval()

    # gated model: same encoder+head weights (warm-started), fresh random
    # gate -- strict=False since mask_attn.{weight,bias} are new keys with
    # nothing to load from a checkpoint trained before this feature existed.
    gated = TBClassifier(ckpt["encoder_name"], pretrained=False, use_mask_attention=True)
    missing, unexpected = gated.load_state_dict(ckpt["state_dict"], strict=False)
    assert unexpected == [] and all(k.startswith("mask_attn") for k in missing), \
        f"unexpected load mismatch: missing={missing} unexpected={unexpected}"

    for p in gated.parameters():
        p.requires_grad = False
    for p in gated.mask_attn.parameters():
        p.requires_grad = True

    xs, masks = [], []
    for iid in fit_ids:
        x, _raw = load_input(args.image_dir, iid, cfg)
        xs.append(x)
        masks.append(load_mask(args.mask_dir, iid, cfg.img_size))
    xs = torch.cat(xs, dim=0)
    masks = torch.stack(masks, dim=0)

    optimizer = torch.optim.Adam(gated.mask_attn.parameters(), lr=args.lr)
    gated.train()
    for step in range(args.steps):
        optimizer.zero_grad()
        _logits, attn = gated(xs, return_attn=True)
        attn_up = F.interpolate(attn.unsqueeze(1), size=masks.shape[-2:],
                                 mode="bilinear", align_corners=False).squeeze(1)
        loss = dice_loss(attn_up, masks)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  dice_loss={loss.item():.4f}")
    gated.eval()

    grid_rows = []
    for iid in show_ids:
        row = df[df["new_id"] == iid].iloc[0]
        label = row["TB/Normal"]

        img_path = find_file(args.image_dir, iid, ".png")
        orig_tile = plain_image_tile(sitk_read(img_path), args.tile_size)

        x, raw = load_input(args.image_dir, iid, cfg)
        cam_before, prob_before = gradcam_timm(baseline, x)
        vis_before = overlay(raw, cam_before, out_size=args.tile_size)

        cam_after, attn_map, prob_after = gradcam_gated(gated, x)
        vis_after = overlay(raw, cam_after, out_size=args.tile_size)
        vis_attn = overlay(raw, attn_map, out_size=args.tile_size)

        cells = [
            ("original", None, orig_tile),
            ("baseline Grad-CAM", prob_before, vis_before),
            ("learned gate", prob_after, vis_attn),
            ("gated Grad-CAM", prob_after, vis_after),
        ]
        grid_rows.append(([f"{iid}  true={label}", f"modality={args.fold_tag}"], cells))
        print(f"{iid}: P(TB) before={prob_before:.3f} after={prob_after:.3f}")

    columns = ["original", "baseline Grad-CAM", "learned gate", "gated Grad-CAM"]
    save_grid(grid_rows, columns, args.out, args.tile_size)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
