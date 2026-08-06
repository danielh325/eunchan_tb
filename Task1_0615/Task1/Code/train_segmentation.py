#!/usr/bin/env python
"""Train the frozen-foundation-model segmentation decoder (segmentation_model.py)
on the SAME 444-train/111-test split everything else in this project uses, and
report real held-out Dice against the exact same test-set ground truth nnU-Net
is scored against -- directly comparable to nnUNet_results/.../summary.json's
0.306 and to whatever sbatch/retrain_nnunet.sbatch produces.

Data: Task1/Data/train/{CXR,CXR_label} for training (our own project copy,
no cross-directory dependency). Task1/Data/test/CXR_label doesn't exist in
this project (only the organizer's private test masks do) -- for evaluation
we reuse the same masks already surfaced via nnU-Net's nnUNet_raw/labelsTr
(confirmed earlier to hold ground truth for all 555 train+test images
combined), same as Code/filter_gt_masks.py already does for evaluate.py.

Held-out selection during training uses a small dev slice carved out of the
444 TRAINING images only (via common.py's make_folds, fold 0) -- this never
touches the 111 test images, which are reserved purely for the final
reported Dice number (stricter than nnU-Net's own setup, which uses the test
split itself for checkpoint selection -- see the chat's data-leakage note).

Usage:
    python train_segmentation.py --encoder rad_dino
    python train_segmentation.py --encoder chexfound_vitl16
    python train_segmentation.py --encoder eva_x_base
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ID_COL, find_file, make_folds, preprocess_image, set_seed, sitk_read
from ordfused import ENCODER_FAMILIES
from segmentation_model import FoundationSegModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))
_DEFAULT_NNUNET_LABELS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data",
    "nnUNet_raw", "Dataset001_Task1", "labelsTr"))


def load_mask(path, size, transpose=False):
    """transpose defaults to False and should stay that way -- an earlier
    version of this function set transpose=True for TEST masks (sourced from
    nnU-Net's labelsTr) based on a single noisy grid-search reading. Re-swept
    offline against job 117713's saved test_probs.pt across all 8 dihedral
    transforms for all three encoders (rad_dino/chexfound_vitl16/eva_x_base):
    identity is the best transform by a wide margin (dice 0.27-0.29) and
    transpose is one of the worst (dice ~0.05, matching the ~0.04 collapse
    seen in that job's actual TEST dice). No transform is needed -- the mask
    and image are already read via the same sitk.GetArrayFromImage row/col
    convention."""
    m = sitk.GetArrayFromImage(sitk.ReadImage(path))
    m = np.squeeze(m)
    if transpose:
        m = m.T
    m = (m > 0).astype(np.uint8)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.float32)


def _rotate_scale(x, y, angle, scale):
    """x: (3,H,W) normalized image, y: (H,W) binary mask -- applies the SAME
    rotation+scale to both. Bilinear + reflect-padding for the image (avoids
    an artificial black ring at the border); nearest + zero-padding for the
    mask (keeps it exactly binary; nothing rotated in from outside the
    original frame can be a real lesion, so zero-fill there is correct)."""
    h, w = y.shape
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    x_hwc = np.transpose(x, (1, 2, 0))
    x_hwc = cv2.warpAffine(x_hwc, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    x = np.transpose(x_hwc, (2, 0, 1))
    y = cv2.warpAffine(y, m, (w, h), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return x.astype(np.float32), y.astype(np.float32)


def _brightness_contrast(x, contrast=0.15, brightness=0.1):
    """x-only (never the mask), applied post-normalization -- small enough
    not to swamp the real signal. One shared factor across all 3 channels
    since preprocess_image duplicates a single grayscale channel into RGB."""
    c = 1.0 + np.random.uniform(-contrast, contrast)
    b = np.random.uniform(-brightness, brightness)
    return (x * c + b).astype(np.float32)


def compute_pos_weight(df, mask_dir, img_size, max_weight=100.0):
    """Foreground:background pixel ratio across the training set's own masks
    -- used as BCE's pos_weight. Cavity pixels are under 1% of the image
    (measured on this exact dataset via nnU-Net's own summary.json: mean
    n_ref ~1590px out of ~230k), so unweighted BCE is dominated by the easy
    background class; this fixes that directly rather than relying on Focal
    Tversky alone to carry the whole imbalance-correction burden. Clamped to
    max_weight to avoid an unstable loss if a fold happens to have very few
    foreground pixels overall."""
    pos = neg = 0
    for _id in df[ID_COL]:
        mp = os.path.join(mask_dir, f"{_id}.nii.gz")
        if not os.path.exists(mp):
            neg += img_size * img_size
            continue
        m = load_mask(mp, img_size)
        pos += int(m.sum())
        neg += int(m.size - m.sum())
    weight = min(neg / max(pos, 1), max_weight)
    print(f"[seg] pos_weight={weight:.2f} (fg={pos:,}px bg={neg:,}px over {len(df)} images)", flush=True)
    return weight


class CFG:
    def __init__(self, args):
        fam = ENCODER_FAMILIES[args.encoder]
        self.img_size = fam["img_size"]
        self.norm_mean, self.norm_std = fam["mean"], fam["std"]
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip, self.clahe_grid = 2.0, 8
        self.n_folds = args.n_folds


class SegDataset(Dataset):
    def __init__(self, df, cfg, image_dir, mask_dir, image_ext=".dcm", train=True, mask_transpose=False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = image_ext
        self.train = train
        self.mask_transpose = mask_transpose

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        _id = row[ID_COL]
        ip = find_file(self.image_dir, _id, self.image_ext)
        if ip is None:
            raise FileNotFoundError(f"no image for {ID_COL}={_id} under {self.image_dir}")
        x = preprocess_image(sitk_read(ip), self.cfg)

        mp = os.path.join(self.mask_dir, f"{_id}.nii.gz")
        y = load_mask(mp, self.cfg.img_size, transpose=self.mask_transpose) if os.path.exists(mp) else np.zeros(
            (self.cfg.img_size, self.cfg.img_size), dtype=np.float32)

        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, ::-1].copy()
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-15, 15))
                scale = float(np.random.uniform(0.9, 1.1))
                x, y = _rotate_scale(x, y, angle, scale)
            if np.random.rand() < 0.5:
                x = _brightness_contrast(x)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()).unsqueeze(0), str(_id)


def focal_tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-5):
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * target).sum(dim=(1, 2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return ((1 - tversky).clamp(min=1e-6) ** gamma).mean()


def dice_score_np(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return np.nan
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * np.logical_and(pred, gt).sum() / (ps + gs)


@torch.no_grad()
def predict_probs(model, dl, device, tta=True):
    """Single pass over `dl` -> (probs_by_id, gt_by_id), both
    {str(_id): (H,W) np array}. Collecting GT here (rather than re-iterating
    dl a second time elsewhere) avoids relying on the DataLoader being safely
    re-iterable in the same order and keeps this the one place batch/sample
    tensors get unpacked. tta=True averages the original image and its
    horizontal flip (flipped back before averaging) -- the one augmentation
    actually used in training, same principle as predict_ordfused.py's TTA."""
    model.eval()
    probs_by_id, gt_by_id = {}, {}
    for x, y, ids in dl:
        x = x.to(device)
        probs = torch.sigmoid(model(x))
        if tta:
            probs_flip = torch.sigmoid(model(x.flip(-1))).flip(-1)
            probs = (probs + probs_flip) / 2
        probs = probs.cpu().numpy()
        gt = y.numpy()
        for i, _id in enumerate(ids):
            probs_by_id[_id] = probs[i, 0]
            gt_by_id[_id] = gt[i, 0]
    return probs_by_id, gt_by_id


@torch.no_grad()
def evaluate_dice(model, dl, device, threshold=0.5, tta=True):
    probs_by_id, gt_by_id = predict_probs(model, dl, device, tta=tta)
    dices = [dice_score_np((probs_by_id[i] >= threshold).astype(np.uint8), (gt_by_id[i] > 0.5).astype(np.uint8))
             for i in probs_by_id]
    return float(np.nanmean(dices))


def sweep_threshold(model, dl, device, grid=None, tta=True, smooth_window=3):
    """Sweeps the binarization threshold on `dl` (must be the DEV split, never
    test) and returns (best_threshold, best_dice) -- same discipline as
    common.py's sweep_tau_accuracy elsewhere in this project: never tuned on
    the held-out test set.

    Dev here is only 89 images, and a raw single-point argmax over a 17-point
    grid is prone to picking up sampling noise rather than real signal (found
    empirically: t*=0.90 beat t=0.5 by only +0.007-0.011 dev dice despite the
    huge threshold jump, and for one encoder the "optimal" swept threshold was
    *worse* than the untuned default -- a sign the selected point doesn't
    generalize, which is exactly what then produced a catastrophic TEST-set
    collapse in job 117713's run). To guard against this, selection is driven
    by a smoothed curve (moving average over `smooth_window` neighboring grid
    points) rather than the raw per-point dice -- a threshold only wins if its
    whole neighborhood is good, not just one noisy point."""
    grid = grid if grid is not None else np.linspace(0.1, 0.9, 17)
    probs_by_id, gt_by_id = predict_probs(model, dl, device, tta=tta)
    raw_dice = []
    for t in grid:
        dices = [dice_score_np((probs_by_id[i] >= t).astype(np.uint8), (gt_by_id[i] > 0.5).astype(np.uint8))
                  for i in probs_by_id]
        raw_dice.append(float(np.nanmean(dices)))
    raw_dice = np.array(raw_dice)

    half = smooth_window // 2
    smoothed = np.array([
        raw_dice[max(0, i - half):min(len(grid), i + half + 1)].mean()
        for i in range(len(grid))
    ])
    best_idx = int(np.argmax(smoothed))
    best_t, best_dice = float(grid[best_idx]), float(raw_dice[best_idx])
    print(f"[seg] threshold sweep raw dice: " +
          ", ".join(f"{t:.2f}:{d:.4f}" for t, d in zip(grid, raw_dice)), flush=True)
    print(f"[seg] threshold sweep smoothed-selected t*={best_t:.2f} "
          f"(raw dev_dice={best_dice:.4f}, smoothed={smoothed[best_idx]:.4f})", flush=True)
    return best_t, best_dice


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=["rad_dino", "chexfound_vitl16", "eva_x_base"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decoder-lr-mult", type=float, default=1.0)
    ap.add_argument("--lora-lr-mult", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--nnunet-labels-dir", default=_DEFAULT_NNUNET_LABELS,
                    help="ground-truth masks for the 111 test cases (not in Task1/Data/test)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs_seg"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.out_dir, args.encoder)
    os.makedirs(out_dir, exist_ok=True)

    cfg = CFG(args)

    df_tr_all = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_tr_all = make_folds(df_tr_all, cfg, group_col=None, seed=args.seed)
    dev_fold = 0
    tr = df_tr_all[df_tr_all.fold != dev_fold]
    dev = df_tr_all[df_tr_all.fold == dev_fold]

    image_dir = os.path.join(args.data_dir, "train", "CXR")
    mask_dir = os.path.join(args.data_dir, "train", "CXR_label")
    dl_tr = DataLoader(SegDataset(tr, cfg, image_dir, mask_dir, train=True), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=True)
    dl_dev = DataLoader(SegDataset(dev, cfg, image_dir, mask_dir, train=False), batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)

    df_te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    dl_te = DataLoader(SegDataset(df_te, cfg, os.path.join(args.data_dir, "test", "CXR"),
                                   args.nnunet_labels_dir, train=False, mask_transpose=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    pos_weight = torch.tensor(compute_pos_weight(tr, mask_dir, cfg.img_size), device=device)

    model = FoundationSegModel(args.encoder).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[seg/{args.encoder}] {n_trainable:,}/{n_total:,} trainable params "
          f"({100 * n_trainable / n_total:.2f}%), train={len(tr)} dev={len(dev)} test={len(df_te)}",
          flush=True)

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    opt = torch.optim.AdamW([
        {"params": [p for p in encoder_params if p.requires_grad], "lr": args.lr * args.lora_lr_mult},
        {"params": decoder_params, "lr": args.lr * args.decoder_lr_mult},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dice, best_path = -1.0, os.path.join(out_dir, "best.pt")
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        train_loss = 0.0
        n_batches = len(dl_tr)
        for bi, (x, y, _ids) in enumerate(dl_tr):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = focal_tversky_loss(logits, y) + \
                    F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            scaler.scale(loss / args.accum_steps).backward()
            train_loss += loss.item() * x.size(0)
            if (bi + 1) % args.accum_steps == 0 or bi == n_batches - 1:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        sched.step()
        train_loss /= max(len(dl_tr.dataset), 1)

        # Fixed threshold=0.5, no TTA during per-epoch monitoring -- cheap,
        # and only needs to be a consistent *relative* ranking signal for
        # checkpoint selection, not the final optimal threshold (that's swept
        # properly on dev, once, after training -- see below).
        dev_dice = evaluate_dice(model, dl_dev, device, threshold=0.5, tta=False)
        print(f"[seg/{args.encoder}] ep{ep:02d} loss={train_loss:.4f} dev_dice={dev_dice:.4f}", flush=True)
        if dev_dice > best_dice:
            best_dice = dev_dice
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "dev_dice": best_dice}, best_path)

    print(f"[seg/{args.encoder}] BEST dev_dice={best_dice:.4f} (threshold=0.5, no TTA) -> {best_path}", flush=True)

    sd = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    # Real final numbers: TTA on, threshold swept on dev (never on test).
    best_t, dev_dice_swept = sweep_threshold(model, dl_dev, device, tta=True)
    print(f"[seg/{args.encoder}] dev threshold sweep: t*={best_t:.2f} dev_dice={dev_dice_swept:.4f} (TTA on)",
          flush=True)
    test_dice = evaluate_dice(model, dl_te, device, threshold=best_t, tta=True)
    print(f"[seg/{args.encoder}] TEST dice={test_dice:.4f} (real held-out, same 111 cases as nnU-Net, "
          f"threshold={best_t:.2f} from dev, TTA on)", flush=True)

    # Sanity cross-check: TEST dice at the untuned default threshold (0.5),
    # never fit on anything. If this differs wildly from the swept-threshold
    # number above, the swept threshold overfit the 89-image dev split and
    # didn't generalize -- surface that immediately instead of only
    # discovering it after the fact.
    test_dice_t50 = evaluate_dice(model, dl_te, device, threshold=0.5, tta=True)
    print(f"[seg/{args.encoder}] TEST dice={test_dice_t50:.4f} (sanity check, fixed threshold=0.5, "
          f"TTA on -- compare against the swept-threshold number above)", flush=True)

    probs_te, _gt_te = predict_probs(model, dl_te, device, tta=True)
    probs_path = os.path.join(out_dir, "test_probs.pt")
    torch.save({"probs": probs_te, "threshold": best_t, "test_dice": test_dice}, probs_path)

    # Also save DEV-set probabilities (same TTA setting) -- ensemble_seg_predictions.py
    # needs these to sweep the ensemble's own threshold on dev, never on test,
    # same no-leakage discipline as every threshold decision elsewhere in this project.
    probs_dev, _gt_dev = predict_probs(model, dl_dev, device, tta=True)
    dev_probs_path = os.path.join(out_dir, "dev_probs.pt")
    torch.save({"probs": probs_dev}, dev_probs_path)
    print(f"[seg/{args.encoder}] wrote test/dev probability maps -> {probs_path}, {dev_probs_path} "
          f"(feeds ensemble_seg_predictions.py)", flush=True)


if __name__ == "__main__":
    main()
