#!/usr/bin/env python
"""Train one model on one CV fold for Task 1 (cavity detection).

Usage (single fold):
    python train.py --model convnext_tiny --fold 0

Usage (all 5 folds, sequential, one process — what the sbatch scripts call):
    python train.py --model resnet50d --fold all

Writes, under --out-dir/<model>/:
    fold{N}_best.pt        checkpoint of the best-scoring epoch for fold N
    fold{N}_history.csv    per-epoch score/acc/dice/tau log for fold N
    cv_summary.csv         one row per fold with the best metrics (after all folds run)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CavityDataset,
    CNNDetector,
    ID_COL,
    challenge_score,
    make_folds,
    set_seed,
    sweep_tau,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))

# Per-model training hyperparameters. resnet50d/densenet201/tf_efficientnetv2_s
# (BatchNorm-based) train stably on a flat lr=3e-4 with no warmup -- that's
# left untouched below. convnext_tiny (LayerNorm-based, timm's ConvNeXt) is
# known to be far more sensitive to LR/warmup when fine-tuning on a small
# dataset (~355 train images/fold): a flat 3e-4 with no warmup let 2 of 5
# folds get stuck predicting the majority class for all 25 epochs (score
# frozen at the trivial baseline, tau* pinned at 0.95). Fix: a lower peak LR,
# a linear warmup before the cosine schedule, and a smaller LR multiplier on
# the pretrained encoder relative to the freshly-initialized head so the head
# can adapt before the backbone is perturbed.
DEFAULT_HP = dict(lr=3e-4, weight_decay=1e-4, warmup_epochs=0, encoder_lr_mult=1.0)
MODEL_HP = {
    "convnext_tiny": dict(lr=1e-4, weight_decay=5e-2, warmup_epochs=3, encoder_lr_mult=0.1),
}


class CFG:
    def __init__(self, args):
        self.csv_path = os.path.join(args.data_dir, "train.csv")
        self.image_dir = os.path.join(args.data_dir, "train", "CXR")
        self.mask_dir = os.path.join(args.data_dir, "train", "CXR_label")
        self.image_ext = ".dcm"
        self.mask_ext = ".nii.gz"

        self.img_size = args.img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

        self.n_folds = args.n_folds
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        hp = dict(MODEL_HP.get(args.model, DEFAULT_HP))
        if args.lr is not None:
            hp["lr"] = args.lr
        if args.weight_decay is not None:
            hp["weight_decay"] = args.weight_decay
        if args.warmup_epochs is not None:
            hp["warmup_epochs"] = args.warmup_epochs
        if args.encoder_lr_mult is not None:
            hp["encoder_lr_mult"] = args.encoder_lr_mult
        self.lr = hp["lr"]
        self.weight_decay = hp["weight_decay"]
        self.warmup_epochs = hp["warmup_epochs"]
        self.encoder_lr_mult = hp["encoder_lr_mult"]
        self.grad_clip = args.grad_clip
        self.max_retries = args.max_retries

        self.encoder = args.model
        self.freeze_encoder = args.freeze_encoder


def build_optimizer(model, lr, weight_decay, encoder_lr_mult):
    # Discriminative LR: the pretrained encoder moves slower than the
    # freshly-initialized head, so the head can adapt before it starts
    # perturbing the backbone's features (main lever against convnext's
    # collapse-to-majority-class failure mode).
    groups = [
        {"params": model.encoder.parameters(), "lr": lr * encoder_lr_mult},
        {"params": model.head.parameters(), "lr": lr},
    ]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(opt, epochs, warmup_epochs):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - warmup_epochs, 1))
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def trivial_ceiling(ys):
    # Best score achievable by a degenerate model that predicts one constant
    # class for everyone -- used to detect a fold that collapsed rather than
    # learned anything.
    neg_frac = float((ys == 0).mean())
    pos_frac = 1.0 - neg_frac
    return max(neg_frac, 0.88 * pos_frac)


def run_one_fold(cfg, df, fold, device, out_dir, num_workers, seed):
    tr = df[df.fold != fold]
    va = df[df.fold == fold]

    pos = (tr.cavity != "none").sum()
    neg = (tr.cavity == "none").sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    dl_tr = DataLoader(
        CavityDataset(tr, cfg, True), batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    dl_va = DataLoader(
        CavityDataset(va, cfg, False), batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ckpt_path = os.path.join(out_dir, f"fold{fold}_best.pt")
    ceiling = None

    for attempt in range(cfg.max_retries + 1):
        set_seed(seed + fold * 1000 + attempt)
        model = CNNDetector(cfg.encoder, pretrained=True, freeze=cfg.freeze_encoder).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[{cfg.encoder}] fold {fold} attempt {attempt}: {n_params:.1f}M params, "
              f"train={len(tr)} val={len(va)} pos_weight={pos_weight.item():.3f} "
              f"lr={cfg.lr:.1e} encoder_lr_mult={cfg.encoder_lr_mult} "
              f"warmup_epochs={cfg.warmup_epochs}", flush=True)

        opt = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.encoder_lr_mult)
        sched = build_scheduler(opt, cfg.epochs, cfg.warmup_epochs)

        best = {"score": -1.0}
        history = []

        for ep in range(cfg.epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            for x, yb, _yo in dl_tr:
                x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad()
                logit = model(x)
                loss = lossfn(logit, yb)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                train_loss += loss.item() * x.size(0)
            sched.step()
            train_loss /= max(len(dl_tr.dataset), 1)

            model.eval()
            probs, ys = [], []
            with torch.no_grad():
                for x, yb, _yo in dl_va:
                    p = torch.sigmoid(model(x.to(device))).cpu().numpy()
                    probs.append(p)
                    ys.append(yb.numpy())
            probs = np.concatenate(probs)
            ys = np.concatenate(ys).astype(int)
            if ceiling is None:
                ceiling = trivial_ceiling(ys)
            tau, score = sweep_tau(ys, probs)
            s, acc, dice = challenge_score(ys, probs, tau)

            dt = time.time() - t0
            row = {"epoch": ep, "train_loss": train_loss, "score": score, "acc": acc,
                   "dice": dice, "tau": tau, "seconds": dt}
            history.append(row)
            print(f"[{cfg.encoder}] fold{fold} ep{ep:02d}  loss={train_loss:.4f}  "
                  f"score={score:.4f} acc={acc:.4f} dice={dice:.4f} tau*={tau:.3f}  ({dt:.1f}s)", flush=True)

            if score > best["score"]:
                best = {"score": score, "tau": tau, "acc": acc, "dice": dice, "epoch": ep}
                torch.save({"model": model.state_dict(), "cfg_encoder": cfg.encoder,
                            "img_size": cfg.img_size, "fold": fold, "best": best}, ckpt_path)

        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"fold{fold}_history.csv"), index=False)

        if best["score"] > ceiling + 1e-4 or attempt == cfg.max_retries:
            break
        print(f"[{cfg.encoder}] fold{fold} attempt {attempt} collapsed "
              f"(best score {best['score']:.4f} <= trivial ceiling {ceiling:.4f}) "
              f"-- retrying with a new init", flush=True)

    print(f"[{cfg.encoder}] fold{fold} BEST: {best}  -> {ckpt_path}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="timm model name, e.g. convnext_tiny, resnet50d, densenet201, tf_efficientnetv2_s")
    ap.add_argument("--fold", default="all", help="fold index (0-based) or 'all' to run every fold sequentially")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=None,
                     help="override the per-model default LR (see MODEL_HP)")
    ap.add_argument("--weight-decay", type=float, default=None,
                     help="override the per-model default weight decay")
    ap.add_argument("--warmup-epochs", type=int, default=None,
                     help="override the per-model default linear warmup length")
    ap.add_argument("--encoder-lr-mult", type=float, default=None,
                     help="override the per-model default encoder/head LR ratio")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                     help="max gradient norm (<=0 disables clipping)")
    ap.add_argument("--max-retries", type=int, default=2,
                     help="re-init and retry a fold this many times if it "
                          "collapses to the trivial majority-class solution")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device index (alternative to CUDA_VISIBLE_DEVICES)")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model} device={device} fold_arg={args.fold}", flush=True)

    cfg = CFG(args)
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    assert ID_COL in df.columns, f"expected id column '{ID_COL}' in {cfg.csv_path}, got {df.columns.tolist()}"
    df = make_folds(df, cfg, group_col=None, seed=args.seed)
    print(df.groupby(["fold", "cavity"]).size().unstack(fill_value=0), flush=True)

    folds = range(cfg.n_folds) if args.fold == "all" else [int(args.fold)]
    results = {}
    for f in folds:
        results[f] = run_one_fold(cfg, df, f, device, out_dir, args.num_workers, args.seed)

    if args.fold == "all":
        summary = pd.DataFrame([{"fold": f, **r} for f, r in results.items()])
        summary.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)
        print(f"[{args.model}] CV mean score={summary['score'].mean():.4f} "
              f"+/- {summary['score'].std():.4f}", flush=True)

    with open(os.path.join(out_dir, f"results_fold_{args.fold}.json"), "w") as fh:
        json.dump({str(k): v for k, v in results.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
