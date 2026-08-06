#!/usr/bin/env python
"""Train HybridFused-CXR (joint CNN+ViT fusion) for Task 1, 5-fold CV.

Same fold splits / preprocessing / challenge metric / CORN ordinal head as
train_ordfused.py, so numbers are directly comparable to every single-stream
OrdFused checkpoint. The only architectural change is the encoder: instead of
one timm backbone, HybridFusedModel runs a CNN branch and a ViT branch in the
same forward pass and fuses their features with a learned gate before the
shared head (see hybrid_fused.py's module docstring for the paper this is
based on).

Usage:
    python train_hybrid_fused.py --fold all
    python train_hybrid_fused.py --cnn-encoder densenet201 --vit-encoder eva_x_base --no-tabular
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
from common import ID_COL, challenge_score, make_folds, set_seed, sweep_tau_accuracy
from hybrid_fused import DEFAULT_CNN_ENCODER, DEFAULT_VIT_ENCODER, HybridFusedDataset, HybridFusedModel
from ordfused import ENCODER_FAMILIES, corn_cavity_prob, corn_loss
from train_ordfused import build_scheduler, trivial_ceiling  # reuse, no need to duplicate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))

_CNN_MEAN, _CNN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class BranchCFG:
    """Minimal CFG-shaped object for one branch's preprocessing (img_size/
    norm/clip/clahe) — HybridFusedDataset needs one of these per branch."""

    def __init__(self, image_dir, image_ext, img_size, mean, std):
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.img_size = img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        self.norm_mean = mean
        self.norm_std = std


class CFG:
    """Fold-split config (make_folds only reads img_size/n_folds off this) +
    the two branch CFGs + optimizer/schedule knobs."""

    def __init__(self, args):
        self.csv_path = os.path.join(args.data_dir, "train.csv")
        image_dir = os.path.join(args.data_dir, "train", "CXR")
        image_ext = ".dcm"

        vit_family = ENCODER_FAMILIES[args.vit_encoder]
        self.img_size = args.cnn_img_size  # used only by make_folds's stratification, harmless
        self.cnn_cfg = BranchCFG(image_dir, image_ext, args.cnn_img_size, _CNN_MEAN, _CNN_STD)
        self.vit_cfg = BranchCFG(image_dir, image_ext, vit_family["img_size"],
                                  vit_family["mean"], vit_family["std"])

        self.n_folds = args.n_folds
        self.batch_size = args.batch_size or vit_family["batch_size"]
        self.accum_steps = args.accum_steps if args.batch_size else vit_family["accum_steps"]
        self.epochs = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_epochs = args.warmup_epochs
        self.cnn_lr_mult = args.cnn_lr_mult
        self.vit_lr_mult = args.vit_lr_mult
        self.grad_clip = args.grad_clip
        self.max_retries = args.max_retries

        self.cnn_encoder = args.cnn_encoder
        self.vit_encoder = args.vit_encoder
        self.use_tabular = not args.no_tabular
        self.use_mixup = args.mixup_alpha > 0
        self.mixup_alpha = args.mixup_alpha
        self.swa_frac = args.swa_frac


def build_optimizer(model, lr, weight_decay, cnn_lr_mult, vit_lr_mult):
    # The CNN branch (densenet201, full fine-tune, BatchNorm-based) and the ViT
    # branch (LoRA-only, foundation-pretrained) have opposite optimal LR
    # treatment -- see train.py's DEFAULT_HP comment: BatchNorm CNNs train
    # stably at a flat encoder_lr_mult=1.0 with no warmup, while the LoRA/ViT
    # recipe (encoder_lr_mult=0.3, warmup=3, from train_ordfused.py) exists to
    # protect a fragile frozen-backbone adapter. A single shared mult (the old
    # behavior) silently undertrained the CNN branch relative to its standalone
    # counterpart -- give each branch its own group instead.
    cnn_ids = {id(p) for p in model.cnn_encoder.parameters()}
    vit_ids = {id(p) for p in model.vit_encoder.parameters()}
    cnn_enc, vit_enc, rest = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in cnn_ids:
            cnn_enc.append(p)
        elif id(p) in vit_ids:
            vit_enc.append(p)
        else:
            rest.append(p)
    groups = [
        {"params": cnn_enc, "lr": lr * cnn_lr_mult},
        {"params": vit_enc, "lr": lr * vit_lr_mult},
        {"params": rest, "lr": lr},
    ]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def update_bn_tabular(model, loader, device):
    momenta = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None
    if not momenta:
        return
    was_training = model.training
    model.train()
    with torch.no_grad():
        for x_cnn, x_vit, tab, _yb, _yo in loader:
            model(x_cnn.to(device), x_vit.to(device), tab.to(device))
    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


def evaluate(model, dl, device):
    model.eval()
    probs, ys, fgates, tgates = [], [], [], []
    with torch.no_grad():
        for x_cnn, x_vit, tab, yb, _yo in dl:
            logits, fg, tg = model(x_cnn.to(device), x_vit.to(device), tab.to(device), return_gate=True)
            probs.append(corn_cavity_prob(logits).cpu().numpy())
            ys.append(yb.numpy())
            fgates.append(fg.cpu().numpy())
            tgates.append(tg.cpu().numpy())
    return (np.concatenate(probs), np.concatenate(ys).astype(int),
            np.concatenate(fgates), np.concatenate(tgates))


def run_one_fold(cfg, df, fold, device, out_dir, num_workers, seed):
    tr = df[df.fold != fold]
    va = df[df.fold == fold]

    pos = (tr.cavity != "none").sum()
    neg = (tr.cavity == "none").sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    dl_tr = DataLoader(HybridFusedDataset(tr, cfg.cnn_cfg, cfg.vit_cfg, True),
                       batch_size=cfg.batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True, pin_memory=True)
    dl_va = DataLoader(HybridFusedDataset(va, cfg.cnn_cfg, cfg.vit_cfg, False),
                       batch_size=cfg.batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    ckpt_path = os.path.join(out_dir, f"fold{fold}_best.pt")
    ceiling = None

    for attempt in range(cfg.max_retries + 1):
        set_seed(seed + fold * 1000 + attempt)
        model = HybridFusedModel(cfg.cnn_encoder, cfg.vit_encoder,
                                  use_tabular=cfg.use_tabular, pretrained=True).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[hybrid/{cfg.cnn_encoder}+{cfg.vit_encoder}] fold {fold} attempt {attempt}: "
              f"{n_params:.1f}M params, train={len(tr)} val={len(va)} tabular={cfg.use_tabular} "
              f"pos_weight={pos_weight.item():.3f} lr={cfg.lr:.1e} "
              f"cnn_mult={cfg.cnn_lr_mult} vit_mult={cfg.vit_lr_mult} warmup={cfg.warmup_epochs}",
              flush=True)

        opt = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.cnn_lr_mult, cfg.vit_lr_mult)
        sched = build_scheduler(opt, cfg.epochs, cfg.warmup_epochs)

        swa_start = int(cfg.epochs * (1 - cfg.swa_frac)) if cfg.swa_frac > 0 else cfg.epochs
        swa_model = torch.optim.swa_utils.AveragedModel(model) if cfg.swa_frac > 0 else None
        swa_scheduler = None
        if swa_model is not None:
            swa_scheduler = torch.optim.swa_utils.SWALR(
                opt, swa_lr=[g["lr"] for g in opt.param_groups])

        use_amp = device == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best = {"score": -1.0, "acc": -1.0}
        history = []
        for ep in range(cfg.epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            opt.zero_grad()
            n_batches = len(dl_tr)
            for bi, (x_cnn, x_vit, tab, _yb, yo) in enumerate(dl_tr):
                x_cnn, x_vit, tab, yo = (x_cnn.to(device), x_vit.to(device),
                                         tab.to(device), yo.to(device))
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    if cfg.use_mixup:
                        lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                        perm = torch.randperm(x_cnn.size(0), device=device)
                        x_cnn = lam * x_cnn + (1 - lam) * x_cnn[perm]
                        x_vit = lam * x_vit + (1 - lam) * x_vit[perm]
                        tab = lam * tab + (1 - lam) * tab[perm]
                        logits = model(x_cnn, x_vit, tab)
                        loss = (lam * corn_loss(logits, yo, node0_pos_weight=pos_weight)
                                + (1 - lam) * corn_loss(logits, yo[perm], node0_pos_weight=pos_weight))
                    else:
                        logits = model(x_cnn, x_vit, tab)
                        loss = corn_loss(logits, yo, node0_pos_weight=pos_weight)
                scaler.scale(loss / cfg.accum_steps).backward()
                train_loss += loss.item() * x_cnn.size(0)
                is_last = bi == n_batches - 1
                if (bi + 1) % cfg.accum_steps == 0 or is_last:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            if swa_model is not None and ep >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                sched.step()
            train_loss /= max(len(dl_tr.dataset), 1)

            probs, ys, fgates, tgates = evaluate(model, dl_va, device)
            if ceiling is None:
                ceiling = trivial_ceiling(ys)
            # tau is chosen to maximize raw accuracy directly, not the old
            # 0.7*acc+0.3*presence_dice proxy: presence_dice is a fixed
            # per-cell constant (1.0 for a correct negative, 0.60 for a
            # correct positive), not real mask overlap, and real Dice now
            # comes from nnU-Net downstream in gate_segmentation.py, fully
            # decoupled from this classifier's threshold. Sweeping tau
            # against the proxy biased it toward predicting "none" more
            # often (chasing the proxy's higher reward for true negatives)
            # instead of the accuracy-optimal point. `score`/`dice` below are
            # still computed and logged for continuity, but no longer drive
            # checkpoint or threshold selection.
            tau, _ = sweep_tau_accuracy(ys, probs)
            score, acc, dice = challenge_score(ys, probs, tau)

            dt = time.time() - t0
            history.append({"epoch": ep, "train_loss": train_loss, "score": score, "acc": acc,
                            "dice": dice, "tau": tau, "fusion_gate_mean": float(fgates.mean()),
                            "tab_gate_mean": float(tgates.mean()), "seconds": dt})
            print(f"[hybrid/{cfg.cnn_encoder}+{cfg.vit_encoder}] fold{fold} ep{ep:02d}  "
                  f"loss={train_loss:.4f}  score={score:.4f} acc={acc:.4f} dice={dice:.4f} "
                  f"tau*={tau:.3f} fg={fgates.mean():.3f} tg={tgates.mean():.3f}  ({dt:.1f}s)",
                  flush=True)

            if acc > best["acc"]:
                best = {"score": score, "tau": tau, "acc": acc, "dice": dice,
                        "fusion_gate_mean": float(fgates.mean()),
                        "tab_gate_mean": float(tgates.mean()), "epoch": ep}
                torch.save({"model": model.state_dict(), "cnn_encoder": cfg.cnn_encoder,
                            "vit_encoder": cfg.vit_encoder, "use_tabular": cfg.use_tabular,
                            "cnn_img_size": cfg.cnn_cfg.img_size, "vit_img_size": cfg.vit_cfg.img_size,
                            "fold": fold, "best": best}, ckpt_path)

        if swa_model is not None:
            update_bn_tabular(swa_model, dl_tr, device)
            probs, ys, fgates, tgates = evaluate(swa_model, dl_va, device)
            tau, _ = sweep_tau_accuracy(ys, probs)
            score, acc, dice = challenge_score(ys, probs, tau)
            print(f"[hybrid/{cfg.cnn_encoder}+{cfg.vit_encoder}] fold{fold} SWA  score={score:.4f} "
                  f"acc={acc:.4f} dice={dice:.4f} tau*={tau:.3f} fg={fgates.mean():.3f} "
                  f"tg={tgates.mean():.3f}", flush=True)
            if acc > best["acc"]:
                best = {"score": score, "tau": tau, "acc": acc, "dice": dice,
                        "fusion_gate_mean": float(fgates.mean()),
                        "tab_gate_mean": float(tgates.mean()), "epoch": "swa"}
                torch.save({"model": swa_model.module.state_dict(), "cnn_encoder": cfg.cnn_encoder,
                            "vit_encoder": cfg.vit_encoder, "use_tabular": cfg.use_tabular,
                            "cnn_img_size": cfg.cnn_cfg.img_size, "vit_img_size": cfg.vit_cfg.img_size,
                            "fold": fold, "best": best}, ckpt_path)

        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"fold{fold}_history.csv"), index=False)
        if best["acc"] > ceiling + 1e-4 or attempt == cfg.max_retries:
            break
        print(f"[hybrid/{cfg.cnn_encoder}+{cfg.vit_encoder}] fold{fold} attempt {attempt} collapsed "
              f"(best {best['score']:.4f} <= ceiling {ceiling:.4f}) -- retrying", flush=True)

    print(f"[hybrid/{cfg.cnn_encoder}+{cfg.vit_encoder}] fold{fold} BEST: {best}  -> {ckpt_path}",
          flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn-encoder", default=DEFAULT_CNN_ENCODER,
                    help="plain timm CNN backbone for the local-feature branch")
    ap.add_argument("--vit-encoder", default=DEFAULT_VIT_ENCODER,
                    choices=sorted(ENCODER_FAMILIES),
                    help="foundation ViT backbone for the global-feature branch")
    ap.add_argument("--fold", default="all", help="fold index or 'all'")
    ap.add_argument("--no-tabular", action="store_true", help="image-only ablation")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override physical batch size; default follows the ViT encoder's "
                         "registry entry (the CNN branch is cheap by comparison)")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation steps; only used together with an "
                         "explicit --batch-size")
    ap.add_argument("--cnn-img-size", type=int, default=224)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--swa-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--cnn-lr-mult", type=float, default=1.0,
                    help="LR multiplier for the CNN branch (full fine-tune, BatchNorm-based "
                         "-- trains stably at 1.0/no-warmup per train.py's DEFAULT_HP, unlike "
                         "the LoRA ViT branch)")
    ap.add_argument("--vit-lr-mult", type=float, default=0.3,
                    help="LR multiplier for the ViT branch (LoRA-only, foundation-pretrained "
                         "-- matches train_ordfused.py's encoder_lr_mult default)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.cnn_encoder}_{args.vit_encoder}"
    if args.no_tabular:
        tag += "_notab"
    print(f"HybridFused cnn={args.cnn_encoder} vit={args.vit_encoder} "
          f"tabular={not args.no_tabular} device={device} fold_arg={args.fold}", flush=True)

    cfg = CFG(args)
    out_dir = os.path.join(args.out_dir, f"hybrid_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    assert ID_COL in df.columns, f"expected id column '{ID_COL}' in {cfg.csv_path}"
    df = make_folds(df, cfg, group_col=None, seed=args.seed)
    print(df.groupby(["fold", "cavity"]).size().unstack(fill_value=0), flush=True)

    folds = range(cfg.n_folds) if args.fold == "all" else [int(args.fold)]
    results = {f: run_one_fold(cfg, df, f, device, out_dir, args.num_workers, args.seed) for f in folds}

    if args.fold == "all":
        summary = pd.DataFrame([{"fold": f, **r} for f, r in results.items()])
        summary.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)
        print(f"[hybrid/{tag}] CV mean score={summary['score'].mean():.4f} "
              f"+/- {summary['score'].std():.4f}  (mean fusion_gate={summary['fusion_gate_mean'].mean():.3f})",
              flush=True)

    with open(os.path.join(out_dir, f"results_fold_{args.fold}.json"), "w") as fh:
        json.dump({str(k): v for k, v in results.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
