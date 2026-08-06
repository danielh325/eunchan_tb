#!/usr/bin/env python
"""Train OrdFused-CXR for Task 1 (TB cavity detection), 5-fold CV.

Same fold splits, preprocessing, and challenge metric as the CNN baselines
(train.py), so numbers are directly comparable. Adds: CORN ordinal supervision,
gated tabular fusion, discriminative encoder/head LR, LR warmup, gradient
clipping, and a collapse-to-majority-class retry — everything needed for a
LayerNorm-heavy fused model to train stably on 444 images.

Usage:
    python train_ordfused.py --encoder tf_efficientnetv2_s --fold all
    python train_ordfused.py --encoder tf_efficientnetv2_s --no-tabular   # ablation
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
from ordfused import (
    ENCODER_FAMILIES,
    FOUNDATION_ENCODER_NAMES,
    OrdFusedDataset,
    OrdFusedModel,
    TAB_DIM,
    corn_cavity_prob,
    corn_loss,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))


class CFG:
    def __init__(self, args):
        self.csv_path = os.path.join(args.data_dir, "train.csv")
        self.image_dir = os.path.join(args.data_dir, "train", "CXR")
        self.image_ext = ".dcm"

        # Foundation-model encoders (EVA-X/RAD-DINO/CheXFound) each need their
        # own input resolution + normalization stats (not the CNN-baseline
        # ImageNet 224 defaults) — pull from the registry when applicable, so
        # --img-size only overrides when the user explicitly passes it.
        family_cfg = ENCODER_FAMILIES.get(args.encoder)
        if args.img_size is not None:
            self.img_size = args.img_size
        elif family_cfg is not None:
            self.img_size = family_cfg["img_size"]
        else:
            self.img_size = 224
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        if family_cfg is not None:
            self.norm_mean = family_cfg["mean"]
            self.norm_std = family_cfg["std"]
        else:
            self.norm_mean = (0.485, 0.456, 0.406)
            self.norm_std = (0.229, 0.224, 0.225)

        self.n_folds = args.n_folds
        # RAD-DINO/CheXFound need a much smaller physical batch (high-res ViT
        # attention memory) — accum_steps keeps the effective batch size
        # comparable across encoders. --batch-size overrides both when passed.
        if args.batch_size is not None:
            self.batch_size = args.batch_size
            self.accum_steps = args.accum_steps
        elif family_cfg is not None:
            self.batch_size = family_cfg["batch_size"]
            self.accum_steps = family_cfg["accum_steps"]
        else:
            self.batch_size = 16
            self.accum_steps = 1
        self.epochs = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_epochs = args.warmup_epochs
        self.encoder_lr_mult = args.encoder_lr_mult
        self.grad_clip = args.grad_clip
        self.max_retries = args.max_retries

        self.encoder = args.encoder
        self.use_tabular = not args.no_tabular
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout

        # Mixup only on foundation-model runs (CNN runs already score best
        # without it — see plan's research-pass note); SWA applies to all.
        self.use_mixup = args.encoder in FOUNDATION_ENCODER_NAMES and args.mixup_alpha > 0
        self.mixup_alpha = args.mixup_alpha
        self.swa_frac = args.swa_frac


def build_optimizer(model, lr, weight_decay, encoder_lr_mult):
    # Pretrained encoder moves slower than the freshly-initialized fusion/head
    # so the new parameters adapt before perturbing the backbone (the main
    # lever against collapse-to-majority-class on small data). Only trainable
    # params are included — with LoRA, most encoder params are frozen, so this
    # avoids handing AdamW a pile of dead (requires_grad=False) param groups.
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc, rest = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (enc if id(p) in enc_ids else rest).append(p)
    groups = [
        {"params": enc, "lr": lr * encoder_lr_mult},
        {"params": rest, "lr": lr},
    ]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(opt, epochs, warmup_epochs):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1))
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def update_bn_tabular(model, loader, device):
    """Like torch.optim.swa_utils.update_bn, but calls model(x, tab) — the
    built-in version only ever passes a single tensor, which would silently
    skip TabularMLP's BatchNorm1d (its `forward` branch only runs when `tab`
    is given) and leave its running stats reset-but-never-refilled."""
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
        for x, tab, _yb, _yo in loader:
            model(x.to(device), tab.to(device))
    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


def trivial_ceiling(ys):
    neg = float((ys == 0).mean())
    return max(neg, 0.88 * (1.0 - neg))


def evaluate(model, dl, device):
    model.eval()
    probs, ys, gates = [], [], []
    with torch.no_grad():
        for x, tab, yb, _yo in dl:
            logits, g = model(x.to(device), tab.to(device), return_gate=True)
            probs.append(corn_cavity_prob(logits).cpu().numpy())
            ys.append(yb.numpy())
            gates.append(g.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys).astype(int), np.concatenate(gates)


def run_one_fold(cfg, df, fold, device, out_dir, num_workers, seed):
    tr = df[df.fold != fold]
    va = df[df.fold == fold]

    pos = (tr.cavity != "none").sum()
    neg = (tr.cavity == "none").sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    dl_tr = DataLoader(OrdFusedDataset(tr, cfg, True), batch_size=cfg.batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True, pin_memory=True)
    dl_va = DataLoader(OrdFusedDataset(va, cfg, False), batch_size=cfg.batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    ckpt_path = os.path.join(out_dir, f"fold{fold}_best.pt")
    ceiling = None

    for attempt in range(cfg.max_retries + 1):
        set_seed(seed + fold * 1000 + attempt)
        model = OrdFusedModel(cfg.encoder, use_tabular=cfg.use_tabular, pretrained=True,
                               lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                               lora_dropout=cfg.lora_dropout).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[ordfused/{cfg.encoder}] fold {fold} attempt {attempt}: {n_params:.1f}M params, "
              f"train={len(tr)} val={len(va)} tabular={cfg.use_tabular} "
              f"pos_weight={pos_weight.item():.3f} lr={cfg.lr:.1e} "
              f"enc_mult={cfg.encoder_lr_mult} warmup={cfg.warmup_epochs}", flush=True)

        opt = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.encoder_lr_mult)
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
            for bi, (x, tab, _yb, yo) in enumerate(dl_tr):
                x, tab, yo = x.to(device), tab.to(device), yo.to(device)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    if cfg.use_mixup:
                        lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                        perm = torch.randperm(x.size(0), device=device)
                        x = lam * x + (1 - lam) * x[perm]
                        tab = lam * tab + (1 - lam) * tab[perm]
                        logits = model(x, tab)
                        loss = (lam * corn_loss(logits, yo, node0_pos_weight=pos_weight)
                                + (1 - lam) * corn_loss(logits, yo[perm], node0_pos_weight=pos_weight))
                    else:
                        logits = model(x, tab)
                        loss = corn_loss(logits, yo, node0_pos_weight=pos_weight)
                # Gradient accumulation (RAD-DINO/CheXFound run a small physical
                # batch to fit V100 memory at their native resolution; this
                # keeps the effective batch size — and thus training dynamics —
                # comparable to the CNN/EVA-X recipe).
                scaler.scale(loss / cfg.accum_steps).backward()
                train_loss += loss.item() * x.size(0)
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

            probs, ys, gates = evaluate(model, dl_va, device)
            if ceiling is None:
                ceiling = trivial_ceiling(ys)
            # tau maximizes raw accuracy directly, not the old 0.7*acc+0.3*
            # presence_dice proxy -- see common.py's presence_dice/sweep_tau_
            # accuracy docstrings and train_hybrid_fused.py's identical fix.
            # `score`/`dice` are still computed and logged for continuity but
            # no longer drive checkpoint or threshold selection.
            tau, _ = sweep_tau_accuracy(ys, probs)
            score, acc, dice = challenge_score(ys, probs, tau)

            dt = time.time() - t0
            history.append({"epoch": ep, "train_loss": train_loss, "score": score, "acc": acc,
                            "dice": dice, "tau": tau, "gate_mean": float(gates.mean()), "seconds": dt})
            print(f"[ordfused/{cfg.encoder}] fold{fold} ep{ep:02d}  loss={train_loss:.4f}  "
                  f"score={score:.4f} acc={acc:.4f} dice={dice:.4f} tau*={tau:.3f} "
                  f"g={gates.mean():.3f}  ({dt:.1f}s)", flush=True)

            if acc > best["acc"]:
                best = {"score": score, "tau": tau, "acc": acc, "dice": dice,
                        "gate_mean": float(gates.mean()), "epoch": ep}
                torch.save({"model": model.state_dict(), "encoder": cfg.encoder,
                            "use_tabular": cfg.use_tabular, "img_size": cfg.img_size,
                            "fold": fold, "best": best}, ckpt_path)

        if swa_model is not None:
            update_bn_tabular(swa_model, dl_tr, device)
            probs, ys, gates = evaluate(swa_model, dl_va, device)
            tau, _ = sweep_tau_accuracy(ys, probs)
            score, acc, dice = challenge_score(ys, probs, tau)
            print(f"[ordfused/{cfg.encoder}] fold{fold} SWA  score={score:.4f} acc={acc:.4f} "
                  f"dice={dice:.4f} tau*={tau:.3f} g={gates.mean():.3f}", flush=True)
            if acc > best["acc"]:
                best = {"score": score, "tau": tau, "acc": acc, "dice": dice,
                        "gate_mean": float(gates.mean()), "epoch": "swa"}
                torch.save({"model": swa_model.module.state_dict(), "encoder": cfg.encoder,
                            "use_tabular": cfg.use_tabular, "img_size": cfg.img_size,
                            "fold": fold, "best": best}, ckpt_path)

        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"fold{fold}_history.csv"), index=False)
        if best["acc"] > ceiling + 1e-4 or attempt == cfg.max_retries:
            break
        print(f"[ordfused/{cfg.encoder}] fold{fold} attempt {attempt} collapsed "
              f"(best {best['score']:.4f} <= ceiling {ceiling:.4f}) -- retrying", flush=True)

    print(f"[ordfused/{cfg.encoder}] fold{fold} BEST: {best}  -> {ckpt_path}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="tf_efficientnetv2_s",
                    help="timm backbone (default: the strongest CNN baseline here)")
    ap.add_argument("--fold", default="all", help="fold index or 'all'")
    ap.add_argument("--no-tabular", action="store_true", help="image-only CORN ablation")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override physical batch size; default is 16 for plain-timm "
                         "CNNs/EVA-X, or a smaller size + gradient accumulation from the "
                         "foundation encoder registry (RAD-DINO/CheXFound need much less "
                         "per-step memory at their native resolution)")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation steps; only used together with an "
                         "explicit --batch-size (registry defaults set this automatically)")
    ap.add_argument("--img-size", type=int, default=None,
                    help="override input resolution; default is 224 for plain-timm "
                         "CNNs, or the encoder's own native size from the foundation "
                         "encoder registry (e.g. 518 for rad_dino, 512 for chexfound)")
    ap.add_argument("--mixup-alpha", type=float, default=0.2,
                    help="Beta(a,a) mixup strength; only active for foundation-model "
                         "encoders (0 disables)")
    ap.add_argument("--swa-frac", type=float, default=0.25,
                    help="fraction of epochs (from the end) to run stochastic weight "
                         "averaging over; 0 disables")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.3)
    ap.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank for foundation-encoder adapters. Default matches every "
                         "prior run (16) for comparability -- literature on small medical-"
                         "imaging datasets (444 images here) suggests rank as low as 2-4 may "
                         "generalize better; not yet ablated, override to try it")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
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
    tag = args.encoder if not args.no_tabular else f"{args.encoder}_notab"
    print(f"OrdFused encoder={args.encoder} tabular={not args.no_tabular} "
          f"device={device} tab_dim={TAB_DIM} fold_arg={args.fold}", flush=True)

    cfg = CFG(args)
    out_dir = os.path.join(args.out_dir, f"ordfused_{tag}")
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
        print(f"[ordfused/{tag}] CV mean score={summary['score'].mean():.4f} "
              f"+/- {summary['score'].std():.4f}  (mean gate={summary['gate_mean'].mean():.3f})", flush=True)

    with open(os.path.join(out_dir, f"results_fold_{args.fold}.json"), "w") as fh:
        json.dump({str(k): v for k, v in results.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
