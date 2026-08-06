"""CLI training entrypoint for GlobalLocalTBClassifier (dual-stream global+
local fusion, see model.py). Structurally mirrors train_task2.py -- same
leave-one-modality-out default, same domain-aware mixup, same gradient
accumulation pattern -- but consumes GlobalLocalTBDataset's (x_global,
x_local, y, domain) tuples and both encoders' batch/accum settings must be
reconciled (uses the smaller of the two ENCODER_FAMILIES batch sizes, since
the pair now shares one physical batch across two backbones at once).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (LABEL_COL, LABEL_MAP, make_leave_one_modality_out_folds,
                     make_stratified_folds, safe_auc, set_seed, sweep_tau_accuracy)
from dataset import GlobalLocalCfg, GlobalLocalTBDataset
from domain_gen import domain_aware_mixup
from encoders import ENCODER_FAMILIES
from model import GlobalLocalTBClassifier


def build_optimizer(model, lr, weight_decay=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_one_fold(args, train_df, val_df, fold_tag, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_fam = ENCODER_FAMILIES.get(args.global_encoder, {})
    local_fam = ENCODER_FAMILIES.get(args.local_encoder, {})
    batch_size = args.batch_size or min(global_fam.get("batch_size", 16), local_fam.get("batch_size", 16))
    accum_steps = args.accum_steps or max(global_fam.get("accum_steps", 1), local_fam.get("accum_steps", 1))

    cfg = GlobalLocalCfg(args.global_encoder, args.local_encoder,
                          global_image_dir=args.global_image_dir, local_image_dir=args.local_image_dir,
                          use_bone_suppression=args.bone_suppress)
    train_ds = GlobalLocalTBDataset(train_df, cfg, train=True, aug_strength=args.aug_strength)
    val_ds = GlobalLocalTBDataset(val_df, cfg, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = GlobalLocalTBClassifier(args.global_encoder, args.local_encoder, pretrained=True).to(device)
    optimizer = build_optimizer(model, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        for i, (xg, xl, y, dom) in enumerate(train_loader):
            xg = xg.to(device, non_blocking=True)
            xl = xl.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # domain-aware mixup applied identically (same lam, same partner
            # permutation) to both streams so a mixed sample's global and
            # local views stay consistent with each other and with the
            # mixed label -- computed once on xg to get the permutation/lam,
            # then re-applied to xl.
            if args.domain_mixup_alpha > 0:
                B = xg.size(0)
                lam = float(np.random.beta(args.domain_mixup_alpha, args.domain_mixup_alpha))
                domains = np.asarray(dom)
                perm = np.empty(B, dtype=np.int64)
                for j in range(B):
                    candidates = np.where(domains != domains[j])[0]
                    perm[j] = np.random.choice(candidates) if len(candidates) > 0 else np.random.choice(B)
                perm_t = torch.from_numpy(perm).to(device)
                xg = lam * xg + (1 - lam) * xg[perm_t]
                xl = lam * xl + (1 - lam) * xl[perm_t]
                y = lam * y + (1 - lam) * y[perm_t]

            logits = model(xg, xl)
            loss = criterion(logits, y) / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * accum_steps * xg.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xg, xl, y, _dom in val_loader:
                xg = xg.to(device, non_blocking=True)
                xl = xl.to(device, non_blocking=True)
                p = torch.sigmoid(model(xg, xl)).cpu().numpy()
                probs.append(p)
                labels.append(y.numpy())
        probs = np.concatenate(probs) if probs else np.array([])
        labels = np.concatenate(labels) if labels else np.array([])
        tau, val_acc = sweep_tau_accuracy(labels, probs) if len(probs) else (0.5, float("nan"))
        val_auc = safe_auc(labels, probs) if len(probs) else float("nan")
        tag = f"{args.global_encoder}+{args.local_encoder}"
        print(f"[{tag}][{fold_tag}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} tau={tau:.2f}",
              flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = dict(
                state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                encoder_name=f"dual_{args.global_encoder}_{args.local_encoder}",
                global_encoder=args.global_encoder, local_encoder=args.local_encoder,
                global_img_size=cfg.global_cfg.img_size, local_img_size=cfg.local_cfg.img_size,
                global_norm_mean=cfg.global_cfg.norm_mean, global_norm_std=cfg.global_cfg.norm_std,
                local_norm_mean=cfg.local_cfg.norm_mean, local_norm_std=cfg.local_cfg.norm_std,
                fold_tag=fold_tag, val_acc=val_acc, val_auc=val_auc, tau=tau,
            )

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"dual_{args.global_encoder}_{args.local_encoder}_{fold_tag}.pth")
    torch.save(best_state, ckpt_path)
    print(f"saved {ckpt_path} (best val_acc={best_acc:.4f})")
    return best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-encoder", required=True, choices=list(ENCODER_FAMILIES),
                     help="backbone for the full/uncropped image stream")
    ap.add_argument("--local-encoder", required=True, choices=list(ENCODER_FAMILIES),
                     help="backbone for the lung-cropped image stream")
    ap.add_argument("--csv", default="/workspace/Data/train.csv")
    ap.add_argument("--global-image-dir", required=True,
                     help="dir of RAW uncropped {id}.png images (e.g. Data/train)")
    ap.add_argument("--local-image-dir", required=True,
                     help="dir of preprocess.py's ch0 (lung-cropped) images")
    ap.add_argument("--out-dir", default="/workspace/checkpoints_dual")
    ap.add_argument("--fold-mode", choices=["leave_one_modality_out", "stratified"],
                     default="leave_one_modality_out")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--aug-strength", choices=["none", "light", "strong"], default="strong")
    ap.add_argument("--bone-suppress", action="store_true")
    ap.add_argument("--domain-mixup-alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(args.csv)
    df = df[df[LABEL_COL].astype(str).str.strip().str.lower().isin(LABEL_MAP)].reset_index(drop=True)

    if args.fold_mode == "leave_one_modality_out":
        df, modalities = make_leave_one_modality_out_folds(df)
        fold_names = modalities
    else:
        df = make_stratified_folds(df, n_folds=args.n_folds, seed=args.seed)
        fold_names = [f"fold{i}" for i in range(args.n_folds)]

    results = []
    for f, tag in enumerate(fold_names):
        train_df = df[df["fold"] != f].reset_index(drop=True)
        val_df = df[df["fold"] == f].reset_index(drop=True)
        if len(val_df) == 0:
            print(f"skip fold {tag}: empty")
            continue
        best = run_one_fold(args, train_df, val_df, tag, args.out_dir)
        results.append(dict(fold=tag, val_acc=best["val_acc"], val_auc=best["val_auc"]))

    res_df = pd.DataFrame(results)
    print(res_df)
    print(f"mean val_acc={res_df['val_acc'].mean():.4f} mean val_auc={res_df['val_auc'].mean():.4f}")


if __name__ == "__main__":
    main()
