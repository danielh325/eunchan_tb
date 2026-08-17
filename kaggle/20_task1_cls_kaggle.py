#!/usr/bin/env python
"""STEP 2 -- Kaggle GPU notebook (T4 x2). Task1 cavity PRESENCE classifier.

WHY THIS EXISTS
---------------
Current OOF accuracy is 0.8446 (test 0.8468) with AUC ~0.90, from 4-6 members
each fully fine-tuned on 444 images. Re-scoring the saved OOF files in
Code/oof/ shows two things:

  * The threshold is not the problem. Every sensible combination of the six
    saved members lands at 0.84-0.85 OOF accuracy, and sweeping tau moves it by
    under a point. Best 5-member average = 0.8491 vs the shipped 4-member
    0.8423 -- three images, i.e. noise.
  * The ceiling is AUC. At AUC 0.90 and 44% prevalence there is essentially no
    accuracy left to extract by reweighting or rethresholding what is already
    trained. Accuracy only moves if the representation gets better.

train_densenet201.log shows why the representation is stuck: training loss
reaches 0.004 by epoch 21 while validation accuracy plateaus around 0.80-0.85.
That is a 444-image full fine-tune memorising its training set.

The largest untapped asset in this repo is Task2: 7757 labelled TB/Normal CXRs
from the same cohort and the same four modalities (CR/DX/XC/XA), sitting
unused by Task1. This script pretrains the backbone on that (stage `pre`), then
fine-tunes on the 444 cavity labels (stage `ft`). Same backbone family, ~17x
more in-domain supervision before it ever sees a cavity label.

Two smaller fixes also included:
  * Train prevalence is 44.37% but test prevalence is 52.25% (measured from the
    GT masks). A threshold tuned on OOF is tuned for the wrong prior. `ft`
    reports both the raw OOF-optimal tau and a prior-shift-corrected one.
  * EMA + mixup + a short schedule, because the failure mode is memorisation.

KAGGLE SETUP
------------
  Accelerator: GPU T4 x2, Internet ON (timm weights).
  !pip -q install timm

  # ~50 min per backbone on one T4; run two backbones on the two cards at once
  !python 20_task1_cls_kaggle.py --stage pre --backbone convnext_small --gpu 0 &
  !python 20_task1_cls_kaggle.py --stage pre --backbone densenet201    --gpu 1
  !python 20_task1_cls_kaggle.py --stage ft  --backbone convnext_small --gpu 0 &
  !python 20_task1_cls_kaggle.py --stage ft  --backbone densenet201    --gpu 1

  # controls: identical fine-tune, ImageNet init only -- this is the ablation
  # that tells you whether the Task2 pretraining actually bought anything
  !python 20_task1_cls_kaggle.py --stage ft --backbone convnext_small --no-pretrain --gpu 0
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

PACK = os.environ.get("MMTB_PACK", "/kaggle/input/mmtb-2026-pack")
WORK = os.environ.get("MMTB_WORK", "/kaggle/working")
GRADE2ORD = {"none": 0, "small": 1, "medium": 2, "large": 3}
MEAN = np.array([.485, .456, .406]).reshape(3, 1, 1)
STD = np.array([.229, .224, .225]).reshape(3, 1, 1)


def prep(u8, size, train):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    u8 = clahe.apply(u8)
    u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    if train:
        if np.random.rand() < 0.5:
            u8 = u8[:, ::-1].copy()
        M = cv2.getRotationMatrix2D((size / 2, size / 2),
                                    np.random.uniform(-12, 12),
                                    np.random.uniform(.88, 1.12))
        M[0, 2] += np.random.uniform(-.06, .06) * size
        M[1, 2] += np.random.uniform(-.06, .06) * size
        u8 = cv2.warpAffine(u8, M, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        u8 = np.clip(u8.astype(np.float32) * np.random.uniform(.85, 1.15)
                     + np.random.uniform(-18, 18), 0, 255)
        u8 = 255.0 * (np.asarray(u8) / 255.0) ** np.random.uniform(.85, 1.15)
    x = np.float32(u8) / 255.0
    x = np.stack([x, x, x], 0)
    return ((x - MEAN) / STD).astype(np.float32)


class T2DS(Dataset):
    """Task2 TB/Normal, used only as pretraining."""

    def __init__(self, df, size, train, split):
        self.df, self.size, self.train, self.split = df.reset_index(drop=True), size, train, split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        u8 = cv2.imread(f"{PACK}/task2/{self.split}/{r.new_id}.jpg", cv2.IMREAD_GRAYSCALE)
        y = 1.0 if r["TB/Normal"] == "TB" else 0.0
        return torch.from_numpy(prep(u8, self.size, self.train)), torch.tensor(y)


class T1DS(Dataset):
    def __init__(self, ids, labels, size, train):
        self.ids, self.labels, self.size, self.train = list(ids), list(labels), size, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        u8 = cv2.imread(f"{PACK}/task1/img/{self.ids[i]}.png", cv2.IMREAD_GRAYSCALE)
        return (torch.from_numpy(prep(u8, self.size, self.train)),
                torch.tensor(float(self.labels[i])))


class Net(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.enc = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(nn.LayerNorm(self.enc.num_features),
                                  nn.Dropout(0.3), nn.Linear(self.enc.num_features, 1))

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(1)


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model):
        model.load_state_dict({k: v.to(dtype=p.dtype) for (k, v), p
                               in zip(self.shadow.items(), model.state_dict().values())})


def run_epoch(model, dl, opt, scaler, sched, dev, ema, mixup):
    model.train()
    tot = 0.0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if mixup > 0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup, mixup)
            idx = torch.randperm(x.size(0), device=dev)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)
        tot += loss.item()
    return tot / max(len(dl), 1)


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    out = []
    with torch.autocast("cuda", dtype=torch.float16):
        for x, _ in dl:
            x = x.to(dev)
            p = torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, [3])))
            out.append((p / 2).float().cpu().numpy())
    return np.concatenate(out)


def stage_pre(backbone, size, epochs, bs, lr, gpu):
    dev = torch.device(f"cuda:{gpu}")
    tr = pd.read_csv(f"{PACK}/task2/train.csv")
    va = pd.read_csv(f"{PACK}/task2/test.csv")
    model = Net(backbone, True).to(dev)
    ema = EMA(model)
    dl_tr = DataLoader(T2DS(tr, size, True, "train"), batch_size=bs, shuffle=True,
                       num_workers=4, drop_last=True, pin_memory=True)
    dl_va = DataLoader(T2DS(va, size, False, "test"), batch_size=bs, num_workers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * len(dl_tr), pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler()
    yv = (va["TB/Normal"] == "TB").astype(int).values
    for ep in range(epochs):
        t0 = time.time()
        loss = run_epoch(model, dl_tr, opt, scaler, sched, dev, ema, mixup=0.4)
        p = predict(model, dl_va, dev)
        print(f"[pre/{backbone}] ep{ep} loss={loss:.4f} auc={roc_auc_score(yv,p):.5f} "
              f"acc={( (p>=.5)==yv ).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    ema.copy_to(model)
    torch.save(model.enc.state_dict(), f"{WORK}/pre_{backbone}.pt")
    print(f"[pre/{backbone}] saved encoder -> {WORK}/pre_{backbone}.pt")


def stage_ft(backbone, size, epochs, bs, lr, folds, seeds, gpu, use_pre):
    dev = torch.device(f"cuda:{gpu}")
    meta = pd.read_csv(f"{PACK}/task1/meta.csv")
    tr = meta[meta.split == "train"].reset_index(drop=True)
    te = meta[meta.split == "test"].reset_index(drop=True)
    ytr = (tr.cavity != "none").astype(int).values
    yte = (te.cavity != "none").astype(int).values
    strat = tr.cavity.map(GRADE2ORD).values

    oof = np.zeros(len(tr))
    testp = np.zeros(len(te))
    for seed in range(seeds):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42 + seed)
        for fold, (ti, vi) in enumerate(skf.split(tr, strat)):
            model = Net(backbone, True).to(dev)
            if use_pre and os.path.exists(f"{WORK}/pre_{backbone}.pt"):
                model.enc.load_state_dict(torch.load(f"{WORK}/pre_{backbone}.pt",
                                                     map_location=dev))
                tag = "task2pre"
            else:
                tag = "imagenet"
            ema = EMA(model)
            dl_tr = DataLoader(T1DS(tr.our_id[ti], ytr[ti], size, True), batch_size=bs,
                               shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
            dl_va = DataLoader(T1DS(tr.our_id[vi], ytr[vi], size, False), batch_size=bs,
                               num_workers=4)
            dl_te = DataLoader(T1DS(te.our_id, yte, size, False), batch_size=bs, num_workers=4)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=lr, total_steps=epochs * max(len(dl_tr), 1), pct_start=0.25)
            scaler = torch.cuda.amp.GradScaler()
            for ep in range(epochs):
                run_epoch(model, dl_tr, opt, scaler, sched, dev, ema, mixup=0.4)
            ema.copy_to(model)
            pv = predict(model, dl_va, dev)
            oof[vi] += pv / seeds
            testp += predict(model, dl_te, dev) / (folds * seeds)
            print(f"[ft/{backbone}/{tag}] s{seed} f{fold} "
                  f"valauc={roc_auc_score(ytr[vi],pv):.4f}", flush=True)

    grid = np.linspace(0.05, 0.95, 181)
    tau = max(((ytr == (oof >= t)).mean(), t) for t in grid)[1]
    # prior-shift correction: train prevalence 0.4437, test prevalence 0.5225
    pi_s, pi_t = ytr.mean(), 0.5225
    w = (pi_t / pi_s) * oof / ((pi_t / pi_s) * oof + ((1 - pi_t) / (1 - pi_s)) * (1 - oof))
    tau_pc = max(((ytr == (w >= t)).mean(), t) for t in grid)[1]
    print(f"\n[ft/{backbone}] OOF auc={roc_auc_score(ytr,oof):.4f} "
          f"acc@0.5={((oof>=.5)==ytr).mean():.4f} acc@tau*={((oof>=tau)==ytr).mean():.4f} tau*={tau:.3f}")
    print(f"[ft/{backbone}] TEST auc={roc_auc_score(yte,testp):.4f} "
          f"acc@tau*={((testp>=tau)==yte).mean():.4f}  "
          f"(prior-corrected tau={tau_pc:.3f})")
    np.savez(f"{WORK}/clsprob_{backbone}{'' if use_pre else '_nopre'}.npz",
             oof=oof, test=testp, tau=tau,
             train_ids=tr.our_id.values, test_ids=te.our_id.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pre", "ft"])
    ap.add_argument("--backbone", default="convnext_small")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=0, help="0 = stage default")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = stage default")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-pretrain", action="store_true")
    a = ap.parse_args()
    if a.stage == "pre":
        stage_pre(a.backbone, a.size, a.epochs or 6, a.bs, a.lr or 2e-4, a.gpu)
    else:
        stage_ft(a.backbone, a.size, a.epochs or 12, a.bs, a.lr or 5e-5,
                 a.folds, a.seeds, a.gpu, not a.no_pretrain)


if __name__ == "__main__":
    main()
