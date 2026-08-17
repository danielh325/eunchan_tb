#!/usr/bin/env python
"""STEP 1 -- Kaggle GPU notebook (T4 x2). Task1 cavity SEGMENTATION, rebuilt.

WHY THIS EXISTS (read before running)
-------------------------------------
The current pipeline scores 0.229 test Dice (best foundation decoder) / 0.306
(nnU-Net). Published TB-lesion segmentation on chest radiographs lands at
0.55-0.65 Dice, and the two papers that get there both do the same two things
this project does NOT:

  1. They segment on the LUNG-CROPPED image, not the whole radiograph.
     preprocess_lung_crop.py already exists here but was only ever wired into
     the *classifier*; the segmentation path still sees full frames, where a
     median cavity is 1.07% of the pixels (measured on this dataset).
  2. They run at 256x256, not 512/518. Rajaraman et al. swept resolution
     directly for this exact task: 256 -> 0.654 Dice, 512 -> 0.649,
     768 -> 0.622, 1024 -> 0.506. Higher resolution actively hurts at this
     sample size. This project runs its three decoders at 518/512/224.

So this script trains plain ImageNet U-Nets on lung crops at 256 -- deliberately
"less sophisticated" than the frozen-foundation-decoder stack, because the
evidence says the frozen ViT decoders are being asked to do fine-grained dense
prediction at a resolution and framing where they lose to a ResNet-34 U-Net.

It also fixes the ground-truth question that task1_paper.tex and
train_segmentation.py's docstring disagree about. Verified locally on all 555
cases: Data/train/CXR_label and Data/CXR_label store masks at NATIVE
resolution with shapes matching their images exactly. No transpose, no
resampling. The nnUNet_raw/labelsTr store (fixed 512x512) is the only thing
that ever needed a transpose, and it should simply not be used as GT.

Scoring here is the ORGANIZERS' metric, reimplemented from
Code/Task1_evaluation_code.ipynb, evaluated at NATIVE resolution:
    accuracy = (mask non-empty) == (gt non-empty), per case
    dice     = NaN when both empty (excluded from the mean!), 0 when exactly
               one is empty, else 2|A&B|/(|A|+|B|)
    score    = 0.7*mean_accuracy + 0.3*nanmean(dice)
The NaN rule matters enormously and is easy to miss: a false-positive mask on a
cavity-negative case does not just cost accuracy, it also injects a 0.0 into
the Dice mean that would otherwise have been excluded. Optimising Dice in
isolation therefore gives the wrong answer -- 30_task1_fuse_score.py does the
joint optimisation.

KAGGLE SETUP
------------
  Accelerator: GPU T4 x2      Internet: ON (needed for pip + torchxrayvision
  weights on the first run only)
  Add your 'mmtb-2026-pack' dataset (from 00_pack_local.py).

  !pip -q install segmentation-models-pytorch torchxrayvision
  !python 10_task1_seg_kaggle.py --stage lungmask
  !python 10_task1_seg_kaggle.py --stage train --encoder resnet34 --gpu 0 &
  !python 10_task1_seg_kaggle.py --stage train --encoder timm-efficientnet-b0 --gpu 1
  !python 10_task1_seg_kaggle.py --stage train --encoder se_resnext50_32x4d --gpu 0

Two encoders run concurrently, one per T4. That is the right way to use the
2xT4 here -- DDP/DataParallel would only split a batch of 16 at 256px across
two cards and spend more time on sync than on compute. Whole sweep is well
under 2h of the 12h session.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

PACK = os.environ.get("MMTB_PACK", "/kaggle/input/mmtb-2026-pack")
WORK = os.environ.get("MMTB_WORK", "/kaggle/working")
SIZE = 256
GRADE2ORD = {"none": 0, "small": 1, "medium": 2, "large": 3}


# ---------------------------------------------------------------------------
# Stage 1: lung bounding boxes (once, cached to lungbox.json)
# ---------------------------------------------------------------------------

def stage_lungmask(pad=0.04):
    """torchxrayvision's PSPNet lung segmenter -> a padded bbox per case.

    Falls back to a centred 90% box if xrv is unavailable (no internet), so the
    rest of the pipeline still runs -- but the crop is the whole point here, so
    prefer to fix the internet toggle rather than accept the fallback.
    """
    meta = pd.read_csv(f"{PACK}/task1/meta.csv")
    out = {}
    try:
        import torchxrayvision as xrv
        seg = xrv.baseline_models.chestx_det.PSPNet().eval().cuda()
        targets = seg.targets
        li = [i for i, t in enumerate(targets) if "Lung" in t]
        print(f"[lungmask] xrv PSPNet ok, lung channels={li}")
        use_xrv = True
    except Exception as e:                                    # noqa: BLE001
        print(f"[lungmask] !! torchxrayvision unavailable ({e}); using fallback box")
        use_xrv = False

    for n, _id in enumerate(meta.our_id, 1):
        if not use_xrv:
            out[str(_id)] = [int(0.05 * SIZE * 4), int(0.95 * SIZE * 4),
                             int(0.05 * SIZE * 4), int(0.95 * SIZE * 4)]
            continue
        img = cv2.imread(f"{PACK}/task1/img/{_id}.png", cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x / 255.0) * 2048.0 - 1024.0                     # xrv expects [-1024,1024]
        t = torch.from_numpy(x)[None, None].cuda()
        with torch.no_grad():
            pred = seg(t)[0, li].sigmoid().max(0).values.cpu().numpy()
        m = (pred > 0.5).astype(np.uint8)
        if m.sum() < 100:
            y0, y1, x0, x1 = 26, 486, 26, 486
        else:
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            dy, dx = int(pad * 512), int(pad * 512)
            y0, y1 = max(0, y0 - dy), min(512, y1 + dy)
            x0, x1 = max(0, x0 - dx), min(512, x1 + dx)
        # store in the 1024-pack coordinate frame
        s = 1024 / 512
        out[str(_id)] = [int(y0 * s), int(y1 * s), int(x0 * s), int(x1 * s)]
        if n % 50 == 0:
            print(f"  [lungmask] {n}/{len(meta)}", flush=True)

    with open(f"{WORK}/lungbox.json", "w") as f:
        json.dump(out, f)
    # sanity: how much of each GT mask survives the crop?
    kept = []
    for _id in meta.our_id:
        gm = cv2.imread(f"{PACK}/task1/mask/{_id}.png", cv2.IMREAD_GRAYSCALE)
        if gm is None or gm.sum() == 0:
            continue
        y0, y1, x0, x1 = out[str(_id)]
        kept.append((gm[y0:y1, x0:x1] > 0).sum() / max((gm > 0).sum(), 1))
    kept = np.array(kept)
    print(f"[lungmask] GT retained by crop: mean={kept.mean():.4f} "
          f"min={kept.min():.4f} frac<0.95: {(kept < 0.95).mean():.3f}")
    print("  (if a lot of mask is being cut off, raise --pad)")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SegDS(Dataset):
    def __init__(self, ids, boxes, train):
        self.ids, self.boxes, self.train = list(ids), boxes, train

    def __len__(self):
        return len(self.ids)

    def _load(self, _id):
        img = cv2.imread(f"{PACK}/task1/img/{_id}.png", cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(f"{PACK}/task1/mask/{_id}.png", cv2.IMREAD_GRAYSCALE)
        y0, y1, x0, x1 = self.boxes[str(_id)]
        img, msk = img[y0:y1, x0:x1], msk[y0:y1, x0:x1]
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        msk = cv2.resize((msk > 0).astype(np.uint8), (SIZE, SIZE),
                         interpolation=cv2.INTER_NEAREST)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img), msk.astype(np.float32)

    def __getitem__(self, i):
        _id = self.ids[i]
        img, msk = self._load(_id)
        if self.train:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            ang = np.random.uniform(-12, 12)
            sc = np.random.uniform(0.88, 1.12)
            tx, ty = np.random.uniform(-.05, .05, 2) * SIZE
            M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, sc)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            msk = cv2.warpAffine(msk, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            img = np.clip(img.astype(np.float32) * np.random.uniform(.85, 1.15)
                          + np.random.uniform(-18, 18), 0, 255)
        x = np.float32(img) / 255.0
        x = np.stack([x, x, x], 0)
        x = (x - np.array([.485, .456, .406]).reshape(3, 1, 1)) / \
            np.array([.229, .224, .225]).reshape(3, 1, 1)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(msk[None])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalTverskyBCE(nn.Module):
    """Same loss family the current segmentation stack uses (alpha=0.3,
    beta=0.7, gamma=1.33 + class-weighted BCE), kept identical so this
    experiment isolates crop+resolution+architecture, not the objective."""

    def __init__(self, pos_weight, alpha=0.3, beta=0.7, gamma=1.33):
        super().__init__()
        self.a, self.b, self.g = alpha, beta, gamma
        self.register_buffer("pw", torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)
        p = torch.sigmoid(logits)
        tp = (p * target).sum((1, 2, 3))
        fp = (p * (1 - target)).sum((1, 2, 3))
        fn = ((1 - p) * target).sum((1, 2, 3))
        tv = (tp + 1e-5) / (tp + self.a * fp + self.b * fn + 1e-5)
        return bce + ((1 - tv) ** self.g).mean()


# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------

def dice_np(p, g):
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return np.nan
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * np.logical_and(p, g).sum() / (ps + gs)


def stage_train(encoder, epochs, folds, gpu, lr, bs, seed):
    import segmentation_models_pytorch as smp
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(f"cuda:{gpu}")
    boxes = json.load(open(f"{WORK}/lungbox.json"))
    meta = pd.read_csv(f"{PACK}/task1/meta.csv")
    tr = meta[meta.split == "train"].reset_index(drop=True)
    te = meta[meta.split == "test"].reset_index(drop=True)

    # pos_weight from the actual cropped/resized training masks
    pos = neg = 0
    ds_all = SegDS(tr.our_id, boxes, train=False)
    for i in range(len(ds_all)):
        _, m = ds_all._load(ds_all.ids[i])
        pos += int(m.sum())
        neg += int(m.size - m.sum())
    pw = min(neg / max(pos, 1), 100.0)
    print(f"[{encoder}] pos_weight={pw:.2f}", flush=True)

    y = tr.cavity.map(GRADE2ORD).values
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    oof = np.zeros((len(tr), SIZE, SIZE), np.float16)
    test_acc = np.zeros((len(te), SIZE, SIZE), np.float32)

    for fold, (ti, vi) in enumerate(skf.split(tr, y)):
        model = smp.Unet(encoder, encoder_weights="imagenet", in_channels=3,
                         classes=1, decoder_attention_type="scse").to(dev)
        crit = FocalTverskyBCE(pw).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        dl_tr = DataLoader(SegDS(tr.our_id[ti], boxes, True), batch_size=bs,
                           shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
        dl_va = DataLoader(SegDS(tr.our_id[vi], boxes, False), batch_size=bs,
                           shuffle=False, num_workers=2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * max(len(dl_tr), 1), pct_start=0.2)
        scaler = torch.cuda.amp.GradScaler()          # T4 is fp16-only, no bf16
        best, best_state = -1, None
        for ep in range(epochs):
            model.train()
            t0, tot = time.time(), 0.0
            for x, m in dl_tr:
                x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = crit(model(x), m)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            model.eval()
            P, G = [], []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for x, m in dl_va:
                    p = torch.sigmoid(model(x.to(dev))).float().cpu().numpy()[:, 0]
                    P.append(p)
                    G.append(m.numpy()[:, 0])
            P, G = np.concatenate(P), np.concatenate(G)
            # selection metric: mean Dice over cases with any GT -- checkpoint
            # selection should not be entangled with the emptiness decision,
            # which 30_task1_fuse_score.py optimises separately.
            d = np.nanmean([dice_np(P[i] > 0.5, G[i] > 0.5)
                            for i in range(len(P)) if G[i].sum() > 0])
            if d > best:
                best, best_state = d, {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}
            print(f"[{encoder}] f{fold} ep{ep:02d} loss={tot/max(len(dl_tr),1):.4f} "
                  f"devdice={d:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)

        model.load_state_dict(best_state)
        model.eval()

        def infer(ids):
            dl = DataLoader(SegDS(ids, boxes, False), batch_size=bs, num_workers=2)
            out = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for x, _ in dl:
                    x = x.to(dev)
                    p = torch.sigmoid(model(x))
                    p = p + torch.flip(torch.sigmoid(model(torch.flip(x, [3]))), [3])
                    out.append((p / 2).float().cpu().numpy()[:, 0])
            return np.concatenate(out)

        oof[vi] = infer(tr.our_id[vi]).astype(np.float16)
        test_acc += infer(te.our_id)
        torch.save(best_state, f"{WORK}/seg_{encoder}_f{fold}.pt")

    np.savez_compressed(f"{WORK}/segprob_{encoder}.npz",
                        oof=oof, test=(test_acc / folds).astype(np.float16),
                        train_ids=tr.our_id.values, test_ids=te.our_id.values)
    print(f"[{encoder}] wrote {WORK}/segprob_{encoder}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["lungmask", "train"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.04)
    a = ap.parse_args()
    if a.stage == "lungmask":
        stage_lungmask(a.pad)
    else:
        stage_train(a.encoder, a.epochs, a.folds, a.gpu, a.lr, a.bs, a.seed)


if __name__ == "__main__":
    main()
