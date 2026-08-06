"""Shared data loading, preprocessing, model, and metric code for Task 1
(cavity detection from CXR) — used by both train.py and predict.py so the
baseline and the other CNN backbones (resnet50d / densenet201 /
tf_efficientnetv2_s) all run through one, tested code path.

Ported from Task1/ordfused_baseline.ipynb with one bug fix: the real CSV's
id column is `our_id`, not `_id`.
"""
import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import timm
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

GRADE2ORD = {"none": 0, "small": 1, "medium": 2, "large": 3}
ORD2GRADE = {v: k for k, v in GRADE2ORD.items()}

ID_COL = "our_id"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Image loading (SimpleITK-first, per the organizers' warning about
# pydicom/nibabel silently flipping images relative to masks)
# ---------------------------------------------------------------------------

def find_file(d, _id, ext):
    for cand in [
        os.path.join(d, f"{_id}{ext}"),
        os.path.join(d, f"{_id}.png"),
        os.path.join(d, f"{_id}.dcm"),
        os.path.join(d, f"{_id}.nii.gz"),
    ]:
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(d, f"{_id}.*"))
    return hits[0] if hits else None


def sitk_read(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
    return arr.astype(np.float32)


def augment_u8(u8):
    """Geometric + photometric augmentation on a resized uint8 grayscale image.
    Applied before normalization so brightness/contrast/gamma jitter operate on
    the true 0-255 range rather than on already mean/std-shifted floats.
    """
    h, w = u8.shape
    if random.random() < 0.5:
        u8 = u8[:, ::-1]

    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-0.05, 0.05) * w
    ty = random.uniform(-0.05, 0.05) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    u8 = cv2.warpAffine(u8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    contrast = random.uniform(0.85, 1.15)
    brightness = random.uniform(-15, 15)
    u8 = np.clip(u8.astype(np.float32) * contrast + brightness, 0, 255)

    gamma = random.uniform(0.85, 1.15)
    u8 = 255.0 * (u8 / 255.0) ** gamma

    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 8), size=u8.shape)
        u8 = u8 + noise

    return np.clip(u8, 0, 255).astype(np.uint8)


def _clahe_u8(arr, cfg):
    """Percentile clip + normalize + (optional) CLAHE, at native resolution --
    the shared preamble before any per-branch resize/normalize."""
    lo, hi = np.percentile(arr, cfg.clip_lo), np.percentile(arr, cfg.clip_hi)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    u8 = (arr * 255).astype(np.uint8)
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        u8 = clahe.apply(u8)
    return u8


def _resize_normalize(u8, img_size, mean, std):
    u8 = cv2.resize(u8, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = u8.astype(np.float32) / 255.0
    x = np.stack([x, x, x], 0)
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    return ((x - mean) / std).astype(np.float32)


def preprocess_image(arr, cfg, augment=False):
    u8 = _clahe_u8(arr, cfg)
    if augment:
        u8 = augment_u8(u8)
    return _resize_normalize(u8, cfg.img_size, cfg.norm_mean, cfg.norm_std)


def preprocess_image_dual(arr, cfg_a, cfg_b, augment=False):
    """Like preprocess_image, but returns two views (e.g. a CNN branch at
    224px/ImageNet stats and a ViT branch at its own resolution/stats) built
    from the SAME clip+CLAHE+augmentation pass, so a hybrid CNN+ViT model sees
    the identical flip/rotation/crop on both branches instead of independently
    randomized augmentation (which would desynchronize the two streams)."""
    u8 = _clahe_u8(arr, cfg_a)
    if augment:
        u8 = augment_u8(u8)
    x_a = _resize_normalize(u8, cfg_a.img_size, cfg_a.norm_mean, cfg_a.norm_std)
    x_b = _resize_normalize(u8, cfg_b.img_size, cfg_b.norm_mean, cfg_b.norm_std)
    return x_a, x_b


class CavityDataset(Dataset):
    def __init__(self, df, cfg, train=True):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        _id = row[ID_COL]
        ip = find_file(self.cfg.image_dir, _id, self.cfg.image_ext)
        if ip is None:
            raise FileNotFoundError(f"no image for {ID_COL}={_id} under {self.cfg.image_dir}")
        arr = sitk_read(ip)
        x = preprocess_image(arr, self.cfg, augment=self.train)
        ordl = GRADE2ORD[row.cavity]
        binl = 1 if ordl > 0 else 0
        return (
            torch.from_numpy(x),
            torch.tensor(binl, dtype=torch.float32),
            torch.tensor(ordl, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Patient-grouped, grade-stratified split
# ---------------------------------------------------------------------------

def make_folds(df, cfg, group_col=None, seed=42):
    df = df.copy()
    df["ord"] = df.cavity.map(GRADE2ORD)
    groups = df[group_col] if group_col else df[ID_COL]
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for f, (_, val_idx) in enumerate(sgkf.split(df, df["ord"], groups)):
        df.loc[df.index[val_idx], "fold"] = f
    return df


# ---------------------------------------------------------------------------
# Model — generic timm encoder + binary head (works for convnext_tiny,
# resnet50d, densenet201, tf_efficientnetv2_s, or any other timm name)
# ---------------------------------------------------------------------------

class CNNDetector(nn.Module):
    def __init__(self, name, pretrained=True, freeze=False):
        super().__init__()
        self.encoder = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat = self.encoder.num_features
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2), nn.Linear(feat, 1))

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(1)


# ---------------------------------------------------------------------------
# The challenge metric — 0.7*Accuracy + 0.3*Dice (presence-consistent proxy
# until the segmentor is wired in; see ordfused_baseline.ipynb section 8)
# ---------------------------------------------------------------------------

def detection_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def presence_dice(y_true, y_pred):
    d = np.zeros(len(y_true))
    d[(y_true == 0) & (y_pred == 0)] = 1.0
    d[(y_true == 1) & (y_pred == 1)] = 0.60
    return float(d.mean())


def challenge_score(y_true, prob, tau):
    y_pred = (prob >= tau).astype(int)
    acc = detection_accuracy(y_true, y_pred)
    dice = presence_dice(y_true, y_pred)
    return 0.7 * acc + 0.3 * dice, acc, dice


def sweep_tau(y_true, prob, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 181)
    best = max((challenge_score(y_true, prob, t)[0], t) for t in grid)
    # cast off numpy scalar types: a np.float64 tau ends up inside the
    # checkpoint's `best` dict via torch.save, which breaks weights_only=True
    # unpickling on newer PyTorch (>=2.6 default).
    return float(best[1]), float(best[0])


def sweep_tau_accuracy(y_true, prob, grid=None):
    """Like sweep_tau, but maximizes plain detection_accuracy directly instead of
    the blended 0.7*acc+0.3*presence_dice proxy. Use this (not sweep_tau) for any
    model whose pred_cavity feeds gate_segmentation.py's classifier-gates-nnUNet
    pipeline: real Dice now comes from nnUNet, fully decoupled from the
    classifier's threshold, so tuning tau against the old proxy would bias the
    threshold toward the proxy's higher reward for true negatives (1.0) vs true
    positives (0.60) rather than the accuracy-optimal point."""
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 181)
    best = max((detection_accuracy(y_true, (prob >= t).astype(int)), t) for t in grid)
    return float(best[1]), float(best[0])
