"""Shared data loading, preprocessing, and metric code for Task 2 (TB/Normal
binary classification from CXR).

Ported from Task1_0615/Task1/Code/common.py -- preprocess_image/find_file/
sitk_read/set_seed are generic and reused verbatim. Task1's `our_id`/`cavity`
column names and Dice-blended challenge_score are replaced with Task2's
`new_id`/`TB/Normal` schema and plain binary accuracy/AUC metrics (no
segmentation component in Task2).
"""
import glob
import os
import random

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ID_COL = "new_id"
LABEL_COL = "TB/Normal"
LABEL_MAP = {"normal": 0, "tb": 1}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Image loading (SimpleITK-first, matches Task1's approach for geometry-safe
# loading; Task2's images are flat PNGs so this is mostly a thin, tested
# wrapper rather than a hard requirement, but keeping it consistent avoids a
# second image-loading code path in the same repo).
# ---------------------------------------------------------------------------

def find_file(d, _id, ext=".png"):
    for cand in [
        os.path.join(d, f"{_id}{ext}"),
        os.path.join(d, f"{_id}.png"),
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


def preprocess_image(arr, cfg, mask_arr=None):
    """Percentile-clip -> normalize -> per-channel-mode processing -> resize
    -> mean/std normalize. Config-driven off cfg.img_size/cfg.norm_mean/
    cfg.norm_std exactly like Task1, so every backbone (densenet121 224px
    ImageNet norm, rad_dino 518px own norm, chexfound_vitl16 512px own norm)
    shares this one function via encoders.ENCODER_FAMILIES lookups in
    dataset.py.

    `arr` is the raw (or lung-cropped, see lung_crop.py/preprocess.py)
    grayscale image -- all per-channel processing happens here, on the fly,
    every call (matches Task1's common.py: cheap CPU filters aren't worth
    precomputing/caching to disk, only the lung-crop's GPU segmenter pass is).

    cfg.channel_mode controls what the 3 channels actually hold:
      - "duplicate" (default, matches every checkpoint trained before
        2026-07-31): one processed view (CLAHE applied if cfg.use_clahe,
        bone-suppression applied if cfg.use_bone_suppression), the SAME
        result copied into all 3 channels. Kept as the default so existing
        checkpoints' expected input distribution doesn't silently change.
      - "multi_view" (added 2026-07-31): 3 GENUINELY DIFFERENT views instead
        of one view tripled -- channel 0 = original (percentile-clipped
        only, no CLAHE/bone-suppress), channel 1 = CLAHE-enhanced, channel 2
        = bone-suppressed (applied ON TOP of the CLAHE'd view, matching
        bone_suppression.py's own documented assumption that its input is
        already CLAHE'd). Motivated by "Chest X-ray Bone Suppression for
        Improving Classification of Tuberculosis-Consistent Findings"
        (PMC8151767/arXiv 2104.04518) -- bone suppression alone gave 5-11
        AUC points on these EXACT Shenzhen/Montgomery benchmarks, and other
        work reports CLAHE + bone-suppression together improving
        discriminative ability further. Every checkpoint using this mode
        MUST be scored with the same mode at predict time (same class of
        train/predict consistency requirement as --mask-dir/--bone-suppress
        already have -- see predict_task2.py's own warnings on that).
      cfg.use_clahe/cfg.use_bone_suppression are ignored in "multi_view"
      mode -- both views are always computed, since the whole point is
      giving the model all three simultaneously rather than choosing one.

    `mask_arr`: optional (H, W) bool/float lung mask, same resolution as
    `arr` (i.e. already cropped to match, if cfg used --lung-crop). When
    given, replaces the 3rd of the 3 stacked channels (whatever that
    channel would otherwise hold -- a duplicate in "duplicate" mode, the
    bone-suppressed view in "multi_view" mode) instead of adding a 4th
    channel -- keeps every backbone's pretrained 3-channel first-conv
    weights valid. Normalized independently of the real image channels (its
    own 0.5/0.5 mean/std, mapping {0,1}->{-1,1}) since it's not photographic
    content and using the image channels' mean/std on it wouldn't be
    meaningful.
    """
    lo, hi = np.percentile(arr, cfg.clip_lo), np.percentile(arr, cfg.clip_hi)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    u8 = (arr * 255).astype(np.uint8)

    channel_mode = getattr(cfg, "channel_mode", "duplicate")

    if channel_mode == "multi_view":
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        u8_clahe = clahe.apply(u8)
        from bone_suppression import suppress_bones
        u8_bone = suppress_bones(u8_clahe)

        resize = lambda a: cv2.resize(a, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
        x0 = resize(u8).astype(np.float32) / 255.0        # channel 0: original
        x1 = resize(u8_clahe).astype(np.float32) / 255.0  # channel 1: CLAHE
        x2 = resize(u8_bone).astype(np.float32) / 255.0   # channel 2: bone-suppressed
        x = np.stack([x0, x1, x2], 0)
    else:
        if cfg.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
            u8 = clahe.apply(u8)
        if getattr(cfg, "use_bone_suppression", False):
            from bone_suppression import suppress_bones
            u8 = suppress_bones(u8)
        u8 = cv2.resize(u8, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
        x = u8.astype(np.float32) / 255.0
        x = np.stack([x, x, x], 0)

    mean = np.array(cfg.norm_mean).reshape(3, 1, 1)
    std = np.array(cfg.norm_std).reshape(3, 1, 1)
    x = (x - mean) / std

    if mask_arr is not None:
        m = cv2.resize(mask_arr.astype(np.float32), (cfg.img_size, cfg.img_size),
                        interpolation=cv2.INTER_NEAREST)
        x[2] = (m - 0.5) / 0.5

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Fold splits. Two flavors: plain stratified (secondary/sanity metric,
# comparable to the baseline notebook's reported number) and
# leave-one-modality-out (the primary metric per the plan -- Modality_DICOM
# is a real domain signal already present in the data, so holding it out
# is a genuine proxy for "how does this generalize to an unseen acquisition
# domain" unlike a random split).
# ---------------------------------------------------------------------------

def make_stratified_folds(df, n_folds=5, seed=42):
    df = df.copy()
    y = df[LABEL_COL].astype(str).str.strip().str.lower().map(LABEL_MAP)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, y)):
        df.loc[df.index[val_idx], "fold"] = f
    return df


def make_leave_one_modality_out_folds(df, modality_col="Modality_DICOM"):
    """Each distinct modality value becomes one held-out fold; every other
    modality is that fold's training set. Returns df with a `fold` column
    where fold value == index into the sorted modality list (so callers can
    map fold -> modality name via `sorted(df[modality_col].unique())`)."""
    df = df.copy()
    modalities = sorted(df[modality_col].unique())
    mod_to_fold = {m: i for i, m in enumerate(modalities)}
    df["fold"] = df[modality_col].map(mod_to_fold)
    return df, modalities


# ---------------------------------------------------------------------------
# Metrics -- plain binary accuracy/AUC/F1 (no Dice/segmentation component in
# Task2, unlike Task1's 0.7*Acc+0.3*Dice challenge_score).
# ---------------------------------------------------------------------------

def detection_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def sweep_tau_accuracy(y_true, prob, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 181)
    best = max((detection_accuracy(y_true, (prob >= t).astype(int)), t) for t in grid)
    return float(best[1]), float(best[0])


def safe_auc(y_true, prob):
    """roc_auc_score requires both classes present; a leave-one-modality-out
    fold can be single-class (e.g. a modality only ever seen on TB-positive
    cases) -- return NaN rather than raising, so callers can skip/report it."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def safe_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, zero_division=0))
