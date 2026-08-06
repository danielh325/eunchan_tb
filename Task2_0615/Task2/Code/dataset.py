"""TBDataset -- wires common.preprocess_image to per-backbone img_size/norm
stats (from encoders.ENCODER_FAMILIES, same pattern as Task1's
train_ordfused.py CFG construction) and applies domain_gen's multi-level
augmentation. Image-only per the plan's metadata decision -- Modality_DICOM
is deliberately never fed to the model (it's the shortcut, not a feature);
age/gender fusion is a stretch, not v1.
"""
import cv2
import torch
from torch.utils.data import Dataset

from common import LABEL_COL, LABEL_MAP, find_file, sitk_read, preprocess_image
from domain_gen import build_augmentation
from encoders import ENCODER_FAMILIES


class Cfg:
    """Minimal attribute bag consumed by common.preprocess_image, filled from
    ENCODER_FAMILIES so every backbone gets its own img_size/norm stats."""

    def __init__(self, encoder_name, image_dir, image_ext=".png",
                 clip_lo=1.0, clip_hi=99.0, use_clahe=True, clahe_clip=1.0, clahe_grid=8,
                 use_bone_suppression=False, mask_dir=None, channel_mode="duplicate"):
        fam = ENCODER_FAMILIES.get(encoder_name, dict(img_size=224, mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225)))
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.img_size = fam["img_size"]
        self.norm_mean = fam["mean"]
        self.norm_std = fam["std"]
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.use_bone_suppression = use_bone_suppression
        # "duplicate" (default, matches every checkpoint trained before
        # 2026-07-31): one processed view tripled into all 3 channels.
        # "multi_view": channel 0=original, 1=CLAHE, 2=bone-suppressed --
        # see common.py's preprocess_image docstring for the motivation
        # (bone suppression alone gave 5-11 AUC points on these exact
        # Shenzhen/Montgomery benchmarks in the cited literature). MUST
        # match between train and predict, same requirement as
        # use_bone_suppression/mask_dir already have.
        self.channel_mode = channel_mode
        # dir of preprocess.py's precomputed lung masks (its ch2 output, only
        # produced when preprocess.py was run with --mask-channel); None
        # disables the mask-channel feature entirely (default 3x-duplicate
        # channel stack, matching the baseline notebook's ch2 convention).
        self.mask_dir = mask_dir


class TBDataset(Dataset):
    def __init__(self, df, cfg: Cfg, train=True, aug_strength="strong"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.aug = build_augmentation(cfg.img_size, aug_strength if train else "none")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        _id = row["new_id"]
        ip = find_file(self.cfg.image_dir, _id, self.cfg.image_ext)
        if ip is None:
            raise FileNotFoundError(f"no image for new_id={_id} under {self.cfg.image_dir}")
        raw = sitk_read(ip)

        mask_arr = None
        if self.cfg.mask_dir is not None:
            mp = find_file(self.cfg.mask_dir, _id, self.cfg.image_ext)
            if mp is None:
                raise FileNotFoundError(f"mask_dir set but no mask for new_id={_id} under {self.cfg.mask_dir}")
            mask_arr = (sitk_read(mp) > 127)  # preprocess.py writes masks as 0/255 uint8

        x = preprocess_image(raw, self.cfg, mask_arr=mask_arr)
        x = torch.from_numpy(x)
        if self.train:
            x = self.aug(x)
        label = LABEL_MAP[str(row[LABEL_COL]).strip().lower()]
        # Modality_DICOM returned alongside (image, label) -- NOT fed to the
        # model (see this file's docstring: metadata is the shortcut, not a
        # feature), only used by train_task2.py's domain-aware mixup to pick
        # cross-domain partners within a batch. default_collate handles a
        # batch of plain strings fine (returns a list, no tensor needed).
        domain = str(row["Modality_DICOM"])
        return x, torch.tensor(label, dtype=torch.float32), domain


class TBDatasetMaskSupervised(TBDataset):
    """Same as TBDataset, plus the real PSPNet lung mask (preprocess.py's
    ch2), resized to img_size, returned as a 4th item -- a supervision
    TARGET for TBClassifier's optional mask-guided attention gate, not
    spliced into channel 3 like Cfg.mask_dir does. That's the passive
    mask-as-input-channel option (model is free to ignore it); this is the
    active option -- the gate is trained with a Dice loss against this mask,
    directly penalizing attention outside the lung silhouette."""

    def __init__(self, df, cfg: Cfg, mask_target_dir, train=True, aug_strength="strong"):
        super().__init__(df, cfg, train=train, aug_strength=aug_strength)
        self.mask_target_dir = mask_target_dir

    def __getitem__(self, i):
        x, label, domain = super().__getitem__(i)
        row = self.df.iloc[i]
        _id = row["new_id"]
        mp = find_file(self.mask_target_dir, _id, self.cfg.image_ext)
        if mp is None:
            raise FileNotFoundError(f"no mask for new_id={_id} under {self.mask_target_dir}")
        mask = (sitk_read(mp) > 127).astype("float32")
        mask = cv2.resize(mask, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)
        return x, label, domain, torch.from_numpy(mask)


class GlobalLocalCfg:
    """Pairs two Cfg instances for GlobalLocalTBDataset -- `global_cfg` points
    at the raw, uncropped image dir (full-frame context: body habitus,
    mediastinum, overall positioning); `local_cfg` points at preprocess.py's
    ch0 (already lung-cropped, precomputed once -- re-running the segmenter
    per __getitem__/epoch would be far too slow). Each stream can use its own
    encoder_name/img_size/norm, since GlobalLocalTBClassifier's two backbones
    are independent (see model.py) -- they don't need matching resolutions."""

    def __init__(self, global_encoder_name, local_encoder_name, global_image_dir, local_image_dir, **kwargs):
        self.global_cfg = Cfg(global_encoder_name, image_dir=global_image_dir, **kwargs)
        self.local_cfg = Cfg(local_encoder_name, image_dir=local_image_dir, **kwargs)


class GlobalLocalTBDataset(Dataset):
    """CheXFound-inspired dual-stream dataset: returns both the full,
    uncropped image (global context) and the lung-cropped image (local
    anatomical detail) for the same sample, so GlobalLocalTBClassifier can
    fuse features from both instead of the single-crop-OR-full choice
    TBDataset/preprocess.py's --lung-crop flag otherwise forces."""

    def __init__(self, df, cfg: GlobalLocalCfg, train=True, aug_strength="strong"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.aug_global = build_augmentation(cfg.global_cfg.img_size, aug_strength if train else "none")
        self.aug_local = build_augmentation(cfg.local_cfg.img_size, aug_strength if train else "none")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        _id = row["new_id"]

        gp = find_file(self.cfg.global_cfg.image_dir, _id, self.cfg.global_cfg.image_ext)
        lp = find_file(self.cfg.local_cfg.image_dir, _id, self.cfg.local_cfg.image_ext)
        if gp is None:
            raise FileNotFoundError(f"no global image for new_id={_id} under {self.cfg.global_cfg.image_dir}")
        if lp is None:
            raise FileNotFoundError(f"no local (lung-cropped) image for new_id={_id} under "
                                     f"{self.cfg.local_cfg.image_dir} -- run preprocess.py --lung-crop first")

        x_global = torch.from_numpy(preprocess_image(sitk_read(gp), self.cfg.global_cfg))
        x_local = torch.from_numpy(preprocess_image(sitk_read(lp), self.cfg.local_cfg))
        if self.train:
            x_global = self.aug_global(x_global)
            x_local = self.aug_local(x_local)

        label = LABEL_MAP[str(row[LABEL_COL]).strip().lower()]
        domain = str(row["Modality_DICOM"])
        return x_global, x_local, torch.tensor(label, dtype=torch.float32), domain
