#!/usr/bin/env python
"""End-to-end Task1 SUBMISSION inference: raw CXR images in --image-dir ->
manifest -> classification ensemble (densenet201, tf_efficientnetv2_s,
resnet50d, ordfused_eva_x_base_notab -- same 4 members as
submission_valid_ensemble_v1.csv, OOF tau*=0.535) -> segmentation ensemble
(rad_dino, chexfound_vitl16, eva_x_base) -> classifier-gated mask (empty mask
when pred_cavity=0, segmentation ensemble's mask when pred_cavity=1, same
principle as gate_segmentation.py but self-contained end-to-end on arbitrary
new images instead of nnU-Net's precomputed validation predictions) ->
writes:
  {out_dir}/submission.csv            (our_id, prob_cavity, pred_cavity)
  {out_dir}/masks/{our_id}.png        (predicted cavity mask, 0/255, only
                                        written for pred_cavity=1 cases;
                                        omitted -- i.e. implicitly empty --
                                        for pred_cavity=0 cases)

Unlike train_segmentation.py/predict.py/predict_ordfused.py (which evaluate
against Task1/Data/{train,test} and need ground truth to sweep/report
metrics), this script takes only a directory of images -- no labels, no
internal Data/ dependency -- so it's the actual deployable submission
container's entrypoint, mirroring Task2/Code/predict_task2.py's role for
Task2.

TODO -- segmentation postprocessing (binarization threshold + connected-
component filtering) is currently a placeholder (threshold=0.5, no CC
filtering). A dev-only grid sweep (never tuned on test) is in progress
(see /Task1_0615/Task1/Code/postprocess_sweep_result.json once written) --
update SEG_THRESHOLD/SEG_MIN_COMPONENT_PX/SEG_KEEP_TOP_K below once that
lands, before the final submission build.

Usage:
    python predict_task1_submission.py --image-dir /input --out-dir /output
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CNNDetector, ID_COL, find_file, preprocess_image, set_seed, sitk_read
from ordfused import ENCODER_FAMILIES, OrdFusedModel, corn_cavity_prob
from segmentation_model import FoundationSegModel

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Classification ensemble: same 4 members as submission_valid_ensemble_v1.csv
# (see Task1_0615/Task1/submissions/submission_valid_ensemble_v1.csv and the
# task1-classifier-decision memory) -- all metadata-free / image-only, valid
# against an images-only test set. Uniform weights, tau swept on OOF=0.535.
CNN_MEMBERS = ["densenet201", "tf_efficientnetv2_s", "resnet50d"]
ORDFUSED_MEMBERS = ["ordfused_eva_x_base_notab"]  # run_dir name under runs/
CLASSIFIER_TAU = 0.535  # OOF-swept, see submission_valid_ensemble_v1 / ensemble_submissions.py

# --- Segmentation ensemble ---
SEG_ENCODERS = ["rad_dino", "chexfound_vitl16", "eva_x_base"]
SEG_THRESHOLD = 0.5           # PLACEHOLDER -- see module docstring TODO
SEG_MIN_COMPONENT_PX = 0      # PLACEHOLDER -- 0 = no connected-component filtering yet
SEG_KEEP_TOP_K = None         # PLACEHOLDER -- None = keep all components above min size
SEG_EVAL_SIZE = 512           # common canvas the 3 encoders' probability maps are resized to before averaging

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_manifest(image_dir):
    exts = (".dcm", ".png", ".jpg", ".jpeg", ".nii.gz")
    ids = sorted(set(
        os.path.basename(p)[: -len(ext)] if p.endswith(ext) else os.path.splitext(os.path.basename(p))[0]
        for ext in exts
        for p in glob.glob(os.path.join(image_dir, f"*{ext}"))
    ))
    if not ids:
        raise SystemExit(f"no images found under {image_dir}")
    return pd.DataFrame({ID_COL: ids})


class ImgCFG:
    def __init__(self, img_size, mean, std, image_dir, image_ext_glob=True):
        self.img_size = img_size
        self.clip_lo, self.clip_hi = 1.0, 99.0
        self.use_clahe = True
        self.clahe_clip, self.clahe_grid = 2.0, 8
        self.norm_mean, self.norm_std = mean, std
        self.image_dir = image_dir


class InferenceImageDataset(torch.utils.data.Dataset):
    """Loads + preprocesses one image per __getitem__ at the given cfg's
    resolution -- shared by both the classification and segmentation passes,
    since both ultimately go through common.py's preprocess_image."""
    def __init__(self, ids, cfg):
        self.ids = list(ids)
        self.cfg = cfg

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        _id = self.ids[i]
        ip = find_file(self.cfg.image_dir, _id, "")
        if ip is None:
            raise FileNotFoundError(f"no image for {ID_COL}={_id} under {self.cfg.image_dir}")
        x = preprocess_image(sitk_read(ip), self.cfg)
        return torch.from_numpy(x), _id


def load_cnn_fold_models(model_dir, model_name, device):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "fold*_best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no fold*_best.pt under {model_dir}")
    models = []
    for p in ckpts:
        sd = torch.load(p, map_location=device, weights_only=False)
        m = CNNDetector(model_name, pretrained=False).to(device)
        m.load_state_dict(sd["model"])
        m.eval()
        models.append(m)
    return models


def load_ordfused_fold_models(model_dir, device):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "fold*_best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no fold*_best.pt under {model_dir}")
    models = []
    for p in ckpts:
        sd = torch.load(p, map_location=device, weights_only=False)
        m = OrdFusedModel(sd["encoder"], use_tabular=sd["use_tabular"], pretrained=False).to(device)
        m.load_state_dict(sd["model"])
        m.eval()
        models.append((sd["encoder"], sd.get("img_size", 224), m))
    return models


@torch.no_grad()
def cnn_ensemble_probs(models, ids, image_dir, device, img_size, batch_size, num_workers, tta=True):
    cfg = ImgCFG(img_size, _DEFAULT_MEAN, _DEFAULT_STD, image_dir)
    dl = DataLoader(InferenceImageDataset(ids, cfg), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs_by_id = {}
    for x, batch_ids in dl:
        x = x.to(device)
        variants = [x, x.flip(-1)] if tta else [x]
        ps = [torch.sigmoid(m(xv)) for xv in variants for m in models]
        p = torch.stack(ps, dim=0).mean(0).squeeze(-1).cpu().numpy()
        for _id, pv in zip(batch_ids, p):
            probs_by_id[_id] = float(pv)
    return probs_by_id


@torch.no_grad()
def ordfused_ensemble_probs(models_with_meta, ids, image_dir, device, batch_size, num_workers, tta=True):
    """models_with_meta: [(encoder_name, img_size_hint, model), ...] -- each
    encoder may need its own resolution (foundation-model members don't use
    224/ImageNet stats), so this groups by (size, mean, std) like
    predict_ordfused.py's group_by_preprocess/weighted_group_probs."""
    groups = {}
    for enc, size_hint, m in models_with_meta:
        fam = ENCODER_FAMILIES.get(enc)
        key = (fam["img_size"], fam["mean"], fam["std"]) if fam else (size_hint, _DEFAULT_MEAN, _DEFAULT_STD)
        groups.setdefault(key, []).append(m)

    total = {}
    total_n = 0
    for (size, mean, std), group_models in groups.items():
        cfg = ImgCFG(size, mean, std, image_dir)
        dl = DataLoader(InferenceImageDataset(ids, cfg), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        n = len(group_models)
        for x, batch_ids in dl:
            x = x.to(device)
            tab = torch.zeros(x.size(0), 0, device=device)  # notab checkpoint: no tabular branch used
            variants = [x, x.flip(-1)] if tta else [x]
            ps = [corn_cavity_prob(m(xv, None)) for xv in variants for m in group_models]
            p = torch.stack(ps, 0).mean(0).cpu().numpy()
            for _id, pv in zip(batch_ids, p):
                total[_id] = total.get(_id, 0.0) + float(pv) * n
        total_n += n
    return {k: v / max(total_n, 1) for k, v in total.items()}


def keep_components(mask, min_size, keep_top_k=None):
    if mask.sum() == 0 or min_size <= 0 and keep_top_k is None:
        return mask
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    if keep_top_k is not None and len(keep_ids) > keep_top_k:
        keep_ids = sorted(keep_ids, key=lambda i: sizes[i - 1], reverse=True)[:keep_top_k]
    return np.isin(lbl, keep_ids).astype(np.uint8)


@torch.no_grad()
def segmentation_ensemble_masks(ids, image_dir, ckpt_dir, device, batch_size, num_workers, tta=True):
    """Only run for images the classifier called cavity-positive (caller
    filters `ids`) -- returns {id: (SEG_EVAL_SIZE,SEG_EVAL_SIZE) uint8 mask}."""
    per_encoder_probs = {}
    for enc in SEG_ENCODERS:
        ckpt_path = os.path.join(ckpt_dir, enc, "best.pt")
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = FoundationSegModel(enc, pretrained=False).to(device)
        model.load_state_dict(sd["model"])
        model.eval()

        fam = ENCODER_FAMILIES[enc]
        cfg = ImgCFG(fam["img_size"], fam["mean"], fam["std"], image_dir)
        dl = DataLoader(InferenceImageDataset(ids, cfg), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        probs_by_id = {}
        for x, batch_ids in dl:
            x = x.to(device)
            probs = torch.sigmoid(model(x))
            if tta:
                probs = (probs + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
            probs = probs.squeeze(1).cpu().numpy()
            for _id, p in zip(batch_ids, probs):
                probs_by_id[_id] = cv2.resize(p, (SEG_EVAL_SIZE, SEG_EVAL_SIZE), interpolation=cv2.INTER_LINEAR)
        per_encoder_probs[enc] = probs_by_id
        del model
        torch.cuda.empty_cache()

    masks = {}
    for _id in ids:
        avg = np.mean([per_encoder_probs[enc][_id] for enc in SEG_ENCODERS], axis=0)
        mask = (avg >= SEG_THRESHOLD).astype(np.uint8)
        mask = keep_components(mask, SEG_MIN_COMPONENT_PX, SEG_KEEP_TOP_K)
        masks[_id] = mask
    return masks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "runs"),
                    help="dir containing <model>/fold*_best.pt for classification members")
    ap.add_argument("--seg-ckpt-dir", default=os.path.join(HERE, "runs_seg"),
                    help="dir containing <encoder>/best.pt for segmentation members")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    df = build_manifest(args.image_dir)
    ids = df[ID_COL].tolist()
    print(f"found {len(ids)} images under {args.image_dir}")

    # --- classification ensemble ---
    n_members = 0
    prob_sum = {i: 0.0 for i in ids}
    for model_name in CNN_MEMBERS:
        models = load_cnn_fold_models(os.path.join(args.ckpt_dir, model_name), model_name, device)
        probs = cnn_ensemble_probs(models, ids, args.image_dir, device, img_size=224,
                                    batch_size=args.batch_size, num_workers=args.num_workers, tta=args.tta)
        for i in ids:
            prob_sum[i] += probs[i]
        n_members += 1
        print(f"[classify] {model_name}: {len(models)} folds loaded")
        del models
        torch.cuda.empty_cache()

    for run_dir in ORDFUSED_MEMBERS:
        models = load_ordfused_fold_models(os.path.join(args.ckpt_dir, run_dir), device)
        probs = ordfused_ensemble_probs(models, ids, args.image_dir, device,
                                         batch_size=args.batch_size, num_workers=args.num_workers, tta=args.tta)
        for i in ids:
            prob_sum[i] += probs[i]
        n_members += 1
        print(f"[classify] {run_dir}: {len(models)} folds loaded")
        del models
        torch.cuda.empty_cache()

    prob_cavity = {i: prob_sum[i] / n_members for i in ids}
    pred_cavity = {i: int(prob_cavity[i] >= CLASSIFIER_TAU) for i in ids}
    n_positive = sum(pred_cavity.values())
    print(f"[classify] {n_positive}/{len(ids)} predicted cavity-positive (tau={CLASSIFIER_TAU})")

    # --- segmentation ensemble, only for classifier-positive cases (gating,
    # same principle as gate_segmentation.py) ---
    positive_ids = [i for i in ids if pred_cavity[i] == 1]
    masks = {}
    if positive_ids:
        masks = segmentation_ensemble_masks(positive_ids, args.image_dir, args.seg_ckpt_dir, device,
                                             batch_size=args.batch_size, num_workers=args.num_workers, tta=args.tta)

    for _id in positive_ids:
        # Resize the (SEG_EVAL_SIZE,SEG_EVAL_SIZE) mask back to the ORIGINAL
        # image's native resolution before writing. Predictions are made on a
        # fixed square grid, but the real image (and any grader comparing
        # against native-resolution ground truth) isn't square -- writing the
        # raw 512x512 mask silently produces a mask at the wrong scale/aspect
        # ratio for every non-square input. Confirmed empirically: scoring
        # the classifier-ensemble + nnU-Net-gated masks with evaluate.py at
        # native resolution collapsed Dice from a previously-reported 0.68
        # (measured against nnU-Net's own resized copy of the GT) to 0.044 --
        # this native-resolution resize is the fix for that class of bug.
        ip = find_file(args.image_dir, _id, ".dcm")
        native_h, native_w = sitk_read(ip).shape if ip else masks[_id].shape
        m = cv2.resize(masks[_id], (native_w, native_h), interpolation=cv2.INTER_NEAREST) * 255
        cv2.imwrite(os.path.join(mask_dir, f"{_id}.png"), m)
    # cavity-negative cases: no mask file written (implicitly empty) --
    # documented in the module docstring; adjust here if the organizers'
    # harness instead expects an explicit all-zero mask file per image.

    out = pd.DataFrame({
        ID_COL: ids,
        "prob_cavity": [prob_cavity[i] for i in ids],
        "pred_cavity": [pred_cavity[i] for i in ids],
    })
    out_csv = os.path.join(args.out_dir, "submission.csv")
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} and {len(positive_ids)} mask file(s) -> {mask_dir}")


if __name__ == "__main__":
    main()