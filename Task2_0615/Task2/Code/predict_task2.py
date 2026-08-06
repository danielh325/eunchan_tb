"""Inference + ensemble for Task 2. Loads every checkpoint under --ckpt-dir
(all fold-held-out-modality checkpoints for every backbone -- an implicit
k-fold ensemble per encoder, plus the 3-way backbone ensemble), groups by
img_size for one dataset/dataloader pass per distinct size (RAD-DINO 518,
CheXFound 512, DenseNet-121 224 all differ), averages probabilities across
every checkpoint at the end -- same late-fusion principle as Task1's
predict_ordfused.py.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ID_COL, LABEL_COL, LABEL_MAP, detection_accuracy, safe_auc, safe_f1, sweep_tau_accuracy
from dataset import Cfg, TBDataset
from model import TBClassifier
from tta import source_free_adapt


def load_checkpoints(ckpt_dir, encoders=None):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    if encoders:
        ckpts = [c for c in ckpts if c["encoder_name"] in encoders]
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--csv", required=True, help="test.csv (no TB/Normal labels required)")
    ap.add_argument("--image-dir", required=True, help="dir of preprocess.py's ch0 (lung-cropped raw) images")
    ap.add_argument("--mask-dir", default=None,
                     help="dir of preprocess.py's ch2 lung masks -- MUST match whatever "
                          "--mask-dir (or its absence) was used at training time, or the "
                          "input channel layout won't match what the checkpoint learned")
    ap.add_argument("--bone-suppress", action="store_true",
                     help="MUST match whatever --bone-suppress (or its absence) was used at "
                          "training time -- this is a preprocessing-input change, not a "
                          "model-architecture one, so an on/off mismatch between train and "
                          "predict silently feeds the model different-looking input than it "
                          "was trained on rather than erroring")
    ap.add_argument("--channel-mode", choices=["duplicate", "multi_view"], default="duplicate",
                     help="MUST match whatever --channel-mode was used at training time -- "
                          "same silent-mismatch risk as --mask-dir/--bone-suppress above. "
                          "'multi_view' checkpoints (channel 0=original, 1=CLAHE, "
                          "2=bone-suppressed) need this set explicitly, or predict will feed "
                          "them the old triplicated-single-view input instead.")
    ap.add_argument("--out", default="submission_task2.csv")
    ap.add_argument("--encoders", nargs="*", default=None,
                     help="restrict ensemble to these encoder names; default: all found")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true",
                     help="apply source-free test-time adaptation (tta.py) to each model "
                          "before scoring -- adapts only normalization-layer affine params + "
                          "the head, using rank-based pseudo-labels from --csv's own images "
                          "(no source data, no external data, works even on a genuinely "
                          "unlabeled file). Compare --tta on/off runs to see if it actually "
                          "helps before trusting it as a default.")
    ap.add_argument("--tta-steps", type=int, default=5)
    ap.add_argument("--tta-lr", type=float, default=1e-4)
    ap.add_argument("--tta-confidence-frac", type=float, default=0.3)
    ap.add_argument("--score", action="store_true",
                     help="print accuracy/AUC/F1 against real TB/Normal labels in --csv, if "
                          "present -- this is the internal-held-out number (e.g. test.csv, "
                          "held out by the organizers from the same source pool as train.csv, "
                          "distinct from both the plain stratified split and true external "
                          "validation -- see the calibration discussion in the plan). No-op "
                          "with a warning if --csv turns out to have no real labels (e.g. a "
                          "genuinely unlabeled external file).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = load_checkpoints(args.ckpt_dir, args.encoders)
    if not ckpts:
        raise SystemExit(f"no checkpoints found under {args.ckpt_dir}")

    df = pd.read_csv(args.csv)
    has_real_labels = LABEL_COL in df.columns
    ids = df[ID_COL].tolist()
    prob_sum = np.zeros(len(df), dtype=np.float64)
    n_models = 0

    # Group by the FULL preprocessing signature (img_size + norm stats), not
    # just img_size -- with the 5-backbone ensemble, densenet121/swin_tiny/
    # convnext_tiny all share img_size=224 (they happen to also share
    # ImageNet norm stats today, but that's a coincidence of the current
    # registry, not something to rely on silently). Grouping on the tuple
    # means a future 224px backbone with different norm stats gets its own
    # group automatically instead of silently reusing the wrong Cfg.
    def _sig(c):
        return (c["img_size"], tuple(c["norm_mean"]), tuple(c["norm_std"]))

    by_sig = {}
    for c in ckpts:
        by_sig.setdefault(_sig(c), []).append(c)

    for sig, group in by_sig.items():
        encoder_name = group[0]["encoder_name"]
        cfg = Cfg(encoder_name, image_dir=args.image_dir, mask_dir=args.mask_dir,
                  use_bone_suppression=args.bone_suppress, channel_mode=args.channel_mode)
        # every checkpoint in this group shares the exact (img_size,
        # norm_mean, norm_std) signature by construction of the grouping
        # above, so one dataset built from any one of their encoder_names
        # is valid for all of them.
        # TBDataset requires a label column to exist (it's unused here, only
        # the image is read) -- test.csv already carries real TB/Normal
        # labels today, but inject a dummy value only if a genuinely
        # unlabeled external test set is ever passed, so real labels are
        # never silently overwritten.
        pred_df = df if has_real_labels else df.assign(**{"TB/Normal": "Normal"})
        # TBDataset reads Modality_DICOM for domain-aware-mixup bookkeeping
        # only (train=False here, so mixup never actually runs) -- but it's
        # still read unconditionally in __getitem__, so a genuinely
        # unlabeled/different-schema external test file needs a placeholder
        # rather than a KeyError at inference time.
        if "Modality_DICOM" not in pred_df.columns:
            pred_df = pred_df.assign(Modality_DICOM="unknown")
        ds = TBDataset(pred_df, cfg, train=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

        for ckpt in group:
            # Explicit fallback of 16/16 here (NOT "look up this encoder's
            # current ENCODER_FAMILIES default") is deliberate: every
            # checkpoint trained before 2026-07-31 used a flat r=16/alpha=16
            # for every encoder and has no lora_r/lora_alpha key at all. If
            # ENCODER_FAMILIES' per-encoder default ever changes again (see
            # encoders.py's history on chexfound_vitl16), a checkpoint-stored
            # value must keep taking priority over "whatever the default is
            # today", or an old checkpoint would reconstruct the wrong LoRA
            # shape and fail to load its own state_dict. New checkpoints
            # always carry their actual resolved rank/alpha (train_task2.py's
            # run_one_fold saves it), so this fallback only matters for
            # pre-existing ones. use_glori_head defaults to False the same
            # way -- only chexfound_vitl16 checkpoints trained with
            # --use-glori-head (2026-07-31 onward) will have it set True.
            model = TBClassifier(ckpt["encoder_name"], pretrained=False,
                                  use_mask_attention=ckpt.get("use_mask_attention", False),
                                  use_glori_head=ckpt.get("use_glori_head", False),
                                  glori_n_layers=ckpt.get("glori_n_layers", 4),
                                  lora_r=ckpt.get("lora_r", 16),
                                  lora_alpha=ckpt.get("lora_alpha", 16)).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            if args.tta:
                # target_loader here is the SAME loader the prediction pass
                # below uses -- source_free_adapt only ever reads images
                # from it (labels ignored), so this is safe/identical
                # whether --csv has real labels or not.
                model = source_free_adapt(model, loader, device, steps=args.tta_steps,
                                           lr=args.tta_lr, confidence_frac=args.tta_confidence_frac)
                model.eval()

            probs = []
            with torch.no_grad():
                for x, _y, _dom in loader:
                    x = x.to(device, non_blocking=True)
                    probs.append(torch.sigmoid(model(x)).cpu().numpy())
            probs = np.concatenate(probs)
            prob_sum += probs
            n_models += 1
            print(f"scored {ckpt['encoder_name']} / {ckpt['fold_tag']} "
                  f"(tau={ckpt['tau']:.2f}, val_acc={ckpt['val_acc']:.4f})")

    prob = prob_sum / n_models
    pred = (prob >= 0.5).astype(int)  # ensemble tau left at 0.5; see note below
    out = pd.DataFrame({ID_COL: ids, "prob_tb": prob, "pred_TB/Normal": np.where(pred == 1, "TB", "Normal")})
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({n_models} models ensembled)")

    if args.score:
        if not has_real_labels:
            print(f"--score requested but {args.csv} has no '{LABEL_COL}' column -- skipping")
        else:
            y_true = df[LABEL_COL].astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy()
            acc_half = detection_accuracy(y_true, pred)
            auc = safe_auc(y_true, prob)
            f1_half = safe_f1(y_true, pred)
            tau_star, acc_tau = sweep_tau_accuracy(y_true, prob)
            print(f"\n=== SCORE against real labels in {args.csv} ({len(y_true)} samples) ===")
            print(f"accuracy@0.5={acc_half:.4f}  AUC={auc:.4f}  F1@0.5={f1_half:.4f}")
            print(f"best-tau={tau_star:.2f}  accuracy@best-tau={acc_tau:.4f}  "
                  f"(tau chosen on this same set -- informational only, not an unbiased "
                  f"estimate; report accuracy@0.5 as the honest number)")


if __name__ == "__main__":
    main()
