#!/usr/bin/env python
"""Gated hybrid mask generation — combine our classifier's presence decision
with nnUNet's raw segmentation, per the plan's section 10.1.

Why: the official scorer (Code/Task1_evaluation_code.ipynb, sibling to this
project) computes accuracy from mask non-emptiness and Dice from real pixel
overlap — nnUNet's own masks alone score only ~0.306 Dice, partly because it
produces false-positive masks on cavity-absent cases (scored dsc=0 today).
Gating: when our classifier says "no cavity", force an empty mask — this
turns a false-positive dsc=0 case into a correct empty/empty match (dsc=NaN,
excluded from the mean, and the accuracy term improves too since our
classifier is a stronger presence-detector than nnUNet's implicit
mask-non-emptiness call). When the classifier says "cavity present", pass
nnUNet's mask through unchanged -- UNLESS --prob-dir is given (see below).

nnUNet's "111 validation cases" are literally our test.csv (confirmed by
cross-referencing case IDs, e.g. case "113" == test.csv our_id 113, grade
"medium") — so a submission_*.csv's `our_id`/`pred_cavity` columns join
directly against nnUNet's validation case IDs, no mapping needed.

--prob-dir / low-threshold recovery (plan section 10.1's second half,
previously undocumented as *why* it matters and never wired up): measured
directly against the 3-way classifier ensemble (accuracy 0.8739 standalone),
the GATED accuracy came out to only 0.8018 -- a real, diagnosed gap, not
noise. Root cause: when the classifier says "positive," nnUNet's raw mask
passes through UNCHANGED at its own default probability threshold (0.5). If
nnUNet's own segmentation predicts an empty mask on a case the classifier
correctly flagged positive, the gate does nothing to rescue it -- it just
inherits nnUNet's miss. That's a false-negative-on-positive-GT case the
classifier alone would have gotten right. Recovery: when the classifier says
positive AND nnUNet's default mask is empty, re-threshold nnUNet's raw
softmax probabilities (channel 1, cavity) at a LOWER, more permissive
threshold (--recovery-threshold, default 0.3) to try to recover a
non-trivial mask instead of accepting the empty one. Requires nnUNet's raw
per-voxel probabilities, which its default validation pass does NOT save
(only the already-thresholded .nii.gz masks) -- generate them with:
    nnUNetv2_predict -i <images> -o <prob_out_dir> -d 1 -c 2d \\
        -tr <TRAINER> -p <PLANS> -f 0 --save_probabilities
then pass --prob-dir <prob_out_dir> here. Gracefully no-ops (falls back to
today's pass-through behavior) if --prob-dir isn't given or a case's .npz
is missing -- this is a recovery mechanism layered on top of the existing
pipeline, not a hard requirement.

Usage:
    python gate_segmentation.py --submission ../../../submissions/submission_densenet201.csv
    python gate_segmentation.py --submission ../../../submissions/submission_densenet201.csv --evaluate
    python gate_segmentation.py --submission ../../../submissions/submission_top3.csv --evaluate \\
        --prob-dir ../../../nnUNet_probabilities --recovery-threshold 0.3
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
# Task1/Code/ -> Task1_0615/ -> IDs/eunchanhwang/ -> Code/baseline/nnUNetv2/...
_DEFAULT_NNUNET_ROOT = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "Code", "baseline", "nnUNetv2", "nnUNet_data"))
_DEFAULT_SUBMISSIONS_DIR = os.path.normpath(os.path.join(HERE, "..", "submissions"))


def load_binary_mask(path):
    mask = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return (np.squeeze(mask) > 0).astype(np.uint8)


def try_recover_from_probabilities(case_id, prob_dir, recovery_threshold, ref_img):
    """Returns a recovered sitk.Image (thresholded at recovery_threshold on
    the cavity-class channel) if `{prob_dir}/{case_id}.npz` exists and its
    shape matches ref_img's array shape, else None -- caller falls back to
    the empty default mask on None, this never raises for a missing/
    malformed file (recovery is a bonus, not a hard requirement).

    nnUNetv2_predict --save_probabilities writes a `probabilities` array of
    shape (num_classes, *spatial_shape) in the same array-index order as the
    corresponding .nii.gz's GetArrayFromImage() -- channel 0 is background,
    channel 1 is the cavity class (this is a 2-class problem, per
    FocalTversky_and_CE_loss's do_bg=False / label_manager setup elsewhere
    in this project). Not verified against a real .npz from this exact
    nnunetv2==2.5.1 pin (none existed yet when this was written) -- if the
    shape/key assumptions are wrong for your version, this prints a warning
    and returns None (safe fallback), it does not silently produce a wrong
    mask.
    """
    npz_path = os.path.join(prob_dir, f"{case_id}.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path)
        probs = data["probabilities"] if "probabilities" in data else data[data.files[0]]
    except Exception as e:
        print(f"  !! {case_id}: failed to load {npz_path} ({e}) -- skipping recovery")
        return None

    ref_shape = sitk.GetArrayFromImage(ref_img).shape
    if probs.ndim != len(ref_shape) + 1 or probs.shape[1:] != ref_shape:
        print(f"  !! {case_id}: probability array shape {probs.shape} doesn't match "
              f"reference image shape {ref_shape} -- skipping recovery (check nnUNet's "
              f"actual --save_probabilities layout for your nnunetv2 version)")
        return None
    if probs.shape[0] < 2:
        print(f"  !! {case_id}: probability array has only {probs.shape[0]} channel(s), "
              f"expected >=2 (bg + cavity) -- skipping recovery")
        return None

    recovered_arr = (probs[1] >= recovery_threshold).astype(np.uint8)
    recovered_img = sitk.GetImageFromArray(recovered_arr)
    recovered_img.CopyInformation(ref_img)
    return recovered_img


def dice_score(pred, gt):
    pred_sum, gt_sum = pred.sum(), gt.sum()
    if pred_sum == 0 and gt_sum == 0:
        return np.nan
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred_sum + gt_sum)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True,
                    help="submission_*.csv providing the classifier's pred_cavity per our_id "
                         "(e.g. submissions/submission_densenet201.csv, or the final ensemble)")
    ap.add_argument("--nnunet-root", default=_DEFAULT_NNUNET_ROOT)
    ap.add_argument("--dataset", default="Dataset001_Task1")
    ap.add_argument("--trainer-config", default="nnUNetTrainer__nnUNetPlans__2d")
    ap.add_argument("--fold", default="fold_0")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "gated_masks"))
    ap.add_argument("--evaluate", action="store_true",
                    help="also score the gated masks against ground truth with the official metric")
    ap.add_argument("--prob-dir", default=None,
                     help="dir of {case_id}.npz raw softmax probabilities from "
                          "nnUNetv2_predict --save_probabilities -- enables low-threshold "
                          "recovery for classifier-positive cases where nnUNet's default mask "
                          "is empty (see module docstring). Omit to keep today's pass-through "
                          "behavior unchanged.")
    ap.add_argument("--recovery-threshold", type=float, default=0.3,
                     help="probability threshold for recovery re-binarization (only used "
                          "when --prob-dir is given), lower than nnUNet's own default of 0.5 "
                          "so it can recover masks nnUNet's default threshold dropped")
    args = ap.parse_args()

    pred_dir = os.path.join(args.nnunet_root, "nnUNet_results", args.dataset,
                             args.trainer_config, args.fold, "validation")
    gt_dir = os.path.join(args.nnunet_root, "nnUNet_raw", args.dataset, "labelsTr")

    sub = pd.read_csv(args.submission)
    assert "our_id" in sub.columns and "pred_cavity" in sub.columns, \
        f"{args.submission} must have our_id and pred_cavity columns"
    presence = dict(zip(sub["our_id"].astype(str), sub["pred_cavity"].astype(int)))

    os.makedirs(args.out_dir, exist_ok=True)
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    if not pred_paths:
        raise FileNotFoundError(f"no nnUNet predictions found under {pred_dir}")

    n_gated_off, n_missing_pred, n_recovered = 0, 0, 0
    case_results = []
    for pred_path in pred_paths:
        case_id = os.path.basename(pred_path).replace(".nii.gz", "")
        if case_id not in presence:
            n_missing_pred += 1
            continue

        img = sitk.ReadImage(pred_path)
        if presence[case_id] == 0:
            gated = sitk.Image(img.GetSize(), img.GetPixelID())
            gated.CopyInformation(img)
            n_gated_off += 1
        else:
            gated = img
            is_empty = sitk.GetArrayFromImage(img).sum() == 0
            if is_empty and args.prob_dir:
                recovered = try_recover_from_probabilities(
                    case_id, args.prob_dir, args.recovery_threshold, img)
                if recovered is not None:
                    gated = recovered
                    n_recovered += 1
        out_path = os.path.join(args.out_dir, f"{case_id}.nii.gz")
        sitk.WriteImage(gated, out_path)

        if args.evaluate:
            gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
            gt = load_binary_mask(gt_path)
            pred_mask = load_binary_mask(out_path)
            case_results.append({
                "case_id": case_id,
                "gt_positive": bool(gt.sum() > 0),
                "pred_positive": bool(pred_mask.sum() > 0),
                "accuracy": float((gt.sum() > 0) == (pred_mask.sum() > 0)),
                "dice": dice_score(pred_mask, gt),
            })

    recovery_note = f", {n_recovered} recovered via low-threshold probabilities" if args.prob_dir else ""
    print(f"wrote {len(pred_paths) - n_missing_pred} gated masks to {args.out_dir} "
          f"({n_gated_off} forced empty by classifier, {n_missing_pred} skipped -- "
          f"no classifier prediction for that case_id{recovery_note})")

    if args.evaluate:
        df = pd.DataFrame(case_results)
        mean_accuracy = df["accuracy"].mean()
        mean_dice = np.nanmean(df["dice"])
        final_score = 0.7 * mean_accuracy + 0.3 * mean_dice
        print(f"\nGated pipeline (classifier={os.path.basename(args.submission)} + nnUNet {args.fold}):")
        print(f"  Mean Accuracy : {mean_accuracy:.4f}")
        print(f"  Mean Dice     : {mean_dice:.4f}")
        print(f"  Final Score   : {final_score:.4f}")

        # Raw nnUNet baseline (no gating) over the same case set, for direct comparison.
        raw_results = []
        for pred_path in pred_paths:
            case_id = os.path.basename(pred_path).replace(".nii.gz", "")
            if case_id not in presence:
                continue
            gt = load_binary_mask(os.path.join(gt_dir, f"{case_id}.nii.gz"))
            pred_mask = load_binary_mask(pred_path)
            raw_results.append({
                "accuracy": float((gt.sum() > 0) == (pred_mask.sum() > 0)),
                "dice": dice_score(pred_mask, gt),
            })
        rdf = pd.DataFrame(raw_results)
        r_acc, r_dice = rdf["accuracy"].mean(), np.nanmean(rdf["dice"])
        print(f"\nRaw nnUNet baseline (no gating), same case set:")
        print(f"  Mean Accuracy : {r_acc:.4f}")
        print(f"  Mean Dice     : {r_dice:.4f}")
        print(f"  Final Score   : {0.7 * r_acc + 0.3 * r_dice:.4f}")


if __name__ == "__main__":
    main()
