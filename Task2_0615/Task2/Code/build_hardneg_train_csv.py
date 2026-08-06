"""Build a hard-negative-augmented Task2 training set.

WHY: with prevalence and threshold both controlled, the model scores F1 0.897 on
Shenzhen / 0.898 on Montgomery but only 0.800 on TBX11K-balanced. The only
remaining difference is what the NEGATIVES are -- Shenzhen/Montgomery negatives
are largely healthy, TBX11K's include sick-but-not-TB patients. So the model has
partly learned "abnormal" rather than "TB", and a worldwide external test set
will look far more like TBX11K's negative pool than Shenzhen's.

WHAT: appends TBX11K's `sick` images (abnormal, TB-negative) to Data/train.csv as
extra `Normal` rows flagged `is_aux=1`. train_task2.py pins is_aux rows to
fold=-1, so they join every leave-one-modality-out fold's TRAINING set and never
appear in any validation fold -- fold metrics stay comparable to earlier runs.

SPLIT DISCIPLINE: only the sick images in TBX11K's OFFICIAL train list are used
(lists/TBX11K_train.txt, 3000 of the 3800). The 800 sick in TBX11K_val.txt are
left untouched so TBX11K_val remains a held-out benchmark. TBX11K's own `test`
split is unusable here -- its 3302 images are unlabeled (held on the authors'
server). Shenzhen and Montgomery are never touched at all and remain the true
never-trained-on external sets.

Only the SICK images are taken, not TBX11K's 600 train TB positives: that
isolates the hard-negative variable (the thing being tested) and keeps the TB
signal coming purely from challenge data.

ALSO FOLDS IN THE OLD INTERNAL TEST SET (--with-test, on by default). Now that
the challenge has moved to the external phase, Data/test.csv's 1940 labelled
images (920 TB / 1020 Normal) are no longer a held-out benchmark -- they are
simply more labelled challenge-distribution data, and they are the single best
source of additional TB signal available. They join as ORDINARY challenge rows
(is_aux=0) because they carry real Modality_DICOM values (CR 866 / XC 537 /
DX 537), so they participate in leave-one-modality-out folding exactly like
train.csv rows and improve per-fold checkpoint selection.

CONSEQUENCE: Data/test.csv STOPS BEING A VALID METRIC the moment this is used.
Any F1 reported on it afterwards is train-on-test. sbatch/train_task2_rad_dino_
hardneg.sbatch drops it from the eval table for exactly this reason, and every
historical test.csv number (e.g. the 0.9913 baseline) is no longer comparable.
Shenzhen and Montgomery remain the true never-trained-on external sets.

ID COLLISION IS TOTAL between train.csv and test.csv -- all 1940 test ids
duplicate a train id (both number from 1). Test rows are therefore re-keyed to
`test_<id>` in the CSV, with matching symlink names pointing back at the
original filenames under Data/Preprocessed/test_images/ch0.

IMAGE DIR: the dataset loader takes a single --image-dir, and the two image pools
live in different trees, so this also builds a merged directory of SYMLINKS
(nothing is copied, nothing is moved, the canonical preprocessed dirs are left
untouched and this is undoable with a single `rm -rf` of the output dir).
TBX11K sick images are already lung-crop preprocessed under
Data/external/tbx11k/preprocessed_full/ch0, matching how the challenge training
images were preprocessed, so no new preprocessing runs here.

ID COLLISION: TBX11K sick ids are s0001.. and challenge ids are bare integers, so
they cannot collide; the script asserts this rather than assuming it.

Usage (from Code/):
  python build_hardneg_train_csv.py
  python build_hardneg_train_csv.py --dry-run     # report only, write nothing
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

DEF_TRAIN_CSV = os.path.join(PROJECT_ROOT, "Data", "train.csv")
DEF_TEST_CSV = os.path.join(PROJECT_ROOT, "Data", "test.csv")
DEF_TBX_ROOT = os.path.join(PROJECT_ROOT, "Data", "external", "tbx11k")
DEF_MAIN_IMG = os.path.join(PROJECT_ROOT, "Data", "Preprocessed", "train_images", "ch0")
DEF_TEST_IMG = os.path.join(PROJECT_ROOT, "Data", "Preprocessed", "test_images", "ch0")
DEF_OUT_CSV = os.path.join(PROJECT_ROOT, "Data", "train_hardneg.csv")
DEF_OUT_IMG = os.path.join(PROJECT_ROOT, "Data", "Preprocessed", "train_hardneg", "ch0")

# Prefix for re-keyed old-internal-test rows. Cannot collide with challenge ids
# (bare integers) or TBX11K sick ids (s0001..).
TEST_PREFIX = "test_"

LABEL_COL = "TB/Normal"
# Distinct Modality_DICOM value so domain_gen.py's MixStyle treats the hard
# negatives as their own domain (which they genuinely are -- different country,
# different scanners) rather than silently merging them into a challenge
# modality. It never becomes a fold: fold assignment runs before these rows are
# concatenated back in (see train_task2.py).
AUX_MODALITY = "TBXSICK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=DEF_TRAIN_CSV)
    ap.add_argument("--test-csv", default=DEF_TEST_CSV)
    ap.add_argument("--tbx-root", default=DEF_TBX_ROOT)
    ap.add_argument("--main-image-dir", default=DEF_MAIN_IMG)
    ap.add_argument("--test-image-dir", default=DEF_TEST_IMG)
    ap.add_argument("--out-csv", default=DEF_OUT_CSV)
    ap.add_argument("--out-image-dir", default=DEF_OUT_IMG)
    ap.add_argument("--no-test", action="store_true",
                     help="do NOT fold the old internal test set into training "
                          "(keeps Data/test.csv usable as a benchmark)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tbx_img_dir = os.path.join(args.tbx_root, "preprocessed_full", "ch0")
    train_list = os.path.join(args.tbx_root, "full", "TBX11K", "lists", "TBX11K_train.txt")

    for p in (args.train_csv, args.main_image_dir, tbx_img_dir, train_list):
        if not os.path.exists(p):
            raise SystemExit(f"missing required input: {p}")

    # utf-8-sig: Data/train.csv carries a BOM, which otherwise corrupts the
    # first column name into '﻿new_id'.
    main_df = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    main_df["new_id"] = main_df["new_id"].astype(str)
    main_df["src"] = "train"

    # Old internal test set, re-keyed. Every one of its 1940 ids duplicates a
    # train id, so the prefix is mandatory, not cosmetic -- without it the
    # merged CSV would carry duplicate keys and the merged image dir would
    # silently resolve half the rows to the wrong picture.
    test_df = None
    if not args.no_test:
        for p in (args.test_csv, args.test_image_dir):
            if not os.path.exists(p):
                raise SystemExit(f"missing required input for --with-test: {p}")
        test_df = pd.read_csv(args.test_csv, encoding="utf-8-sig")
        test_df["src_id"] = test_df["new_id"].astype(str)
        test_df["new_id"] = TEST_PREFIX + test_df["src_id"]
        test_df["src"] = "test"

    with open(train_list) as fh:
        official_train = [ln.strip() for ln in fh if ln.strip()]
    sick_ids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in official_train
        if p.startswith("sick/")
    )
    if not sick_ids:
        raise SystemExit(f"no sick/ entries found in {train_list}")

    # Only keep ids whose preprocessed image actually exists.
    have = {os.path.splitext(f)[0] for f in os.listdir(tbx_img_dir)}
    missing = [i for i in sick_ids if i not in have]
    sick_ids = [i for i in sick_ids if i in have]
    if missing:
        print(f"!! {len(missing)} sick ids in the official train list have no "
              f"preprocessed image under {tbx_img_dir} -- skipped "
              f"(e.g. {missing[:3]})")

    aux_df = pd.DataFrame({
        "new_id": sick_ids,
        LABEL_COL: "Normal",
        "age": pd.NA,
        "gender": pd.NA,
        "Modality_DICOM": AUX_MODALITY,
    })
    aux_df["src"] = "tbx_sick"

    main_df["is_aux"] = 0
    aux_df["is_aux"] = 1
    parts = [main_df]
    if test_df is not None:
        test_df["is_aux"] = 0      # ordinary challenge rows: they join LOMO folds
        parts.append(test_df)
    parts.append(aux_df)
    out = pd.concat(parts, ignore_index=True)

    # Any duplicate key here means two different images would share one symlink
    # name -- silently training on the wrong picture for half of them. Hard fail.
    dupes = out["new_id"][out["new_id"].duplicated()].unique()
    if len(dupes):
        raise SystemExit(f"duplicate new_id after merge: {sorted(dupes)[:5]} "
                         f"({len(dupes)} total) -- refusing to build")

    print("=== composition ===")
    print(f"challenge train : {len(main_df):5d}  "
          f"({(main_df[LABEL_COL] == 'TB').sum()} TB / "
          f"{(main_df[LABEL_COL] == 'Normal').sum()} Normal)")
    if test_df is not None:
        print(f"old internal test: {len(test_df):4d}  "
              f"({(test_df[LABEL_COL] == 'TB').sum()} TB / "
              f"{(test_df[LABEL_COL] == 'Normal').sum()} Normal)  "
              f"[re-keyed {TEST_PREFIX}*, is_aux=0]")
    print(f"aux hard negs   : {len(aux_df):5d}  (all Normal, TBX11K official train split only)")
    print(f"combined        : {len(out):5d}  "
          f"({(out[LABEL_COL] == 'TB').sum()} TB / "
          f"{(out[LABEL_COL] == 'Normal').sum()} Normal)")
    neg = (out[LABEL_COL] == "Normal").sum()
    print(f"  -> {len(aux_df) / neg:.1%} of negatives are now abnormal-but-TB-negative")
    if test_df is not None:
        print("  !! Data/test.csv is now TRAINING DATA -- it is no longer a valid")
        print("     metric, and its historical numbers are no longer comparable.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    # src_id is only scaffolding for building the symlink names; `src` is kept
    # because it makes a row's origin obvious when debugging the merged CSV.
    out.drop(columns=["src_id"], errors="ignore").to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")

    os.makedirs(args.out_image_dir, exist_ok=True)

    # RELATIVE symlinks, deliberately. Absolute ones would break the moment the
    # tree is seen through a different mount point than the one this script ran
    # under -- which is guaranteed here: training runs inside docker with
    # PROJECT_ROOT bind-mounted at /workspace, and the cluster mounts the share
    # at /mnt/nas125/... while this may be built from a Mac at /Volumes/....
    # Relative links resolve correctly under all three.
    #
    # RESUMABLE, not rebuild-from-scratch: this runs over a network share that
    # does time out mid-loop (an earlier build died with ETIMEDOUT at ~7.3k of
    # 10.7k links). Re-running just fills in what is missing, so a timeout costs
    # only the remainder instead of the whole job. Stale links are still pruned,
    # so changing the split and re-running stays correct.
    # (source dir, [(link_name_id, source_file_id), ...]). The two differ only
    # for the re-keyed test rows: link `test_42.png` -> source `42.png`.
    sources = [
        (args.main_image_dir, [(i, i) for i in main_df["new_id"]]),
        (tbx_img_dir, [(i, i) for i in sick_ids]),
    ]
    if test_df is not None:
        sources.append((args.test_image_dir,
                        list(zip(test_df["new_id"], test_df["src_id"]))))

    want = {}
    for src_dir, pairs in sources:
        rel_src_dir = os.path.relpath(src_dir, args.out_image_dir)
        for link_id, src_id in pairs:
            if os.path.exists(os.path.join(src_dir, f"{src_id}.png")):
                want[f"{link_id}.png"] = os.path.join(rel_src_dir, f"{src_id}.png")

    # set() here is LOAD-BEARING, not tidiness. Over the SMB mount this tree
    # lives on, os.listdir returns massive duplicate entries -- a directory
    # holding 10,757 links was observed returning 37,063 names, the same entries
    # repeated. Any logic that counts or iterates the raw list (rather than its
    # unique set) will be wrong, which is why the progress numbers printed below
    # are derived from `want`/`existing` sets and never from len(os.listdir()).
    existing = set(os.listdir(args.out_image_dir))
    stale = existing - set(want)
    for f in stale:
        os.unlink(os.path.join(args.out_image_dir, f))
    if stale:
        print(f"pruned {len(stale)} stale links")

    made = kept = 0
    for name, target in want.items():
        path = os.path.join(args.out_image_dir, name)
        if name in existing:
            # Repair a link that points somewhere else (e.g. built by an older
            # version of this script, which used absolute targets).
            if os.path.islink(path) and os.readlink(path) == target:
                kept += 1
                continue
            os.unlink(path)
        os.symlink(target, path)
        made += 1

    print(f"symlinked {made} new, kept {kept} existing "
          f"({made + kept} total) -> {args.out_image_dir}")
    if made + kept != len(want):
        print(f"!! expected {len(want)} links -- re-run to finish")
    print("\nNext: sbatch sbatch/train_task2_rad_dino_hardneg.sbatch")


if __name__ == "__main__":
    main()
