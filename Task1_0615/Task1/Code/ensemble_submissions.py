#!/usr/bin/env python
"""Combine multiple already-scored models' OOF + test probabilities into one
weighted ensemble, with a tau freshly swept on the combined OOF (never on
test) -- CPU-only, no torch/GPU needed, just merges CSVs already produced by
predict.py / predict_ordfused.py.

Only meant for models that are valid under the real challenge test set, which
gives images only (no metadata): plain CNNs (predict.py) and OrdFused
checkpoints trained with --no-tabular (predict_ordfused.py --no-tabular). Any
submission from a tabular-fusion OrdFused checkpoint assumed metadata was
present at inference and its probabilities are not usable here -- this script
refuses known tabular-only submission filenames by default (see
_TABULAR_ONLY_PATTERN) rather than silently repeating that mistake.

Usage:
    python ensemble_submissions.py \\
        --oof densenet201:oof/densenet201_oof.csv \\
        --oof chexfound_vitl16_notab:oof/chexfound_vitl16_notab_oof.csv \\
        --test densenet201:../submissions/submission_densenet201.csv \\
        --test chexfound_vitl16_notab:../submissions/submission_ordfused_chexfound_vitl16_notab.csv \\
        --name best2

    # weight densenet201 twice as heavily as chexfound_vitl16_notab
    python ensemble_submissions.py --oof ... --test ... \\
        --weights densenet201=2,chexfound_vitl16_notab=1
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ID_COL, challenge_score, sweep_tau, sweep_tau_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "Data"))
DEFAULT_SUBMISSIONS_DIR = os.path.normpath(os.path.join(HERE, "..", "submissions"))

# submission_ordfused[_<encoder>].csv that does NOT end in _notab.csv came from
# a checkpoint trained with use_tabular=True -- its probabilities assumed
# metadata was present at inference and are not valid ensemble inputs for the
# real (image-only) test set.
_TABULAR_ONLY_PATTERN = re.compile(r"^submission_ordfused(_[a-zA-Z0-9]+)*\.csv$")


def _parse_tag_path_pairs(pairs):
    out = {}
    for item in pairs or []:
        if ":" not in item:
            raise ValueError(f"expected TAG:PATH, got {item!r}")
        tag, path = item.split(":", 1)
        out[tag] = path
    return out


def _parse_weights(spec, tags):
    if not spec:
        return {t: 1.0 for t in tags}
    weights = {}
    for item in spec.split(","):
        tag, w = item.split("=")
        weights[tag] = float(w)
    missing = set(tags) - set(weights)
    if missing:
        raise ValueError(f"--weights is missing entries for: {sorted(missing)}")
    return weights


def _check_not_tabular_only(tag, path, allow_tabular):
    base = os.path.basename(path)
    if _TABULAR_ONLY_PATTERN.match(base) and not base.endswith("_notab.csv"):
        if allow_tabular:
            print(f"!! WARNING: {tag} ({base}) looks like a tabular-fusion submission "
                  f"(metadata assumed present at inference) -- included anyway because "
                  f"--allow-tabular was passed.", file=sys.stderr)
        else:
            raise SystemExit(
                f"refusing to ensemble {tag} ({base}): filename matches a tabular-fusion "
                f"OrdFused submission, which assumed metadata was present at test time. "
                f"The real challenge test set gives images only. Use the encoder's "
                f"--no-tabular submission instead (e.g. submission_ordfused_<encoder>_notab.csv), "
                f"or pass --allow-tabular if you are certain this is intentional.")


def weighted_merge(frames_by_tag, value_col, weights):
    """frames_by_tag: {tag: DataFrame[ID_COL, value_col]} -> Series indexed by
    ID_COL with the weighted mean of value_col, restricted to IDs common to
    every tag (inner join -- an ensemble member missing a row is a bug, not
    something to silently paper over)."""
    merged = None
    for tag, df in frames_by_tag.items():
        s = df.set_index(ID_COL)[value_col].rename(tag)
        merged = s.to_frame() if merged is None else merged.join(s, how="inner")
    total_w = sum(weights.values())
    combined = sum(merged[tag] * weights[tag] for tag in frames_by_tag) / total_w
    return combined


def optimize_weights(oof_frames, oof_y, sweep_fn, tags, n_rounds=4, grid=None):
    """Coordinate-ascent search over per-member weights, maximizing the same
    OOF metric sweep_tau*/sweep_tau_accuracy already uses for tau -- never
    touches test. Cycles through members, grid-searching each one's weight
    with the others held fixed, until a full round makes no improvement.
    Cheap: combined_oof is a small in-memory merge, no GPU/model involved."""
    grid = grid if grid is not None else np.linspace(0.0, 2.0, 21)
    weights = {t: 1.0 for t in tags}

    def score_for(w):
        combined = weighted_merge(oof_frames, "oof_prob", w)
        _, s = sweep_fn(oof_y, combined.to_numpy())
        return s

    best_score = score_for(weights)
    for _ in range(n_rounds):
        improved = False
        for t in tags:
            best_w = weights[t]
            for w in grid:
                trial = dict(weights)
                trial[t] = float(w)
                s = score_for(trial)
                if s > best_score:
                    best_score, best_w = s, float(w)
            if best_w != weights[t]:
                weights[t] = best_w
                improved = True
        if not improved:
            break
    return weights, best_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", action="append", required=True, metavar="TAG:PATH",
                    help="repeatable; TAG:PATH to a --dump-oof CSV (our_id, oof_prob)")
    ap.add_argument("--test", action="append", required=True, metavar="TAG:PATH",
                    help="repeatable; TAG:PATH to that model's submission_*.csv "
                         "(must have our_id, prob_cavity)")
    ap.add_argument("--weights", default=None,
                    help="TAG=weight,TAG=weight,... (default: uniform)")
    ap.add_argument("--optimize-weights", action="store_true",
                    help="ignore --weights and instead grid-search per-member weights "
                         "by coordinate ascent on the combined OOF accuracy (never on "
                         "test) -- usually beats flat/uniform averaging once there are "
                         "3+ members of uneven strength")
    ap.add_argument("--tau-objective", default="accuracy", choices=["accuracy", "challenge"])
    ap.add_argument("--allow-tabular", action="store_true",
                    help="skip the tabular-fusion-submission safety check (see module docstring)")
    ap.add_argument("--name", required=True, help="output submission name")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--submissions-dir", default=DEFAULT_SUBMISSIONS_DIR)
    args = ap.parse_args()

    oof_paths = _parse_tag_path_pairs(args.oof)
    test_paths = _parse_tag_path_pairs(args.test)
    if set(oof_paths) != set(test_paths):
        raise SystemExit(f"--oof and --test tag sets must match: "
                          f"oof={sorted(oof_paths)} test={sorted(test_paths)}")
    tags = sorted(oof_paths)
    for tag in tags:
        _check_not_tabular_only(tag, test_paths[tag], args.allow_tabular)

    oof_frames = {t: pd.read_csv(p) for t, p in oof_paths.items()}
    test_frames = {t: pd.read_csv(p) for t, p in test_paths.items()}

    df_tr = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    oof_y_lookup = dict(zip(df_tr[ID_COL], (df_tr.cavity != "none").astype(int)))

    sweep_fn = sweep_tau_accuracy if args.tau_objective == "accuracy" else sweep_tau

    # oof_y needs a fixed row order to score candidate weightings against --
    # derive it from the actual inner-joined ID set (weighted_merge's join
    # structure is the same regardless of the weight values used), not just
    # one member's raw CSV, in case members don't cover identical our_ids.
    joined_ids = weighted_merge(oof_frames, "oof_prob", {t: 1.0 for t in tags}).index
    oof_y_ordered = joined_ids.to_series().map(oof_y_lookup).to_numpy()
    if pd.isna(oof_y_ordered).any():
        raise SystemExit("some OOF our_id values were not found in train.csv -- "
                          "check the --oof CSVs came from the same data-dir")

    if args.optimize_weights:
        weights, _ = optimize_weights(oof_frames, oof_y_ordered.astype(int), sweep_fn, tags)
        print(f"[{args.name}] optimized weights via OOF coordinate ascent: {weights}")
    else:
        weights = _parse_weights(args.weights, tags)

    combined_oof = weighted_merge(oof_frames, "oof_prob", weights)
    oof_y = combined_oof.index.to_series().map(oof_y_lookup).to_numpy()
    if pd.isna(oof_y).any():
        raise SystemExit("some OOF our_id values were not found in train.csv -- "
                          "check the --oof CSVs came from the same data-dir")
    tau, oof_score = sweep_fn(oof_y.astype(int), combined_oof.to_numpy())
    print(f"[{args.name}] combined OOF tau*={tau:.3f} {args.tau_objective}={oof_score:.4f} "
          f"(n={len(combined_oof)}, members={tags}, weights={weights})")

    combined_test = weighted_merge(test_frames, "prob_cavity", weights)
    pred = (combined_test.to_numpy() >= tau).astype(int)

    out = pd.DataFrame({ID_COL: combined_test.index, "prob_cavity": combined_test.to_numpy(),
                         "pred_cavity": pred})

    df_te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    te_true_lookup = dict(zip(df_te[ID_COL], (df_te.cavity != "none").astype(int)))
    y_true = out[ID_COL].map(te_true_lookup)
    if not y_true.isna().any():
        s, acc, dice = challenge_score(y_true.to_numpy(), combined_test.to_numpy(), tau)
        print(f"[{args.name}] TEST score={s:.4f} acc={acc:.4f} dice={dice:.4f} (tau from OOF)")
        out["cavity_true"] = df_te.set_index(ID_COL).loc[out[ID_COL], "cavity"].to_numpy()

    os.makedirs(args.submissions_dir, exist_ok=True)
    out_path = os.path.join(args.submissions_dir, f"submission_{args.name}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
