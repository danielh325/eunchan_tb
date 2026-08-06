"""CLI training entrypoint for Task 2. Structure mirrors Task1's
train_ordfused.py: per-encoder, per-fold loop, gradient accumulation sized
per ENCODER_FAMILIES (RAD-DINO/CheXFound have far more attention tokens than
DenseNet-121's 224px input, so their activation memory is much larger --
smaller physical batch + accumulation keeps effective batch size comparable
without changing training dynamics), checkpoint saved on best val metric.

Default evaluation is leave-one-modality-out (the plan's primary metric --
Modality_DICOM is a real domain signal, unlike a random stratified split
which never tests cross-domain transfer at all); --fold-mode stratified is
available for direct comparability to the baseline notebook's number.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (LABEL_COL, LABEL_MAP, make_leave_one_modality_out_folds,
                     make_stratified_folds, safe_auc, set_seed, sweep_tau_accuracy)
from dataset import Cfg, TBDataset, TBDatasetMaskSupervised
from domain_gen import domain_aware_mixup
from encoders import ENCODER_FAMILIES, FOUNDATION_ENCODER_NAMES
from model import TBClassifier


def dice_loss(pred, target, eps=1e-6):
    """pred, target: (B,H,W) in [0,1]. 1 - soft Dice, averaged over the batch."""
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def irm_penalty(logits, y, criterion):
    """IRMv1 penalty (Arjovsky et al. 2019, "Invariant Risk Minimization"):
    squared gradient of the loss w.r.t. a dummy scale=1.0 multiplying the
    logits. Near-zero when the classifier is equally good regardless of
    which environment produced this subset of the batch -- i.e. NOT
    exploiting an environment-specific shortcut (a modality-specific
    artifact, in our case) that happens to correlate with the label in
    that environment but not others. Reuses whatever forward pass already
    produced `logits`, no extra model call needed."""
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = criterion(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def irm_loss(logits, y, dom, criterion, penalty_weight):
    """Splits a batch by Modality_DICOM value (`dom`, already available for
    domain_aware_mixup) and treats each modality present as one IRM
    "environment": mean per-environment BCE (the ERM term) plus
    penalty_weight * mean per-environment irm_penalty. Environments with
    <2 samples in this batch are skipped (too few for a meaningful
    gradient-based penalty estimate). Falls back to plain BCE if the batch
    only has one usable environment (penalty is 0/undefined with a single
    environment by IRM's own definition -- it measures INVARIANCE ACROSS
    environments).

    Rescales by 1/penalty_weight when penalty_weight > 1, matching the
    reference IRM implementation -- otherwise the total loss magnitude
    balloons with the (large, annealed-up) penalty weight and destabilizes
    the optimizer purely from loss-scale, not from the penalty's actual
    information content.
    """
    envs = {}
    for i, d in enumerate(dom):
        envs.setdefault(d, []).append(i)
    erm_terms, penalty_terms = [], []
    for idxs in envs.values():
        if len(idxs) < 2:
            continue
        idx_t = torch.tensor(idxs, device=logits.device)
        env_logits, env_y = logits[idx_t], y[idx_t]
        erm_terms.append(criterion(env_logits, env_y))
        penalty_terms.append(irm_penalty(env_logits, env_y, criterion))

    if len(erm_terms) < 2:
        # Fewer than 2 usable environments in this batch -- nothing to
        # measure invariance ACROSS, fall back to plain BCE on the full batch.
        return criterion(logits, y)

    erm = torch.stack(erm_terms).mean()
    penalty = torch.stack(penalty_terms).mean()
    total = erm + penalty_weight * penalty
    if penalty_weight > 1.0:
        total = total / penalty_weight
    return total


def build_optimizer(model, lr, weight_decay=1e-4):
    # Only hand AdamW params that actually require grad -- with LoRA, most
    # backbone params are frozen (same rationale as Task1's train_ordfused.py
    # build_optimizer: avoids a pile of dead param groups).
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_one_fold(args, train_df, val_df, encoder_name, fold_tag, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fam = ENCODER_FAMILIES.get(encoder_name, {})
    batch_size = args.batch_size or fam.get("batch_size", 16)
    accum_steps = args.accum_steps or fam.get("accum_steps", 1)
    # Resolve once here (not left implicit inside TBClassifier's constructor)
    # so the ACTUAL rank/alpha used gets stored in the checkpoint below --
    # predict_task2.py needs this to reconstruct the exact same LoRA shape
    # at inference time. Historical default (every checkpoint trained
    # before 2026-07-31) was a flat 16/16 for every encoder regardless of
    # size -- ENCODER_FAMILIES now sets a larger default for chexfound_vitl16
    # specifically (see encoders.py), so this resolution can no longer be
    # left to "whatever TBClassifier defaults to today" without silently
    # breaking old checkpoints' shapes on reload.
    resolved_lora_r = args.lora_r if args.lora_r is not None else fam.get("lora_r", 16)
    resolved_lora_alpha = args.lora_alpha if args.lora_alpha is not None else fam.get("lora_alpha", 16)

    cfg = Cfg(encoder_name, image_dir=args.image_dir, use_bone_suppression=args.bone_suppress,
              mask_dir=args.mask_dir, channel_mode=args.channel_mode)
    use_mask_guide = args.mask_guide_dir is not None
    if use_mask_guide:
        train_ds = TBDatasetMaskSupervised(train_df, cfg, args.mask_guide_dir,
                                            train=True, aug_strength=args.aug_strength)
    else:
        train_ds = TBDataset(train_df, cfg, train=True, aug_strength=args.aug_strength)
    val_ds = TBDataset(val_df, cfg, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = TBClassifier(encoder_name, pretrained=True, use_mixstyle=not args.no_mixstyle,
                          use_mask_attention=use_mask_guide,
                          lora_r=resolved_lora_r, lora_alpha=resolved_lora_alpha,
                          lora_dropout=args.lora_dropout,
                          use_glori_head=args.use_glori_head,
                          glori_n_layers=args.glori_n_layers).to(device)
    optimizer = build_optimizer(model, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Checkpoint path resolved BEFORE the loop: every improvement is flushed to
    # disk immediately (see below) rather than only after all epochs finish.
    # The old write-once-at-the-end behaviour meant a scancel mid-fold threw
    # away the entire fold -- job 120352 lost a 29-epoch DX fold whose best
    # epoch (1) had been reached in the first 20 minutes.
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{encoder_name}_{fold_tag}.pth")

    best_acc, best_state, epochs_since_best = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            if use_mask_guide:
                x, y, dom, mask_t = batch
                mask_t = mask_t.to(device, non_blocking=True)
            else:
                x, y, dom = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # NOTE: domain-aware mixup blends x/y across samples -- skip it
            # when mask-guiding (mixing images would also require mixing
            # masks, not wired here) OR when IRM is on: IRM's per-environment
            # split needs each sample cleanly attributable to ONE modality,
            # which mixup's cross-domain blending directly breaks.
            if not use_mask_guide and not args.use_irm:
                x, y = domain_aware_mixup(x, y, dom, alpha=args.domain_mixup_alpha)

            if use_mask_guide:
                logits, attn = model(x, return_attn=True)
                # attn is at the (coarse) last-feature-map resolution, e.g.
                # 7x7 for densenet121/convnext_tiny at 224px; mask_t is at
                # full img_size (TBDatasetMaskSupervised resizes the real
                # mask, not the model's downsampled attention) -- upsample
                # attn to match before the Dice loss, same convention
                # gradcam_task2.py uses for visualization upsampling.
                attn_up = F.interpolate(attn.unsqueeze(1), size=mask_t.shape[-2:],
                                         mode="bilinear", align_corners=False).squeeze(1)
            else:
                logits = model(x)

            if args.use_irm:
                penalty_weight = args.irm_lambda if epoch >= args.irm_anneal_epochs else 1.0
                loss = irm_loss(logits, y, dom, criterion, penalty_weight)
            else:
                loss = criterion(logits, y)

            if use_mask_guide:
                loss = loss + args.mask_guide_lambda * dice_loss(attn_up, mask_t)
            loss = loss / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * accum_steps * x.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y, _dom in val_loader:
                x = x.to(device, non_blocking=True)
                p = torch.sigmoid(model(x)).cpu().numpy()
                probs.append(p)
                labels.append(y.numpy())
        probs = np.concatenate(probs) if probs else np.array([])
        labels = np.concatenate(labels) if labels else np.array([])
        tau, val_acc = sweep_tau_accuracy(labels, probs) if len(probs) else (0.5, float("nan"))
        val_auc = safe_auc(labels, probs) if len(probs) else float("nan")
        print(f"[{encoder_name}][{fold_tag}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} tau={tau:.2f}",
              flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_since_best = 0
            best_state = dict(
                state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                encoder_name=encoder_name, img_size=cfg.img_size,
                norm_mean=cfg.norm_mean, norm_std=cfg.norm_std,
                fold_tag=fold_tag, val_acc=val_acc, val_auc=val_auc, tau=tau,
                use_mask_attention=use_mask_guide,
                lora_r=resolved_lora_r, lora_alpha=resolved_lora_alpha,
                use_glori_head=args.use_glori_head, glori_n_layers=args.glori_n_layers,
                channel_mode=args.channel_mode,
            )
            # Write through a temp file + atomic rename so an interrupt during
            # the (multi-GB for ViT-L) save can't leave a truncated .pth that
            # predict_task2.py / --skip-existing-folds would later pick up as
            # if it were a finished fold.
            tmp_path = ckpt_path + ".tmp"
            torch.save(best_state, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f"  checkpointed {ckpt_path} (val_acc={val_acc:.4f})", flush=True)
        else:
            epochs_since_best += 1
            # Opt-in only (--patience 0 = off, the default), so every run
            # trained before this existed stays reproducible.
            if args.patience > 0 and epochs_since_best >= args.patience:
                print(f"[{encoder_name}][{fold_tag}] early stop at epoch {epoch+1}: "
                      f"no val_acc improvement in {args.patience} epochs "
                      f"(best={best_acc:.4f})", flush=True)
                break

    print(f"saved {ckpt_path} (best val_acc={best_acc:.4f})")
    return best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODER_FAMILIES))
    ap.add_argument("--csv", default="/workspace/Data/train.csv")
    ap.add_argument("--image-dir", required=True, help="dir of preprocess.py's ch0 (lung-cropped raw) images")
    ap.add_argument("--mask-dir", default=None,
                     help="dir of preprocess.py's ch2 lung masks (only if it was run with "
                          "--mask-channel) -- replaces the redundant 3rd duplicate channel; "
                          "omit to disable (default 3x-duplicate channel stack). Passive: model "
                          "is free to ignore it. Mutually exclusive in practice with "
                          "--mask-guide-dir (that one actively supervises attention instead).")
    ap.add_argument("--mask-guide-dir", default=None,
                     help="dir of preprocess.py's ch2 lung masks, used to ACTIVELY supervise "
                          "TBClassifier's mask-guided attention gate via a Dice loss (see "
                          "model.py's use_mask_attention) -- penalizes attention outside the "
                          "lung silhouette during training, rather than passively offering the "
                          "mask as an extra input channel. Works for CNN backbones (spatial "
                          "1x1-conv gate) and rad_dino/chexfound_vitl16 (per-patch-token linear "
                          "gate, added 2026-07-30 -- see model.py's _forward_vit_gated, NOT yet "
                          "smoke-tested end to end); omit to disable.")
    ap.add_argument("--mask-guide-lambda", type=float, default=1.0,
                     help="weight on the Dice attention-supervision loss, added to the BCE "
                          "classification loss; only used when --mask-guide-dir is set")
    ap.add_argument("--out-dir", default="/workspace/checkpoints")
    ap.add_argument("--skip-existing-folds", action="store_true",
                     help="resume a partially-completed run: reuse any fold whose "
                          "<out-dir>/<encoder>_<fold>.pth already exists instead of "
                          "retraining it, taking its val_acc/val_auc from the checkpoint "
                          "for the summary table. Added 2026-08-02 for the chexfound_glori "
                          "run that was scancel'd 12h in (job 119810) with CR and DX "
                          "already saved. A reused checkpoint is REJECTED (hard error, not "
                          "a silent skip) unless the training-config fields it stores "
                          "match this invocation -- reusing a fold trained under different "
                          "settings would quietly corrupt the comparison. Default off, so "
                          "every other job's behavior is unchanged.")
    ap.add_argument("--fold-mode", choices=["leave_one_modality_out", "stratified"],
                     default="leave_one_modality_out")
    ap.add_argument("--n-folds", type=int, default=5, help="only used for --fold-mode stratified")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop a fold after this many epochs with no val_acc "
                          "improvement (0 = off, the historical behaviour). Best "
                          "checkpoint selection is unaffected -- the best epoch is "
                          "flushed to disk as it happens either way.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=None, help="override ENCODER_FAMILIES default")
    ap.add_argument("--accum-steps", type=int, default=None, help="override ENCODER_FAMILIES default")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--aug-strength", choices=["none", "light", "strong"], default="strong")
    ap.add_argument("--no-mixstyle", action="store_true")
    ap.add_argument("--lora-r", type=int, default=None,
                     help="LoRA rank for foundation encoders (rad_dino/chexfound_vitl16); "
                          "default (None) defers to this encoder's own ENCODER_FAMILIES "
                          "lora_r -- BOTH are 16 (chexfound's r=32 was reverted 2026-07-31; "
                          "see encoders.py). Override to sweep rank explicitly, e.g. for an "
                          "ablation.")
    ap.add_argument("--lora-alpha", type=int, default=None,
                     help="LoRA alpha; default (None) defers to ENCODER_FAMILIES lora_alpha "
                          "the same way as --lora-r.")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--use-glori-head", action="store_true",
                     help="chexfound_vitl16 ONLY -- CheXFound's own validated downstream-"
                          "adaptation recipe (the GLoRI paper): backbone stays FULLY frozen "
                          "(no LoRA at all, --lora-r/--lora-alpha ignored), and the head reads "
                          "concatenated CLS tokens from the last --glori-n-layers transformer "
                          "blocks plus one learned-query attention-pooled 'local' feature over "
                          "the last layer's patch tokens (a single-query simplification of "
                          "GLoRI's disease-specific multi-query cross-attention, scoped down "
                          "from its original 40-disease multi-label setting to our single "
                          "TB/Normal binary label). Mutually exclusive with --mask-guide-dir "
                          "(forward() takes a completely different path, mask-guided attention "
                          "gating never runs). See model.py's TBClassifier._forward_glori.")
    ap.add_argument("--glori-n-layers", type=int, default=4,
                     help="number of final transformer blocks to read CLS tokens from; only "
                          "used when --use-glori-head. 4 matches CheXFound's own paper.")
    ap.add_argument("--bone-suppress", action="store_true",
                     help="ignored when --channel-mode multi_view (bone suppression is always "
                          "computed as channel 2 in that mode); applies the old single-view "
                          "on/off behavior otherwise.")
    ap.add_argument("--channel-mode", choices=["duplicate", "multi_view"], default="duplicate",
                     help="'duplicate' (default, matches every checkpoint trained before "
                          "2026-07-31): one processed view copied into all 3 channels. "
                          "'multi_view': channel 0=original, 1=CLAHE, 2=bone-suppressed -- "
                          "see common.py's preprocess_image docstring. MUST match at predict "
                          "time (predict_task2.py's own --channel-mode), same requirement as "
                          "--bone-suppress/--mask-dir already have.")
    ap.add_argument("--domain-mixup-alpha", type=float, default=0.0,
                     help="Beta(alpha,alpha) mixing strength for domain-aware cross-modality "
                          "mixup; 0 disables (the DEFAULT since 2026-08-04). Pairs each sample "
                          "with a partner from a different Modality_DICOM value when the batch "
                          "has one available. Ignored (forced off) when --use-irm is set -- see "
                          "that flag's help. DEFAULT CHANGED 0.2 -> 0.0: the DG-machinery "
                          "ablation (jobs 120343/120353/120354/120576, "
                          "rad_dino_dgablation_*_eval_summary.csv) showed mixup does not earn "
                          "its place -- removing it from the full recipe improved external "
                          "F1@0.5 on both Shenzhen (0.8954 -> 0.8966) and Montgomery (0.8780 -> "
                          "0.8976), and mixup with no image-level augmentation was the WORST of "
                          "all 7 configs tested (Montgomery 0.7451), below the no-DG floor it "
                          "is supposed to improve on. Pass --domain-mixup-alpha 0.2 explicitly "
                          "only to reproduce pre-2026-08-04 runs such as the original "
                          "checkpoints/ baseline.")
    ap.add_argument("--use-irm", action="store_true",
                     help="Invariant Risk Minimization (Arjovsky et al. 2019) over "
                          "Modality_DICOM as the environment variable -- treats each modality "
                          "present in the TRAINING split (e.g. 3 of 4 under "
                          "--fold-mode leave_one_modality_out) as one IRM environment, and "
                          "penalizes the classifier for learning anything that predicts TB "
                          "differently depending on which one produced the image (a "
                          "modality-specific shortcut), rather than a genuinely invariant "
                          "TB signal. See train_task2.py's irm_loss/irm_penalty. Forces "
                          "domain-aware mixup off (mixup's cross-domain blending breaks IRM's "
                          "clean per-environment split). Compatible with --mask-guide-dir "
                          "(the Dice term just adds on top); NOT yet run end to end -- verify "
                          "with a short 1-2 epoch sanity pass before a full job.")
    ap.add_argument("--irm-lambda", type=float, default=10.0,
                     help="target IRM penalty weight, applied after --irm-anneal-epochs "
                          "(annealed UP from 1.0, not down -- per the original IRM paper's own "
                          "recipe: applying the full penalty from epoch 0 tends to prevent the "
                          "model from learning anything useful at all before it has a "
                          "reasonable representation to regularize). Only used when --use-irm.")
    ap.add_argument("--irm-anneal-epochs", type=int, default=5,
                     help="epochs of penalty_weight=1.0 (plain multi-environment BCE, no real "
                          "invariance pressure yet) before switching to --irm-lambda. Only used "
                          "when --use-irm.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.use_glori_head:
        if args.encoder != "chexfound_vitl16":
            raise SystemExit("--use-glori-head only supports --encoder chexfound_vitl16 "
                              "(see model.py's TBClassifier guard).")
        if args.mask_guide_dir is not None:
            raise SystemExit("--use-glori-head and --mask-guide-dir are mutually exclusive -- "
                              "forward() takes the GLoRI path unconditionally when "
                              "use_glori_head is set, so mask-guided attention gating "
                              "would silently never run.")

    # No more hard block here as of 2026-07-30: model.py's TBClassifier now
    # has a ViT-token gate path (_forward_vit_gated) for rad_dino/chexfound,
    # via encoders.py's _RadDinoWrapper/_CheXFoundWrapper.forward_features
    # (added alongside it). Not yet smoke-tested end to end -- run a short
    # 1-2 epoch sanity pass before committing a full multi-day job.

    set_seed(args.seed)
    df = pd.read_csv(args.csv)
    df = df[df[LABEL_COL].astype(str).str.strip().str.lower().isin(LABEL_MAP)].reset_index(drop=True)

    if args.fold_mode == "leave_one_modality_out":
        df, modalities = make_leave_one_modality_out_folds(df)
        fold_names = modalities
    else:
        df = make_stratified_folds(df, n_folds=args.n_folds, seed=args.seed)
        fold_names = [f"fold{i}" for i in range(args.n_folds)]

    results = []
    for f, tag in enumerate(fold_names):
        train_df = df[df["fold"] != f].reset_index(drop=True)
        val_df = df[df["fold"] == f].reset_index(drop=True)
        if len(val_df) == 0:
            print(f"skip fold {tag}: empty")
            continue

        ckpt_path = os.path.join(args.out_dir, f"{args.encoder}_{tag}.pth")
        if args.skip_existing_folds and os.path.exists(ckpt_path):
            prev = torch.load(ckpt_path, map_location="cpu")
            # A fold is only reusable if it was trained the way THIS run trains.
            # Anything else silently mixes configs across folds, which is exactly
            # the "changed several things at once" failure the ablations exist to
            # avoid -- so mismatches abort rather than warn.
            expected = dict(
                encoder_name=args.encoder,
                use_glori_head=args.use_glori_head,
                glori_n_layers=args.glori_n_layers,
                channel_mode=args.channel_mode,
            )
            mismatched = {
                k: (prev.get(k), v) for k, v in expected.items() if prev.get(k) != v
            }
            if mismatched:
                raise SystemExit(
                    f"!! refusing to reuse {ckpt_path}: it was trained with "
                    + ", ".join(f"{k}={got!r} (this run wants {want!r})"
                                for k, (got, want) in mismatched.items())
                    + ". Delete it to retrain this fold, or drop --skip-existing-folds."
                )
            print(f"reuse fold {tag}: {ckpt_path} "
                  f"(val_acc={prev['val_acc']:.4f} val_auc={prev['val_auc']:.4f} tau={prev['tau']:.2f})",
                  flush=True)
            results.append(dict(fold=tag, val_acc=prev["val_acc"], val_auc=prev["val_auc"]))
            continue

        best = run_one_fold(args, train_df, val_df, args.encoder, tag, args.out_dir)
        results.append(dict(fold=tag, val_acc=best["val_acc"], val_auc=best["val_auc"]))

    res_df = pd.DataFrame(results)
    print(res_df)
    print(f"mean val_acc={res_df['val_acc'].mean():.4f} mean val_auc={res_df['val_auc'].mean():.4f}")


if __name__ == "__main__":
    main()
