"""SWAD -- Stochastic Weight Averaging Densely (Cha et al., NeurIPS 2021).

Averages weights over a *dense* window of training iterates, chosen from the
validation-loss curve, to land in a flat minimum. Flat minima have a smaller
domain-generalization gap, which is why this is the single most reliable
non-architectural DG intervention published: reported +3.6pp over ERM, +2.8pp
over CORAL and +1.0pp over SAM, on the five DomainBed benchmarks, and the paper
notes it is "readily adaptable to existing DG methods without modification."

Why it fits this project better than anything else on the list:

  * Zero architecture change and **zero inference cost** -- the output is one
    set of weights, so the shipped container is unchanged in shape and speed.
  * With a frozen backbone only LoRA + head are trainable, so a snapshot is a
    few MB, not 400. Keeping a dense queue in RAM is free here in a way it
    isn't for a full fine-tune.
  * It composes with the token stylization in dg_layers.py -- one acts in
    weight space, the other in feature space, so they don't compete.
  * It replaces something you already do with a strictly better version. The
    pipeline currently probability-averages 4 LOMO fold checkpoints; that is an
    *output*-space ensemble costing 4 forward passes. SWAD averages in *weight*
    space for one.

The window selection ("LossValley" in the paper) is implemented faithfully in
shape: find where validation loss settles into its valley, and stop before it
climbs back out. Defaults are the paper's: n_converge=3, n_tolerance=6,
tolerance_ratio=0.3.
"""
import copy

import torch


class SWAD:
    def __init__(self, model, n_converge=3, n_tolerance=6, tolerance_ratio=0.3,
                 trainable_only=True):
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.trainable_only = trainable_only
        self.keys = [n for n, p in model.named_parameters()
                     if p.requires_grad or not trainable_only]
        self.snapshots = []      # list of (step, val_loss, state_dict-on-cpu)
        print(f"[swad] tracking {len(self.keys)} tensors "
              f"({'trainable only' if trainable_only else 'all params'})")

    def _snap(self, model):
        sd = dict(model.named_parameters())
        return {k: sd[k].detach().float().cpu().clone() for k in self.keys}

    def record(self, model, step, val_loss):
        """Call after each validation pass.

        DENSITY CAVEAT, stated plainly: the "Densely" in SWAD means sampling
        many times per epoch, and the paper's gain over plain SWA comes partly
        from that. Wired into train_task2.py's one-validation-per-epoch loop,
        this records ~15 snapshots per fold, which is closer to SWA than to
        true SWAD. It still helps -- weight averaging over the valley is the
        load-bearing idea -- but do not expect the paper's full +3.6pp from an
        epoch-level schedule, and do not report it as SWAD-proper without
        saying this.

        Making it dense needs a cheap validation signal several times per
        epoch. That is a worthwhile change if a run has time to spare; it is
        not worth the added risk three days out.
        """
        self.snapshots.append((step, float(val_loss), self._snap(model)))

    # -- window selection --------------------------------------------------
    def _window(self):
        losses = [l for _, l, _ in self.snapshots]
        n = len(losses)
        if n < self.n_converge:
            return 0, n

        # The valley is defined relative to its own floor: everything within
        # tolerance_ratio of the best loss ever seen counts as "in the valley".
        floor = min(losses)
        thresh = floor * (1.0 + self.tolerance_ratio)
        bottom = int(min(range(n), key=lambda i: losses[i]))

        # ts: where the curve first ENTERS the valley (on the way down), not
        # where it bottoms out. Averaging must include the descent side --
        # starting at the minimum throws away half the window and biases it
        # toward the overfitting side, which is the opposite of the intent.
        ts = next((i for i in range(n) if losses[i] <= thresh), bottom)

        # te: where it climbs back out, confirmed by n_tolerance consecutive
        # records above threshold, then rolled back to where the rise began.
        # Without the rollback the window swallows the overfitting tail.
        te, bad = n, 0
        for i in range(bottom, n):
            bad = bad + 1 if losses[i] > thresh else 0
            if bad >= self.n_tolerance:
                te = i - self.n_tolerance + 1
                break
        else:
            # Never confirmed (run ended first). Cut at the first crossing
            # after the bottom rather than averaging in a partial climb.
            te = next((i for i in range(bottom, n) if losses[i] > thresh), n)
        return ts, max(te, ts + 1)

    def averaged_state(self):
        """The SWAD weights: a uniform average over the selected window."""
        if not self.snapshots:
            raise RuntimeError("no snapshots recorded -- call record() during training")
        ts, te = self._window()
        window = self.snapshots[ts:te]
        acc = {k: torch.zeros_like(v) for k, v in window[0][2].items()}
        for _, _, sd in window:
            for k, v in sd.items():
                acc[k] += v
        for k in acc:
            acc[k] /= len(window)
        print(f"[swad] averaging {len(window)}/{len(self.snapshots)} snapshots "
              f"(steps {window[0][0]}..{window[-1][0]}, "
              f"val loss {window[0][1]:.4f}..{window[-1][1]:.4f})")
        return acc

    def apply_to(self, model):
        """Load the averaged weights in place. Do this once, after training,
        before saving the checkpoint you ship."""
        avg = self.averaged_state()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in avg:
                    p.copy_(avg[n].to(p.device, p.dtype))
        return model


def weight_soup(state_dicts, keys=None):
    """Uniform 'soup' over independently trained checkpoints.

    Use this to merge the 3 LOMO fold checkpoints into ONE model instead of
    probability-averaging them at inference. Averaging in weight space is what
    DiWA and Model Soups do for OOD, and it cuts inference cost by 3x, which
    matters for a container the organizers run on their own hardware.

    IMPORTANT caveat: weight averaging only works between checkpoints that
    stayed in the same loss basin -- i.e. fine-tuned from a *shared*
    initialization. That holds here (all folds start from the same frozen
    RAD-DINO and the same head init) but it is the assumption to check first if
    the soup underperforms the probability average. Validate on Shenzhen and
    Montgomery before shipping it; if it loses, keep probability-averaging.
    """
    keys = keys or list(state_dicts[0].keys())
    out = {}
    for k in keys:
        vs = [sd[k].float() for sd in state_dicts if k in sd]
        out[k] = sum(vs) / len(vs)
    return out
