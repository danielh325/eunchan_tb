#!/usr/bin/env python
"""Wires SWAD + token stylization into train_task2.py.

Adds two flags, both OFF by default so an unpatched-equivalent run is still
reachable:

    --tfs-vit          enable ATFS token stylization during training
    --tfs-layers N     blocks stylized per forward pass (default 3)
    --tfs-p P          probability of stylizing a given batch (default 0.5)
    --tfs-token-frac F fraction of tokens restyled (default 0.5)
    --swad             average weights over the validation-loss valley

Every edit is verified to match its anchor exactly once, and the script aborts
rather than guessing if train_task2.py has moved on. Run 02_launch_train.py
first (it applies --only-fold); this patches the same working copy.

    python 05_patch_dg.py --code /kaggle/working/Code
"""
import argparse
import os

EDITS = [
    # 1. imports
    ("import numpy as np\n",
     "import numpy as np\n"
     "from dg_layers import TokenStylizer          # ATFS-ViT token stylization\n"
     "from swad import SWAD                        # flat-minima weight averaging\n",
     "imports"),

    # 2. per-fold setup: build the stylizer + SWAD tracker
    ("    best_acc, best_state, epochs_since_best = -1.0, None, 0\n",
     "    _stylizer = None\n"
     "    if getattr(args, 'tfs_vit', False):\n"
     "        # cls_tokens: Dinov2/RAD-DINO has 1 CLS; CheXFound adds 4 register\n"
     "        # tokens, so 5 leading tokens must be protected there.\n"
     "        _n_cls = 5 if encoder_name == 'chexfound_vitl16' else 1\n"
     "        _stylizer = TokenStylizer(model, n_layers=args.tfs_layers, p=args.tfs_p,\n"
     "                                  token_frac=args.tfs_token_frac,\n"
     "                                  attention_aware=True, cls_tokens=_n_cls)\n"
     "    _swad = SWAD(model) if getattr(args, 'swad', False) else None\n"
     "    best_acc, best_state, epochs_since_best = -1.0, None, 0\n",
     "fold setup"),

    # 3. training mode on
    ("    for epoch in range(args.epochs):\n        model.train()\n",
     "    for epoch in range(args.epochs):\n        model.train()\n"
     "        if _stylizer is not None:\n            _stylizer.train()\n",
     "train mode"),

    # 4. resample which blocks stylize, once per batch
    ("        for i, batch in enumerate(train_loader):\n",
     "        for i, batch in enumerate(train_loader):\n"
     "            if _stylizer is not None:\n                _stylizer.resample_layers()\n",
     "per-batch resample"),

    # 5. eval mode off -- hooks inert, so validation and inference are
    #    bit-identical to an unpatched run
    ("        model.eval()\n        probs, labels = [], []\n",
     "        model.eval()\n"
     "        if _stylizer is not None:\n            _stylizer.eval()\n"
     "        probs, labels = [], []\n",
     "eval mode"),

    # 6. record a SWAD snapshot against validation BCE (a loss, not accuracy --
    #    SWAD's window selection needs a smooth curve, and val_acc on a small
    #    fold is a step function)
    ("        val_auc = safe_auc(labels, probs) if len(probs) else float(\"nan\")\n",
     "        val_auc = safe_auc(labels, probs) if len(probs) else float(\"nan\")\n"
     "        if _swad is not None and len(probs):\n"
     "            _p = np.clip(probs, 1e-6, 1 - 1e-6)\n"
     "            _vl = float(-(labels * np.log(_p) + (1 - labels) * np.log(1 - _p)).mean())\n"
     "            _swad.record(model, epoch, _vl)\n",
     "swad record"),

    # 7. after the epoch loop: overwrite the checkpoint with the SWAD average.
    #    The averaged weights are NOT the best epoch's weights, so val_acc /
    #    val_auc / tau must be recomputed against them -- otherwise the
    #    checkpoint reports metrics belonging to a different model, and the
    #    SWAD-vs-control comparison silently compares the control's best epoch
    #    against the control's best epoch. (predict_task2.py decides at a hard
    #    0.5 and only prints the stored tau, so this is a reporting bug rather
    #    than an inference one -- but the whole point of the run is the report.)
    ("    print(f\"saved {ckpt_path} (best val_acc={best_acc:.4f})\")\n",
     "    if _swad is not None and _swad.snapshots:\n"
     "        _swad.apply_to(model)\n"
     "        model.eval()\n"
     "        if _stylizer is not None:\n            _stylizer.eval()\n"
     "        _pr, _lb = [], []\n"
     "        with torch.no_grad():\n"
     "            for _x, _y, _d in val_loader:\n"
     "                _pr.append(torch.sigmoid(model(_x.to(device, non_blocking=True))).cpu().numpy())\n"
     "                _lb.append(_y.numpy())\n"
     "        _pr = np.concatenate(_pr) if _pr else np.array([])\n"
     "        _lb = np.concatenate(_lb) if _lb else np.array([])\n"
     "        if len(_pr):\n"
     "            _tau, _acc = sweep_tau_accuracy(_lb, _pr)\n"
     "            _auc = safe_auc(_lb, _pr)\n"
     "            print(f\"  [swad] re-evaluated averaged weights: \"\n"
     "                  f\"val_acc={_acc:.4f} (best-epoch was {best_acc:.4f}) val_auc={_auc:.4f}\")\n"
     "        else:\n"
     "            _tau, _acc, _auc = 0.5, float('nan'), float('nan')\n"
     "        if best_state is None:\n"
     "            raise RuntimeError(f'fold {fold_tag}: no epoch ever produced a valid '\n"
     "                               f'val_acc, so there is no checkpoint metadata to '\n"
     "                               f'attach SWAD weights to. Check the validation split.')\n"
     "        best_state = dict(best_state)\n"
     "        best_state['state_dict'] = {k: v.cpu() for k, v in model.state_dict().items()}\n"
     "        best_state['val_acc'], best_state['val_auc'], best_state['tau'] = _acc, _auc, _tau\n"
     "        best_state['swad'] = True\n"
     "        _tmp = ckpt_path + '.tmp'\n"
     "        torch.save(best_state, _tmp)\n"
     "        os.replace(_tmp, ckpt_path)\n"
     "        best_acc = _acc\n"
     "        print(f\"  overwrote {ckpt_path} with SWAD-averaged weights\")\n"
     "    if _stylizer is not None:\n        _stylizer.remove()\n"
     "    print(f\"saved {ckpt_path} (best val_acc={best_acc:.4f})\")\n",
     "swad apply"),

    # 8. CLI
    ("    ap.add_argument(\"--seed\", type=int, default=42)\n",
     "    ap.add_argument(\"--seed\", type=int, default=42)\n"
     "    ap.add_argument(\"--tfs-vit\", action=\"store_true\",\n"
     "                     help=\"ATFS-ViT token-level feature stylization (dg_layers.py)\")\n"
     "    ap.add_argument(\"--tfs-layers\", type=int, default=3)\n"
     "    ap.add_argument(\"--tfs-p\", type=float, default=0.5)\n"
     "    ap.add_argument(\"--tfs-token-frac\", type=float, default=0.5)\n"
     "    ap.add_argument(\"--swad\", action=\"store_true\",\n"
     "                     help=\"SWAD flat-minima weight averaging (swad.py)\")\n",
     "cli"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="/kaggle/working/Code")
    a = ap.parse_args()
    p = os.path.join(a.code, "train_task2.py")
    src = open(p).read()

    if "--tfs-vit" in src:
        print("[patch] already applied")
        return
    for anchor, body, what in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(
                f"[patch] anchor for '{what}' matched {n} times, expected 1.\n"
                f"train_task2.py has changed -- apply this edit by hand instead of\n"
                f"letting the script guess. Anchor was:\n{anchor!r}")
        src = src.replace(anchor, body)
    open(p, "w").write(src)

    for f in ("dg_layers.py", "swad.py", "t3a.py"):
        s = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
        if os.path.exists(s):
            open(os.path.join(a.code, f), "w").write(open(s).read())
    print(f"[patch] applied {len(EDITS)} edits to {p} and copied dg_layers/swad/t3a")
    print("\nRun the A/B:")
    print("  python train_task2.py ... --encoder rad_dino                # control")
    print("  python train_task2.py ... --encoder rad_dino --swad         # +SWAD")
    print("  python train_task2.py ... --encoder rad_dino --swad --tfs-vit  # +both")


if __name__ == "__main__":
    main()
