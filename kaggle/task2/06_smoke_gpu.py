#!/usr/bin/env python
"""STEP 0 -- run this FIRST on Kaggle. ~5 minutes. Answers the three questions
that decide whether a multi-hour run is worth starting.

Everything in dg_layers.py / swad.py was verified against synthetic ViTs built
to mimic Dinov2's module naming. Nothing has touched real RAD-DINO weights on a
GPU. This closes that gap before you spend session time:

  1. Do the hooks find and wrap the REAL Dinov2 blocks under peft's wrapper?
  2. Does it survive fp16 autocast on a T4, and how much extra GPU memory does
     the fp32 statistics cast cost? (That cast fixes a genuine fp16 crash but
     increases activation memory -- an OOM risk I introduced.)
  3. How long is one training step, and therefore one fold?

Exit code is 0 only if all checks pass.

    python 06_smoke_gpu.py --code <repo>/Task2_0615/Task2/Code
"""
import argparse
import os
import sys
import time

import torch


def hr(t):
    return f"{t/3600:.1f} h" if t >= 3600 else f"{t/60:.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="Task2/Code (patched copy)")
    ap.add_argument("--weights", default=os.environ.get("TASK2_WEIGHTS_DIR", ""))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=5817,
                    help="images per fold (7757*3/4 by default)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=3)
    a = ap.parse_args()

    sys.path.insert(0, a.code)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.weights:
        os.environ["TASK2_WEIGHTS_DIR"] = a.weights

    ok = True

    # -- 0. environment ----------------------------------------------------
    print("=" * 64)
    if not torch.cuda.is_available():
        print("FAIL: no CUDA. Set Accelerator = GPU T4 x2.")
        return 1
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {p.name}  {p.total_memory/2**30:.1f} GB  sm_{cap[0]}{cap[1]}  "
          f"count={torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()} "
          f"(T4 = False, so fp16 autocast is the only option)")

    # -- 1. build the real model ------------------------------------------
    print("=" * 64)
    print("[1/3] building real rad_dino + LoRA ...")
    try:
        from model import TBClassifier
    except Exception as e:                                       # noqa: BLE001
        print(f"FAIL: cannot import model.py from {a.code}: {e}")
        return 1
    t0 = time.time()
    try:
        net = TBClassifier("rad_dino", pretrained=True).cuda()
    except Exception as e:                                       # noqa: BLE001
        print(f"FAIL: TBClassifier('rad_dino') -- {type(e).__name__}: {e}")
        print("      Usually TASK2_WEIGHTS_DIR is unset or rad_dino/ is missing.")
        return 1
    ntr = sum(q.numel() for q in net.parameters() if q.requires_grad)
    print(f"  built in {time.time()-t0:.0f}s  trainable={ntr:,} / "
          f"{sum(q.numel() for q in net.parameters()):,}")

    # -- 2. hooks against the REAL module tree -----------------------------
    print("=" * 64)
    print("[2/3] hooking real Dinov2 blocks ...")
    from dg_layers import TokenStylizer, find_transformer_blocks
    try:
        blocks = find_transformer_blocks(net)
    except Exception as e:                                       # noqa: BLE001
        print(f"FAIL: {e}")
        print("      Dump names with: [n for n,_ in net.named_modules()]")
        return 1
    print(f"  found {len(blocks)} blocks: {blocks[0][0]}  ...  {blocks[-1][0]}")
    if len(blocks) != 12:
        print(f"  WARNING: RAD-DINO is ViT-B/14 = 12 blocks, got {len(blocks)}. "
              f"Check for chunked/duplicated matches before trusting this.")
        ok = False

    x = torch.randn(a.batch_size, 3, 518, 518, device="cuda")
    st = TokenStylizer(net, n_layers=3, p=1.0, token_frac=0.5, attention_aware=True)

    # eval must be bit-identical -- this is the claim that inference is free
    st.eval()
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        r1 = net(x).float()
        r2 = net(x).float()
    same = torch.allclose(r1, r2)
    print(f"  eval bit-identical (inference unaffected): {same}")
    ok &= bool(same)

    # train mode under fp16 autocast -- the path that crashed before the fix
    net.train()
    st.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        st.resample_layers()
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
        d = (out.float() - r1).abs().mean().item()
        print(f"  fp16 train fwd OK, finite={bool(torch.isfinite(out).all())}, "
              f"mean|delta| vs eval = {d:.4f}")
        ok &= bool(torch.isfinite(out).all())
    except Exception as e:                                       # noqa: BLE001
        print(f"FAIL: fp16 forward with stylization -- {type(e).__name__}: {e}")
        return 1
    mem_on = torch.cuda.max_memory_allocated() / 2**30

    st.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        net(x)
    mem_off = torch.cuda.max_memory_allocated() / 2**30
    print(f"  peak GPU mem: stylization OFF {mem_off:.2f} GB -> ON {mem_on:.2f} GB "
          f"(+{mem_on-mem_off:.2f} GB from the fp32 stats cast)")
    if mem_on > 0.85 * p.total_memory / 2**30:
        print(f"  WARNING: within 15% of the card. Lower --batch-size or --tfs-layers.")
        ok = False

    # -- 3. timing ---------------------------------------------------------
    print("=" * 64)
    print("[3/3] timing a real training step ...")
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    y = torch.randint(0, 2, (a.batch_size,), device="cuda").float()
    net.train()

    def step(stylize):
        if stylize:
            st.train()
            st.resample_layers()
        else:
            st.eval()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for mode in (False, True):
        for _ in range(2):                       # warm-up
            step(mode)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            step(mode)
        torch.cuda.synchronize()
        per = (time.time() - t0) / 5
        ips = a.batch_size / per
        ep = a.n_train / ips
        fold = ep * a.epochs
        print(f"  stylization {'ON ' if mode else 'OFF'}: {per*1000:6.0f} ms/step  "
              f"{ips:5.1f} img/s  epoch~{hr(ep)}  fold~{hr(fold)}  "
              f"{a.folds} folds on 2 GPUs ~{hr(fold*((a.folds+1)//2))}")

    print("=" * 64)
    print(f"RESULT: {'PASS -- safe to launch' if ok else 'PROBLEMS ABOVE -- read them first'}")
    print("If the 2-GPU fold estimate exceeds ~10 h, cut --epochs to 10 before")
    print("committing a session. Kaggle kills the notebook at 12 h.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
