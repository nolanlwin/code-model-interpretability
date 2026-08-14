"""GPU sanity gate for the causal hooks. Run this BEFORE trusting any number.

`causal.py verify` proves the case-construction logic against a fake model.
It cannot prove that HFRunner's forward hooks write where they claim to --
that needs a real model, and a mis-wired hook does not raise, it quietly
patches the wrong site and returns plausible numbers.

Four checks, each of which fails loudly if the wiring is wrong:

  1. SELF-PATCH IS A NO-OP. Capture the residual at a layer, write the exact
     same values back at the same positions, and the logits must be
     unchanged. This is the single most important check: if it fails, the
     hook is writing to the wrong tensor, the wrong positions, or the wrong
     layer, and everything downstream is noise.
  2. CAPTURE ROUND-TRIPS at every layer, including 0 (embeddings) and the
     last block -- the two ends of the off-by-one that layer indexing invites.
  3. A LARGE PERTURBATION MOVES THE LOGITS at every layer. If some layer
     never moves, its hook is not attached.
  4. POSITION SPECIFICITY. Editing position p must change the prediction
     more than editing an unrelated position q. If not, the position
     arithmetic is wrong even though the hook fires.

    python scripts/causal.py sanity --model-id Qwen/Qwen2.5-Coder-1.5B
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

PROBE_CODE = (
    "def total(values):\n"
    "    acc = 0\n"
    "    for v in values:\n"
    "        acc += v\n"
    "    return acc\n"
)


def run_sanity(args) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("sanity needs torch and a loaded model, so it only runs where the\n"
              "experiment runs (Colab/GPU). `causal.py verify` covers everything\n"
              "checkable without a model and does run here.")
        return 2
    from causal import HFRunner

    runner = HFRunner(args.model_id, args.device, args.dtype)
    ids, offsets, _ = runner.tokenize(PROBE_CODE, 2048)
    n = len(ids)
    layers = sorted({0, 1, runner.n_layers // 2, runner.n_layers})
    print(f"model={args.model_id}  n_layers={runner.n_layers}  seq={n}")
    print(f"checking layers {layers}\n")

    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'OK  ' if cond else 'FAIL'} {name}{'' if cond else '  ' + extra}")

    base_logits, cap = runner.run(ids, want_resid_layers=layers)
    check("forward pass produced logits", base_logits.shape[0] == n,
          f"got {base_logits.shape}")

    # Measure the model's OWN run-to-run variation first. fp16 kernels are not
    # bitwise reproducible, so a fixed tolerance either masks a real bug or
    # fails on noise. Everything below is judged against this floor.
    again, _ = runner.run(ids)
    noise = float(np.abs(again - base_logits).max())
    tol = max(10 * noise, 1e-3)
    print(f"run-to-run noise floor: {noise:.6f}  ->  self-patch tolerance {tol:.6f}\n")

    for ly in layers:
        got = cap.get(ly)
        shape_ok = got is not None and got.ndim == 2 and got.shape[0] == n
        check(f"layer {ly} residual captured with shape [seq, hidden]", shape_ok,
              f"got {None if got is None else got.shape}, expected ({n}, hidden) -- "
              "a 1-D capture means the module output was unwrapped one level too far")

    pos = list(range(max(1, n // 3), max(2, n // 3 + 2)))
    for ly in layers:
        if ly not in cap:
            continue
        resid = cap[ly]

        # 1 + 2: writing captured values back must change nothing beyond the
        # model's own nondeterminism.
        same, _ = runner.run(ids, edits=[(ly, pos, resid[pos])])
        delta = float(np.abs(same - base_logits).max())
        check(f"layer {ly}: self-patch is a no-op", delta <= tol,
              f"max |logit delta| = {delta:.6f} vs noise floor {noise:.6f}. "
              f"{'That is ~' + str(round(delta / max(noise, 1e-9))) + 'x the floor, so it is structural: ' if delta > 10 * max(noise, 1e-9) else ''}"
              "the hook is writing to the wrong tensor, positions, or layer")

        # 3: a large perturbation must move the logits.
        big = resid[pos] + 50.0
        moved, _ = runner.run(ids, edits=[(ly, pos, big)])
        d_moved = float(np.abs(moved - base_logits).max())
        check(f"layer {ly}: a large edit moves the logits", d_moved > 1e-3,
              f"max |logit delta| = {d_moved:.6f} -- hook not attached")

    # 4: editing the position just before the readout must matter more than
    # editing the first position, at the last position's logits.
    ly = layers[len(layers) // 2]
    if ly in cap:
        resid = cap[ly]
        near = [n - 2]
        far = [0]
        a, _ = runner.run(ids, edits=[(ly, near, resid[near] + 50.0)])
        b, _ = runner.run(ids, edits=[(ly, far, resid[far] + 50.0)])
        da = float(np.abs(a[-1] - base_logits[-1]).max())
        db = float(np.abs(b[-1] - base_logits[-1]).max())
        check(f"layer {ly}: nearby edit outweighs a distant one",
              da > db, f"near={da:.4f} far={db:.4f} -- position arithmetic suspect")

    print("\nSANITY PASS -- hooks are wired correctly" if ok else
          "\nSANITY FAILED -- do NOT trust causal numbers from this build")
    return 0 if ok else 1
