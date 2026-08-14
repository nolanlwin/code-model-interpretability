"""Causal interventions on variable-role representations: patching, ablation,
steering. Works for every role and every XLCoST language.

WHY THIS SHAPE
--------------
A probe shows information is *present*; it cannot show the model *uses* it
(Belinkov 2022; Hewitt & Liang 2019). Causal work needs a behavioural
readout, not the probe's own output -- scoring an intervention with the
probe that motivated it is circular.

The readout here is a real next-token prediction the model already makes:

    prompt      = the program up to (not including) a variable's LAST occurrence
    metric      = logit(target variable's first token)
                - logit(distractor variable's first token)

The distractor is another variable in the same program holding a DIFFERENT
role. Roles constrain where a variable may legally appear -- an accumulator
follows ``+=``, an index sits inside ``[]``, a flag heads an ``if`` -- so
this logit difference is a behaviour that depends on role, and it is a logit
difference rather than a probability (Zhang & Nanda, ICLR 2024).

Interventions act on the variable's EARLIER occurrences, where its role was
established, never on the readout position itself. So the causal question
is: does what the model built at those earlier sites change what it predicts
later? That needs no renaming, which is what makes this work for all seven
languages rather than Python alone.

    patch    interchange: overwrite the target's earlier-occurrence residuals
             with the distractor's, taken from the SAME forward pass (so
             positions need no cross-run alignment). Reported both
             directions: target<-distractor and distractor<-target.
    ablate   mean ablation against a stated reference distribution, or
             resample ablation from another program. NEVER zero ablation --
             zeros are off-distribution and inflate apparent importance.
    steer    add alpha * direction, where direction is the difference of role
             means computed on a HELD-OUT split. Controls use a random
             direction at matched norm.

CONTROLS (run by default, --no-controls to skip)
    random_position  same edit, same number of positions, at randomly chosen
                     non-variable positions. Separates "this variable's
                     representation matters" from "editing anything matters".
    random_direction steering only: matched-norm random vector.

    python scripts/causal.py run --occurrences outputs/role_occ/accumulator_python_train.jsonl \
        --canonical data/xlcost/python_train.jsonl --model-id Qwen/Qwen2.5-Coder-1.5B \
        --intervention patch --layers 0,4,8,12,16,20,24,28 --output outputs/causal/acc_py.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

INTERVENTIONS = ("patch", "ablate", "steer")


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or "(unknown)"
    except Exception:
        return "(unknown)"


# --------------------------------------------------------------------------
# Case construction (pure; no torch)
# --------------------------------------------------------------------------

def build_cases(occ_rows: list[dict], code_by_pid: dict, min_occ: int = 2,
                target_role: str | None = None) -> list[dict]:
    """One case per (program, target variable) with a different-role distractor.

    ``occ_rows`` must contain occurrences for MORE THAN ONE role: the
    distractor is by definition a variable holding a different role, so a
    single-role file yields zero cases. ``target_role`` selects which role is
    intervened on; every other role in the file remains eligible as a
    distractor.

    Requires the target to occur at least ``min_occ`` times: the last is the
    readout, the earlier ones are where the intervention lands.
    """
    by_pid: dict = defaultdict(list)
    for r in occ_rows:
        by_pid[r["problem_id"]].append(r)

    cases: list[dict] = []
    for pid, rows in by_pid.items():
        code = code_by_pid.get(pid)
        if not code:
            continue
        # Bind by (enclosing function, name), NOT by spelling. Two functions
        # in one program routinely both use `i` or `res`; merging them would
        # let an intervention edit a different binding than the one scored,
        # and would offer a distractor that is not in scope at the readout.
        # When no parser gave us a scope (C#), refuse rather than pretend the
        # program is one flat namespace.
        if any(r.get("scope_known") is False for r in rows):
            continue
        by_var: dict = defaultdict(list)
        for r in rows:
            by_var[(r.get("function"), r["variable"])].append(r)
        for v in by_var:
            by_var[v].sort(key=lambda r: r["source_span"][0])

        roles = {v: rows_[0].get("role") for v, rows_ in by_var.items()}
        for tgt, trows in by_var.items():
            if len(trows) < min_occ:
                continue
            if target_role is not None and roles.get(tgt) != target_role:
                continue
            # Distractor: a different variable with a DIFFERENT role, whose
            # own occurrences all start before the readout point (so it is
            # already in scope and a legal continuation there).
            readout_start = trows[-1]["source_span"][0]
            # The distractor must share the target's scope and appear before
            # the readout, so it is a legal continuation at that point.
            cand = [
                d for d, drows in by_var.items()
                if d != tgt and d[0] == tgt[0]
                and roles.get(d) != roles.get(tgt)
                and drows[0]["source_span"][0] < readout_start
            ]
            if not cand:
                continue
            dist = sorted(cand, key=lambda k: (k[1], str(k[0])))[0]
            cases.append({
                "problem_id": pid,
                "function": tgt[0],
                "target": tgt[1],
                "target_role": roles.get(tgt),
                "distractor": dist[1],
                "distractor_role": roles.get(dist),
                "readout_char": readout_start,
                "target_spans": [r["source_span"] for r in trows[:-1]],
                "distractor_spans": [r["source_span"] for r in by_var[dist]],
                "occurrence_id": trows[-1].get("occurrence_id"),
            })
    return cases


def effect_size(clean: float, intervened: float, floor: float | None = None) -> float:
    """Fraction of the clean logit difference destroyed by the intervention.

    With a floor (the fully-corrupted reference) this is the standard
    recovery fraction; without one it is the relative change against clean.
    Returns nan when the denominator is degenerate rather than a huge number
    off a near-zero clean effect.
    """
    denom = (clean - floor) if floor is not None else clean
    if not np.isfinite(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((clean - intervened) / denom)


def pick_control_positions(n: int, seq_len: int, avoid: set[int], rng) -> list[int]:
    """``n`` positions outside ``avoid`` -- the random-position control."""
    pool = [p for p in range(seq_len) if p not in avoid]
    if len(pool) < n:
        return []
    return sorted(rng.sample(pool, n))


def matched_norm_random(direction: np.ndarray, rng_np) -> np.ndarray:
    """A random vector with the same L2 norm as ``direction``."""
    v = rng_np.normal(size=direction.shape)
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.zeros_like(direction)
    return v * (np.linalg.norm(direction) / nv)


# --------------------------------------------------------------------------
# Model runner (torch; the only part that needs a GPU)
# --------------------------------------------------------------------------

class HFRunner:
    """Residual-stream reads and writes for a causal LM.

    LAYER INDEXING matches probe.py: layer 0 is the embedding output and
    layer i (1..L) is the output of transformer block i-1. Getting this
    off by one silently patches the wrong site, so the mapping lives in one
    place -- ``_module_for_layer`` -- and nowhere else.
    """

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto"):
        import torch
        from extract_activations import load_model_and_tokenizer
        self.torch = torch
        self.model, self.tok, self.device = load_model_and_tokenizer(model_id, device, dtype)
        base = getattr(self.model, "model", self.model)
        self.blocks = base.layers
        self.embed = base.embed_tokens
        self.n_layers = len(self.blocks)

    def _module_for_layer(self, layer: int):
        if layer == 0:
            return self.embed, False      # embeddings: output is a plain tensor
        return self.blocks[layer - 1], True   # blocks return a tuple

    def tokenize(self, code: str, max_length: int = 2048):
        from token_alignment import tokenize_for_alignment
        return tokenize_for_alignment(self.tok, code, max_length=max_length)

    def run(self, ids, edits=None, want_resid_layers=None):
        """Forward pass. ``edits`` = [(layer, positions, values)] applied to
        the residual stream, with ``values`` a NUMPY array -- callers never
        touch torch, which is what lets the driver run against a fake runner
        in tests on a machine with no GPU.
        Returns (logits[seq, vocab], {layer: resid[seq, hidden]})."""
        torch = self.torch
        handles, captured = [], {}

        def make_hook(layer, is_tuple, positions, values):
            def hook(_m, _inp, out):
                h = out[0] if is_tuple else out
                if values is not None:
                    h = h.clone()
                    v = torch.as_tensor(np.asarray(values), dtype=h.dtype, device=h.device)
                    h[0, positions, :] = v
                    return (h,) + tuple(out[1:]) if is_tuple else h
                return out
            return hook

        def make_capture(layer, is_tuple):
            def hook(_m, _inp, out):
                h = out[0] if is_tuple else out
                captured[layer] = h[0].detach().float().cpu().numpy()
            return hook

        for layer, positions, values in (edits or []):
            mod, is_tuple = self._module_for_layer(layer)
            handles.append(mod.register_forward_hook(
                make_hook(layer, is_tuple, positions, values)))
        for layer in (want_resid_layers or []):
            mod, is_tuple = self._module_for_layer(layer)
            handles.append(mod.register_forward_hook(make_capture(layer, is_tuple)))

        try:
            with torch.no_grad():
                enc = torch.tensor([ids], device=self.device)
                out = self.model(enc, use_cache=False)
            logits = out.logits[0].float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        return logits, captured


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--occurrences", required=True)
    r.add_argument("--canonical", required=True)
    r.add_argument("--model-id", required=True)
    r.add_argument("--intervention", required=True, choices=INTERVENTIONS)
    r.add_argument("--target-role", default=None,
                   help="which role to intervene on; the occurrences file must "
                        "carry several roles so a different-role distractor exists")
    r.add_argument("--layers", default="", help="comma-separated; default = every 4th")
    r.add_argument("--alpha", type=float, default=1.0, help="steering coefficient")
    r.add_argument("--max-cases", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-controls", action="store_true")
    r.add_argument("--device", default="auto")
    r.add_argument("--dtype", default="auto")
    r.add_argument("--max-length", type=int, default=2048)
    r.add_argument("--output", required=True)
    sub.add_parser("verify", help="self-check the pure logic, no model needed")
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        from causal_verify import run_verify
        return run_verify()
    from causal_run import run_experiment
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
