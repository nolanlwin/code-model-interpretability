"""Self-check for the causal pipeline that needs no GPU and no model.

Runs the REAL driver (causal_run.run_experiment) against a fake runner whose
behaviour is known exactly, so a wiring mistake -- patching the wrong
positions, reading the wrong layer, scoring the wrong token -- shows up as a
failed assertion rather than as a plausible-looking number on a GPU.

    python scripts/causal.py verify
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

HIDDEN, VOCAB = 8, 64
T_TOK, D_TOK = 11, 22


class FakeRunner:
    """A model whose logit difference is an EXACTLY known function of the
    residual stream, so every assertion below is a computed number rather
    than a shape check.

        resid[p, 0] = p                       (varies by position)
        logits[p, v] = mean(resid[:, 0]) * (v % 7)
        metric       = mean0 * ((t_id % 7) - (d_id % 7))

    An intervention that changes channel 0 at the edited positions therefore
    changes the metric by a predictable amount. An earlier version of this
    fake made every logit zero, so clean and intervened were both 0.0 and the
    no-op assertion passed vacuously -- the whole point of these numbers is
    that a wiring bug cannot hide behind them."""

    n_layers = 4

    def __init__(self):
        self.calls = 0
        self.seen_edits: list = []

    def tokenize(self, code, max_length=2048):
        # one token per character: offsets are trivially exact, so position
        # arithmetic in the driver is tested rather than the tokenizer's.
        ids = [ord(c) % VOCAB for c in code]
        offsets = [(i, i + 1) for i in range(len(code))]
        return ids, offsets, None

    def run(self, ids, edits=None, want_resid_layers=None):
        self.calls += 1
        seq = len(ids)
        resid = np.zeros((seq, HIDDEN), dtype=np.float32)
        resid[:, 0] = np.arange(seq, dtype=np.float32)   # position-dependent
        for (layer, positions, values) in (edits or []):
            self.seen_edits.append((layer, tuple(positions)))
            if values is not None:
                resid[list(positions), :] = np.asarray(values)
        mean0 = float(resid[:, 0].mean())
        logits = np.zeros((seq, VOCAB), dtype=np.float32)
        for v in range(VOCAB):
            logits[:, v] = mean0 * (v % 7)
        cap = {ly: resid.copy() for ly in (want_resid_layers or [])}
        return logits, cap


def run_verify() -> int:
    from causal import build_cases, effect_size, matched_norm_random, pick_control_positions
    from causal_run import run_experiment

    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'OK  ' if cond else 'FAIL'} {name}{'' if cond else '  ' + extra}")

    # ---- pure helpers -----------------------------------------------------
    check("effect_size: full destruction is 1.0", effect_size(2.0, 0.0) == 1.0)
    check("effect_size: no change is 0.0", effect_size(2.0, 2.0) == 0.0)
    check("effect_size: with floor uses recovery denominator",
          abs(effect_size(3.0, 2.0, floor=1.0) - 0.5) < 1e-9)
    check("effect_size: degenerate denominator -> nan",
          np.isnan(effect_size(0.0, 0.0)))
    d = np.array([3.0, 4.0])
    mn = matched_norm_random(d, np.random.default_rng(0))
    check("matched_norm_random preserves L2 norm",
          abs(np.linalg.norm(mn) - 5.0) < 1e-9, f"got {np.linalg.norm(mn)}")
    import random as _r
    pos = pick_control_positions(3, 10, {0, 1, 2}, _r.Random(0))
    check("control positions avoid the variable's own tokens",
          len(pos) == 3 and not ({0, 1, 2} & set(pos)), f"got {pos}")
    check("control positions refuse when the pool is too small",
          pick_control_positions(9, 10, set(range(9)), _r.Random(0)) == [])

    # ---- case construction ------------------------------------------------
    occ = [
        {"problem_id": "p1", "variable": "acc", "role": "accumulator",
         "source_span": [0, 3], "occurrence_id": "p1:f0:b0:o0"},
        {"problem_id": "p1", "variable": "idx", "role": "index_key",
         "source_span": [5, 8], "occurrence_id": "p1:f0:b1:o0"},
        {"problem_id": "p1", "variable": "acc", "role": "accumulator",
         "source_span": [10, 13], "occurrence_id": "p1:f0:b0:o1"},
    ]
    cases = build_cases(occ, {"p1": "acc  idx  acc  "})
    check("one case built for the two-occurrence target", len(cases) == 1, f"got {len(cases)}")
    if cases:
        c = cases[0]
        check("readout is the LAST occurrence", c["readout_char"] == 10, f"got {c['readout_char']}")
        check("intervention spans exclude the readout", c["target_spans"] == [[0, 3]],
              f"got {c['target_spans']}")
        check("distractor holds a different role", c["distractor_role"] != c["target_role"])
    check("a single-occurrence target yields no case",
          build_cases(occ[:2], {"p1": "acc  idx  "}) == [])
    same_role = [dict(r, role="accumulator") for r in occ]
    check("same-role-only program yields no case",
          build_cases(same_role, {"p1": "acc  idx  acc  "}) == [])
    # The distractor must hold a different role, so the target-role filter
    # selects the target without removing distractor candidates.
    check("target_role selects the intervened role",
          len(build_cases(occ, {"p1": "acc  idx  acc  "}, target_role="accumulator")) == 1)
    check("target_role with no matching target yields no case",
          build_cases(occ, {"p1": "acc  idx  acc  "}, target_role="iterator") == [])

    # ---- end-to-end through the real driver -------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        code = "acc  idx  acc  "
        (td / "occ.jsonl").write_text("\n".join(json.dumps(r) for r in occ))
        (td / "canon.jsonl").write_text(json.dumps({"problem_id": "p1", "code": code}))
        args = SimpleNamespace(
            occurrences=str(td / "occ.jsonl"), canonical=str(td / "canon.jsonl"),
            model_id="fake", intervention="ablate", layers="0,1", alpha=1.0,
            max_cases=10, seed=0, no_controls=False, device="cpu", dtype="fp32",
            max_length=2048, output=str(td / "out.json"), target_role=None)
        import causal_run
        causal_run.HFRunner = lambda *a, **k: FakeRunner()
        rc = run_experiment(args)
        res = json.loads((td / "out.json").read_text())
        check("driver returns 0", rc == 0)
        check("driver scored the case", res["n_cases_scored"] == 1,
              f"got {res['n_cases_scored']} skipped={res['skipped']}")
        check("both requested layers reported",
              sorted(s["layer"] for s in res["summary_by_layer"]) == [0, 1])
        check("provenance stamped", res["git_commit"] not in ("", None))
        check("controls recorded", res["controls"] is True)
        if res["summary_by_layer"]:
            s0 = res["summary_by_layer"][0]
            # Worked by hand: prompt is code[:10], so seq=10 and
            # mean(channel0) = 4.5. Target first token is 'a' (97 % 64 = 33,
            # 33 % 7 = 5); distractor is 'i' (105 % 64 = 41, 41 % 7 = 6).
            # clean = 4.5 * (5 - 6) = -4.5.
            check("clean metric is exactly -4.5",
                  abs(s0["clean_mean"] + 4.5) < 1e-6, f"got {s0['clean_mean']}")
            # Mean ablation writes 4.5 into positions 0,1,2, so channel 0
            # becomes [4.5,4.5,4.5,3..9] with mean 5.55 and the metric -5.55.
            # 1e-4, not 1e-6: the fake's residual is float32, so the
            # hand-computed 5.55 lands at 5.5500030517578125. That gap is the
            # dtype, not the wiring -- tightening it further would only test
            # numpy's float32.
            check("mean ablation moves the metric to exactly -5.55",
                  abs(s0["intervened_mean"] + 5.55) < 1e-4, f"got {s0['intervened_mean']}")
            check("effect fraction is exactly -0.2333",
                  abs(s0["effect_mean"] + (1.05 / 4.5)) < 1e-6, f"got {s0['effect_mean']}")
            check("random-position control was run and differs from clean",
                  "control_random_position_mean" in s0
                  and abs(s0["control_random_position_mean"] - s0["clean_mean"]) > 1e-9)

        # A single-role occurrence file can only ever yield zero cases; the
        # driver must say so rather than silently reporting n=0.
        single = [dict(r, role="accumulator") for r in occ]
        (td / "single.jsonl").write_text("\n".join(json.dumps(r) for r in single))
        args.occurrences = str(td / "single.jsonl")
        args.output = str(td / "single.json")
        try:
            run_experiment(args)
            check("single-role file is refused", False, "it was accepted")
        except SystemExit as e:
            check("single-role file is refused with a reason",
                  "single role" in str(e))
        args.occurrences = str(td / "occ.jsonl")

        # patch must edit BOTH directions
        args.intervention, args.output = "patch", str(td / "patch.json")
        run_experiment(args)
        pres = json.loads((td / "patch.json").read_text())
        check("patch reports the reverse direction",
              any("reverse_mean" in s for s in pres["summary_by_layer"]))

    print("\nALL PASS" if ok else "\nFAILURES")
    return 0 if ok else 1
