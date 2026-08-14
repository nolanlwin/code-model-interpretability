"""Driver for scripts/causal.py: builds cases, runs the three interventions
with their controls, and writes a results file with problem-clustered CIs.

Kept separate from causal.py so the intervention algebra there stays
importable (and testable) without torch installed.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from causal import (  # noqa: E402
    HFRunner, build_cases, effect_size, git_commit,
    matched_norm_random, pick_control_positions,
)
from token_alignment import char_span_to_token_indices  # noqa: E402


def _read_jsonl(p: Path):
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            yield json.loads(ln)


def _first_token_id(ids, offsets, char_start):
    """Token id whose span begins the identifier at ``char_start``."""
    for i, (s, e) in enumerate(offsets):
        if s == char_start and e > s:
            return ids[i], i
    hits = char_span_to_token_indices(offsets, char_start, char_start + 1)
    return (ids[hits[0]], hits[0]) if hits else (None, None)


def _positions_for(spans, offsets):
    out: list[int] = []
    for s, e in spans:
        out.extend(char_span_to_token_indices(offsets, int(s), int(e)))
    return sorted(set(out))


def run_experiment(args) -> int:
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    code_by_pid = {r["problem_id"]: r["code"] for r in _read_jsonl(Path(args.canonical))}
    occ = list(_read_jsonl(Path(args.occurrences)))
    roles_present = {r.get("role") for r in occ}
    if len(roles_present) < 2:
        raise SystemExit(
            f"{args.occurrences} carries a single role ({roles_present}). The "
            "distractor must hold a DIFFERENT role, so this file can only "
            "produce zero cases. Extract with --role all."
        )
    cases = build_cases(occ, code_by_pid, target_role=args.target_role)
    rng.shuffle(cases)
    print(f"cases available: {len(cases)}; running up to {args.max_cases}", flush=True)

    runner = HFRunner(args.model_id, args.device, args.dtype)
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              or list(range(0, runner.n_layers + 1, max(1, runner.n_layers // 7))))

    # Steering direction: difference of role means, estimated on the FIRST
    # half of cases and applied only to the second half, so the direction is
    # never fitted on the cases it is scored on.
    holdout = len(cases) // 2 if args.intervention == "steer" else 0
    dir_by_layer: dict = {}
    if holdout:
        acc_t = {ly: [] for ly in layers}
        acc_d = {ly: [] for ly in layers}
        for c in cases[:holdout]:
            code = code_by_pid[c["problem_id"]]
            ids, offsets, _ = runner.tokenize(code, args.max_length)
            tp = _positions_for(c["target_spans"], offsets)
            dp = _positions_for(c["distractor_spans"], offsets)
            if not tp or not dp:
                continue
            _, cap = runner.run(ids, want_resid_layers=layers)
            for ly in layers:
                if ly in cap:
                    acc_t[ly].append(cap[ly][tp].mean(0))
                    acc_d[ly].append(cap[ly][dp].mean(0))
        for ly in layers:
            if acc_t[ly] and acc_d[ly]:
                dir_by_layer[ly] = (np.mean(acc_d[ly], 0) - np.mean(acc_t[ly], 0))
        print(f"steering directions fitted on {holdout} held-out cases", flush=True)

    rows: list[dict] = []
    skipped = defaultdict(int)
    for c in cases[holdout:holdout + args.max_cases]:
        code = code_by_pid[c["problem_id"]]
        prompt = code[: c["readout_char"]]
        if not prompt.strip():
            skipped["empty_prompt"] += 1
            continue
        ids, offsets, _ = runner.tokenize(code, args.max_length)
        t_id, _ = _first_token_id(ids, offsets, c["readout_char"])
        d_id, _ = _first_token_id(ids, offsets, c["distractor_spans"][0][0])
        if t_id is None or d_id is None or t_id == d_id:
            skipped["token_ids_unusable"] += 1
            continue
        p_ids, p_offsets, _ = runner.tokenize(prompt, args.max_length)
        if len(p_ids) < 2:
            skipped["prompt_too_short"] += 1
            continue
        tpos = _positions_for(c["target_spans"], p_offsets)
        dpos = _positions_for(c["distractor_spans"], p_offsets)
        if not tpos or not dpos:
            skipped["no_positions_in_prompt"] += 1
            continue

        def metric(logits):
            return float(logits[-1, t_id] - logits[-1, d_id])

        clean_logits, cap = runner.run(p_ids, want_resid_layers=layers)
        clean = metric(clean_logits)

        for ly in layers:
            if ly not in cap:
                continue
            resid = cap[ly]
            avoid = set(tpos) | set(dpos)
            ctrl_pos = pick_control_positions(len(tpos), len(p_ids), avoid, rng)

            def apply(positions, values):
                lg, _ = runner.run(p_ids, edits=[(ly, positions, np.asarray(values))])
                return metric(lg)

            entry = {"problem_id": c["problem_id"], "occurrence_id": c["occurrence_id"],
                     "target_role": c["target_role"], "distractor_role": c["distractor_role"],
                     "layer": ly, "clean": clean}

            if args.intervention == "patch":
                # Interchange, both directions, from the same forward pass.
                n = min(len(tpos), len(dpos))
                entry["intervened"] = apply(tpos[:n], resid[dpos[:n]])
                entry["reverse"] = apply(dpos[:n], resid[tpos[:n]])
            elif args.intervention == "ablate":
                # Mean ablation against this program's own residual mean --
                # an on-distribution reference. Never zeros.
                mean_vec = resid.mean(0, keepdims=True).repeat(len(tpos), 0)
                entry["intervened"] = apply(tpos, mean_vec)
            else:  # steer
                d = dir_by_layer.get(ly)
                if d is None:
                    continue
                entry["intervened"] = apply(tpos, resid[tpos] + args.alpha * d)
                if not args.no_controls:
                    rd = matched_norm_random(d, rng_np)
                    entry["control_random_direction"] = apply(tpos, resid[tpos] + args.alpha * rd)

            if not args.no_controls and ctrl_pos and args.intervention != "steer":
                src = resid[dpos[:len(ctrl_pos)]] if args.intervention == "patch" \
                    else resid.mean(0, keepdims=True).repeat(len(ctrl_pos), 0)
                entry["control_random_position"] = apply(ctrl_pos, src)

            entry["effect"] = effect_size(clean, entry["intervened"])
            rows.append(entry)

    by_layer: dict = defaultdict(list)
    for r_ in rows:
        by_layer[r_["layer"]].append(r_)

    def agg(vals):
        v = [x for x in vals if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    summary = []
    for ly in sorted(by_layer):
        g = by_layer[ly]
        s = {"layer": ly, "n": len(g),
             "clean_mean": agg([x["clean"] for x in g]),
             "intervened_mean": agg([x["intervened"] for x in g]),
             "effect_mean": agg([x["effect"] for x in g])}
        for k in ("reverse", "control_random_position", "control_random_direction"):
            if any(k in x for x in g):
                s[k + "_mean"] = agg([x[k] for x in g if k in x])
        summary.append(s)

    result = {
        "protocol_version": "1.0",
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "intervention": args.intervention,
        "occurrences": args.occurrences,
        "canonical": args.canonical,
        "layers": layers,
        "alpha": args.alpha if args.intervention == "steer" else None,
        "seed": args.seed,
        "controls": not args.no_controls,
        "n_cases_available": len(cases),
        "n_cases_scored": len({(r_["problem_id"], r_["occurrence_id"]) for r_ in rows}),
        "steering_holdout_cases": holdout,
        "skipped": dict(skipped),
        "metric": "logit(target first token) - logit(distractor first token) at the readout position",
        "summary_by_layer": summary,
        "cases": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{args.intervention} | {args.model_id}")
    print(f"{'layer':>6}{'n':>6}{'clean':>10}{'intervened':>12}{'effect':>9}"
          f"{'ctrl-pos':>10}{'ctrl-dir':>10}")
    for s in summary:
        print(f"{s['layer']:>6}{s['n']:>6}{s['clean_mean']:>10.3f}"
              f"{s['intervened_mean']:>12.3f}{s['effect_mean']:>9.3f}"
              f"{s.get('control_random_position_mean', float('nan')):>10.3f}"
              f"{s.get('control_random_direction_mean', float('nan')):>10.3f}")
    print(f"\nskipped: {dict(skipped) or 'none'}")
    print(f"wrote {out}")
    return 0
