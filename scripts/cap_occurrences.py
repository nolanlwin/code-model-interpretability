"""Cap an occurrence file per role, sampling WHOLE PROBLEMS.

Activation stores are [n_occurrences, layers+1, hidden] in fp16, so their size
is linear in occurrence count: the three LP4FM languages uncapped come to
~3.6 GB for a 1.5B model and ~11 GB for a 7B one. Capping keeps a GPU session
and its Drive checkpoint tractable.

Whole problems, not individual occurrences, and a fixed seed independent of
any probe seed -- the same rule probe.py uses for its own cap. Sampling
occurrences directly would split a program across the cap boundary and leave
some of its variables unrepresented; sampling problems keeps each program
either wholly in or wholly out, which is also what the grouped split
requires.

    python scripts/cap_occurrences.py --input outputs/role_occ/all_python_train.jsonl \
        --output outputs/role_occ/capped_python_train.jsonl \
        --roles accumulator iterator index_key --max-per-role 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def cap(rows: list[dict], roles: list[str], max_per_role: int, seed: int) -> tuple[list[dict], dict]:
    by_role: dict = defaultdict(list)
    for r in rows:
        if not roles or r.get("role") in roles:
            by_role[r["role"]].append(r)

    rng = np.random.default_rng(seed)
    keep_ids: set = set()
    stats: dict = {}
    for role, rrows in sorted(by_role.items()):
        problems = sorted({r["problem_id"] for r in rrows})
        rng.shuffle(problems)
        per_problem: dict = defaultdict(int)
        for r in rrows:
            per_problem[r["problem_id"]] += 1
        taken, n = [], 0
        for p in problems:
            if n >= max_per_role:
                break
            taken.append(p)
            n += per_problem[p]
        keep_ids |= {(role, p) for p in taken}
        stats[role] = {"available": len(rrows), "kept": n, "problems": len(taken)}

    out = [r for r in rows
           if (not roles or r.get("role") in roles) and (r["role"], r["problem_id"]) in keep_ids]
    return out, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--roles", nargs="*", default=[],
                    help="keep only these roles; empty keeps all")
    ap.add_argument("--max-per-role", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed and independent of probe seeds, so the same "
                         "sample is reproduced on every run")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    out, stats = cap(rows, args.roles, args.max_per_role, args.seed)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(dst) + ".stats.json")
    marker.unlink(missing_ok=True)          # never outlive the file it describes
    dst.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    summary = {"input": args.input, "output": str(dst), "seed": args.seed,
               "max_per_role": args.max_per_role, "roles": args.roles or "all",
               "occurrences_in": len(rows), "occurrences_out": len(out),
               "problems_out": len({r["problem_id"] for r in out}), "per_role": stats}
    marker.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
