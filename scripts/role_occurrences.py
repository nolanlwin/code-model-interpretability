"""Role-agnostic variable occurrences for ALL five roles, every XLCoST language.

The boolean workstream's ``xlcost_occurrences.py`` is AST-precise but only
knows ``boolean_flag``. Causal work needs the other four roles too, and the
role definitions for them live in ``pipeline/roles.py`` (Python AST, regex
elsewhere). This module bridges the two: it asks pipeline for
``{role: {names}}`` and then locates every real occurrence of those names in
the source, emitting the SAME row schema the boolean pipeline uses so
downstream tools do not care which extractor produced a row.

Occurrences are found by scanning identifiers outside strings and comments
(reusing the ``_code_mask`` lexer that ``baselines.py`` already relies on),
never by naive substring search -- ``i`` would otherwise match inside
``if``, ``print``, and every string literal in the program.

    python scripts/role_occurrences.py extract \
        --input data/xlcost/python_train.jsonl \
        --role accumulator \
        --output outputs/role_occ/accumulator_python_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from baselines import _code_mask  # noqa: E402  (string/comment-aware lexer)
from pipeline.roles import extract_roles  # noqa: E402

ROLES = ("index_key", "accumulator", "iterator", "boolean", "class_struct")

_IDENT = re.compile(r"[A-Za-z_]\w*")


def occurrence_rows(code: str, language: str, role: str, problem_id: str) -> list[dict]:
    """Every in-code occurrence of every variable pipeline assigns to ``role``."""
    try:
        by_role = extract_roles(code, language)
    except Exception:
        return []
    names = set(by_role.get(role) or ())
    if not names:
        return []

    is_code, _ = _code_mask(code)
    rows: list[dict] = []
    for m in _IDENT.finditer(code):
        name = m.group(0)
        if name not in names:
            continue
        s, e = m.start(), m.end()
        # Reject identifiers inside strings/comments. Checking only the first
        # character is enough: the lexer marks a literal contiguously.
        if not is_code[s]:
            continue
        line = code.count("\n", 0, s) + 1
        line_start = code.rfind("\n", 0, s) + 1
        rows.append({
            "problem_id": problem_id,
            "language": language,
            "variable": name,
            "role": role,
            "source_span": [s, e],
            "line": line,
            "col_offset": s - line_start,
            "end_col_offset": e - line_start,
        })

    rows.sort(key=lambda r: r["source_span"][0])
    # Stable ids in the protocol's shape. There is no reliable enclosing
    # function for the regex languages, so the function slot is fixed at f0
    # and the binding index orders variables by first appearance.
    order: dict = {}
    counter: Counter = Counter()
    for r in rows:
        v = r["variable"]
        if v not in order:
            order[v] = len(order)
        b, o = order[v], counter[v]
        counter[v] += 1
        r["occurrence_id"] = f"{problem_id}:{language}:f0:b{b}:o{o}"
    return rows


def cmd_extract(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_prog = n_rows = n_with = 0
    per_var = Counter()
    with out.open("w", encoding="utf-8") as f:
        for ln in Path(args.input).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            n_prog += 1
            wanted = ROLES if args.role == "all" else (args.role,)
            rows = [
                row
                for rl in wanted
                for row in occurrence_rows(rec["code"], rec["language"], rl, rec["problem_id"])
            ]
            if rows:
                n_with += 1
            for r in rows:
                # Span integrity, same gate as the boolean path: the span must
                # slice back to the variable or the row is a lie.
                s, e = r["source_span"]
                if rec["code"][s:e] != r["variable"]:
                    continue
                r["split"] = rec.get("split")
                f.write(json.dumps(r) + "\n")
                n_rows += 1
                per_var[r["variable"]] += 1

    stats = {
        "input": args.input,
        "role": args.role,
        "programs": n_prog,
        "programs_with_role": n_with,
        "occurrences_written": n_rows,
        "distinct_variables": len(per_var),
        "output": str(out),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--input", required=True, help="canonical XLCoST jsonl")
    e.add_argument("--role", required=True, choices=(*ROLES, "all"),
                   help="'all' writes every role into one file, which is what "
                        "the causal path needs: its distractor must hold a "
                        "different role than the target")
    e.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    return cmd_extract(args)


if __name__ == "__main__":
    sys.exit(main())
