"""Scope-correct identifier renaming for the C1-C5 conditions (PROTOCOL.md §2).

Python only (v1). Renames are SPAN EDITS on the original source — never
``ast.unparse`` — so formatting, comments and strings are untouched and the
model sees the same code with only identifiers changed.

Conditions:
  C1 neutral numeric  every renameable local -> v1, v2, ... (declaration order)
  C2 single char      every renameable local -> a, b, c, ...
  C3 all-same         every TARGET-role variable -> one identifier (x)
  C4 random nouns     every renameable local -> seeded noun, disjoint pools
  C5 misleading       TARGET variables -> names from the PARTNER role's pool
                      (boolean's partner is INDEX, per the protocol table)

Scope discipline (v1): a function is renameable only if its symtable scope has
NO child scopes (no nested def/lambda/comprehension) and no global/nonlocal —
anything else is left verbatim and counted. Renaming targets function-local
bindings only: parameters and locally-assigned names; globals, builtins,
attributes, imports and other functions' names are never touched.

Gates, applied per program and per function:
  - every edited span must slice back to the old identifier (else drop program)
  - renamed program must re-parse
  - the boolean occurrence-id set of TOUCHED functions must be IDENTICAL to
    the original's (computed by the same pipeline: xlcost_occurrences.
    program_occurrence_rows) — renaming must not change what is detected
  - a new-name collision with any identifier in the program -> next candidate

Output: renamed canonical JSONL (same problem_id) + renamed occurrence JSONL
(same occurrence_ids, new spans) + stats. Paired deltas join on occurrence_id.

    uv run python scripts/rename_corpus.py run --condition C5 \
        --canonical data/xlcost/python_valid.jsonl \
        --occurrences outputs/xlcost_occ/python_valid.jsonl \
        --out-canonical data/xlcost_renamed/C5/python_valid.jsonl \
        --out-occurrences outputs/xlcost_occ_renamed/C5/python_valid.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import symtable
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from variable_occurrences import line_start_indices, name_char_span  # noqa: E402
from xlcost_occurrences import program_occurrence_rows  # noqa: E402

CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]

NOUN_POOL = [
    "apple", "brick", "candle", "drum", "engine", "fabric", "garden", "harbor",
    "island", "jacket", "kettle", "ladder", "marble", "needle", "orchard",
    "pillow", "quartz", "ribbon", "saddle", "timber", "urn", "violin",
    "wagon", "yarn", "zephyr", "anchor", "basket", "canyon", "dagger",
    "ember", "falcon", "glacier", "hammer", "ivory", "jungle", "kernel_",
    "lantern", "meadow", "nectar", "oyster", "parcel", "quiver", "rocket",
    "sculpt", "tunnel", "umbrella", "velvet", "walnut", "xylo", "zinc",
]

# Partner-role pools for C5 (PROTOCOL §2: boolean -> index). Must stay disjoint
# from other roles' pools when those land; asserted at import of both.
INDEX_POOL = ["i", "j", "k", "idx", "index", "pos", "ii", "jj", "kk", "ind", "ix", "iy"]

PARTNER_POOL = {"boolean": INDEX_POOL}


def _seeded_rng(problem_id: str, condition: str) -> np.random.Generator:
    h = int(hashlib.sha1(f"{problem_id}:{condition}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def _arg_char_span(code: str, starts: list[int], node: ast.arg) -> tuple[int, int]:
    s = starts[node.lineno - 1] + node.col_offset
    return s, s + len(node.arg)


def _all_identifiers(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
    return out


def renameable_functions(code: str) -> tuple[list, set[str], dict]:
    """Top-level functions whose scope is flat and safe to rename.

    Returns (list of (ast fn node, ordered local names), skip counters).
    """
    tree = ast.parse(code)
    table = symtable.symtable(code, "<prog>", "exec")
    scopes = {}
    for child in table.get_children():
        if child.get_type() == "function":
            scopes[(child.get_name(), child.get_lineno())] = child
    kw_call_names = {
        k.arg
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg
    }
    skips = {"nested_scope": 0, "global_nonlocal": 0, "kwarg_coupling": 0, "unsupported_binder": 0}
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        st = scopes.get((node.name, node.lineno))
        if st is None or st.get_children():
            skips["nested_scope"] += 1
            continue
        symbols = st.get_symbols()
        if any(s.is_global() and s.is_assigned() for s in symbols) or any(
            isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node)
        ):
            skips["global_nonlocal"] += 1
            continue
        # Binding forms our span editor does not rewrite (the bound name is a
        # bare string on the node, not an ast.Name): `except E as err`,
        # match-case captures. symtable still lists them as locals, so C1/C2/C4
        # would rename their USES but not the binding -> unbound code.
        if any(
            (isinstance(n, ast.ExceptHandler) and n.name)
            or (isinstance(n, getattr(ast, "MatchAs", ())) and getattr(n, "name", None))
            or (isinstance(n, getattr(ast, "MatchStar", ())) and getattr(n, "name", None))
            for n in ast.walk(node)
        ):
            skips["unsupported_binder"] += 1
            continue
        local_names = [
            s.get_name()
            for s in symbols
            if s.is_local() and not s.is_imported() and not s.is_namespace()
        ]
        if any(nm in kw_call_names for nm in local_names):
            # A call somewhere uses keyword=..., matching a local we would
            # rename; if that call targets THIS function the program breaks.
            skips["kwarg_coupling"] += 1
            continue
        # Order by first appearance (params first, then body order).
        order: dict[str, int] = {}
        for a in ast.walk(ast.Module(body=[node], type_ignores=[])):
            nm = None
            if isinstance(a, ast.arg):
                nm = a.arg
            elif isinstance(a, ast.Name):
                nm = a.id
            if nm in local_names and nm not in order:
                order[nm] = (getattr(a, "lineno", 0), getattr(a, "col_offset", 0)) and len(order)
        ordered = sorted(local_names, key=lambda nm: order.get(nm, 10**9))
        out.append((node, ordered))
    return out, _all_identifiers(tree), skips


def build_mapping(
    condition: str,
    ordered_locals: list[str],
    targets: set[str],
    taken: set[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    """old -> new for one function. Only keys being renamed appear."""
    which = ordered_locals if condition in ("C1", "C2", "C4") else [
        nm for nm in ordered_locals if nm in targets
    ]
    mapping: dict[str, str] = {}
    used = set(taken)

    def free(base: list[str]):
        """First unused candidate; falls back to numbered variants, unbounded."""
        def candidates():
            yield from base
            k = 2
            while k < 10_000:
                for b in base:
                    yield f"{b}{k}"
                k += 1

        for c in candidates():
            if c not in used and not keyword.iskeyword(c):
                used.add(c)
                return c
        raise RuntimeError("name pool exhausted")

    for n, old in enumerate(which):
        if condition == "C1":
            new = free([f"v{n + 1}", f"v{n + 1}_"])
        elif condition == "C2":
            letters = [chr(ord("a") + (n + i) % 26) for i in range(26)]
            new = free(letters)
        elif condition == "C3":
            new = mapping.get("__same__") or free(["x", "xx"])
            mapping["__same__"] = new
        elif condition == "C4":
            pool = list(NOUN_POOL)
            rng.shuffle(pool)
            new = free(pool)
        else:  # C5
            pool = list(PARTNER_POOL["boolean"])
            rng.shuffle(pool)
            new = free(pool)
        if old != new:
            mapping[old] = new
    mapping.pop("__same__", None)
    return mapping


def rename_program(
    code: str, problem_id: str, condition: str, target_vars_by_fn: dict[str, set[str]]
) -> tuple[str | None, dict]:
    """Return (renamed code or None, stats). None => program dropped/unchanged."""
    stats = {"functions_renamed": 0, "functions_skipped": 0, "edits": 0}
    try:
        fns, all_ids, skips = renameable_functions(code)
    except (SyntaxError, ValueError, RecursionError):
        return None, {**stats, "drop_reason": "parse"}
    stats["functions_skipped"] = sum(skips.values())
    stats.update({f"skip_{k}": v for k, v in skips.items()})
    if not fns:
        return None, {**stats, "drop_reason": "no_renameable_function"}

    starts = line_start_indices(code)
    rng = _seeded_rng(problem_id, condition)
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement)
    taken = set(all_ids)
    for node, ordered_locals in fns:
        targets = target_vars_by_fn.get(node.name, set())
        if condition in ("C3", "C5") and not targets:
            continue
        try:
            mapping = build_mapping(condition, ordered_locals, targets, taken, rng)
        except RuntimeError:
            return None, {**stats, "drop_reason": "pool_exhausted"}
        if not mapping:
            continue
        taken |= set(mapping.values())
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in mapping:
                s, e = name_char_span(code, sub)
                if code[s:e] != sub.id:
                    return None, {**stats, "drop_reason": "span_mismatch"}
                edits.append((s, e, mapping[sub.id]))
            elif isinstance(sub, ast.arg) and sub.arg in mapping:
                s, e = _arg_char_span(code, starts, sub)
                if code[s:e] != sub.arg:
                    return None, {**stats, "drop_reason": "span_mismatch"}
                edits.append((s, e, mapping[sub.arg]))
        stats["functions_renamed"] += 1

    if not edits:
        return None, {**stats, "drop_reason": "no_edits"}
    stats["edits"] = len(edits)
    edits = sorted(edits, key=lambda t: t[0])
    new_code = code
    for s, e, repl in reversed(edits):
        new_code = new_code[:s] + repl + new_code[e:]
    try:
        ast.parse(new_code)
    except SyntaxError:
        return None, {**stats, "drop_reason": "reparse_failed"}
    return new_code, {**stats, "edits_list": edits}


def map_span(edits: list[tuple[int, int, str]], s: int, e: int):
    """Map an original [s, e) span into the edited string.

    Edits are whole-identifier spans, so an occurrence span either coincides
    with one edit exactly (a renamed identifier — returns the new span and
    name) or overlaps none (shifted only). Partial overlap returns None.
    """
    shift = 0
    for es, ee, repl in edits:
        if ee <= s:
            shift += len(repl) - (ee - es)
            continue
        if (es, ee) == (s, e):
            return s + shift, s + shift + len(repl), repl
        if es >= e:
            break
        return None  # partial overlap — malformed
    return s + shift, e + shift, None


def cmd_run(args: argparse.Namespace) -> int:
    canon = {}
    for ln in Path(args.canonical).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            canon.setdefault(r["problem_id"], r)

    occ_by_problem: dict[str, list[dict]] = {}
    for ln in Path(args.occurrences).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            occ_by_problem.setdefault(r["problem_id"], []).append(r)

    wanted_problems = set(occ_by_problem)
    if args.sample_ids:
        ids = set(json.loads(Path(args.sample_ids).read_text(encoding="utf-8")))
        wanted_problems = {i.split(":")[0] for i in ids}

    out_c = Path(args.out_canonical)
    out_o = Path(args.out_occurrences)
    out_c.parent.mkdir(parents=True, exist_ok=True)
    out_o.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_drop = n_id_mismatch = 0
    drop_reasons: dict[str, int] = {}
    with out_c.open("w", encoding="utf-8") as fc, out_o.open("w", encoding="utf-8") as fo:
        for pid in sorted(wanted_problems):
            rec = canon.get(pid)
            rows = occ_by_problem.get(pid)
            if rec is None or not rows:
                continue
            language = rec["language"]
            if language != "Python":
                raise SystemExit("v1 renames Python only")
            targets: dict[str, set[str]] = {}
            for r in rows:
                # rename_program looks targets up by ast node.name, i.e. the
                # BARE function name. role_occurrences.py puts a scope id
                # ("name@start-end") in `function` to keep same-named functions
                # distinct, and carries the bare name in `function_name`.
                # Keying on the scope id would match nothing, so C3/C5 -- the
                # conditions that rename only the targets -- would drop every
                # program with drop_reason "no_edits". C1/C2/C4 rename all
                # locals and would mask the bug.
                fn_key = str(r.get("function_name") or r.get("function"))
                targets.setdefault(fn_key, set()).add(str(r["variable"]))
            new_code, st = rename_program(rec["code"], pid, args.condition, targets)
            if new_code is None:
                n_drop += 1
                dr = st.get("drop_reason", "unknown")
                drop_reasons[dr] = drop_reasons.get(dr, 0) + 1
                continue
            edits = st["edits_list"]

            # Carry ORIGINAL occurrence_ids by span mapping (C3 merges bindings,
            # so recomputed ids cannot match — the id must travel with the span).
            # Gate: extraction on the renamed code must find an occurrence at
            # every mapped span with the same label.
            #
            # The re-extraction has to use the SAME producer that made the
            # input, or the gate compares two different labelling schemes and
            # rejects everything. The boolean workstream writes
            # `occurrence_type` via program_occurrence_rows;
            # role_occurrences.py writes `role` and finds a different set of
            # sites entirely.
            if args.label_field == "role":
                from role_occurrences import ROLES as _ROLES
                from role_occurrences import occurrence_rows as _role_rows
                new_rows = [row for rl in _ROLES
                            for row in _role_rows(new_code, language, rl, pid)]
                pstats = {"parse_error": not new_rows}
            else:
                new_rows, pstats = program_occurrence_rows(language, new_code, pid)
            if pstats["parse_error"]:
                n_id_mismatch += 1
                continue
            sites = {tuple(r["source_span"]): r[args.label_field] for r in new_rows}
            mapped, gate_ok = [], True
            for r in rows:
                m = map_span(edits, int(r["source_span"][0]), int(r["source_span"][1]))
                if m is None:
                    gate_ok = False
                    break
                ns, ne, new_name = m
                var = new_name if new_name is not None else r["variable"]
                if new_code[ns:ne] != var or sites.get((ns, ne)) != r[args.label_field]:
                    gate_ok = False
                    break
                mapped.append(
                    {
                        **r,
                        "variable": var,
                        "original_variable": r["variable"],
                        "source_span": [ns, ne],
                        "condition": args.condition,
                        "split": rec.get("split"),
                    }
                )
            if not gate_ok:
                n_id_mismatch += 1
                continue
            fc.write(json.dumps({**rec, "code": new_code, "condition": args.condition}) + "\n")
            for r in mapped:
                fo.write(json.dumps(r) + "\n")
            n_ok += 1

    stats = {
        "condition": args.condition,
        "problems_considered": len(wanted_problems),
        "programs_renamed": n_ok,
        "programs_dropped": n_drop,
        "drop_reasons": drop_reasons,
        "id_preservation_failures": n_id_mismatch,
        "out_canonical": str(out_c),
        "out_occurrences": str(out_o),
    }
    Path(str(out_o) + ".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    code = (
        "def demo(flag, items):\n"
        "    ok = True\n"
        '    msg = "flag ok"  # flag in string/comment must NOT change\n'
        "    if flag and ok:\n"
        "        return flag\n"
        "    return ok\n"
        "\n"
        "def uses_comp(xs):\n"
        "    return [x * 2 for x in xs]\n"
        "\n"
        "print(demo(True, []))\n"
    )
    targets = {"demo": {"flag", "ok"}}
    orig_rows = program_occurrence_rows("Python", code, "deadbeef")[0]
    checks = []
    for cond, expect in [("C1", "v1"), ("C2", "a"), ("C3", "x"), ("C5", None)]:
        new, st = rename_program(code, "deadbeef", cond, targets)
        okay = new is not None and '"flag ok"' in new and "# flag in string" in new
        if new is not None:
            okay &= ast.parse(new) is not None
            # Note: on Python 3.12+ comprehensions are inlined (PEP 709), so
            # uses_comp is legitimately renameable under C1/C2/C4.
            okay &= st["functions_renamed"] >= 1
            if expect:
                okay &= expect in new
            # Span-mapped id carryover: every original occurrence must map to a
            # site the extractor finds in the renamed code, same type.
            edits = st["edits_list"]
            sites = {
                tuple(r["source_span"]): r["occurrence_type"]
                for r in program_occurrence_rows("Python", new, "deadbeef")[0]
            }
            for r in orig_rows:
                m = map_span(edits, *[int(x) for x in r["source_span"]])
                okay &= m is not None and sites.get((m[0], m[1])) == r["occurrence_type"]
        checks.append((cond, okay))
    exc_code = (
        "def risky(flag):\n"
        "    try:\n"
        "        return flag\n"
        "    except Exception as err:\n"
        "        return err\n"
    )
    fns, _, skips = renameable_functions(exc_code)
    checks.append(("except-as function skipped", not fns and skips["unsupported_binder"] == 1))
    det = rename_program(code, "deadbeef", "C4", targets)[0]
    checks.append(("C4 deterministic", det is not None and det == rename_program(code, "deadbeef", "C4", targets)[0]))
    ok = True
    for name, passed in checks:
        print(f"  {'OK ' if passed else 'FAIL'} {name}")
        ok &= passed
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--condition", required=True, choices=CONDITIONS)
    r.add_argument("--canonical", required=True)
    r.add_argument("--occurrences", required=True)
    r.add_argument("--out-canonical", required=True)
    r.add_argument("--out-occurrences", required=True)
    r.add_argument("--sample-ids", help="restrict to problems containing these occurrence ids")
    r.add_argument("--label-field", default="occurrence_type",
                   choices=["occurrence_type", "role"],
                   help="which field the identity gate compares, and therefore "
                        "which extractor re-reads the renamed code: "
                        "'occurrence_type' for the boolean workstream, 'role' "
                        "for role_occurrences.py output")
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    return cmd_verify(args) if args.cmd == "verify" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
