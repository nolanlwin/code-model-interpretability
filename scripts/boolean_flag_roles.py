"""
Boolean flags extraction

Definition: variable used primarily as a conditional control signal.

Heuristics (AST):
- If / elif tests, While tests, IfExp tests: truthiness Name, `not` Name, BoolOp
  trees, and Compare that ties a Name to True/False via ==, !=, is, is not.
- Assign / AnnAssign: RHS is True/False, BoolOp, or `not Name`; targets and
  relevant RHS names are candidates.

Exclusions (approximate):
- Names that appear only inside generic comparisons (e.g. `if i < n`) are not
  taken from Compare unless paired with bool literals as above.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Run as `python scripts/boolean_flag_roles.py` from repo root.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from csn_function_ast import (  # noqa: E402
    build_parent_map,
    iter_top_level_functions,
    names_in_assignment_target,
    parse_module,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELED_OUT = PROJECT_ROOT / "outputs" / "labeled"


@dataclass
class FlagHit:
    variable: str
    line: int
    pattern: str
    node: ast.AST


def _is_load_name(n: ast.AST) -> bool:
    return isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)


def compare_boolean_flag_names(node: ast.Compare) -> set[str]:
    """Names compared to True/False with ==, !=, is, is not (symmetric)."""
    out: set[str] = set()
    allowed = {"Eq", "Is", "NotEq", "IsNot"}
    cur: ast.expr = node.left
    for op, comp in zip(node.ops, node.comparators, strict=True):
        op_name = type(op).__name__
        if op_name not in allowed:
            cur = comp
            continue
        for a, b in ((cur, comp), (comp, cur)):
            # Same trap as _bool_literal: `x == 0` must not be read as
            # `x == False`. Comparing an int is not a boolean-flag test.
            if _is_load_name(a) and isinstance(b, ast.Constant) and _is_bool_constant(b.value):
                out.add(a.id)
        cur = comp
    return out


def names_in_boolean_test(expr: ast.expr) -> set[str]:
    """Names used as boolean conditions (excludes generic comparison-only names)."""
    if _is_load_name(expr):
        return {expr.id}
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return names_in_boolean_test(expr.operand)
    if isinstance(expr, ast.BoolOp):
        s: set[str] = set()
        for v in expr.values:
            s |= names_in_boolean_test(v)
        return s
    if isinstance(expr, ast.Compare):
        return compare_boolean_flag_names(expr)
    if isinstance(expr, ast.IfExp):
        return names_in_boolean_test(expr.test)
    return set()


def bool_expression_load_names(expr: ast.expr) -> set[str]:
    """Load-names appearing in a nested boolean-shaped expression."""
    if _is_load_name(expr):
        return {expr.id}
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return bool_expression_load_names(expr.operand)
    if isinstance(expr, ast.BoolOp):
        s: set[str] = set()
        for v in expr.values:
            s |= bool_expression_load_names(v)
        return s
    if isinstance(expr, ast.Compare):
        return compare_boolean_flag_names(expr)
    return set()


def _is_bool_constant(value) -> bool:
    """True only for the literals ``True``/``False``.

    NOT ``value in (True, False)``. Python defines ``0 == False`` and
    ``1 == True``, so that membership test also accepts the integers 0 and 1
    (and the floats 0.0/1.0). Every ``i = 0`` and ``res = 1`` in the corpus
    was therefore detected as a boolean-flag assignment: 7,897 of 8,432
    detections on XLCoST Python -- 93.7% -- had an int or float right-hand
    side, and the resulting class made up 47% of the whole boolean sample.

    ``isinstance`` is the correct test because ``bool`` is a subclass of
    ``int``: True and False are instances of bool, 0 and 1 are not.
    """
    return isinstance(value, bool)


def _bool_literal(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and _is_bool_constant(expr.value)


def hits_from_assign(node: ast.Assign, _source: str) -> list[FlagHit]:
    hits: list[FlagHit] = []
    rhs = node.value
    if _bool_literal(rhs):
        for t in node.targets:
            for vid in names_in_assignment_target(t):
                hits.append(
                    FlagHit(vid, node.lineno, "assign_bool_literal", node)
                )
        return hits
    if isinstance(rhs, ast.BoolOp):
        names_rhs = bool_expression_load_names(rhs)
        for t in node.targets:
            for vid in names_in_assignment_target(t):
                hits.append(
                    FlagHit(vid, node.lineno, "assign_boolop_lhs", node)
                )
        for vid in names_rhs:
            hits.append(FlagHit(vid, node.lineno, "assign_boolop_rhs", node))
        return hits
    if isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.Not):
        inner = rhs.operand
        if _is_load_name(inner):
            for t in node.targets:
                for vid in names_in_assignment_target(t):
                    hits.append(
                        FlagHit(vid, node.lineno, "assign_not_name", node)
                    )
            hits.append(
                FlagHit(inner.id, node.lineno, "assign_not_name_inner", node)
            )
    return hits


def hits_from_annassign(node: ast.AnnAssign, _source: str) -> list[FlagHit]:
    if node.value is None:
        return []
    if not _bool_literal(node.value):
        return []
    if node.target is None:
        return []
    hits: list[FlagHit] = []
    for vid in names_in_assignment_target(node.target):
        hits.append(FlagHit(vid, node.lineno, "annassign_bool_literal", node))
    return hits


def _iter_if_tests(node: ast.If) -> Iterator[ast.expr]:
    cur: ast.If | None = node
    while cur is not None:
        yield cur.test
        nxt = None
        if cur.orelse and len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            nxt = cur.orelse[0]
        cur = nxt


def inside_nested_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    node: ast.AST,
    parents: dict[ast.AST, ast.AST | None],
) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not fn:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        cur = parents.get(cur)
    return False


def collect_flag_hits(func: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> list[FlagHit]:
    parents = build_parent_map(func)
    hits: list[FlagHit] = []
    for node in ast.walk(func):
        if node is func:
            continue
        if inside_nested_function(func, node, parents):
            continue
        if isinstance(node, ast.If):
            for test in _iter_if_tests(node):
                for vid in names_in_boolean_test(test):
                    hits.append(FlagHit(vid, test.lineno, "if_test", test))
        elif isinstance(node, ast.While):
            test = node.test
            for vid in names_in_boolean_test(test):
                hits.append(FlagHit(vid, test.lineno, "while_test", test))
        elif isinstance(node, ast.IfExp):
            test = node.test
            for vid in names_in_boolean_test(test):
                hits.append(FlagHit(vid, test.lineno, "if_exp_test", test))
        elif isinstance(node, ast.Assign):
            hits.extend(hits_from_assign(node, source))
        elif isinstance(node, ast.AnnAssign):
            hits.extend(hits_from_annassign(node, source))
    return hits


def _snippet(source: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(source, node)
    if seg is None:
        return ""
    line = seg.strip().splitlines()
    return line[0] if line else seg.strip()


def extract_boolean_flags(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
) -> list[dict[str, Any]]:
    hits = collect_flag_hits(func, source)
    by_var: dict[str, list[FlagHit]] = defaultdict(list)
    for h in hits:
        by_var[h.variable].append(h)

    out: list[dict[str, Any]] = []
    for var in sorted(by_var):
        hs = by_var[var]
        hs.sort(key=lambda h: (h.line, h.pattern))
        first = hs[0]
        out.append(
            {
                "variable": var,
                "role": "boolean_flag",
                "line": first.line,
                "code": _snippet(source, first.node),
                "function": func.name,
            }
        )
    return out


def labeled_rows_from_code(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        tree = parse_module(code)
    except SyntaxError as e:
        return [], f"{e.msg} (line {e.lineno})"

    funcs = list(iter_top_level_functions(tree))
    if not funcs:
        return [], "no top-level function definition in module"

    rows: list[dict[str, Any]] = []
    for fn in funcs:
        for ex in extract_boolean_flags(fn, code):
            if repo is not None:
                ex["repo"] = repo
            if path is not None:
                ex["path"] = path
            if source_row is not None:
                ex["source_row"] = source_row
            rows.append(ex)
    return rows, None


def _cmd_extract(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_err = 0
    max_rows = args.max_rows

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(desc="boolean flags", unit="row", total=max_rows)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            pbar.update(1)
            row = json.loads(line)
            code = row.get("code") or ""
            repo = row.get("repo")
            path = row.get("path")
            records, err = labeled_rows_from_code(
                code, repo=repo, path=path, source_row=n_in
            )
            if err is not None:
                n_err += 1
                err_obj = {
                    "parse_error": err,
                    "variable": None,
                    "role": None,
                    "line": None,
                    "code": None,
                    "function": None,
                    "source_row": n_in,
                }
                if repo is not None:
                    err_obj["repo"] = repo
                if path is not None:
                    err_obj["path"] = path
                fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            else:
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
            if max_rows is not None and n_in >= max_rows:
                break
        pbar.close()

    print(
        f"read_lines={n_in} written_labels={n_out} parse_errors={n_err} -> {out_path}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Day 3 (Role 3 only): extract boolean-flag variables to JSONL."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser(
        "extract",
        help="Read canonical JSONL (repo,path,code); write one line per flag variable.",
    )
    ex.add_argument("--input", type=str, required=True)
    ex.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_LABELED_OUT / "boolean_flags.jsonl"),
        help=f"Default: {DEFAULT_LABELED_OUT / 'boolean_flags.jsonl'}",
    )
    ex.add_argument("--max-rows", type=int, default=None)
    ex.set_defaults(func=_cmd_extract)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
