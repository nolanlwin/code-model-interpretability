"""
Day 13: structural boolean-flag variable occurrences (per Name site).

Each output row is one occurrence with AST context, optional Qwen token indices
(via ``token_alignment``), and a coarse ``occurrence_type`` aligned to AGENDA2
(definition, assignment, conditional_use, loop_use, return_use, indexing_use).

``update`` (e.g. AugAssign) is not emitted yet — current boolean heuristics do
not flag those sites.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from boolean_flag_roles import (  # noqa: E402
    FlagHit,
    collect_flag_hits,
    inside_nested_function,
    names_in_boolean_test,
)
from csn_function_ast import build_parent_map, iter_top_level_functions, parse_module  # noqa: E402
from go_variable_occurrences import occurrence_rows_from_go_code  # noqa: E402
from java_variable_occurrences import occurrence_rows_from_java_code  # noqa: E402
from javascript_variable_occurrences import occurrence_rows_from_javascript_code  # noqa: E402
from php_variable_occurrences import occurrence_rows_from_php_code  # noqa: E402
from ruby_variable_occurrences import occurrence_rows_from_ruby_code  # noqa: E402
import token_alignment as _tokalign  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "occurrences" / "boolean_flag_occurrences.jsonl"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
SUPPORTED_LANGUAGES = ("python", "java", "go", "javascript", "php", "ruby")

_NAME_IN_BOOL_PATTERNS = frozenset(
    {
        "if_test",
        "while_test",
        "if_exp_test",
        "assign_boolop_rhs",
        "assign_not_name_inner",
    }
)
_TARGET_PATTERNS = frozenset(
    {
        "assign_bool_literal",
        "assign_boolop_lhs",
        "assign_not_name",
        "annassign_bool_literal",
    }
)

_STMT_SKIP = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def line_start_indices(code: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def name_char_span(code: str, name: ast.Name) -> tuple[int, int]:
    ln = name.lineno
    starts = line_start_indices(code)
    if ln < 1 or ln > len(starts):
        return 0, max(0, len(name.id))
    base = starts[ln - 1]
    c0 = name.col_offset if name.col_offset is not None else 0
    c1 = getattr(name, "end_col_offset", None)
    if c1 is None:
        c1 = c0 + len(name.id)
    return base + c0, base + c1


def parent_ast_type(parents: dict[ast.AST, ast.AST | None], node: ast.AST) -> str | None:
    p = parents.get(node)
    return type(p).__name__ if p is not None else None


def enclosing_statement_type(parents: dict[ast.AST, ast.AST | None], node: ast.AST) -> str | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.stmt) and not isinstance(cur, _STMT_SKIP):
            return type(cur).__name__
        cur = parents.get(cur)
    return None


def _is_indexing_name(name: ast.Name, parents: dict[ast.AST, ast.AST | None]) -> bool:
    return isinstance(parents.get(name), ast.Subscript)


def occurrence_type(pattern: str, name: ast.Name, parents: dict[ast.AST, ast.AST | None]) -> str:
    if _is_indexing_name(name, parents):
        return "indexing_use"
    if pattern == "while_test":
        return "loop_use"
    if pattern == "return_bool":
        return "return_use"
    if pattern in ("if_test", "if_exp_test", "assign_boolop_rhs", "assign_not_name_inner"):
        return "conditional_use"
    if pattern == "annassign_bool_literal":
        return "definition"
    return "assignment"


def iter_hit_names(h: FlagHit) -> Iterator[ast.Name]:
    var, node, pat = h.variable, h.node, h.pattern
    if pat == "return_bool" and isinstance(node, ast.Name) and node.id == var:
        yield node
        return
    if pat in _NAME_IN_BOOL_PATTERNS:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == var and isinstance(sub.ctx, ast.Load):
                yield sub
        return
    if pat in _TARGET_PATTERNS:
        if isinstance(node, ast.AnnAssign) and node.target is not None:
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name) and sub.id == var:
                    yield sub
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name) and sub.id == var:
                        yield sub
        return


def collect_return_hits(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST | None],
) -> list[FlagHit]:
    hits: list[FlagHit] = []
    for node in ast.walk(func):
        if node is func:
            continue
        if inside_nested_function(func, node, parents):
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            allowed = names_in_boolean_test(node.value)
            if not allowed:
                continue
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Name)
                    and isinstance(sub.ctx, ast.Load)
                    and sub.id in allowed
                ):
                    hits.append(FlagHit(sub.id, sub.lineno, "return_bool", sub))
    return hits


def _token_positions_for_span(
    code: str,
    span: tuple[int, int],
    offset_mapping: Sequence[tuple[int, int]],
) -> list[int] | None:
    s, e = span
    if s >= e or s < 0 or e > len(code):
        return None
    if code[s:e] == "":
        return None
    return _tokalign.char_span_to_token_indices(offset_mapping, s, e)



def _byte_to_char_map(code: str) -> dict[int, int]:
    m, bi = {}, 0
    for ci, ch in enumerate(code):
        m[bi] = ci
        bi += len(ch.encode("utf-8"))
    m[bi] = len(code)
    return m


def _normalize_treesitter_rows(
    code: str, rows: list[dict[str, Any]], tokenizer, max_length: int
) -> list[dict[str, Any]]:
    """Tree-sitter extractors emit BYTE offsets; everything downstream consumes
    CHARACTER offsets (identical only on pure-ASCII source). Convert spans,
    recompute token positions where the byte-based ones were wrong, and drop
    any row whose span still does not slice back to its variable name (PHP
    spans legitimately include the ``$`` sigil)."""
    if not rows:
        return rows
    is_ascii = code.isascii()
    bmap = offset_mapping = None
    if not is_ascii:
        bmap = _byte_to_char_map(code)
        if tokenizer is not None and getattr(tokenizer, "is_fast", False):
            _, offset_mapping, _ = _tokalign.tokenize_for_alignment(
                tokenizer, code, max_length=max_length
            )
    kept = []
    for r in rows:
        span = r.get("source_span") or (0, 0)
        s, e = int(span[0]), int(span[1])
        if bmap is not None:
            if s not in bmap or e not in bmap:
                continue  # mid-codepoint span: unrecoverable, drop
            s, e = bmap[s], bmap[e]
            r["source_span"] = [s, e]
            if offset_mapping is not None:
                r["token_positions"] = _token_positions_for_span(
                    code, (s, e), offset_mapping
                )
        var = str(r.get("variable") or "")
        if 0 <= s < e <= len(code) and code[s:e] in (var, f"${var}"):
            kept.append(r)
    return kept


def occurrence_rows_from_code(
    code: str,
    *,
    language: str = "python",
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
    tokenizer=None,
    max_length: int = 2048,
) -> tuple[list[dict[str, Any]], str | None]:
    if language == "java":
        rows, err = occurrence_rows_from_java_code(
            code,
            repo=repo,
            path=path,
            source_row=source_row,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if err is None:
            rows = _normalize_treesitter_rows(code, rows, tokenizer, max_length)
        return rows, err
    if language == "go":
        rows, err = occurrence_rows_from_go_code(
            code,
            repo=repo,
            path=path,
            source_row=source_row,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if err is None:
            rows = _normalize_treesitter_rows(code, rows, tokenizer, max_length)
        return rows, err
    if language == "javascript":
        rows, err = occurrence_rows_from_javascript_code(
            code,
            repo=repo,
            path=path,
            source_row=source_row,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if err is None:
            rows = _normalize_treesitter_rows(code, rows, tokenizer, max_length)
        return rows, err
    if language == "php":
        rows, err = occurrence_rows_from_php_code(
            code,
            repo=repo,
            path=path,
            source_row=source_row,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if err is None:
            rows = _normalize_treesitter_rows(code, rows, tokenizer, max_length)
        return rows, err
    if language == "ruby":
        rows, err = occurrence_rows_from_ruby_code(
            code,
            repo=repo,
            path=path,
            source_row=source_row,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if err is None:
            rows = _normalize_treesitter_rows(code, rows, tokenizer, max_length)
        return rows, err
    if language != "python":
        return [], f"unsupported language: {language!r} (choose {SUPPORTED_LANGUAGES})"
    notes: list[str] = []
    try:
        tree = parse_module(code)
    except SyntaxError as e:
        return [], f"{e.msg} (line {e.lineno})"

    funcs = list(iter_top_level_functions(tree))
    if not funcs:
        return [], "no top-level function definition in module"

    offset_mapping: Sequence[tuple[int, int]] | None = None
    if tokenizer is not None:
        if tokenizer.is_fast:
            _, offset_mapping, _ = _tokalign.tokenize_for_alignment(
                tokenizer, code, max_length=max_length
            )
        else:
            notes.append("slow_tokenizer_no_token_positions")

    rows: list[dict[str, Any]] = []
    for fn in funcs:
        parents = build_parent_map(fn)
        hits = collect_flag_hits(fn, code)
        hits.extend(collect_return_hits(fn, parents))
        for h in hits:
            for name in iter_hit_names(h):
                span = name_char_span(code, name)
                s, e = span
                occ_type = occurrence_type(h.pattern, name, parents)
                tok_pos: list[int] | None = None
                if offset_mapping is not None and s < e and e <= len(code):
                    tok_pos = _token_positions_for_span(code, span, offset_mapping)
                rec: dict[str, Any] = {
                    "variable": name.id,
                    "role": "boolean_flag",
                    "occurrence_type": occ_type,
                    "line": name.lineno,
                    "col_offset": name.col_offset,
                    "end_col_offset": getattr(name, "end_col_offset", None),
                    "source_span": [s, e],
                    "token_positions": tok_pos,
                    "detection_pattern": h.pattern,
                    "parent_ast_type": parent_ast_type(parents, name),
                    "enclosing_statement": enclosing_statement_type(parents, name),
                    "function": fn.name,
                    "function_lineno": int(fn.lineno),
                    "is_async": isinstance(fn, ast.AsyncFunctionDef),
                }
                if repo is not None:
                    rec["repo"] = repo
                if path is not None:
                    rec["path"] = path
                if source_row is not None:
                    rec["source_row"] = source_row
                rows.append(rec)

    rows.sort(key=lambda r: (r["line"], r["col_offset"] or 0, r["variable"]))
    if notes:
        for r in rows:
            r.setdefault("notes", []).extend(notes)
    return rows, None


def _maybe_tokenizer(model_id: str | None):
    if not model_id:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def cmd_extract(args: argparse.Namespace) -> int:
    tok = _maybe_tokenizer(args.model_id) if not args.no_tokens else None

    def write_rows(rows: list[dict[str, Any]], fout) -> None:
        for rec in rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.code_file:
        try:
            code = Path(args.code_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"no such file: {args.code_file}", file=sys.stderr)
            return 1
        rows, err = occurrence_rows_from_code(
            code,
            language=args.language,
            tokenizer=tok,
            max_length=args.max_length,
        )
        if err:
            print(err, file=sys.stderr)
            return 1
        out_dest = None if args.output == "-" else Path(args.output)
        if out_dest is None:
            write_rows(rows, sys.stdout)
        else:
            out_dest.parent.mkdir(parents=True, exist_ok=True)
            with out_dest.open("w", encoding="utf-8") as fout:
                write_rows(rows, fout)
            print(f"wrote {len(rows)} occurrences -> {out_dest}")
        return 0

    in_path = Path(args.input)
    try:
        fin = in_path.open(encoding="utf-8")
    except FileNotFoundError:
        print(f"no such file: {args.input}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    n_err = 0
    max_rows = args.max_rows

    with fin, out_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(desc="occurrences", unit="row", total=max_rows)
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
            records, err = occurrence_rows_from_code(
                code,
                language=args.language,
                repo=repo,
                path=path,
                source_row=n_in,
                tokenizer=tok,
                max_length=args.max_length,
            )
            if err is not None:
                n_err += 1
                err_obj: dict[str, Any] = {
                    "parse_error": err,
                    "variable": None,
                    "role": None,
                    "occurrence_type": None,
                    "line": None,
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

    print(f"read_lines={n_in} written_occurrences={n_out} parse_errors={n_err} -> {out_path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    _FIXTURES: dict[str, tuple[str, set[str], int]] = {
        "java": ("boolean_occurrence_sample.java", {"assignment", "conditional_use", "return_use", "loop_use"}, 6),
        "go": ("boolean_occurrence_sample.go", {"assignment", "conditional_use", "return_use", "loop_use"}, 6),
        "javascript": ("boolean_occurrence_sample.js", {"assignment", "conditional_use", "return_use", "loop_use"}, 6),
        "php": ("boolean_occurrence_sample.php", {"assignment", "conditional_use", "return_use", "loop_use"}, 6),
        "ruby": ("boolean_occurrence_sample.rb", {"assignment", "conditional_use", "return_use", "loop_use"}, 6),
        "python": ("boolean_occurrence_sample.py", {"definition", "assignment", "conditional_use", "return_use", "loop_use"}, 6),
    }
    fname, need, min_rows = _FIXTURES[args.language]
    sample = PROJECT_ROOT / "fixtures" / fname
    code = sample.read_text(encoding="utf-8")
    rows, err = occurrence_rows_from_code(
        code, language=args.language, tokenizer=None, max_length=2048
    )
    if err:
        print(err, file=sys.stderr)
        return 1
    types_found = {r["occurrence_type"] for r in rows}
    missing = need - types_found
    if missing:
        print(f"verify missing occurrence_types: {missing} (have {types_found})", file=sys.stderr)
        return 1
    if len(rows) < min_rows:
        print(f"verify expected at least {min_rows} rows, got {len(rows)}", file=sys.stderr)
        return 1
    print(
        f"variable_occurrences verify ({args.language}): ok "
        f"({len(rows)} rows, types {sorted(types_found)})"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Day 13: per-site boolean-flag variable occurrences with AST + token span."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser(
        "extract",
        help="JSONL (repo,path,code) like boolean_flags, or --code-file for one snippet.",
    )
    src = ex.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Canonical JSONL path.")
    src.add_argument("--code-file", type=str, help="Single UTF-8 source file.")
    ex.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"JSONL path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)}), or '-' with --code-file for stdout.",
    )
    ex.add_argument("--max-rows", type=int, default=None)
    ex.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=f"HF model id for tokenizer (default: omit token_positions). Typical: {DEFAULT_MODEL_ID}",
    )
    ex.add_argument(
        "--no-tokens",
        action="store_true",
        help="Do not load a tokenizer; token_positions will be null.",
    )
    ex.add_argument("--max-length", type=int, default=2048)
    ex.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="python",
        help="Source language for AST/heuristics (default: python).",
    )
    ex.set_defaults(func=cmd_extract)

    v = sub.add_parser("verify", help="Run checks on language fixtures.")
    v.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="python",
        help="Fixture language to verify (default: python).",
    )
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
