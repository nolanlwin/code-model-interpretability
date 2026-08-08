"""
Per-site boolean-flag variable occurrences for PHP (tree-sitter).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tree_sitter import Node

from php_boolean_flag_roles import (
    PhpFlagHit,
    collect_flag_hits,
    collect_return_hits,
)
from php_csn_parse import (
    PhpFunction,
    build_parent_map,
    iter_top_level_functions,
    parse_php_module,
    span_in_original,
    variable_label,
)
import token_alignment as _tokalign  # noqa: E402

_NAME_IN_BOOL_PATTERNS = frozenset(
    {
        "if_test",
        "while_test",
        "assign_boolop_rhs",
        "assign_not_name_inner",
    }
)
_TARGET_PATTERNS = frozenset(
    {
        "assign_bool_literal",
        "assign_boolop_lhs",
        "assign_not_name",
    }
)


def parent_node_type(parents: dict[Node, Node | None], node: Node) -> str | None:
    p = parents.get(node)
    return p.type if p is not None else None


def enclosing_statement_type(parents: dict[Node, Node | None], node: Node) -> str | None:
    skip = frozenset(
        {
            "program",
            "function_definition",
            "method_declaration",
            "class_declaration",
            "compound_statement",
        }
    )
    cur = parents.get(node)
    while cur is not None:
        if cur.type.endswith("_statement") or cur.type == "assignment_expression":
            if cur.type not in skip:
                return cur.type
        cur = parents.get(cur)
    return None


def _is_indexing_name(node: Node, parents: dict[Node, Node | None]) -> bool:
    p = parents.get(node)
    return p is not None and p.type in {
        "subscript_expression",
        "member_access_expression",
        "member_call_expression",
    }


def occurrence_type(pattern: str, node: Node, parents: dict[Node, Node | None]) -> str:
    if _is_indexing_name(node, parents):
        return "indexing_use"
    if pattern == "while_test":
        return "loop_use"
    if pattern == "return_bool":
        return "return_use"
    if pattern in ("if_test", "assign_boolop_rhs", "assign_not_name_inner"):
        return "conditional_use"
    return "assignment"


def _walk_variables(node: Node) -> list[Node]:
    out: list[Node] = []
    if node.type == "variable_name":
        out.append(node)
    for i in range(node.child_count):
        out.extend(_walk_variables(node.child(i)))
    return out


def iter_hit_names(h: PhpFlagHit) -> list[Node]:
    var, node, pat = h.variable, h.node, h.pattern
    if pat == "return_bool":
        name = variable_label(node)
        if name == var:
            return [node]
        return []
    if pat in _NAME_IN_BOOL_PATTERNS:
        return [sub for sub in _walk_variables(node) if variable_label(sub) == var]
    if pat in _TARGET_PATTERNS:
        return [sub for sub in _walk_variables(node) if variable_label(sub) == var]
    return []


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


def occurrence_rows_from_php_code(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
    tokenizer=None,
    max_length: int = 2048,
) -> tuple[list[dict[str, Any]], str | None]:
    notes: list[str] = []
    module = parse_php_module(code)
    if module.tree.root_node.has_error:
        return [], "php parse error"

    funcs = list(iter_top_level_functions(module))
    if not funcs:
        return [], "no top-level function declaration"

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
        parents = build_parent_map(fn.node)
        hits = collect_flag_hits(fn, code)
        hits.extend(collect_return_hits(fn, code))
        for h in hits:
            for name_node in iter_hit_names(h):
                s, e = span_in_original(module, name_node)
                occ_type = occurrence_type(h.pattern, name_node, parents)
                tok_pos: list[int] | None = None
                if offset_mapping is not None and 0 <= s < e <= len(code):
                    tok_pos = _token_positions_for_span(code, (s, e), offset_mapping)
                rec: dict[str, Any] = {
                    "variable": h.variable,
                    "role": "boolean_flag",
                    "occurrence_type": occ_type,
                    "line": name_node.start_point[0] + 1,
                    "col_offset": name_node.start_point[1],
                    "end_col_offset": name_node.end_point[1],
                    "source_span": [s, e],
                    "token_positions": tok_pos,
                    "detection_pattern": h.pattern,
                    "parent_ast_type": parent_node_type(parents, name_node),
                    "enclosing_statement": enclosing_statement_type(parents, name_node),
                    "function": fn.name,
                    "function_lineno": fn.start_line,
                    "language": "php",
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
