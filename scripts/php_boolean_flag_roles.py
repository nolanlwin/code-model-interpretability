"""
Boolean flags extraction for PHP (tree-sitter).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from tree_sitter import Node

from php_csn_parse import (
    PhpFunction,
    PhpModule,
    assignment_target_names,
    build_parent_map,
    inside_nested_function,
    iter_top_level_functions,
    parse_php_module,
    span_in_original,
    variable_label,
    variable_nodes_in,
)

_BOOL_BIN_OPS = frozenset({"&&", "||"})
_BOOL_COMPARE_OPS = frozenset({"==", "!=", "===", "!=="})


@dataclass
class PhpFlagHit:
    variable: str
    line: int
    pattern: str
    node: Node


def _node_text(code: str, node: Node) -> str:
    return code[node.start_byte : node.end_byte]


def _is_bool_literal(node: Node) -> bool:
    if node.type == "boolean":
        return node.text.decode("utf-8").lower() in ("true", "false")
    return False


def _binary_operator(node: Node) -> str | None:
    for i in range(node.child_count):
        child = node.child(i)
        if child.type in _BOOL_BIN_OPS | _BOOL_COMPARE_OPS:
            return child.type
    return None


def _unwrap_parens(node: Node) -> Node:
    if node.type == "parenthesized_expression":
        for i in range(node.child_count):
            child = node.child(i)
            if child.type not in {"(", ")"}:
                return _unwrap_parens(child)
    return node


def compare_boolean_flag_names(node: Node) -> set[str]:
    node = _unwrap_parens(node)
    if node.type != "binary_expression":
        return set()
    if _binary_operator(node) not in _BOOL_COMPARE_OPS:
        return set()
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
    if left is None or right is None:
        return set()
    out: set[str] = set()
    for a, b in ((left, right), (right, left)):
        name = variable_label(a)
        if name and _is_bool_literal(b):
            out.add(name)
    return out


def names_in_boolean_test(node: Node) -> set[str]:
    node = _unwrap_parens(node)
    if node.type == "variable_name":
        name = variable_label(node)
        return {name} if name else set()
    if node.type == "unary_op_expression":
        op = node.child(0)
        operand = node.child(1) if node.child_count > 1 else None
        if op is not None and op.type == "!" and operand is not None:
            return names_in_boolean_test(operand)
        return set()
    if node.type == "binary_expression" and _binary_operator(node) in _BOOL_BIN_OPS:
        out: set[str] = set()
        for sub in (node.child_by_field_name("left"), node.child_by_field_name("right")):
            if sub is not None:
                out |= names_in_boolean_test(sub)
        return out
    if node.type == "binary_expression":
        return compare_boolean_flag_names(node)
    return set()


def bool_expression_load_names(node: Node) -> set[str]:
    node = _unwrap_parens(node)
    if node.type == "variable_name":
        name = variable_label(node)
        return {name} if name else set()
    if node.type == "unary_op_expression":
        op = node.child(0)
        operand = node.child(1) if node.child_count > 1 else None
        if op is not None and op.type == "!" and operand is not None:
            return bool_expression_load_names(operand)
        return set()
    if node.type == "binary_expression" and _binary_operator(node) in _BOOL_BIN_OPS:
        out: set[str] = set()
        for sub in (node.child_by_field_name("left"), node.child_by_field_name("right")):
            if sub is not None:
                out |= bool_expression_load_names(sub)
        return out
    if node.type == "binary_expression":
        return compare_boolean_flag_names(node)
    return set()


def _if_condition(node: Node) -> Node | None:
    if node.type != "if_statement":
        return None
    for i in range(node.child_count):
        child = node.child(i)
        if child.type not in {"if", "else", "compound_statement", ";" }:
            return child
    return None


def _hits_from_value(
    value: Node, targets: list[str], line: int, container: Node
) -> list[PhpFlagHit]:
    hits: list[PhpFlagHit] = []
    value = _unwrap_parens(value)
    if _is_bool_literal(value):
        for vid in targets:
            hits.append(PhpFlagHit(vid, line, "assign_bool_literal", container))
    elif _binary_operator(value) in _BOOL_BIN_OPS:
        names_rhs = bool_expression_load_names(value)
        for vid in targets:
            hits.append(PhpFlagHit(vid, line, "assign_boolop_lhs", container))
        for vid in names_rhs:
            hits.append(PhpFlagHit(vid, line, "assign_boolop_rhs", value))
    elif value.type == "unary_op_expression":
        op = value.child(0)
        operand = value.child(1) if value.child_count > 1 else None
        if op is not None and op.type == "!" and operand is not None:
            inner_name = variable_label(operand)
            if inner_name:
                for vid in targets:
                    hits.append(PhpFlagHit(vid, line, "assign_not_name", container))
                hits.append(PhpFlagHit(inner_name, line, "assign_not_name_inner", value))
    return hits


def hits_from_assignment(node: Node, code: str) -> list[PhpFlagHit]:
    if node.type != "assignment_expression":
        return []
    value = node.child_by_field_name("right")
    if value is None:
        return []
    return _hits_from_value(
        value, assignment_target_names(node), node.start_point[0] + 1, node
    )


def _iter_if_tests(node: Node) -> Iterator[Node]:
    if node.type != "if_statement":
        return
    cond = _if_condition(node)
    if cond is not None:
        yield cond
    for i in range(node.child_count):
        child = node.child(i)
        if child.type == "else":
            for j in range(child.child_count):
                sub = child.child(j)
                if sub.type == "if_statement":
                    yield from _iter_if_tests(sub)


def collect_flag_hits(fn: PhpFunction, code: str) -> list[PhpFlagHit]:
    parents = build_parent_map(fn.node)
    hits: list[PhpFlagHit] = []

    def walk(node: Node) -> None:
        if node is fn.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_function(fn, node, parents):
            return
        if node.type == "if_statement":
            for test in _iter_if_tests(node):
                for vid in names_in_boolean_test(test):
                    hits.append(PhpFlagHit(vid, test.start_point[0] + 1, "if_test", test))
        elif node.type == "while_statement":
            for i in range(node.child_count):
                child = node.child(i)
                if child.type not in {"while", "compound_statement", ";"}:
                    for vid in names_in_boolean_test(child):
                        hits.append(
                            PhpFlagHit(vid, child.start_point[0] + 1, "while_test", child)
                        )
        elif node.type == "assignment_expression":
            hits.extend(hits_from_assignment(node, code))
        for i in range(node.child_count):
            walk(node.child(i))

    walk(fn.node)
    return hits


def collect_return_hits(fn: PhpFunction, code: str) -> list[PhpFlagHit]:
    parents = build_parent_map(fn.node)
    hits: list[PhpFlagHit] = []

    def walk(node: Node) -> None:
        if node is fn.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_function(fn, node, parents):
            return
        if node.type == "return_statement":
            value = None
            for i in range(node.child_count):
                child = node.child(i)
                if child.type not in {"return", ";"}:
                    value = child
                    break
            if value is not None:
                value = _unwrap_parens(value)
                allowed = names_in_boolean_test(value)
                if allowed:
                    for sub in variable_nodes_in(value):
                        name = variable_label(sub)
                        if name and name in allowed:
                            hits.append(
                                PhpFlagHit(name, sub.start_point[0] + 1, "return_bool", sub)
                            )
        for i in range(node.child_count):
            walk(node.child(i))

    walk(fn.node)
    return hits


def extract_boolean_flags(fn: PhpFunction, module: PhpModule) -> list[dict[str, Any]]:
    code = module.original
    hits = collect_flag_hits(fn, code)
    by_var: dict[str, list[PhpFlagHit]] = defaultdict(list)
    for h in hits:
        by_var[h.variable].append(h)

    out: list[dict[str, Any]] = []
    for var in sorted(by_var):
        hs = by_var[var]
        hs.sort(key=lambda h: (h.line, h.pattern))
        first = hs[0]
        s, e = span_in_original(module, first.node)
        snippet = code[s:e].strip().splitlines()
        out.append(
            {
                "variable": var,
                "role": "boolean_flag",
                "line": first.line,
                "code": snippet[0] if snippet else "",
                "function": fn.name,
            }
        )
    return out


def labeled_rows_from_php_code(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    module = parse_php_module(code)
    if module.tree.root_node.has_error:
        return [], "php parse error"

    funcs = list(iter_top_level_functions(module))
    if not funcs:
        return [], "no top-level function declaration"

    rows: list[dict[str, Any]] = []
    for fn in funcs:
        for ex in extract_boolean_flags(fn, module):
            if repo is not None:
                ex["repo"] = repo
            if path is not None:
                ex["path"] = path
            if source_row is not None:
                ex["source_row"] = source_row
            rows.append(ex)
    return rows, None
