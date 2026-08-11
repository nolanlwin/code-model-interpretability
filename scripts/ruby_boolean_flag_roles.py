"""
Boolean flags extraction for Ruby (tree-sitter).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from tree_sitter import Node

from ruby_csn_parse import (
    RubyMethod,
    assignment_target_names,
    build_parent_map,
    identifier_nodes_in,
    inside_nested_method,
    iter_top_level_methods,
    parse_ruby,
)

_BOOL_LITERAL_TYPES = frozenset({"true", "false"})
_BOOL_BIN_OPS = frozenset({"&&", "||"})
_BOOL_COMPARE_OPS = frozenset({"==", "!="})


@dataclass
class RubyFlagHit:
    variable: str
    line: int
    pattern: str
    node: Node


def _node_text(code: str, node: Node) -> str:
    return code[node.start_byte : node.end_byte]


def _identifier_name(node: Node) -> str | None:
    if node.type != "identifier":
        return None
    return node.text.decode("utf-8")


def _is_bool_literal(node: Node) -> bool:
    return node.type in _BOOL_LITERAL_TYPES


def _binary_operator(node: Node) -> str | None:
    for i in range(node.child_count):
        child = node.child(i)
        if child.type in _BOOL_BIN_OPS | _BOOL_COMPARE_OPS:
            return child.type
    return None


def _binary_operands(node: Node) -> tuple[Node | None, Node | None]:
    if node.type != "binary":
        return None, None
    left = right = None
    for child in node.children:
        if child.type in _BOOL_BIN_OPS | _BOOL_COMPARE_OPS:
            continue
        if left is None:
            left = child
        else:
            right = child
    return left, right


def compare_boolean_flag_names(node: Node) -> set[str]:
    if node.type != "binary":
        return set()
    if _binary_operator(node) not in _BOOL_COMPARE_OPS:
        return set()
    left, right = _binary_operands(node)
    if left is None or right is None:
        return set()
    out: set[str] = set()
    for a, b in ((left, right), (right, left)):
        name = _identifier_name(a)
        if name and _is_bool_literal(b):
            out.add(name)
    return out


def names_in_boolean_test(node: Node) -> set[str]:
    if node.type == "identifier":
        name = _identifier_name(node)
        return {name} if name else set()
    if node.type == "unary":
        op = node.child(0)
        operand = node.child(1) if node.child_count > 1 else None
        if op is not None and op.type == "!" and operand is not None:
            return names_in_boolean_test(operand)
        return set()
    if node.type == "binary" and _binary_operator(node) in _BOOL_BIN_OPS:
        left, right = _binary_operands(node)
        out: set[str] = set()
        if left is not None:
            out |= names_in_boolean_test(left)
        if right is not None:
            out |= names_in_boolean_test(right)
        return out
    if node.type == "binary":
        return compare_boolean_flag_names(node)
    if node.type == "parenthesized_statements":
        for i in range(node.child_count):
            child = node.child(i)
            if child.type not in {"(", ")"}:
                return names_in_boolean_test(child)
    return set()


def bool_expression_load_names(node: Node) -> set[str]:
    if node.type == "identifier":
        name = _identifier_name(node)
        return {name} if name else set()
    if node.type == "unary":
        op = node.child(0)
        operand = node.child(1) if node.child_count > 1 else None
        if op is not None and op.type == "!" and operand is not None:
            return bool_expression_load_names(operand)
        return set()
    if node.type == "binary" and _binary_operator(node) in _BOOL_BIN_OPS:
        left, right = _binary_operands(node)
        out: set[str] = set()
        if left is not None:
            out |= bool_expression_load_names(left)
        if right is not None:
            out |= bool_expression_load_names(right)
        return out
    if node.type == "binary":
        return compare_boolean_flag_names(node)
    return set()


def _if_condition(node: Node) -> Node | None:
    if node.type != "if":
        return None
    for i in range(node.child_count):
        child = node.child(i)
        if child.type not in {"if", "then", "end", ";", "else"}:
            return child
    return None


def _while_condition(node: Node) -> Node | None:
    if node.type != "while":
        return None
    for i in range(node.child_count):
        child = node.child(i)
        if child.type not in {"while", "do", "end", ";"}:
            return child
    return None


def _hits_from_value(
    value: Node, targets: list[str], line: int, container: Node
) -> list[RubyFlagHit]:
    hits: list[RubyFlagHit] = []
    if _is_bool_literal(value):
        for vid in targets:
            hits.append(RubyFlagHit(vid, line, "assign_bool_literal", container))
    elif _binary_operator(value) in _BOOL_BIN_OPS:
        names_rhs = bool_expression_load_names(value)
        for vid in targets:
            hits.append(RubyFlagHit(vid, line, "assign_boolop_lhs", container))
        for vid in names_rhs:
            hits.append(RubyFlagHit(vid, line, "assign_boolop_rhs", value))
    elif value.type == "unary":
        op = value.child(0)
        operand = value.child(1) if value.child_count > 1 else None
        if op is not None and op.type == "!" and operand is not None:
            inner_name = _identifier_name(operand)
            if inner_name:
                for vid in targets:
                    hits.append(RubyFlagHit(vid, line, "assign_not_name", container))
                hits.append(RubyFlagHit(inner_name, line, "assign_not_name_inner", value))
    return hits


def hits_from_assignment(node: Node, code: str) -> list[RubyFlagHit]:
    if node.type != "assignment":
        return []
    value = node.child_by_field_name("right")
    if value is None and node.child_count >= 3:
        value = node.child(2)
    if value is None:
        return []
    return _hits_from_value(
        value, assignment_target_names(node), node.start_point[0] + 1, node
    )


def _iter_if_tests(node: Node) -> Iterator[Node]:
    if node.type != "if":
        return
    cond = _if_condition(node)
    if cond is not None:
        yield cond
    for i in range(node.child_count):
        child = node.child(i)
        if child.type == "else":
            for j in range(child.child_count):
                sub = child.child(j)
                if sub.type == "if":
                    yield from _iter_if_tests(sub)


def collect_flag_hits(method: RubyMethod, code: str) -> list[RubyFlagHit]:
    parents = build_parent_map(method.node)
    hits: list[RubyFlagHit] = []

    def walk(node: Node) -> None:
        if node is method.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_method(method, node, parents):
            return
        if node.type == "if":
            for test in _iter_if_tests(node):
                for vid in names_in_boolean_test(test):
                    hits.append(RubyFlagHit(vid, test.start_point[0] + 1, "if_test", test))
        elif node.type == "while":
            test = _while_condition(node)
            if test is not None:
                for vid in names_in_boolean_test(test):
                    hits.append(RubyFlagHit(vid, test.start_point[0] + 1, "while_test", test))
        elif node.type == "assignment":
            hits.extend(hits_from_assignment(node, code))
        for i in range(node.child_count):
            walk(node.child(i))

    walk(method.node)
    return hits


def collect_return_hits(method: RubyMethod, code: str) -> list[RubyFlagHit]:
    parents = build_parent_map(method.node)
    hits: list[RubyFlagHit] = []

    def walk(node: Node) -> None:
        if node is method.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_method(method, node, parents):
            return
        if node.type == "return":
            value = None
            for i in range(node.child_count):
                child = node.child(i)
                if child.type == "argument_list":
                    for j in range(child.child_count):
                        inner = child.child(j)
                        if inner.type == "identifier":
                            value = inner
                            break
                    break
                if child.type == "identifier":
                    value = child
                    break
            if value is not None:
                allowed = names_in_boolean_test(value)
                if allowed:
                    name = _identifier_name(value)
                    if name and name in allowed:
                        hits.append(
                            RubyFlagHit(name, value.start_point[0] + 1, "return_bool", value)
                        )
        for i in range(node.child_count):
            walk(node.child(i))

    walk(method.node)
    return hits


def extract_boolean_flags(method: RubyMethod, code: str) -> list[dict[str, Any]]:
    hits = collect_flag_hits(method, code)
    by_var: dict[str, list[RubyFlagHit]] = defaultdict(list)
    for h in hits:
        by_var[h.variable].append(h)

    out: list[dict[str, Any]] = []
    for var in sorted(by_var):
        hs = by_var[var]
        hs.sort(key=lambda h: (h.line, h.pattern))
        first = hs[0]
        snippet = _node_text(code, first.node).strip().splitlines()
        out.append(
            {
                "variable": var,
                "role": "boolean_flag",
                "line": first.line,
                "code": snippet[0] if snippet else "",
                "function": method.name,
            }
        )
    return out


def labeled_rows_from_ruby_code(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    tree = parse_ruby(code)
    if tree.root_node.has_error:
        return [], "ruby parse error"

    methods = list(iter_top_level_methods(tree.root_node))
    if not methods:
        return [], "no top-level method definition"

    rows: list[dict[str, Any]] = []
    for method in methods:
        for ex in extract_boolean_flags(method, code):
            if repo is not None:
                ex["repo"] = repo
            if path is not None:
                ex["path"] = path
            if source_row is not None:
                ex["source_row"] = source_row
            rows.append(ex)
    return rows, None
