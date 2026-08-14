"""
Boolean flags extraction for C++ (tree-sitter).

Mirrors the Java heuristics in ``java_boolean_flag_roles.py`` so the two
languages label the same construct the same way:
- if / while / conditional-expression tests
- assignments to true/false, boolean operators, or ``!name``
- return of boolean-shaped expressions

Where C++ and Java diverge, ``cpp_csn_parse`` documents the mapping. The
one behavioural difference worth stating here: C++ exposes ``operator`` as
a named field on ``binary_expression`` and ``unary_expression``, so this
module reads the field instead of scanning children for an operator token
as the Java port does. Same result, fewer ways to be wrong.

PRECISION AUDIT (XLCoST cpp_train, 8,406 parsed programs, 6,428
occurrences). Only **13.9%** of the variables this module labels
``boolean_flag`` are declared ``bool``. The residue is not a C++ porting
defect -- Java measures 14.9% on the same audit -- but a property of the
shared heuristics, and it has two sources:

* ``collect_return_hits`` accepts ``return <identifier>`` for any
  identifier, because ``names_in_boolean_test`` maps a bare name to
  itself. Every returned variable in every program becomes a
  ``return_use`` flag; that class is 76% of C++ occurrences and 4,827 of
  Java's 4,851 return_use rows are not boolean.
* C++ additionally admits any scalar in a condition (``if (count)``),
  which Java's type system rejects. This widens ``conditional_use`` here
  relative to Java.

Anyone comparing ``boolean_flag`` results across languages, or reading a
probe trained on this label as evidence about boolean reasoning, needs
those numbers first. See also the ``_bool_literal`` defect in
``boolean_flag_roles.py`` (the Python extractor), which is a separate and
larger problem.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from tree_sitter import Node

from cpp_csn_parse import (
    CppFunction,
    assignment_target_names,
    build_parent_map,
    condition_expression,
    identifier_nodes_in,
    inside_nested_function,
    iter_top_level_functions,
    parse_cpp,
)


@dataclass
class CppFlagHit:
    variable: str
    line: int
    pattern: str
    node: Node


_BOOL_LITERAL_TYPES = frozenset({"true", "false"})
_BOOL_BIN_OPS = frozenset({"&&", "||"})
_BOOL_COMPARE_OPS = frozenset({"==", "!="})


def _node_text(code: str, node: Node) -> str:
    return code[node.start_byte : node.end_byte]


def _identifier_name(node: Node) -> str | None:
    if node.type != "identifier":
        return None
    return node.text.decode("utf-8")


def _is_bool_literal(node: Node) -> bool:
    return node.type in _BOOL_LITERAL_TYPES


def _operator(node: Node) -> str | None:
    op = node.child_by_field_name("operator")
    return op.type if op is not None else None


def _unary_operand(node: Node) -> Node | None:
    """The operand of a logical negation, or None if this is not one."""
    if node.type != "unary_expression" or _operator(node) != "!":
        return None
    return node.child_by_field_name("argument")


def compare_boolean_flag_names(node: Node) -> set[str]:
    """Names compared to true/false with == or != (symmetric)."""
    if node.type != "binary_expression" or _operator(node) not in _BOOL_COMPARE_OPS:
        return set()
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
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
    operand = _unary_operand(node)
    if operand is not None:
        return names_in_boolean_test(operand)
    if node.type == "unary_expression":
        return set()
    if node.type == "binary_expression" and _operator(node) in _BOOL_BIN_OPS:
        out: set[str] = set()
        for sub in (node.child_by_field_name("left"), node.child_by_field_name("right")):
            if sub is not None:
                out |= names_in_boolean_test(sub)
        return out
    if node.type == "parenthesized_expression":
        for i in range(node.child_count):
            child = node.child(i)
            if child.type not in {"(", ")"}:
                return names_in_boolean_test(child)
        return set()
    if node.type == "binary_expression":
        return compare_boolean_flag_names(node)
    if node.type == "conditional_expression":
        cond = node.child_by_field_name("condition")
        return names_in_boolean_test(cond) if cond is not None else set()
    return set()


def bool_expression_load_names(node: Node) -> set[str]:
    if node.type == "identifier":
        name = _identifier_name(node)
        return {name} if name else set()
    operand = _unary_operand(node)
    if operand is not None:
        return bool_expression_load_names(operand)
    if node.type == "unary_expression":
        return set()
    if node.type == "binary_expression" and _operator(node) in _BOOL_BIN_OPS:
        out: set[str] = set()
        for sub in (node.child_by_field_name("left"), node.child_by_field_name("right")):
            if sub is not None:
                out |= bool_expression_load_names(sub)
        return out
    if node.type == "parenthesized_expression":
        for i in range(node.child_count):
            child = node.child(i)
            if child.type not in {"(", ")"}:
                return bool_expression_load_names(child)
        return set()
    if node.type == "binary_expression":
        return compare_boolean_flag_names(node)
    return set()


def _hits_for_value(
    value: Node, targets: list[str], line: int, anchor: Node
) -> list[CppFlagHit]:
    """Boolean-shaped right-hand side -> hits for its targets and loads.

    Shared by declaration and assignment so the two cannot drift apart;
    ``anchor`` is the node whose span represents the target occurrence.
    """
    hits: list[CppFlagHit] = []
    if _is_bool_literal(value):
        for vid in targets:
            hits.append(CppFlagHit(vid, line, "assign_bool_literal", anchor))
        return hits
    if value.type == "binary_expression" and _operator(value) in _BOOL_BIN_OPS:
        for vid in targets:
            hits.append(CppFlagHit(vid, line, "assign_boolop_lhs", anchor))
        for vid in bool_expression_load_names(value):
            hits.append(CppFlagHit(vid, line, "assign_boolop_rhs", value))
        return hits
    operand = _unary_operand(value)
    if operand is not None:
        inner_name = _identifier_name(operand)
        if inner_name:
            for vid in targets:
                hits.append(CppFlagHit(vid, line, "assign_not_name", anchor))
            hits.append(CppFlagHit(inner_name, line, "assign_not_name_inner", value))
    return hits


def hits_from_declaration(node: Node, code: str) -> list[CppFlagHit]:
    hits: list[CppFlagHit] = []
    line = node.start_point[0] + 1
    for i in range(node.child_count):
        child = node.child(i)
        if child.type != "init_declarator":
            continue
        value = child.child_by_field_name("value")
        if value is None:
            continue
        hits.extend(_hits_for_value(value, assignment_target_names(child), line, child))
    return hits


def hits_from_assignment(node: Node, code: str) -> list[CppFlagHit]:
    if node.type != "assignment_expression":
        return []
    # Compound assignment (``ok &= cond``) is not a boolean-flag definition
    # under these heuristics; only plain ``=`` establishes the flag.
    if _operator(node) != "=":
        return []
    value = node.child_by_field_name("right")
    if value is None:
        return []
    line = node.start_point[0] + 1
    return _hits_for_value(value, assignment_target_names(node), line, node)


def _iter_if_tests(node: Node) -> Iterator[Node]:
    """Yield this ``if``'s tested expression -- and only this one.

    Deliberately does NOT recurse into ``else_clause``. In C++ an
    ``else if`` is a real ``if_statement`` nested inside a real
    ``else_clause`` node, and ``collect_flag_hits`` already walks every
    node in the function, so recursing here would collect each chained
    test a second time (and a third at depth two). The Java port appears
    to recurse but cannot: tree-sitter-java's ``else`` is a bare keyword
    token with no children, so its loop never fires. Emitting once here
    is what makes the two languages agree.
    """
    if node.type != "if_statement":
        return
    cond = condition_expression(node)
    if cond is not None:
        yield cond


def collect_flag_hits(func: CppFunction, code: str) -> list[CppFlagHit]:
    parents = build_parent_map(func.node)
    hits: list[CppFlagHit] = []

    def walk(node: Node) -> None:
        if node is func.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_function(func, node, parents):
            return
        if node.type == "if_statement":
            for test in _iter_if_tests(node):
                for vid in names_in_boolean_test(test):
                    hits.append(CppFlagHit(vid, test.start_point[0] + 1, "if_test", test))
        elif node.type == "while_statement":
            test = condition_expression(node)
            if test is not None:
                for vid in names_in_boolean_test(test):
                    hits.append(
                        CppFlagHit(vid, test.start_point[0] + 1, "while_test", test)
                    )
        elif node.type == "conditional_expression":
            cond = node.child_by_field_name("condition")
            if cond is not None:
                for vid in names_in_boolean_test(cond):
                    hits.append(
                        CppFlagHit(vid, cond.start_point[0] + 1, "if_exp_test", cond)
                    )
        elif node.type == "declaration":
            hits.extend(hits_from_declaration(node, code))
        elif node.type == "assignment_expression":
            hits.extend(hits_from_assignment(node, code))
        for i in range(node.child_count):
            walk(node.child(i))

    walk(func.node)
    return hits


def collect_return_hits(func: CppFunction, code: str) -> list[CppFlagHit]:
    parents = build_parent_map(func.node)
    hits: list[CppFlagHit] = []

    def walk(node: Node) -> None:
        if node is func.node:
            for i in range(node.child_count):
                walk(node.child(i))
            return
        if inside_nested_function(func, node, parents):
            return
        if node.type == "return_statement":
            value = None
            for i in range(node.child_count):
                child = node.child(i)
                if child.type not in {"return", ";"}:
                    value = child
                    break
            if value is not None:
                allowed = names_in_boolean_test(value)
                if allowed:
                    for sub in identifier_nodes_in(value):
                        name = _identifier_name(sub)
                        if name and name in allowed:
                            hits.append(
                                CppFlagHit(name, sub.start_point[0] + 1, "return_bool", sub)
                            )
        for i in range(node.child_count):
            walk(node.child(i))

    walk(func.node)
    return hits


def extract_boolean_flags(func: CppFunction, code: str) -> list[dict[str, Any]]:
    hits = collect_flag_hits(func, code)
    by_var: dict[str, list[CppFlagHit]] = defaultdict(list)
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
                "function": func.name,
            }
        )
    return out


def labeled_rows_from_cpp_code(
    code: str,
    *,
    repo: str | None = None,
    path: str | None = None,
    source_row: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    tree = parse_cpp(code)
    if tree.root_node.has_error:
        return [], "cpp parse error"

    funcs = list(iter_top_level_functions(tree.root_node))
    if not funcs:
        return [], "no top-level function definition"

    rows: list[dict[str, Any]] = []
    for func in funcs:
        for ex in extract_boolean_flags(func, code):
            if repo is not None:
                ex["repo"] = repo
            if path is not None:
                ex["path"] = path
            if source_row is not None:
                ex["source_row"] = source_row
            rows.append(ex)
    return rows, None
