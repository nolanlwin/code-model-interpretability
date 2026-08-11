"""
Tree-sitter helpers for JavaScript CodeSearchNet snippets (single function per row).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Node, Parser, Tree

_FUNC_DECL_TYPES = frozenset({"function_declaration", "method_definition"})
_NESTED_FUNC_TYPES = frozenset({*_FUNC_DECL_TYPES, "arrow_function"})


@dataclass(frozen=True)
class JavaScriptFunction:
    node: Node
    name: str
    start_byte: int
    end_byte: int
    start_line: int


def _javascript_language() -> Language:
    return Language(tsjs.language())


@lru_cache(maxsize=1)
def javascript_parser() -> Parser:
    return Parser(_javascript_language())


def parse_javascript(code: str) -> Tree:
    return javascript_parser().parse(bytes(code, "utf-8"))


def build_parent_map(root: Node) -> dict[Node, Node | None]:
    parents: dict[Node, Node | None] = {}

    def visit(node: Node, parent: Node | None) -> None:
        parents[node] = parent
        for i in range(node.child_count):
            visit(node.child(i), node)

    visit(root, None)
    return parents


def function_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is None:
        return None
    return name.text.decode("utf-8")


def _yield_function(node: Node) -> Iterator[JavaScriptFunction]:
    name = function_name(node)
    if not name:
        return
    yield JavaScriptFunction(
        node=node,
        name=name,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_line=node.start_point[0] + 1,
    )


def iter_top_level_functions(root: Node) -> Iterator[JavaScriptFunction]:
    for i in range(root.child_count):
        child = root.child(i)
        if child.type == "function_declaration":
            yield from _yield_function(child)
        elif child.type == "class_declaration":
            body = child.child_by_field_name("body")
            if body is None:
                continue
            for j in range(body.child_count):
                member = body.child(j)
                if member.type == "method_definition":
                    yield from _yield_function(member)


def inside_nested_function(
    fn: JavaScriptFunction, node: Node, parents: dict[Node, Node | None]
) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not fn.node:
        if cur.type in _NESTED_FUNC_TYPES:
            return True
        cur = parents.get(cur)
    return False


def identifier_nodes_in(node: Node) -> Iterator[Node]:
    if node.type == "identifier":
        yield node
    for i in range(node.child_count):
        yield from identifier_nodes_in(node.child(i))


def assignment_target_names(node: Node) -> list[str]:
    if node.type == "variable_declarator":
        name = node.child_by_field_name("name")
        if name is not None and name.type == "identifier":
            return [name.text.decode("utf-8")]
        return []
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        if left is not None and left.type == "identifier":
            return [left.text.decode("utf-8")]
        return []
    if node.type in {"lexical_declaration", "variable_declaration"}:
        out: list[str] = []
        for i in range(node.child_count):
            child = node.child(i)
            if child.type == "variable_declarator":
                out.extend(assignment_target_names(child))
        return out
    return []
