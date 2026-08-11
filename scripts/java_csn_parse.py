"""
Tree-sitter helpers for Java CodeSearchNet snippets (single method per row).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser, Tree


@dataclass(frozen=True)
class JavaMethod:
    node: Node
    name: str
    start_byte: int
    end_byte: int
    start_line: int


def _java_language() -> Language:
    return Language(tsjava.language())


@lru_cache(maxsize=1)
def java_parser() -> Parser:
    return Parser(_java_language())


def parse_java(code: str) -> Tree:
    return java_parser().parse(bytes(code, "utf-8"))


def build_parent_map(root: Node) -> dict[Node, Node | None]:
    parents: dict[Node, Node | None] = {}

    def visit(node: Node, parent: Node | None) -> None:
        parents[node] = parent
        for i in range(node.child_count):
            visit(node.child(i), node)

    visit(root, None)
    return parents


def method_name(method: Node) -> str | None:
    for i in range(method.child_count):
        child = method.child(i)
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return None


def iter_top_level_methods(root: Node) -> Iterator[JavaMethod]:
    """Yield outermost method declarations at any class-nesting depth.

    CodeSearchNet rows are bare methods (direct children of the root);
    XLCoST programs wrap methods in ``class GFG { ... }``. Walk the whole
    tree but skip methods nested inside another method (anonymous-class
    bodies), so each occurrence belongs to exactly one yielded method.
    """

    def walk(node: Node, inside_method: bool) -> Iterator[Node]:
        for i in range(node.child_count):
            child = node.child(i)
            if child.type == "method_declaration":
                if not inside_method:
                    yield child
                yield from walk(child, True)
            else:
                yield from walk(child, inside_method)

    for child in walk(root, False):
        name = method_name(child)
        if not name:
            continue
        yield JavaMethod(
            node=child,
            name=name,
            start_byte=child.start_byte,
            end_byte=child.end_byte,
            start_line=child.start_point[0] + 1,
        )


def inside_nested_method(method: JavaMethod, node: Node, parents: dict[Node, Node | None]) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not method.node:
        if cur.type == "method_declaration":
            return True
        cur = parents.get(cur)
    return False


def identifier_nodes_in(node: Node) -> Iterator[Node]:
    if node.type == "identifier":
        yield node
    for i in range(node.child_count):
        yield from identifier_nodes_in(node.child(i))


def assignment_target_names(node: Node) -> list[str]:
    """Variable declarator or assignment_expression left-hand names."""
    if node.type == "variable_declarator":
        name = node.child_by_field_name("name")
        if name is not None and name.type == "identifier":
            return [name.text.decode("utf-8")]
        return []
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        if left is None:
            return []
        if left.type == "identifier":
            return [left.text.decode("utf-8")]
        return []
    names: list[str] = []
    for i in range(node.child_count):
        names.extend(assignment_target_names(node.child(i)))
    return names
