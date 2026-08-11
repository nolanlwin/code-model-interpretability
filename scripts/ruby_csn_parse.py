"""
Tree-sitter helpers for Ruby CodeSearchNet snippets (single method per row).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_ruby as tsrb
from tree_sitter import Language, Node, Parser, Tree

_METHOD_TYPES = frozenset({"method"})


@dataclass(frozen=True)
class RubyMethod:
    node: Node
    name: str
    start_byte: int
    end_byte: int
    start_line: int


def _ruby_language() -> Language:
    return Language(tsrb.language())


@lru_cache(maxsize=1)
def ruby_parser() -> Parser:
    return Parser(_ruby_language())


def parse_ruby(code: str) -> Tree:
    return ruby_parser().parse(bytes(code, "utf-8"))


def build_parent_map(root: Node) -> dict[Node, Node | None]:
    parents: dict[Node, Node | None] = {}

    def visit(node: Node, parent: Node | None) -> None:
        parents[node] = parent
        for i in range(node.child_count):
            visit(node.child(i), node)

    visit(root, None)
    return parents


def method_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is None:
        return None
    return name.text.decode("utf-8")


def iter_top_level_methods(root: Node) -> Iterator[RubyMethod]:
    for i in range(root.child_count):
        child = root.child(i)
        if child.type != "method":
            continue
        name = method_name(child)
        if not name:
            continue
        yield RubyMethod(
            node=child,
            name=name,
            start_byte=child.start_byte,
            end_byte=child.end_byte,
            start_line=child.start_point[0] + 1,
        )


def inside_nested_method(
    method: RubyMethod, node: Node, parents: dict[Node, Node | None]
) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not method.node:
        if cur.type in _METHOD_TYPES:
            return True
        if cur.type == "lambda" or cur.type == "block":
            return True
        cur = parents.get(cur)
    return False


def identifier_nodes_in(node: Node) -> Iterator[Node]:
    if node.type == "identifier":
        yield node
    for i in range(node.child_count):
        yield from identifier_nodes_in(node.child(i))


def assignment_target_names(node: Node) -> list[str]:
    if node.type != "assignment":
        return []
    left = node.child_by_field_name("left")
    if left is None and node.child_count > 0:
        left = node.child(0)
    if left is not None and left.type == "identifier":
        return [left.text.decode("utf-8")]
    return []
