"""
Tree-sitter helpers for PHP CodeSearchNet snippets (single function per row).

CSN rows are often bare ``public function ...`` bodies; those are wrapped in a
synthetic class before parsing. Byte spans are mapped back to the original code.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_php as tsphp
from tree_sitter import Language, Node, Parser, Tree

_FUNC_DECL_TYPES = frozenset({"function_definition", "method_declaration"})
_WRAP_PREFIX = "<?php\nclass __CSN__ {\n"
_WRAP_SUFFIX = "\n}\n"


@dataclass(frozen=True)
class PhpFunction:
    node: Node
    name: str
    start_byte: int
    end_byte: int
    start_line: int


@dataclass(frozen=True)
class PhpModule:
    tree: Tree
    span_offset: int
    original: str


def _php_language() -> Language:
    return Language(tsphp.language_php())


@lru_cache(maxsize=1)
def php_parser() -> Parser:
    return Parser(_php_language())


def parse_php_module(code: str) -> PhpModule:
    stripped = code.lstrip()
    if stripped.startswith("<?php"):
        return PhpModule(php_parser().parse(bytes(code, "utf-8")), 0, code)
    wrapped = _WRAP_PREFIX + code + _WRAP_SUFFIX
    return PhpModule(php_parser().parse(bytes(wrapped, "utf-8")), len(_WRAP_PREFIX), code)


def span_in_original(module: PhpModule, node: Node) -> tuple[int, int]:
    return node.start_byte - module.span_offset, node.end_byte - module.span_offset


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


def _yield_function(node: Node, module: PhpModule) -> Iterator[PhpFunction]:
    name = function_name(node)
    if not name:
        return
    s, e = span_in_original(module, node)
    yield PhpFunction(
        node=node,
        name=name,
        start_byte=s,
        end_byte=e,
        start_line=node.start_point[0] + 1,
    )


def iter_top_level_functions(module: PhpModule) -> Iterator[PhpFunction]:
    root = module.tree.root_node

    def walk(node: Node) -> Iterator[PhpFunction]:
        if node.type in _FUNC_DECL_TYPES:
            yield from _yield_function(node, module)
            return
        for i in range(node.child_count):
            yield from walk(node.child(i))

    yield from walk(root)


def inside_nested_function(
    fn: PhpFunction, node: Node, parents: dict[Node, Node | None]
) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not fn.node:
        if cur.type in _FUNC_DECL_TYPES:
            return True
        if cur.type == "anonymous_function_creation_expression":
            return True
        cur = parents.get(cur)
    return False


def variable_nodes_in(node: Node) -> Iterator[Node]:
    if node.type == "variable_name":
        yield node
    for i in range(node.child_count):
        yield from variable_nodes_in(node.child(i))


def variable_label(node: Node) -> str | None:
    if node.type != "variable_name":
        return None
    text = node.text.decode("utf-8")
    return text[1:] if text.startswith("$") else text


def assignment_target_names(node: Node) -> list[str]:
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        if left is None:
            return []
        if left.type == "variable_name":
            label = variable_label(left)
            return [label] if label else []
        if left.type == "list_literal":
            out: list[str] = []
            for i in range(left.child_count):
                child = left.child(i)
                if child.type == "variable_name":
                    label = variable_label(child)
                    if label:
                        out.append(label)
            return out
    return []
