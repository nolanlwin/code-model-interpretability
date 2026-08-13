"""
Tree-sitter helpers for C++ programs (XLCoST rows: free functions + main).

Named ``*_csn_parse`` to match the java/javascript/php/ruby modules it sits
beside; the CodeSearchNet lineage is historical and C++ is XLCoST-only.

The grammar differs from Java in ways that matter to every consumer here,
so they are named once, up front:

===========================  ==================================
Java                         C++
===========================  ==================================
``method_declaration``       ``function_definition``
``variable_declarator``      ``init_declarator``
``local_variable_declaration``  ``declaration``
``ternary_expression``       ``conditional_expression``
``array_access``             ``subscript_expression``
``field_access``             ``field_expression``
``parenthesized_expression`` ``condition_clause`` (if/while tests)
===========================  ==================================

Two further C++-only shapes are handled explicitly because getting them
wrong fails silently rather than loudly:

* A function's name sits behind a declarator chain -- ``bool* f()`` nests
  ``pointer_declarator > function_declarator > identifier`` -- and
  ``reference_declarator`` exposes its inner declarator as a plain child
  rather than through the ``declarator`` field. ``function_name`` walks
  both.
* In ``v[i]`` the index ``i`` is a child of ``subscript_argument_list``,
  not of ``subscript_expression`` as Java's ``array_access`` would give.
  Indexing detection must accept both.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser, Tree


@dataclass(frozen=True)
class CppFunction:
    node: Node
    name: str
    start_byte: int
    end_byte: int
    start_line: int


#: Declarator wrappers between ``function_definition`` and the name.
_DECLARATOR_WRAPPERS = frozenset(
    {"pointer_declarator", "reference_declarator", "array_declarator"}
)
#: Node types that can carry a function or variable name in C++.
_NAME_TYPES = frozenset({"identifier", "qualified_identifier", "field_identifier"})
#: Function-like bodies that nest inside another function.
_NESTED_FUNCTION_TYPES = frozenset({"lambda_expression", "function_definition"})


def _cpp_language() -> Language:
    return Language(tscpp.language())


@lru_cache(maxsize=1)
def cpp_parser() -> Parser:
    return Parser(_cpp_language())


def parse_cpp(code: str) -> Tree:
    return cpp_parser().parse(bytes(code, "utf-8"))


def build_parent_map(root: Node) -> dict[Node, Node | None]:
    parents: dict[Node, Node | None] = {}

    def visit(node: Node, parent: Node | None) -> None:
        parents[node] = parent
        for i in range(node.child_count):
            visit(node.child(i), node)

    visit(root, None)
    return parents


def _unwrap_declarator(node: Node | None) -> Node | None:
    """Walk a declarator chain down to the node carrying the name.

    ``bool* f()`` and ``bool& f()`` both wrap ``function_declarator``, but
    ``reference_declarator`` does not expose it through the ``declarator``
    field -- it appears as an ordinary child. Try the field first, then
    fall back to scanning children, so both shapes resolve.
    """
    seen = 0
    while node is not None and node.type not in _NAME_TYPES and seen < 8:
        seen += 1
        nxt = node.child_by_field_name("declarator")
        if nxt is None:
            nxt = next(
                (
                    node.child(i)
                    for i in range(node.child_count)
                    if node.child(i).type in _DECLARATOR_WRAPPERS
                    or node.child(i).type == "function_declarator"
                    or node.child(i).type in _NAME_TYPES
                ),
                None,
            )
        if nxt is None:
            return None
        node = nxt
    return node if node is not None and node.type in _NAME_TYPES else None


def function_name(func: Node) -> str | None:
    name = _unwrap_declarator(func.child_by_field_name("declarator"))
    if name is None:
        return None
    return name.text.decode("utf-8")


def iter_top_level_functions(root: Node) -> Iterator[CppFunction]:
    """Yield outermost function definitions at any nesting depth.

    XLCoST C++ rows are free functions plus ``int main()``; class member
    functions appear inside ``class_specifier``. Walk the whole tree but
    skip definitions nested inside another function body or a lambda, so
    every occurrence belongs to exactly one yielded function.
    """

    def walk(node: Node, inside: bool) -> Iterator[Node]:
        for i in range(node.child_count):
            child = node.child(i)
            if child.type == "function_definition":
                if not inside:
                    yield child
                yield from walk(child, True)
            elif child.type == "lambda_expression":
                yield from walk(child, True)
            else:
                yield from walk(child, inside)

    for child in walk(root, False):
        name = function_name(child)
        if not name:
            continue
        yield CppFunction(
            node=child,
            name=name,
            start_byte=child.start_byte,
            end_byte=child.end_byte,
            start_line=child.start_point[0] + 1,
        )


def inside_nested_function(
    func: CppFunction, node: Node, parents: dict[Node, Node | None]
) -> bool:
    cur = parents.get(node)
    while cur is not None and cur is not func.node:
        if cur.type in _NESTED_FUNCTION_TYPES:
            return True
        cur = parents.get(cur)
    return False


def identifier_nodes_in(node: Node) -> Iterator[Node]:
    if node.type == "identifier":
        yield node
    for i in range(node.child_count):
        yield from identifier_nodes_in(node.child(i))


def condition_expression(node: Node) -> Node | None:
    """The tested expression of an ``if``/``while``.

    C++ wraps it in ``condition_clause``, whose ``value`` field is absent
    when the condition is a declaration (``if (int x = f())``); return
    None there rather than guessing.
    """
    for i in range(node.child_count):
        child = node.child(i)
        if child.type == "condition_clause":
            return child.child_by_field_name("value")
    return None


def assignment_target_names(node: Node) -> list[str]:
    """Names bound by an ``init_declarator`` or ``assignment_expression``."""
    if node.type == "init_declarator":
        name = _unwrap_declarator(node.child_by_field_name("declarator"))
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
