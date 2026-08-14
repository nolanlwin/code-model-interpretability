"""Role-agnostic variable occurrences for ALL five roles, every XLCoST language.

The boolean workstream's ``xlcost_occurrences.py`` is AST-precise but only
knows ``boolean_flag``. Causal work needs the other four roles too, and the
role definitions for them live in ``pipeline/roles.py`` (Python AST, regex
elsewhere). This module bridges the two: it asks pipeline for
``{role: {names}}`` and then locates every real occurrence of those names in
the source, emitting the SAME row schema the boolean pipeline uses so
downstream tools do not care which extractor produced a row.

Occurrences are found by scanning identifiers outside strings and comments
(reusing the ``_code_mask`` lexer that ``baselines.py`` already relies on),
never by naive substring search -- ``i`` would otherwise match inside
``if``, ``print``, and every string literal in the program.

    python scripts/role_occurrences.py extract \
        --input data/xlcost/python_train.jsonl \
        --role accumulator \
        --output outputs/role_occ/accumulator_python_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from baselines import _code_mask  # noqa: E402  (string/comment-aware lexer)
from pipeline.roles import extract_roles  # noqa: E402

ROLES = ("index_key", "accumulator", "iterator", "boolean", "class_struct")

_IDENT = re.compile(r"[A-Za-z_]\w*")


def function_spans(code: str, language: str) -> list[tuple[int, int, str]]:
    """``(start_char, end_char, name)`` for each function, innermost last.

    Scope matters here: two functions in one program routinely reuse ``i``
    or ``res``, and treating those as one binding would let an intervention
    edit a DIFFERENT variable than the one being scored. Returns [] when the
    language has no parser (C#), and callers must treat that as "scope
    unknown" rather than "one global scope".

    Spans are CHARACTER offsets. tree-sitter reports bytes, and a third of
    the Java/PHP corpus carries multi-byte U+2581, so the conversion is not
    optional.
    """
    if language == "Python":
        import ast as _ast
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return []
        starts = [0]
        for ln in code.splitlines(keepends=True):
            starts.append(starts[-1] + len(ln))

        def off(lineno, col):
            return starts[lineno - 1] + col if lineno - 1 < len(starts) else 0

        out = []
        for n in _ast.walk(tree):
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if n.end_lineno is None:
                    continue
                out.append((off(n.lineno, n.col_offset),
                            off(n.end_lineno, n.end_col_offset), n.name))
        return sorted(out, key=lambda t: (t[1] - t[0]), reverse=True)

    # PHP is NOT like the others: it parses to a module (the source may be
    # wrapped in <?php, shifting every offset), it takes that module rather
    # than a root node, and its spans must go through span_in_original.
    # Treating it like Java silently returned zero functions, which made
    # every PHP case unscoped and therefore refused.
    try:
        from cpp_csn_parse import byte_to_char
        to_char = byte_to_char(code)
        out = []
        if language == "PHP":
            import php_csn_parse as _php
            module = _php.parse_php_module(code)
            for fn in _php.iter_top_level_functions(module):
                s0, e0 = _php.span_in_original(module, fn.node)
                out.append((to_char(max(0, s0)), to_char(max(0, e0)), fn.name))
        else:
            mods = {
                "Java": ("java_csn_parse", "parse_java", "iter_top_level_methods"),
                "Javascript": ("javascript_csn_parse", "parse_javascript",
                               "iter_top_level_functions"),
                "C++": ("cpp_csn_parse", "parse_cpp", "iter_top_level_functions"),
                "C": ("cpp_csn_parse", "parse_cpp", "iter_top_level_functions"),
            }
            spec = mods.get(language)
            if spec is None:
                return []
            modname, parsefn, iterfn = spec
            mod = __import__(modname)
            tree = getattr(mod, parsefn)(code)
            for fn in getattr(mod, iterfn)(tree.root_node):
                out.append((to_char(fn.start_byte), to_char(fn.end_byte), fn.name))
        return sorted(out, key=lambda t: (t[1] - t[0]), reverse=True)
    except Exception:
        return []


def _enclosing(spans: list[tuple[int, int, str]], pos: int):
    """Innermost enclosing scope as ``(scope_id, name)``; widest-first input.

    The id is ``name@start-end``, NOT the bare name. Overloaded methods,
    same-named methods in different classes, and nested functions all share a
    name, so keying on the name alone would merge distinct bindings again --
    the very bug the scope work was meant to fix. A character span is unique
    within a program, so it is the identity.
    """
    found = (None, None)
    for s0, e0, name in spans:
        if s0 <= pos < e0:
            found = (f"{name}@{s0}-{e0}", name)
    return found


#: Languages with BLOCK scoping, where one function can declare the same
#: spelling twice as two independent bindings. Python and PHP are
#: function-scoped, so a repeated assignment there is the same variable.
_BLOCK_SCOPED = {"Java", "Javascript", "C++", "C"}
#: Node type -> field holding the bound name. Loop headers bind without any
#: *_declarator node, so visiting only declarators misses every enhanced-for
#: / range-for / for-of binding -- exactly the repeated `for (int x : a)`
#: shape that makes two independent variables look like one.
_BINDING_FIELDS = {
    "init_declarator": "declarator",       # C, C++
    "variable_declarator": "name",         # Java, JavaScript
    "enhanced_for_statement": "name",      # Java   for (int x : a)
    "for_range_loop": "declarator",        # C, C++ for (int x : a)
    "for_in_statement": "left",            # JS     for (const x of a)
}
#: JS `for (x of a)` with no let/const/var declares nothing -- it assigns to
#: an existing binding, so it must not be counted as a redeclaration.
_JS_DECLARING_KINDS = {"let", "const", "var"}


def ambiguous_bindings(code: str, language: str, fspans) -> dict:
    """``{scope_id: {names declared more than once in that scope}}``.

    In C, C++, Java and JavaScript a function may declare the same spelling
    in two disjoint blocks -- ``for (int i ...)`` twice, say -- and those are
    two independent bindings. Keying occurrences on (function, name) would
    merge them, so the readout could come from the second binding while the
    intervention edits the first: the experiment would score one variable and
    edit another.

    Resolving block scope exactly would mean a per-language scope analyser.
    Instead these variables are DROPPED and counted. That loses data in the
    ~20% of C++/Java programs where it happens, and it cannot mis-attribute
    anything, which is the right side to err on for a causal claim.
    """
    if language not in _BLOCK_SCOPED:
        return {}
    mods = {"Java": ("tree_sitter_java", "language"),
            "Javascript": ("tree_sitter_javascript", "language"),
            "C++": ("tree_sitter_cpp", "language"),
            "C": ("tree_sitter_cpp", "language")}
    spec = mods.get(language)
    if spec is None:
        return {}
    try:
        from tree_sitter import Language, Parser
        mod = __import__(spec[0])
        parser = Parser(Language(getattr(mod, spec[1])()))
    except Exception:
        return {}

    out: dict = {}
    for s0, e0, name in fspans:
        try:
            tree = parser.parse(code[s0:e0].encode("utf-8"))
        except Exception:
            continue
        counts: Counter = Counter()
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            field = _BINDING_FIELDS.get(n.type)
            if field is not None:
                if n.type == "for_in_statement":
                    kind = n.child_by_field_name("kind")
                    kind_txt = kind.text.decode("utf-8") if kind is not None else ""
                    if kind_txt not in _JS_DECLARING_KINDS:
                        stack.extend(n.children)
                        continue
                d = n.child_by_field_name(field)
                if d is None:
                    d = n.child_by_field_name("declarator") or n.child_by_field_name("name")
                seen = 0
                while d is not None and d.type != "identifier" and seen < 8:
                    d = d.child_by_field_name("declarator") or d.child_by_field_name("name")
                    seen += 1
                if d is not None and d.type == "identifier":
                    counts[d.text.decode("utf-8")] += 1
            stack.extend(n.children)
        dup = {k for k, v in counts.items() if v > 1}
        if dup:
            out[f"{name}@{s0}-{e0}"] = dup
    return out


def _scoped_role_map(code: str, language: str, fspans) -> dict:
    """``{scope_id: {role: {names}}}``, plus ``None`` for module level.

    Roles are re-extracted per function rather than once per program. Two
    functions routinely reuse a spelling for different purposes -- ``i`` as
    an index in one and an accumulator in the other -- and a program-wide
    role map cannot represent that. Filtering on the program-wide map would
    then drop the spelling from BOTH scopes even though each is
    individually unambiguous.
    """
    import textwrap
    out: dict = {None: _safe_roles(code, language)}
    for s0, e0, name in fspans:
        sid = f"{name}@{s0}-{e0}"
        slice_ = code[s0:e0]
        # Blank any NESTED function body before extracting this scope's
        # roles: otherwise ast.walk sees the inner function's variables too
        # and an outer name that the inner scope uses differently looks
        # multi-role. (Zero incidence in XLCoST -- 0/800 Python programs
        # contain a nested function -- but the guard is nearly free.)
        for n0, n1, _nm in fspans:
            if n0 > s0 and n1 <= e0:
                a, b = n0 - s0, n1 - s0
                slice_ = slice_[:a] + (" " * (b - a)) + slice_[b:]
        out[sid] = _safe_roles(textwrap.dedent(slice_), language)
    return out


def _safe_roles(src: str, language: str) -> dict:
    try:
        return {k: set(v or ()) for k, v in (extract_roles(src, language) or {}).items()}
    except Exception:
        return {}


def occurrence_rows(code: str, language: str, role: str, problem_id: str,
                    drop_multi_role: bool = True) -> list[dict]:
    """Every in-code occurrence of every variable pipeline assigns to ``role``.

    Variables that satisfy more than one role predicate (a loop counter is
    routinely iterator AND index_key AND accumulator) are DROPPED by default
    and counted. A causal claim about "the accumulator role" is not
    meaningful for a variable that is equally the index, and keeping it would
    make the label depend on which role happened to be extracted first.
    """
    program_roles = _safe_roles(code, language)
    fspans = function_spans(code, language)
    scoped = _scoped_role_map(code, language, fspans)
    # Bail only if NO scope carries the role. Gating on the program-wide map
    # would undo the per-scope filtering below: a variable can be an
    # accumulator inside one function while the whole-program extractor,
    # seeing both uses at once, never assigns that role at all.
    if not any(m.get(role) for m in scoped.values()) and not program_roles.get(role):
        return []

    is_code, _ = _code_mask(code)
    ambiguous = ambiguous_bindings(code, language, fspans)
    rows: list[dict] = []
    for m in _IDENT.finditer(code):
        name = m.group(0)
        s, e = m.start(), m.end()
        sid, sname = _enclosing(fspans, s)
        # Role membership is decided in the occurrence's OWN scope, falling
        # back to the program map only for module-level code.
        rmap = scoped.get(sid) or program_roles
        if name not in rmap.get(role, set()):
            continue
        if drop_multi_role and any(
            name in rmap.get(rl, set()) for rl in rmap if rl != role
        ):
            continue
        # Two block-local bindings of the same spelling in one function are
        # two variables; refuse rather than merge them.
        if name in ambiguous.get(sid, ()):
            continue
        # Reject identifiers inside strings/comments. Checking only the first
        # character is enough: the lexer marks a literal contiguously.
        if not is_code[s]:
            continue
        line = code.count("\n", 0, s) + 1
        line_start = code.rfind("\n", 0, s) + 1
        rows.append({
            "problem_id": problem_id,
            "language": language,
            "variable": name,
            "role": role,
            "function": sid,
            "function_name": sname,
            "scope_known": bool(fspans),
            "source_span": [s, e],
            "line": line,
            "col_offset": s - line_start,
            "end_col_offset": e - line_start,
        })

    rows.sort(key=lambda r: r["source_span"][0])
    # Stable ids in the protocol's shape. There is no reliable enclosing
    # function for the regex languages, so the function slot is fixed at f0
    # and the binding index orders variables by first appearance.
    order: dict = {}
    counter: Counter = Counter()
    for r in rows:
        v = r["variable"]
        if v not in order:
            order[v] = len(order)
        b, o = order[v], counter[v]
        counter[v] += 1
        r["occurrence_id"] = f"{problem_id}:{language}:f0:b{b}:o{o}"
    return rows


def cmd_extract(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_prog = n_rows = n_with = 0
    per_var = Counter()
    with out.open("w", encoding="utf-8") as f:
        for ln in Path(args.input).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            n_prog += 1
            wanted = ROLES if args.role == "all" else (args.role,)
            rows = [
                row
                for rl in wanted
                for row in occurrence_rows(rec["code"], rec["language"], rl, rec["problem_id"])
            ]
            if rows:
                n_with += 1
            for r in rows:
                # Span integrity, same gate as the boolean path: the span must
                # slice back to the variable or the row is a lie.
                s, e = r["source_span"]
                if rec["code"][s:e] != r["variable"]:
                    continue
                r["split"] = rec.get("split")
                f.write(json.dumps(r) + "\n")
                n_rows += 1
                per_var[r["variable"]] += 1

    stats = {
        "input": args.input,
        "role": args.role,
        "programs": n_prog,
        "programs_with_role": n_with,
        "occurrences_written": n_rows,
        "distinct_variables": len(per_var),
        "output": str(out),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--input", required=True, help="canonical XLCoST jsonl")
    e.add_argument("--role", required=True, choices=(*ROLES, "all"),
                   help="'all' writes every role into one file, which is what "
                        "the causal path needs: its distractor must hold a "
                        "different role than the target")
    e.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    return cmd_extract(args)


if __name__ == "__main__":
    sys.exit(main())
