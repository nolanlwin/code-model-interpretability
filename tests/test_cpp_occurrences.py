"""C++ boolean-flag occurrence extraction, with EXACT expected tuples.

Exact matching is deliberate: substring assertions in an earlier test file
certified broken output twice. Each case pins (variable, occurrence_type,
detection_pattern) for every row, and asserts the source span slices back
to the variable name -- the check that catches tree-sitter's byte offsets
being used where character offsets are expected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cpp_variable_occurrences import occurrence_rows_from_cpp_code  # noqa: E402

CASES = [
    (
        "declaration with bool literal",
        "bool f(){bool ok = true; return ok;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "return_use", "return_bool"),
        ],
    ),
    (
        "plain assignment",
        "bool f(){bool ok; ok = false; return ok;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "return_use", "return_bool"),
        ],
    ),
    (
        "if test",
        "int f(){bool ok = true; if(ok) return 1; return 0;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "conditional_use", "if_test"),
        ],
    ),
    (
        "negated if test",
        "int f(){bool ok = true; if(!ok) return 1; return 0;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "conditional_use", "if_test"),
        ],
    ),
    (
        "while test is loop_use",
        "bool f(){bool go = true; while(go){go = false;} return go;}",
        [
            ("go", "assignment", "assign_bool_literal"),
            ("go", "loop_use", "while_test"),
            ("go", "assignment", "assign_bool_literal"),
            ("go", "return_use", "return_bool"),
        ],
    ),
    (
        "boolean operator splits lhs and rhs",
        "bool f(){bool a = true; bool b = false; bool c = a && b; return c;}",
        [
            ("a", "assignment", "assign_bool_literal"),
            ("b", "assignment", "assign_bool_literal"),
            ("c", "assignment", "assign_boolop_lhs"),
            ("a", "conditional_use", "assign_boolop_rhs"),
            ("b", "conditional_use", "assign_boolop_rhs"),
            ("c", "return_use", "return_bool"),
        ],
    ),
    (
        # A returned ternary makes its condition both a test and a return
        # value, so the same span carries two labels. Java does exactly the
        # same; the conflicting-label gate in xlcost_occurrences drops the
        # pair downstream. Pinned here so the agreement stays visible.
        "conditional expression test (dual-labelled, as in Java)",
        "int f(){bool ok = true; return ok ? 1 : 0;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "conditional_use", "if_exp_test"),
            ("ok", "return_use", "return_bool"),
        ],
    ),
    (
        "compared to bool literal",
        "int f(){bool ok = true; if(ok == true) return 1; return 0;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "conditional_use", "if_test"),
        ],
    ),
    (
        "negated-name assignment records both sides",
        "bool f(){bool a = true; bool b = !a; return b;}",
        [
            ("a", "assignment", "assign_bool_literal"),
            ("b", "assignment", "assign_not_name"),
            ("a", "conditional_use", "assign_not_name_inner"),
            ("b", "return_use", "return_bool"),
        ],
    ),
    (
        # Each chained test must be collected EXACTLY once. C++ nests an
        # `else if` inside a real `else_clause` node that the walker already
        # visits, so recursing into it would count depth-1 tests twice and
        # depth-2 tests three times.
        "else-if chain collects each test once",
        "int f(bool n){bool p = true; bool q = false;"
        " if(n) return 0; else if(p) return 1; else if(q) return 2; return 3;}",
        [
            ("p", "assignment", "assign_bool_literal"),
            ("q", "assignment", "assign_bool_literal"),
            ("n", "conditional_use", "if_test"),
            ("p", "conditional_use", "if_test"),
            ("q", "conditional_use", "if_test"),
        ],
    ),
    (
        # KNOWN C++ IMPRECISION, pinned so it cannot change silently.
        # C++ accepts any scalar in a condition, so `if(count)` looks
        # exactly like `if(flag)` to these heuristics. Java's type system
        # rules this out, so C++ recall for `boolean_flag` includes
        # non-boolean scalars that Java would never yield. See the
        # bool-declaration audit in the module docstring.
        "integer in a condition is (wrongly) taken for a flag",
        "int f(int count){if(count) return 1; return 0;}",
        [
            ("count", "conditional_use", "if_test"),
        ],
    ),
    (
        # pointer_declarator wraps function_declarator; the name must still
        # resolve or the whole function yields nothing.
        "pointer return type still finds the function",
        "bool* g(){bool ok = true; if(ok) return 0; return 0;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "conditional_use", "if_test"),
        ],
    ),
    (
        # A lambda body is a nested function: its flags belong to no
        # top-level function and must not leak into the enclosing one.
        "lambda body is skipped",
        "bool f(){bool outer = true; auto g = [](){bool inner = true; return inner;};"
        " return outer;}",
        [
            ("outer", "assignment", "assign_bool_literal"),
            ("outer", "return_use", "return_bool"),
        ],
    ),
    (
        # compound assignment is not a flag definition
        "compound assignment is not a definition",
        "bool f(){bool ok = true; ok &= false; return ok;}",
        [
            ("ok", "assignment", "assign_bool_literal"),
            ("ok", "return_use", "return_bool"),
        ],
    ),
]


def run() -> int:
    failures = 0
    for name, code, expected in CASES:
        rows, err = occurrence_rows_from_cpp_code(code)
        if err is not None:
            print(f"FAIL {name}\n       parse error: {err}")
            failures += 1
            continue
        got = [(r["variable"], r["occurrence_type"], r["detection_pattern"]) for r in rows]
        span_bad = [
            (r["variable"], code[r["source_span"][0] : r["source_span"][1]])
            for r in rows
            if code[r["source_span"][0] : r["source_span"][1]] != r["variable"]
        ]
        ok = sorted(got) == sorted(expected) and not span_bad
        if ok:
            print(f"OK   {name}")
        else:
            failures += 1
            print(f"FAIL {name}")
            if sorted(got) != sorted(expected):
                print(f"       expected {sorted(expected)}")
                print(f"       got      {sorted(got)}")
            if span_bad:
                print(f"       span does not slice to the variable: {span_bad}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
