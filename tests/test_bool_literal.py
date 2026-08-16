"""Pins issue #16: only True/False are boolean literals, never 0 or 1.

Python defines ``0 == False`` and ``1 == True``, so ``value in (True, False)``
silently accepts the integers. That defect made every ``i = 0`` and
``res = 1`` a boolean-flag assignment -- 47% of the XLCoST Python boolean
corpus -- and it hit TWO sites: the assignment test and the comparison test
(``x == 0`` read as ``x == False``).

Exact expected tuples, not substring checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from boolean_flag_roles import _is_bool_constant  # noqa: E402
from xlcost_occurrences import extract_rows  # noqa: E402

# (value, is a boolean literal)
CONSTANTS = [
    (True, True), (False, True),
    (0, False), (1, False),            # the whole bug: 0 == False, 1 == True
    (0.0, False), (1.0, False),
    (2, False), (-1, False), ("", False), (None, False),
]

# (name, source, variables expected to be detected as boolean flags)
PROGRAMS = [
    ("True literal is a flag",      "def f():\n    ok = True\n    return ok\n", {"ok"}),
    ("False literal is a flag",     "def f():\n    ok = False\n    return ok\n", {"ok"}),
    ("int 0 is NOT a flag",         "def f():\n    i = 0\n    return i\n", set()),
    ("int 1 is NOT a flag",         "def f():\n    n = 1\n    return n\n", set()),
    ("float 0.0 is NOT a flag",     "def f():\n    x = 0.0\n    return x\n", set()),
    ("boolean operator still counts",
     "def f(a, b):\n    c = a and b\n    return c\n", {"a", "b", "c"}),
    ("not-name still counts",
     "def f(a):\n    c = not a\n    return c\n", {"a", "c"}),
    ("compare to True is a flag test",
     "def f(ok):\n    if ok == True:\n        return 1\n    return 0\n", {"ok"}),
    ("compare to 0 is NOT a flag test",
     "def f(n):\n    if n == 0:\n        return 1\n    return 0\n", set()),
    ("compare to 1 is NOT a flag test",
     "def f(n):\n    if n == 1:\n        return 1\n    return 0\n", set()),
]


def flags(src: str) -> set:
    rows, err = extract_rows("Python", src)
    assert err is None, f"parse error: {err}"
    # `return <name>` matches any identifier under these heuristics, so it is
    # excluded here: this file is about literal classification, nothing else.
    return {r["variable"] for r in rows if r["detection_pattern"] != "return_bool"}


def run() -> int:
    failures = 0
    for value, want in CONSTANTS:
        got = _is_bool_constant(value)
        ok = got == want
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} _is_bool_constant({value!r}) == {want}"
              f"{'' if ok else f'  got {got}'}")
    for name, src, want in PROGRAMS:
        got = flags(src)
        ok = got == want
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}"
              f"{'' if ok else f'  expected {want or set()}, got {got or set()}'}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
