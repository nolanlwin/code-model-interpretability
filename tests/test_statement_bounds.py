"""Statement-boundary cases, with EXACT expected fragments (no substring
matching — that hid a real failure twice)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from baselines import statement_bounds

Q = '"""'
B = "\\"

CASES = [
    # (name, code, target, exact expected stripped fragment)
    # `\\` + `\"\"\"`: the escape makes the first delimiter literal, so the
    # remaining two do NOT close the literal and its `;`/`{}` stay masked.
    ("escaped triple delimiter (reported)",
     'def f():\n    msg = ' + Q + 'say ' + B + Q + ' and; more {x}' + Q + '\n    ok = True\n',
     "ok = True", "ok = True"),
    # `\\` + four quotes: escape consumes one, the next three legitimately
    # close the literal — Python semantics, so this must NOT be masked.
    ("escaped quote then real close",
     'def f():\n    msg = ' + Q + 'say ' + B + '"' + Q + '\n    ok = True\n',
     "ok = True", "ok = True"),
    ("escaped backslash before close",
     'def f():\n    m = ' + Q + "path C:" + B + B + Q + '\n    ok = True\n',
     "ok = True", "ok = True"),
    ("unbalanced quote in literal",
     'def f():\n    m = ' + Q + 'say "hi; there' + Q + '\n    ok = True\n',
     "ok = True", "ok = True"),
    ("docstring containing ; { }",
     'def f():\n    ' + Q + 'Doc; with {braces}\n    more.' + Q + '\n    ok = True\n',
     "ok = True", "ok = True"),
    ("java text block",
     'class A{void f(){String s = ' + Q + '\n a; b {c}\n ' + Q + '; flag = true;}}',
     "flag = true", "flag = true;"),
    ("escaped quote in single-quoted",
     'x=1; s="a' + B + '";b"; flag=true;', "flag", "flag=true;"),
    ("for-header occurrence",
     'int f(){for(int i=0;i<n;i++){t+=a[i];}}', "i<n", "for(int i=0;i<n;i++){"),
    ("multiline call",
     'int f(){\n g(\n  a,\n  flag\n );\n}', "flag", "g(\n  a,\n  flag\n );"),
    ("plain python",
     'def f(x):\n    ok = True\n    return x\n', "ok = True", "ok = True"),
    ("body statement",
     'int f(){for(int i=0;i<n;i++){t+=a[i];}}', "t+=a[i]", "t+=a[i];"),
]

ok_all = True
for name, code, target, expect in CASES:
    i = code.index(target)
    ss, se = statement_bounds(code, i, i + len(target))
    got = code[ss:se].strip()
    ok = got == expect
    ok_all &= ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"       expected {expect!r}")
        print(f"       got      {got!r}")
print("\nALL PASS" if ok_all else "\nFAILURES")
sys.exit(0 if ok_all else 1)
