"""Every script must parse on the oldest Python the project supports.

pyproject declares requires-python >= 3.11. A backslash inside an f-string
expression is a SyntaxError before 3.12 (PEP 701 relaxed it), so
scripts/make_appendix.py parsed fine on the 3.14 interpreter it was written
with and could not be run at all by anyone on 3.11. ast.parse(feature_version=)
does not catch it either -- that argument controls AST features, not the
tokenizer -- so the check has to be a real interpreter.

Skips rather than fails when no 3.11 interpreter is reachable, so the suite
stays green on machines that have only a newer Python.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def min_python() -> str:
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())
    req = spec.get("project", {}).get("requires-python", ">=3.11")
    m = re.search(r"(\d+)\.(\d+)", req)
    return f"{m.group(1)}.{m.group(2)}" if m else "3.11"


def run() -> int:
    ver = min_python()
    probe = subprocess.run(["uv", "run", "--python", ver, "python", "-c", "pass"],
                           capture_output=True, cwd=ROOT)
    if probe.returncode != 0:
        print(f"  SKIP no Python {ver} available; cannot verify the floor")
        return 0

    files = sorted(list((ROOT / "scripts").glob("*.py"))
                   + list((ROOT / "tests").glob("*.py")))
    script = (
        "import ast,sys\n"
        "bad=[]\n"
        "for p in sys.argv[1:]:\n"
        "    try: ast.parse(open(p).read())\n"
        "    except SyntaxError as e: bad.append((p,e.lineno,e.msg))\n"
        "print('\\n'.join(f'{p}:{l}: {m}' for p,l,m in bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run(["uv", "run", "--python", ver, "python", "-c", script,
                        *[str(f) for f in files]],
                       capture_output=True, text=True, cwd=ROOT)
    ok = r.returncode == 0
    print(f"  {'OK  ' if ok else 'FAIL'} {len(files)} files parse on Python {ver}")
    if not ok:
        for line in (r.stdout or r.stderr).strip().splitlines():
            print(f"       {line}")
    print("\nALL PASS" if ok else "\n1 FAILURE(S)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
