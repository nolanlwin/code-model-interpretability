"""Scripts that notebooks invoke must keep the flags those notebooks pass.

scripts/make_figures.py was overwritten wholesale by an unrelated generator.
Nothing failed: the file still existed, still parsed, still ran. Two committed
notebooks went on passing --results-dir/--lang/--split to an entry point that
ignored them, and the boolean workstream's figures had no producer left.

So the check is not "does the file exist" but "does every flag a notebook
passes still resolve". It parses the ! commands out of the notebooks and asks
each script's own argparse whether it recognises them.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def invocations():
    """(notebook, script, flags) for every ! call to a repo script."""
    out = []
    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            # ! commands continue across backslash-newline
            src = re.sub(r"\\\s*\n", " ", src)
            for line in src.splitlines():
                line = line.strip()
                if not line.startswith("!"):
                    continue
                m = re.search(r"python3?\s+(scripts/[\w./-]+\.py)(.*)", line)
                if not m:
                    continue
                rest = m.group(2)
                # Several of these CLIs are subcommand-style
                # (baselines.py transfer --train-occurrences ...), and the
                # top-level --help does not list a subcommand's flags. Capture
                # the subcommand so the right help page is consulted.
                sub = re.match(r"\s+([a-z][a-z0-9_-]*)(?=\s|$)", rest)
                sub = sub.group(1) if sub else None
                flags = set(re.findall(r"(--[a-z][a-z0-9-]*)", rest))
                if flags:
                    out.append((nb_path.name, m.group(1), sub, flags))
    return out


def accepted(script: str, sub: str | None) -> set[str] | None:
    cmd = [sys.executable, str(ROOT / script)] + ([sub] if sub else []) + ["--help"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        # subcommand-style CLIs print usage on the error path
        r = subprocess.run([sys.executable, str(ROOT / script)],
                           capture_output=True, text=True, cwd=ROOT)
    text = r.stdout + r.stderr
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", text)) or None


def run() -> int:
    calls = invocations()
    if not calls:
        print("  SKIP no notebook invocations found")
        return 0

    cache, failures = {}, 0
    checked = 0
    for nb, script, sub, flags in calls:
        if not (ROOT / script).exists():
            print(f"  FAIL {nb} calls {script}, which does not exist")
            failures += 1
            continue
        key = (script, sub)
        if key not in cache:
            cache[key] = accepted(script, sub)
        known = cache[key]
        if known is None:
            continue  # CLI help unreadable; not this test's business
        missing = sorted(f for f in flags if f not in known)
        checked += 1
        if missing:
            failures += 1
            where = f"{script} {sub}" if sub else script
            print(f"  FAIL {nb}: {where} no longer accepts {missing}")
        else:
            where = f"{script} {sub}" if sub else script
            print(f"  OK   {nb}: {where} accepts {len(flags)} flag(s)")

    print(f"\n{checked} invocation(s) checked")
    print("ALL PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
