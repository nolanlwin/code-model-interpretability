"""The exporter must not silently republish a smaller table.

export_crosslang rewrites summary.csv from whatever inputs are present, so a
session that recomputes only the `original` condition will drop every renaming
row. That happened: an 18-row table replaced a 36-row one, and the paper table
built on the renaming cells kept its numbers while its evidence left the
repository. These cases pin the refusal and the explicit override.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "scripts" / "export_crosslang.py"

ROLES = ("accumulator", "iterator", "index_key")
LANGS = ("python", "javascript", "php")


def _cell(n=400):
    """A minimal out_*.json with every field the exporter reads."""
    scores = {k: {"macro_f1": v} for k, v in (
        ("name_only", 0.80), ("statement_masked", 0.85),
        ("line_masked", 0.83), ("window_masked", 0.81))}
    return {
        "aggregate": scores, "n_train": n, "n_test": n,
        "pairing": f"matched on {n} shared problems",
        "majority_macro_f1": 0.35, "shuffled_label_control_macro_f1": 0.48,
        "test_predictions": [], "git_commit": "0" * 40,
    }


def _write_inputs(src: Path, conditions) -> int:
    n = 0
    for role in ROLES:
        for a in LANGS:
            for b in LANGS:
                if a == b:
                    continue
                for cond in conditions:
                    # C1/C2/C4 were produced with Python as source only --
                    # the renamer supports Python.
                    if cond != "original" and a != "python":
                        continue
                    tag = "" if cond == "original" else f"_{cond}"
                    (src / f"out_{role}{tag}_{a}_to_{b}.json").write_text(
                        json.dumps(_cell()))
                    n += 1
    return n


def _run(src, dst, *extra):
    return subprocess.run(
        [sys.executable, str(EXPORT), "--in", str(src), "--out", str(dst), *extra],
        capture_output=True, text=True, cwd=ROOT)


def run() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, dst = td / "in", td / "out"
        src.mkdir(); dst.mkdir()

        # 1. full table: original + the three renaming conditions
        full = _write_inputs(src, ("original", "C1", "C2", "C4"))
        r = _run(src, dst)
        rows = len((dst / "summary.csv").read_text().strip().splitlines()) - 1
        ok = r.returncode == 0 and rows == full
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} publishes the full {full}-row table "
              f"(got {rows}, rc={r.returncode})")

        # 2. rerun with ONLY the original condition -> must refuse
        for f in src.glob("out_*_C*.json"):
            f.unlink()
        r = _run(src, dst)
        after = len((dst / "summary.csv").read_text().strip().splitlines()) - 1
        ok = (r.returncode == 1 and "REFUSING" in r.stdout and after == full)
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} refuses to drop renaming rows "
              f"(rc={r.returncode}, table still {after} rows)")
        if not ok:
            print(f"        stdout: {r.stdout[:300]}")
        named = all(c in r.stdout for c in ("C1", "C2", "C4"))
        failures += not named
        print(f"  {'OK  ' if named else 'FAIL'} names the conditions it would drop")

        # 3. the table on disk is untouched by a refused run
        ok = after == full
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} refused run leaves summary.csv intact")

        # 4. --allow-drop proceeds, and says so
        r = _run(src, dst, "--allow-drop")
        after = len((dst / "summary.csv").read_text().strip().splitlines()) - 1
        ok = r.returncode == 0 and after == 18 and "dropping" in r.stdout
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} --allow-drop republishes {after} rows "
              f"and reports the loss (rc={r.returncode})")

        # 5. a run that drops nothing is not obstructed
        r = _run(src, dst)
        ok = r.returncode == 0 and "REFUSING" not in r.stdout
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} an identical rerun is not obstructed")

    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0





def run_model_case() -> int:
    """A second model must not silently replace the first model's numbers.

    Store directories and probe filenames carry the model slug; summary.csv
    does not. Without the model in the drop key, exporting model B over model
    A's directory yields the same (role, condition, source, target) cells with
    different numbers and overwrites A in place.
    """
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, dst = td / "in", td / "out"
        src.mkdir(); dst.mkdir()
        _write_inputs(src, ("original",))

        def probe(model):
            for role in ROLES:
                for a in LANGS:
                    for b in LANGS:
                        if a == b:
                            continue
                        slug = model.split("/")[-1].lower().replace(".", "").replace("-", "")
                        (src / f"probe_{role}_{a}_to_{b}_{slug}.json").write_text(json.dumps({
                            "transfer_macro_f1_mean": 0.9, "indomain_macro_f1_mean": 0.93,
                            "shuffled_source_macro_f1_mean": 0.49,
                            "resolution_rho": 0.001, "model_id": model}))

        probe("Qwen/Qwen2.5-Coder-1.5B")
        r = _run(src, dst)
        ok = r.returncode == 0
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} publishes model A (rc={r.returncode})")

        for f in src.glob("probe_*.json"):
            f.unlink()
        probe("bigcode/starcoder2-7b")
        r = _run(src, dst)
        ok = r.returncode == 1 and "REFUSING" in r.stdout
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} refuses to overwrite model A with model B "
              f"(rc={r.returncode})")
        told = "own directory" in r.stdout and "Qwen2.5-Coder-1.5B" in r.stdout
        failures += not told
        print(f"  {'OK  ' if told else 'FAIL'} names the model at risk and the fix")

        r = _run(src, dst / "b")
        ok = r.returncode == 0
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} model B exports cleanly to its own directory")
    return failures


if __name__ == "__main__":
    rc = run()
    print()
    extra = run_model_case()
    print("\nALL PASS" if not extra else f"\n{extra} FAILURE(S)")
    sys.exit(1 if (rc or extra) else 0)
