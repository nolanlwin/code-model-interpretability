"""Cross-lingual probe transfer, against synthetic stores with known answers.

The apparatus cannot be exercised on real activations without a GPU, so these
build small stores whose separability is planted by construction. Each case
has an outcome that follows from the construction rather than from a
threshold someone picked:

  shared direction  -> transfer must clearly beat its shuffled control
  no direction      -> transfer must collapse to the shuffled control
  wrong label_field -> refused (two stores holding different class schemes)
  different model   -> refused (transfer across models is not what this means)

Written after two bugs that reading the code did not catch: the store record
key is "X" not "x", and load_records_from_store returns run stats carrying
model_id but NOT label_field, which lives in the store's meta.json.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAYERS, HIDDEN, N_PROBLEMS = 4, 16, 400


def build_store(root: Path, name: str, signal: float,
                label_field: str = "role", model_id: str = "fake-1b") -> Path:
    """A store whose accumulator/other split is linearly separable iff signal>0."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, X = [], []
    for p in range(N_PROBLEMS):
        for k in range(3):
            role = ("accumulator" if (p + k) % 3 == 0
                    else ("index_key" if k else "iterator"))
            v = rng.normal(size=(LAYERS, HIDDEN)).astype(np.float32)
            v[2, 0] += signal * (1.0 if role == "accumulator" else -1.0)
            X.append(v)
            rows.append({"occurrence_id": f"{p}:{k}", "occurrence_type": role,
                         "problem_id": f"prob{p:04d}", "function": "f",
                         "variable": f"v{k}", "row": len(X) - 1})
    np.save(d / "shard.npy", np.stack(X))
    (d / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / "meta.json").write_text(json.dumps(
        {"model_id": model_id, "label_field": label_field, "dtype": "float16",
         "shape": [len(rows), LAYERS, HIDDEN]}))
    return d


def run(train: Path, test: Path, out: Path) -> tuple[int, dict, str]:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crosslang.py"), "run",
         "--train-store", str(train), "--test-store", str(test),
         "--role", "accumulator", "--seeds", "0", "1", "--output", str(out)],
        capture_output=True, text=True, cwd=REPO)
    data = json.loads(out.read_text()) if out.exists() else {}
    return r.returncode, data, (r.stderr or "") + (r.stdout or "")


def main() -> int:
    failures = 0

    def check(name, cond, extra=""):
        nonlocal failures
        failures += not cond
        print(f"{'OK  ' if cond else 'FAIL'} {name}{'' if cond else '  ' + extra}")

    tmp = Path(tempfile.mkdtemp())
    try:
        a = build_store(tmp, "lang_a", 2.5)
        b = build_store(tmp, "lang_b", 2.5)
        c = build_store(tmp, "lang_c", 0.0)
        wrong_label = build_store(tmp, "lang_wrong", 2.5, label_field="occurrence_type")
        other_model = build_store(tmp, "lang_other", 2.5, model_id="fake-7b")

        rc, d, _ = run(a, b, tmp / "ab.json")
        transfer = d.get("transfer_macro_f1_mean", 0.0)
        shuffled = d.get("shuffled_source_macro_f1_mean", 1.0)
        check("shared direction: run succeeds", rc == 0)
        check("shared direction: transfer clears its shuffled control",
              transfer - shuffled > 0.25, f"transfer={transfer:.3f} shuffled={shuffled:.3f}")
        # The key is named for what it measures. It brackets SEED 0's score,
        # not transfer_macro_f1_mean, which averages over seeds that each pick
        # their own layer; the two diverge and the mean often lands outside it.
        # scripts/probe_intervals.py computes the interval for the reported
        # estimate from test_predictions, which carries every seed.
        check("shared direction: the seed-0 interval is reported and named as such",
              "transfer_ci_seed0_only" in d
              and d["transfer_ci_seed0_only"].get("ci_low") is not None)
        check("shared direction: every seed's predictions are kept",
              len({r["seed"] for r in (d.get("test_predictions") or [])}) > 1)
        check("shared direction: rho is reported", d.get("resolution_rho") is not None)

        rc, d2, _ = run(a, c, tmp / "ac.json")
        t2 = d2.get("transfer_macro_f1_mean", 1.0)
        s2 = d2.get("shuffled_source_macro_f1_mean", 0.0)
        check("no direction in target: transfer collapses to its control",
              abs(t2 - s2) < 0.15, f"transfer={t2:.3f} shuffled={s2:.3f}")
        check("no direction in target: and is far below the shared case",
              transfer - t2 > 0.3, f"{transfer:.3f} vs {t2:.3f}")

        rc, _, log = run(a, wrong_label, tmp / "aw.json")
        check("mismatched label_field is refused",
              rc != 0 and "label-field" in log, log.strip()[-90:])

        rc, _, log = run(a, other_model, tmp / "ao.json")
        check("different model is refused",
              rc != 0 and "different models" in log, log.strip()[-90:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
