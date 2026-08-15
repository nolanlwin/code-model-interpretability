"""Export probing results into the tracked tree: results/boolean/probe/.

make_figures.py already writes PNGs into results/boolean/, but nothing ever
commits them and outputs/probe_results/ is gitignored, so the numbers behind
the figures live only in Drive. This writes the durable part:

  results/boolean/probe/summary.csv    one row per language/split/model
  results/boolean/probe/SUMMARY.md     probe vs baseline, with RESOLUTION
  results/boolean/probe/<run>.json     metadata + aggregate, no predictions

Every row carries rho, the macro-F1 movement caused by ONE test occurrence
of the smallest class changing its prediction. That number decided how §4.4
had to be written -- most probe-minus-baseline margins turned out to be
smaller than a single test instance -- so it is computed here from each run's
own test fold rather than left for someone to rediscover.

    python scripts/export_probe.py --in outputs/probe_results --out results/boolean/probe
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MODEL_LABELS = {
    "qwen2515b": "Qwen2.5-1.5B",
    "qwen25coder15b": "Qwen2.5-Coder-1.5B",
    "starcoder27b": "StarCoder2-7B",
}
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]


def resolution(preds: list[dict]) -> tuple[float | None, str | None, int | None]:
    """(rho, smallest class, its test count) from a run's own predictions.

    Measured, not approximated: start from a perfect prediction on the real
    test fold, flip one occurrence of the smallest class to the majority
    class, and take the macro-F1 drop.
    """
    if not preds:
        return None, None, None
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        return None, None, None
    seed0 = [p for p in preds if int(p.get("seed", 0)) == 0] or preds
    y = [p["y_true"] for p in seed0]
    counts = Counter(y)
    if len(counts) < 2:
        return None, None, None
    small = min(counts, key=counts.get)
    big = max(counts, key=counts.get)
    arr = np.array(y)
    flipped = arr.copy()
    flipped[np.where(arr == small)[0][0]] = big
    labels = sorted(counts)
    rho = (f1_score(arr, arr, average="macro", labels=labels, zero_division=0)
           - f1_score(arr, flipped, average="macro", labels=labels, zero_division=0))
    return float(rho), small, int(counts[small])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="outputs/probe_results")
    ap.add_argument("--out", dest="dst", default="results/boolean/probe")
    args = ap.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    probes = sorted(src.glob("*_problem.json"))
    probes = [p for p in probes if not re.search(r"_C\d_", p.name)]
    if not probes:
        print(f"no probe results in {src}")
        return 1

    rows = []
    for p in probes:
        m = re.match(r"(.+?)_(train|valid|test)_(\w+)_problem\.json$", p.name)
        if not m:
            continue
        lang, split, model = m.groups()
        d = json.loads(p.read_text())
        rho, small, small_n = resolution(d.get("test_predictions") or [])

        base_path = src / f"{lang}_{split}_{model}_baselines_capped.json"
        best = None
        baselines = {}
        if base_path.exists():
            b = json.loads(base_path.read_text())
            best = b.get("strongest_baseline_macro_f1")
            baselines = {k: v.get("macro_f1") for k, v in (b.get("aggregate") or {}).items()}

        f1 = (d.get("aggregate") or {}).get("test_macro_f1_mean")
        row = {
            "language": lang, "split": split, "model": model,
            "macro_f1": None if f1 is None else round(f1, 4),
            "selectivity": (None if d.get("selectivity_macro_f1") is None
                            else round(d["selectivity_macro_f1"], 4)),
            "best_baseline": None if best is None else round(best, 4),
            "probe_minus_baseline": (None if (f1 is None or best is None)
                                     else round(f1 - best, 4)),
            "rho": None if rho is None else round(rho, 4),
            "in_rho_units": (None if (f1 is None or best is None or not rho)
                             else round((f1 - best) / rho, 2)),
            "smallest_test_class": small, "smallest_test_n": small_n,
            "n_records": d.get("n_records"), "n_problems": d.get("n_repos"),
            "classes_used": ",".join(d.get("classes_used") or []),
            "classes_dropped": json.dumps(d.get("classes_dropped") or {}),
            "git_commit": (d.get("git_commit") or "")[:12],
        }
        for k, v in baselines.items():
            row[f"baseline_{k}"] = None if v is None else round(v, 4)
        for c in CONDITIONS:
            dp = src / f"{lang}_{split}_{c}_{model}_delta_vs_C0.json"
            if dp.exists():
                dd = json.loads(dp.read_text())
                row[f"d{c}"] = round(dd.get("delta", float("nan")), 4)
                row[f"d{c}_ci"] = f"[{dd.get('ci_low', 0):.4f}, {dd.get('ci_high', 0):.4f}]"
                row[f"d{c}_excludes_zero"] = dd.get("excludes_zero")
        rows.append(row)

        slim = {k: v for k, v in d.items() if k != "test_predictions"}
        slim["resolution_rho"] = rho
        slim["smallest_test_class"] = {"label": small, "n": small_n}
        (dst / p.name).write_text(json.dumps(slim, indent=1))

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("baseline_"), k))
    with (dst / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = ["# Probing — boolean", "",
             "Generated by `scripts/export_probe.py`.", "",
             "**rho** is the macro-F1 movement caused by ONE test occurrence of the",
             "smallest class changing its prediction, measured on each run's own test",
             "fold. A probe-minus-baseline margin smaller than rho is below what the",
             "sample can resolve and must not be read as a difference.", "",
             "| language | model | macro F1 | select. | best baseline | probe − base | rho | in rho units |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["language"], r["model"])):
        u = r["in_rho_units"]
        flag = "" if u is None else ("  ⚠ below resolution" if abs(u) < 1 else "")
        lines.append(
            f"| {r['language']} | {MODEL_LABELS.get(r['model'], r['model'])} | "
            f"{r['macro_f1']} | {r['selectivity']} | {r['best_baseline']} | "
            f"{r['probe_minus_baseline']} | {r['rho']} | "
            f"{'—' if u is None else f'{u:+.2f}'}{flag} |")
    (dst / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} probe summaries, summary.csv, SUMMARY.md -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
