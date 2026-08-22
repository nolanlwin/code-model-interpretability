"""Recompute probe transfer intervals so they describe the reported estimate.

crosslang.py reports TRANSFER as the mean macro-F1 over five seeds, and prints
beside it a bootstrap interval computed from seed 0's predictions alone. Those
are different quantities. Each seed selects its own layer, so seed variance is
large, and in a fair number of cells the reported point estimate falls outside
its own interval -- which is how the discrepancy was noticed.

This recomputes the interval for the statistic actually reported: resample whole
problems, and within each resample average the per-seed macro-F1 exactly as the
point estimate does. Every seed's predictions are already in the artifact, so
this runs from committed files and costs no GPU time.

    uv run python scripts/probe_intervals.py --in outputs/crosslang \
        --out results/lp4fm/probe_intervals.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np
from sklearn.metrics import f1_score

NAME = re.compile(r"^probe_([a-z_]+)_([a-z]+)_to_([a-z]+)(?:_([A-Za-z0-9]+))?\.json$")


def macro(y, p, labels):
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def cell_ci(preds, n_boot=2000, seed=0):
    """Percentile CI on the seed-averaged macro-F1, resampling problems."""
    seeds = sorted({r["seed"] for r in preds})
    labels = sorted({r["y_true"] for r in preds})
    by_seed = {}
    for s in seeds:
        rows = [r for r in preds if r["seed"] == s]
        rows.sort(key=lambda r: r["occurrence_id"] or "")
        by_seed[s] = (np.array([r["y_true"] for r in rows]),
                      np.array([r["y_pred"] for r in rows]),
                      np.array([r["cluster"] for r in rows]))
    point = float(np.mean([macro(*by_seed[s][:2], labels) for s in seeds]))

    problems = sorted({c for _, _, cl in by_seed.values() for c in cl.tolist()})
    index = {p: i for i, p in enumerate(problems)}
    pos = {s: np.array([index[c] for c in by_seed[s][2]]) for s in seeds}

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        counts = np.zeros(len(problems), dtype=np.int64)
        np.add.at(counts, rng.integers(0, len(problems), len(problems)), 1)
        per = []
        for s in seeds:
            w = counts[pos[s]]
            take = np.repeat(np.arange(len(w)), w)
            if take.size:
                y, p, _ = by_seed[s]
                per.append(macro(y[take], p[take], labels))
        if per:
            draws.append(float(np.mean(per)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_seeds": len(seeds), "n_problems": len(problems)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    rows, bad = [], 0
    for f in sorted(pathlib.Path(args.src).glob("probe_*.json")):
        m = NAME.match(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        preds = d.get("test_predictions") or []
        if not preds:
            continue
        role, a, b, slug = m.groups()
        ci = cell_ci(preds, args.n_boot)
        reported = d.get("transfer_macro_f1_mean")
        old = d.get("transfer_ci") or {}
        outside = (reported is not None and old
                   and not (old["ci_low"] <= reported <= old["ci_high"]))
        bad += bool(outside)
        rows.append({"role": role, "source": a, "target": b, "model_slug": slug,
                     "transfer": round(ci["point"], 4),
                     "ci_low": round(ci["ci_low"], 4),
                     "ci_high": round(ci["ci_high"], 4),
                     "n_seeds": ci["n_seeds"], "n_problems": ci["n_problems"],
                     "old_ci_excluded_point": bool(outside)})
        print(f"  {role:<12} {a[:4]}->{b[:4]:<5} {slug or '?':<28} "
              f"{ci['point']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
              f"{'   (old CI excluded the point)' if outside else ''}")

    if not rows:
        raise SystemExit(f"no probe artifacts with predictions in {args.src}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} cells; {bad} had a seed-0 interval that excluded the "
          f"reported mean\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
