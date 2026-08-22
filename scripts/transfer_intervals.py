"""Problem-level confidence intervals for transfer scores and the boundary effect.

Reviewer point 4. Occurrences inside one program share a forward pass and are
not independent, so the resampling unit is the PROBLEM, never the occurrence.
scripts/bootstrap_ci.py already implements that for a single prediction set;
this adds the interval the paper's headline actually needs, on the DIFFERENCE
between the two groups of cells.

That difference is only bootstrappable because of the intersection restriction
(scripts/build_intersection.py): all six ordered pairs now score the same 869
problems, so one resample of problem ids can be applied to every cell at once
and the group means recomputed under it. Without that, each cell has its own
problem set and there is no common unit to resample.

    uv run python scripts/transfer_intervals.py --in outputs/isect \
        --feature window_masked --out results/lp4fm/transfer_intervals.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import cluster_bootstrap_ci  # noqa: E402

NAME = re.compile(r"^([a-z_]+)_([a-z]+)_to_([a-z]+)\.json$")
LABELS = ["other", "target"]


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def load(path: pathlib.Path, seed: int = 0):
    d = json.loads(path.read_text())
    preds = [p for p in (d.get("test_predictions") or []) if p.get("seed") == seed]
    if not preds:
        return None
    return {
        "y_true": np.array([p["y_true"] for p in preds]),
        "y_pred": np.array([p["y_pred"] for p in preds]),
        "cluster": np.array([p["cluster"] for p in preds]),
        # cmd_transfer records it here; cmd_run uses the other key.
        "feature": d.get("test_predictions_feature")
                   or d.get("test_predictions_baseline"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cells = {}
    for f in sorted(pathlib.Path(args.src).glob("*.json")):
        m = NAME.match(f.name)
        if not m:
            continue
        got = load(f, args.seed)
        if got:
            cells[m.groups()] = got
    if not cells:
        raise SystemExit(f"no cells with test_predictions in {args.src}")
    print(f"  {len(cells)} cells loaded")

    rows = []
    for (role, a, b), c in sorted(cells.items()):
        ci = cluster_bootstrap_ci(c["y_true"], c["y_pred"], c["cluster"], LABELS,
                                  n_boot=args.n_boot, seed=args.seed)
        rows.append({"role": role, "source": a, "target": b,
                     "feature": c["feature"],
                     "macro_f1": round(ci["point"], 4),
                     "ci_low": round(ci["ci_low"], 4), "ci_high": round(ci["ci_high"], 4),
                     "n_problems": int(len(set(c["cluster"].tolist())))})
        print(f"    {role:<12} {a[:4]}->{b[:4]:<11} {ci['point']:.3f} "
              f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    # The boundary effect, resampling problems once for every cell together.
    problems = sorted(set().union(*[set(c["cluster"].tolist()) for c in cells.values()]))
    idx = {p: i for i, p in enumerate(problems)}
    per_cell = {k: {"t": c["y_true"], "p": c["y_pred"],
                    "c": np.array([idx[x] for x in c["cluster"]])}
                for k, c in cells.items()}
    near = [k for k in cells if "python" not in (k[1], k[2])]
    far = [k for k in cells if "python" in (k[1], k[2])]

    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.n_boot):
        take = rng.integers(0, len(problems), len(problems))
        keep = np.zeros(len(problems), dtype=np.int64)
        np.add.at(keep, take, 1)
        def group(keys):
            out = []
            for k in keys:
                d = per_cell[k]
                w = keep[d["c"]]
                sel = np.repeat(np.arange(len(w)), w)
                if sel.size:
                    out.append(macro_f1(d["t"][sel], d["p"][sel]))
            return float(np.mean(out)) if out else np.nan
        draws.append(group(far) - group(near))
    draws = np.array([d for d in draws if np.isfinite(d)])
    point = (float(np.mean([macro_f1(per_cell[k]["t"], per_cell[k]["p"]) for k in far]))
             - float(np.mean([macro_f1(per_cell[k]["t"], per_cell[k]["p"]) for k in near])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rows.append({"role": "ALL", "source": "close-pair", "target": "python-pairs",
                 "feature": "boundary effect", "macro_f1": round(point, 4),
                 "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                 "n_problems": len(problems)})
    print(f"\n  boundary effect {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  ({len(problems)} problems, {len(draws)} draws)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
