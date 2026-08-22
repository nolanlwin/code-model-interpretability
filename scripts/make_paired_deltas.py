"""Paired surface-vs-probe deltas from the committed masked-probe artifacts.

The paper claims the surface baseline beats the context-pooled probe on the
same occurrences, not merely on average. That is a paired statistic, and it
lives here as a committed CSV so the figure test can assert it: for each cell,
macroF1(probe) - macroF1(surface) over the occurrences both scored, with a
problem-clustered interval from one resample applied to both sides.

    uv run python scripts/make_paired_deltas.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import paired_delta_ci  # noqa: E402

D = pathlib.Path("results/lp4fm/masked_probe")
CTX = "qwen25coder15bpoolcontext16"


def main() -> int:
    rows = []
    for pj in sorted(D.glob(f"probe_*_{CTX}.json")):
        m = re.match(rf"probe_([a-z_]+)_([a-z]+)_to_([a-z]+)_{CTX}\.json", pj.name)
        role, a, b = m.groups()
        bj = D / "surface" / f"{role}_{a}_to_{b}.json"
        P = json.loads(pj.read_text())
        B = json.loads(bj.read_text())
        key = lambda rs: {r["occurrence_id"]: r for r in rs
                          if r.get("seed") in (0, "0") and r.get("occurrence_id")}
        kp, kb = key(P["test_predictions"]), key(B["test_predictions"])
        both = sorted(set(kp) & set(kb))
        y = np.array([kp[o]["y_true"] for o in both])
        pa = np.array([kp[o]["y_pred"] for o in both])
        pb = np.array([kb[o]["y_pred"] for o in both])
        cl = np.array([kb[o]["cluster"] for o in both])
        r = paired_delta_ci(y, pa, pb, cl, sorted(set(y.tolist())), n_boot=2000, seed=0)
        rows.append({"role": role, "source": a, "target": b, "n_occurrences": len(both),
                     "delta": round(r["delta"], 4), "ci_low": round(r["ci_low"], 4),
                     "ci_high": round(r["ci_high"], 4),
                     "excludes_zero": r["excludes_zero"]})
        print(f"  {role:<12} {a[:4]}->{b[:4]:<6} {r['delta']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              f"{'  *' if r['excludes_zero'] else ''}")
    out = D / "paired_deltas.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = sum(r["excludes_zero"] for r in rows)
    print(f"\n  {n}/{len(rows)} cells exclude zero; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
