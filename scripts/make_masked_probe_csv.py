"""Derive results/lp4fm/masked_probe/conditions.csv from the probe artifacts.

One row per (condition, role, ordered pair). The paper's Table 1 and the
figure test both read this file, so it is generated, never edited: the JSONs
beside it are the ground truth and this is their summary.

    uv run python scripts/make_masked_probe_csv.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import statistics as st

D = pathlib.Path("results/lp4fm/masked_probe")


def main() -> int:
    rows = []
    for f in sorted(D.glob("probe_*.json")):
        d = json.loads(f.read_text())
        m = re.match(r"probe_([a-z_]+)_([a-z]+)_to_([a-z]+)_(.+)\.json", f.name)
        rows.append({"condition": m.group(4), "role": m.group(1),
                     "source": m.group(2), "target": m.group(3),
                     "transfer": round(d["transfer_macro_f1_mean"], 4),
                     "indomain": round(d["indomain_macro_f1_mean"], 4),
                     "shuffled_source": round(d["shuffled_source_macro_f1_mean"], 4),
                     "model_id": d.get("model_id"),
                     "n_seeds": len({r["seed"] for r in d.get("test_predictions") or []}),
                     "n_shared_problems": d.get("n_shared_problems")})
    for f in sorted((D / "surface").glob("*.json")):
        d = json.loads(f.read_text())
        m = re.match(r"([a-z_]+)_([a-z]+)_to_([a-z]+)\.json", f.name)
        rows.append({"condition": "surface_window_masked", "role": m.group(1),
                     "source": m.group(2), "target": m.group(3),
                     "transfer": round(d["aggregate"]["window_masked"]["macro_f1"], 4),
                     "indomain": None,
                     "shuffled_source": round(d["shuffled_label_control_macro_f1"], 4),
                     "model_id": None, "n_seeds": None,
                     "n_shared_problems": d.get("n_test")})
    out = D / "conditions.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    conds = sorted({r["condition"] for r in rows})
    print(f"  wrote {out}: {len(rows)} cells, {len(conds)} conditions")
    for c in conds:
        g = [r for r in rows if r["condition"] == c]
        near = [r["transfer"] for r in g if "python" not in (r["source"], r["target"])]
        far = [r["transfer"] for r in g if "python" in (r["source"], r["target"])]
        print(f"    {c:<42} {st.mean(near):.3f} / {st.mean(far):.3f}"
              f"  effect {st.mean(far) - st.mean(near):+.3f}  n={len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
