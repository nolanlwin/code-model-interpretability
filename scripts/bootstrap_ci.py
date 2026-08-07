"""Cluster bootstrap confidence intervals (percentile and BCa).

Implements what reviewer QBg8 asked for ("confidence intervals or bootstrap
estimates") the way the protocol freezes it: the resampling unit is the CLUSTER
(problem / repo / function), never the occurrence, because occurrences within a
program share a forward pass and are not independent.

Two entry points:

- ``cluster_bootstrap_ci``: CI on a statistic of one prediction set.
- ``paired_delta_ci``: CI on stat(A) - stat(B) where A and B are the SAME
  occurrences under two conditions (e.g. baseline vs renamed). The same cluster
  resample is applied to both sides, which is what makes the interval tight.

CLI: compare two results.json files produced by scripts/probe.py:

    uv run python scripts/bootstrap_ci.py delta a.json b.json --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import norm  # scipy ships with scikit-learn's dependency tree
from sklearn.metrics import f1_score


def macro_f1_stat(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence) -> float:
    """Macro F1 over a FIXED label set so resamples missing a class stay comparable."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def _cluster_indices(clusters: np.ndarray) -> dict:
    out: dict = {}
    for i, c in enumerate(clusters):
        out.setdefault(c, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _resample_stats(
    stat_fn: Callable[[np.ndarray], float],
    cluster_map: dict,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    keys = list(cluster_map)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([cluster_map[keys[p]] for p in picked])
        stats[b] = stat_fn(idx)
    return stats


def _bca_interval(
    boot: np.ndarray, observed: float, jackknife: np.ndarray, alpha: float
) -> tuple[float, float]:
    """BCa bounds from bootstrap replicates + leave-one-cluster-out jackknife."""
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan"), float("nan")
    # Bias correction: proportion of replicates below the observed value.
    prop = np.clip(np.mean(boot < observed), 1e-9, 1 - 1e-9)
    z0 = norm.ppf(prop)
    # Acceleration from the jackknife skew.
    jm = jackknife.mean()
    num = np.sum((jm - jackknife) ** 3)
    den = 6.0 * (np.sum((jm - jackknife) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def _adj(z: float) -> float:
        return float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo = float(np.quantile(boot, np.clip(_adj(z_lo), 0, 1)))
    hi = float(np.quantile(boot, np.clip(_adj(z_hi), 0, 1)))
    return lo, hi


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    labels: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    method: str = "bca",
) -> dict:
    """CI on macro F1, resampling whole clusters with replacement."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clusters = np.asarray(clusters)
    cmap = _cluster_indices(clusters)
    observed = macro_f1_stat(y_true, y_pred, labels)

    def stat(idx: np.ndarray) -> float:
        return macro_f1_stat(y_true[idx], y_pred[idx], labels)

    boot = _resample_stats(stat, cmap, n_boot, seed)
    keys = list(cmap)
    jack = np.array(
        [
            stat(np.concatenate([cmap[k] for k in keys if k != drop]))
            for drop in keys
        ]
    )
    if method == "bca":
        lo, hi = _bca_interval(boot, observed, jack, alpha)
    else:
        lo, hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
    return {
        "point": observed,
        "ci_low": lo,
        "ci_high": hi,
        "method": method,
        "n_boot": n_boot,
        "n_clusters": len(cmap),
        "max_cluster_share": float(max(len(v) for v in cmap.values()) / len(y_true)),
        "cluster_warning": len(cmap) < 30,
    }


def paired_delta_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    clusters: np.ndarray,
    labels: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    method: str = "bca",
) -> dict:
    """CI on macroF1(A) - macroF1(B) over the SAME occurrences, same resamples."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    clusters = np.asarray(clusters)
    cmap = _cluster_indices(clusters)

    def stat(idx: np.ndarray) -> float:
        return macro_f1_stat(y_true[idx], pred_a[idx], labels) - macro_f1_stat(
            y_true[idx], pred_b[idx], labels
        )

    observed = stat(np.arange(len(y_true)))
    boot = _resample_stats(stat, cmap, n_boot, seed)
    keys = list(cmap)
    jack = np.array(
        [stat(np.concatenate([cmap[k] for k in keys if k != drop])) for drop in keys]
    )
    if method == "bca":
        lo, hi = _bca_interval(boot, observed, jack, alpha)
    else:
        lo, hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
    return {
        "delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "method": method,
        "n_boot": n_boot,
        "n_clusters": len(cmap),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def _load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    preds = res.get("test_predictions")
    if not preds:
        raise SystemExit(f"{path}: no test_predictions block (rerun probe.py with predictions on)")
    return res


def cmd_delta(args: argparse.Namespace) -> int:
    ra, rb = _load_predictions(args.results_a), _load_predictions(args.results_b)
    pa = {p["occurrence_id"]: p for p in ra["test_predictions"]}
    pb = {p["occurrence_id"]: p for p in rb["test_predictions"]}
    shared = sorted(set(pa) & set(pb))
    if not shared:
        raise SystemExit("no shared occurrence_ids between the two results files")
    dropped = len(set(pa) ^ set(pb))
    y = np.array([pa[k]["y_true"] for k in shared])
    yb_check = np.array([pb[k]["y_true"] for k in shared])
    if not np.array_equal(y, yb_check):
        raise SystemExit("y_true disagrees between files for shared occurrences — not the same data")
    a = np.array([pa[k]["y_pred"] for k in shared])
    b = np.array([pb[k]["y_pred"] for k in shared])
    cl = np.array([pa[k]["cluster"] for k in shared])
    labels = sorted(set(y.tolist()))
    out = paired_delta_ci(y, a, b, cl, labels, n_boot=args.n_boot, seed=args.seed)
    out["n_shared"] = len(shared)
    out["n_unpaired_dropped"] = dropped
    print(json.dumps(out, indent=2))
    return 0


def verify() -> int:
    """Synthetic check: known separation should give a positive CI excluding zero."""
    rng = np.random.default_rng(0)
    n_clusters, per = 40, 25
    clusters = np.repeat(np.arange(n_clusters), per)
    y = rng.integers(0, 2, size=n_clusters * per)
    good = np.where(rng.random(y.size) < 0.9, y, 1 - y)  # 90% correct
    bad = np.where(rng.random(y.size) < 0.6, y, 1 - y)  # 60% correct
    ci = cluster_bootstrap_ci(y, good, clusters, [0, 1], n_boot=500, seed=0)
    d = paired_delta_ci(y, good, bad, clusters, [0, 1], n_boot=500, seed=0)
    ok = ci["ci_low"] < ci["point"] < ci["ci_high"] and d["delta"] > 0 and d["excludes_zero"]
    print(json.dumps({"ci": ci, "delta": d}, indent=2))
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("delta", help="paired clustered CI on macroF1(A) - macroF1(B)")
    d.add_argument("results_a")
    d.add_argument("results_b")
    d.add_argument("--n-boot", type=int, default=2000)
    d.add_argument("--seed", type=int, default=0)
    sub.add_parser("verify", help="synthetic self-check")
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        return verify()
    return cmd_delta(args)


if __name__ == "__main__":
    sys.exit(main())
