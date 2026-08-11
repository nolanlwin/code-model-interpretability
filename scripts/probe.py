"""Layer-wise linear probing CLI — replaces the notebook probe cell.

Protocol-compliant probing over cached activations (PROTOCOL.md v1.0):

- GROUPED splits (``--split-policy function|repo|random``); the old stratified
  occurrence-level split leaked (85.5% of test occurrences shared a function
  with train).
- ``StandardScaler`` fitted on train only, per layer.
- Layer SELECTED ON VALIDATION, reported on test. The full per-layer curves are
  stored for figures, but the headline number is the validation-selected layer.
- 5 seeds; per-seed results plus aggregate.
- Classes below ``--min-class-count`` HARD-FAIL unless ``--allow-class-drop``
  is passed, and any drop is recorded in results.json (never silent).
- Duplicate occurrences (identical token span) with CONFLICTING labels are
  removed entirely and counted.
- Hewitt control task (``--control-task``): each variable NAME gets a random
  label drawn from the empirical label marginal; selectivity = real - control.
- Cluster BCa bootstrap CI on the test macro F1 (scripts/bootstrap_ci.py), plus
  per-item test predictions embedded for downstream paired deltas.

Example (existing Java cache):

    uv run python scripts/probe.py run \
      --manifest outputs/activations_java/manifest.jsonl \
      --npz-dir outputs/npz_cache_java \
      --split-policy repo --allow-class-drop \
      --output outputs/probe_results/java_repo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from bootstrap_ci import cluster_bootstrap_ci, macro_f1_stat  # noqa: E402

PROTOCOL_VERSION = "1.0"


def load_records(manifest: Path, npz_dir: Path, pooling: str) -> tuple[list[dict], dict]:
    """Load manifest rows joined to npz tensors, deduped and conflict-filtered."""
    rows = []
    for ln in manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if not r.get("activation_path") or not r.get("occurrence_type"):
            continue
        rows.append(r)

    # Drop duplicate spans; drop ALL copies when labels conflict (audit found
    # occ pairs with identical token_positions and contradictory labels).
    by_span: dict = {}
    for r in rows:
        key = (
            r.get("repo"),
            r.get("path"),
            r.get("source_row"),
            r.get("function"),
            r.get("variable"),
            tuple(r.get("token_positions") or []),
        )
        by_span.setdefault(key, []).append(r)
    kept, dup_dropped, conflict_dropped = [], 0, 0
    for group in by_span.values():
        labels = {g["occurrence_type"] for g in group}
        if len(labels) > 1:
            conflict_dropped += len(group)
            continue
        kept.append(group[0])
        dup_dropped += len(group) - 1

    records, missing = [], 0
    for r in kept:
        p = npz_dir / Path(r["activation_path"]).name
        if not p.is_file():
            missing += 1
            continue
        with np.load(p) as d:
            if pooling not in d:
                continue
            x = d[pooling].astype(np.float32)
        records.append(
            {
                "occurrence_id": Path(r["activation_path"]).stem,
                "y": r["occurrence_type"],
                "X": x,
                "repo": str(r.get("repo") or "?"),
                "function": f'{r.get("repo")}::{r.get("path")}::{r.get("source_row")}::{r.get("function")}',
                "variable": str(r.get("variable") or "?"),
            }
        )
    stats = {
        "manifest_rows_usable": len(rows),
        "duplicate_span_dropped": dup_dropped,
        "conflicting_label_dropped": conflict_dropped,
        "npz_missing": missing,
        "records": len(records),
    }
    return records, stats


def load_records_from_store(store_dir: Path) -> tuple[list[dict], dict]:
    """Load an extract_activations.py memmap store (index.jsonl + shard.npy).

    XLCoST has no repositories; the cluster/grouping unit is the PROBLEM, so
    ``problem_id`` fills the record's "repo" slot (the ``repo`` split policy
    therefore groups by problem — the protocol's split unit).
    """
    meta = json.loads((store_dir / "meta.json").read_text(encoding="utf-8"))
    shard = np.load(store_dir / "shard.npy", mmap_mode="r")
    records, skipped = [], 0
    for ln in (store_dir / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("skip"):
            skipped += 1
            continue
        records.append(
            {
                "occurrence_id": r["occurrence_id"],
                "y": r["occurrence_type"],
                "X": np.asarray(shard[r["row"]], dtype=np.float32),
                "repo": str(r["problem_id"]),
                "function": f'{r["problem_id"]}::{r.get("function")}',
                "variable": str(r.get("variable") or "?"),
            }
        )
    stats = {
        "store": str(store_dir),
        "model_id": meta.get("model_id"),
        "index_skipped_rows": skipped,
        "records": len(records),
    }
    return records, stats


def apply_class_policy(
    records: list[dict], min_class_count: int, allow_drop: bool
) -> tuple[list[dict], dict]:
    counts = Counter(r["y"] for r in records)
    low = {k: v for k, v in counts.items() if v < min_class_count}
    if low and not allow_drop:
        raise SystemExit(
            f"classes below --min-class-count={min_class_count}: {dict(low)}. "
            "Refusing to drop silently — pass --allow-class-drop to drop them "
            "explicitly (the drop is recorded in results.json)."
        )
    if low:
        records = [r for r in records if r["y"] not in low]
    return records, {"class_counts_full": dict(counts), "classes_dropped": dict(low)}


def three_way_split(
    n: int, groups: np.ndarray, y: np.ndarray, policy: str, seed: int,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/10/20 train/val/test, grouped by `policy` unit (or stratified random)."""
    idx = np.arange(n)
    f_train, f_val, f_test = fractions
    if policy == "random":
        trval, test = train_test_split(
            idx, test_size=f_test, random_state=seed, stratify=y
        )
        tr, val = train_test_split(
            trval,
            test_size=f_val / (f_train + f_val),
            random_state=seed,
            stratify=y[trval],
        )
        return tr, val, test
    # Deterministic HASH bucketing per group (not GroupShuffleSplit): every
    # group lands in the same fold for a given seed REGARDLESS of which other
    # groups are present. Renamed corpora drop a few programs; a shuffle-based
    # split would diverge wholesale and leave paired deltas with only an
    # accidental sliver of shared test items (observed: 76 of ~400).
    import hashlib as _hl

    def bucket(g: str) -> float:
        return int(_hl.sha1(f"{g}:{seed}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    fracs = np.array([bucket(str(g)) for g in groups])
    test = idx[fracs < f_test]
    val = idx[(fracs >= f_test) & (fracs < f_test + f_val)]
    tr = idx[fracs >= f_test + f_val]
    return tr, val, test


def run_probe_one_seed(
    X_all: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clusters: np.ndarray,
    occurrence_ids: list[str],
    labels: list[str],
    policy: str,
    seed: int,
    c_reg: float,
    n_boot: int,
) -> dict:
    """One full split -> per-layer val/test curves -> val-selected headline."""
    n, n_layers = X_all.shape[0], X_all.shape[1]
    tr, val, test = three_way_split(n, groups, y, policy, seed)
    # A grouped split can strand a class entirely outside train; record it.
    tr_classes = set(y[tr].tolist())
    absent = [c for c in labels if c not in tr_classes]

    import time

    from joblib import Parallel, delayed

    def fit_layer(layer: int):
        scaler = StandardScaler().fit(X_all[tr, layer])
        Xtr, Xval, Xte = (
            scaler.transform(X_all[s, layer]) for s in (tr, val, test)
        )
        clf = LogisticRegression(max_iter=1000, C=c_reg, random_state=seed)
        clf.fit(Xtr, y[tr])
        pred_te = clf.predict(Xte)
        return (
            macro_f1_stat(y[val], clf.predict(Xval), labels),
            macro_f1_stat(y[test], pred_te, labels),
            float(accuracy_score(y[test], pred_te)),
            pred_te,
        )

    t0 = time.time()
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(fit_layer)(layer) for layer in range(n_layers)
    )
    val_f1 = [r[0] for r in results]
    test_f1 = [r[1] for r in results]
    test_acc = [r[2] for r in results]
    per_layer_models = [r[3] for r in results]
    print(
        f"  seed {seed}: {n_layers} layers fitted in {time.time() - t0:.0f}s "
        f"(train n={len(tr)})",
        flush=True,
    )

    best = int(np.argmax(val_f1))
    pred = per_layer_models[best]
    ci = cluster_bootstrap_ci(
        y[test], pred, clusters[test], labels, n_boot=n_boot, seed=seed
    )
    per_class = f1_score(
        y[test], pred, labels=labels, average=None, zero_division=0
    )
    return {
        "seed": seed,
        "selected_layer": best,
        "normalized_depth": best / (n_layers - 1),
        "val_macro_f1_curve": [round(v, 4) for v in val_f1],
        "test_macro_f1_curve": [round(v, 4) for v in test_f1],
        "test_acc_curve": [round(v, 4) for v in test_acc],
        "test_macro_f1": test_f1[best],
        "test_accuracy": test_acc[best],
        "test_macro_f1_ci": [ci["ci_low"], ci["ci_high"]],
        "ci_meta": {k: ci[k] for k in ("n_clusters", "max_cluster_share", "cluster_warning")},
        "classes_absent_from_train": absent,
        "per_class_f1": {c: round(float(v), 4) for c, v in zip(labels, per_class)},
        "test_support": dict(Counter(y[test].tolist())),
        "confusion_matrix": confusion_matrix(y[test], pred, labels=labels).tolist(),
        "split_sizes": {"train": len(tr), "val": len(val), "test": len(test)},
        "_test_idx": test,
        "_test_pred": pred,
    }


def control_labels(records: list[dict], seed: int) -> np.ndarray:
    """Hewitt control task: one random label per variable NAME, from the empirical marginal."""
    rng = np.random.default_rng(seed)
    labels, counts = zip(*Counter(r["y"] for r in records).items())
    p = np.array(counts, dtype=float) / sum(counts)
    names = sorted({r["variable"] for r in records})
    mapping = {nm: labels[rng.choice(len(labels), p=p)] for nm in names}
    return np.array([mapping[r["variable"]] for r in records])


def cmd_run(args: argparse.Namespace) -> int:
    if args.store:
        records, load_stats = load_records_from_store(Path(args.store))
        source_desc = args.store
    else:
        records, load_stats = load_records(Path(args.manifest), Path(args.npz_dir), args.pooling)
        source_desc = args.manifest
    if not records:
        raise SystemExit("no usable records after join/dedup")
    print(f"loaded {len(records)} records from {source_desc}", flush=True)

    # PROTOCOL §1: occurrence cap, sampled ONCE by problem with a FIXED seed
    # (independent of probe seeds), so every renaming condition can reuse the
    # identical occurrence set. The sampled id list is written next to the
    # results file for that purpose.
    sample_info = {"occurrence_cap": args.occurrence_cap, "cap_applied": False}
    if args.occurrence_cap and len(records) > args.occurrence_cap:
        rng = np.random.default_rng(1234)
        problems = sorted({r["repo"] for r in records})
        rng.shuffle(problems)
        by_problem: dict = {}
        for r in records:
            by_problem.setdefault(r["repo"], []).append(r)
        sampled, n_acc = [], 0
        for p in problems:
            if n_acc >= args.occurrence_cap:
                break
            sampled.extend(by_problem[p])
            n_acc += len(by_problem[p])
        records = sampled
        sample_info.update(
            cap_applied=True,
            sampling_seed=1234,
            n_after_cap=len(records),
            n_problems_sampled=len({r["repo"] for r in records}),
        )
        print(
            f"occurrence cap {args.occurrence_cap}: sampled {len(records)} occurrences "
            f"from {sample_info['n_problems_sampled']} problems (fixed seed 1234)",
            flush=True,
        )

    records, class_stats = apply_class_policy(
        records, args.min_class_count, args.allow_class_drop
    )
    y = np.array([r["y"] for r in records])
    labels = sorted(set(y.tolist()))
    X_all = np.stack([r["X"] for r in records])  # [n, layers, hidden]
    group_key = "repo" if args.split_policy == "repo" else "function"
    groups = np.array([r[group_key] for r in records])
    clusters = np.array([r["repo"] for r in records])  # CI cluster unit: repo
    occ_ids = [r["occurrence_id"] for r in records]

    seeds = args.seeds
    per_seed = [
        run_probe_one_seed(
            X_all, y, groups, clusters, occ_ids, labels,
            args.split_policy, s, args.C, args.n_boot,
        )
        for s in seeds
    ]

    result: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "occurrence_type",
        "source": source_desc,
        "pooling": args.pooling,
        "split_policy": args.split_policy,
        "split_ratio": [0.7, 0.1, 0.2],
        "split_method": "hash_bucket_per_group",
        "ci_cluster_unit": "repo",
        "layer_indexing": "0 = embedding layer (pre-transformer); label axes 'embed,1..L'",
        "layer_selected_on": "validation",
        "seeds": seeds,
        "C": args.C,
        "load_stats": load_stats,
        "sample_info": sample_info,
        **class_stats,
        "classes_used": labels,
        "majority_baseline_acc": max(Counter(y.tolist()).values()) / len(y),
        "n_records": len(records),
        "n_repos": len(set(clusters.tolist())),
        "max_repo_share": max(Counter(clusters.tolist()).values()) / len(y),
        "per_seed": [],
        "aggregate": {},
    }

    f1s = [s["test_macro_f1"] for s in per_seed]
    accs = [s["test_accuracy"] for s in per_seed]
    result["aggregate"] = {
        "test_macro_f1_mean": float(np.mean(f1s)),
        "test_macro_f1_std": float(np.std(f1s)),
        "test_accuracy_mean": float(np.mean(accs)),
        "selected_layers": [s["selected_layer"] for s in per_seed],
        "normalized_depths": [round(s["normalized_depth"], 3) for s in per_seed],
    }

    if args.control_task:
        print("control task (Hewitt): refitting all layers on permuted labels...", flush=True)
        y_ctrl = control_labels(records, seed=1234)
        ctrl_labels_used = sorted(set(y_ctrl.tolist()))
        ctrl = [
            run_probe_one_seed(
                X_all, y_ctrl, groups, clusters, occ_ids, ctrl_labels_used,
                args.split_policy, s, args.C, max(200, args.n_boot // 4),
            )
            for s in seeds
        ]
        cf1 = [s["test_macro_f1"] for s in ctrl]
        result["control_task"] = {
            "test_macro_f1_mean": float(np.mean(cf1)),
            "selected_layers": [s["selected_layer"] for s in ctrl],
        }
        result["selectivity_macro_f1"] = float(np.mean(f1s) - np.mean(cf1))

    # Per-seed test predictions embedded for paired deltas downstream —
    # deltas must aggregate over the same seeds as the headline metric, not
    # describe seed 0 alone.
    result["test_predictions"] = [
        {
            "occurrence_id": occ_ids[i],
            "seed": sd["seed"],
            "y_true": str(y[i]),
            "y_pred": str(sd["_test_pred"][j]),
            "cluster": str(clusters[i]),
        }
        for sd in per_seed
        for j, i in enumerate(sd["_test_idx"])
    ]
    for sd in per_seed:
        sd.pop("_test_idx"), sd.pop("_test_pred")
    result["per_seed"] = per_seed

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    Path(str(out) + ".sample_ids.json").write_text(
        json.dumps(sorted(occ_ids)), encoding="utf-8"
    )

    agg = result["aggregate"]
    print(f"records={len(records)} repos={result['n_repos']} "
          f"max_repo_share={result['max_repo_share']:.2f} classes={labels}")
    if class_stats["classes_dropped"]:
        print(f"DROPPED CLASSES (explicit): {class_stats['classes_dropped']}")
    print(f"[{args.split_policy}] test macroF1 = {agg['test_macro_f1_mean']:.4f} "
          f"± {agg['test_macro_f1_std']:.4f}  acc = {agg['test_accuracy_mean']:.4f}  "
          f"majority = {result['majority_baseline_acc']:.4f}")
    print(f"selected layers per seed: {agg['selected_layers']}")
    if "selectivity_macro_f1" in result:
        print(f"control-task macroF1 = {result['control_task']['test_macro_f1_mean']:.4f}  "
              f"SELECTIVITY = {result['selectivity_macro_f1']:+.4f}")
    print(f"wrote {out}")
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Synthetic end-to-end check: separable classes must probe near 1.0, grouped."""
    rng = np.random.default_rng(0)
    n, layers, hidden = 300, 5, 16
    y = rng.integers(0, 3, size=n)
    X = rng.normal(size=(n, layers, hidden)).astype(np.float32)
    X[:, 3, :3] += np.eye(3)[y] * 4  # layer 3 carries the signal
    groups = np.array([f"g{i % 25}" for i in range(n)])
    res = run_probe_one_seed(
        X, y.astype(str), groups, groups, [f"o{i}" for i in range(n)],
        sorted(set(y.astype(str))), "function", 0, 1.0, 200,
    )
    ok = res["selected_layer"] == 3 and res["test_macro_f1"] > 0.9
    print(f"selected_layer={res['selected_layer']} test_f1={res['test_macro_f1']:.3f}")
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="probe cached activations")
    r.add_argument("--store", help="extract_activations.py output dir (index.jsonl + shard.npy)")
    r.add_argument("--manifest")
    r.add_argument("--npz-dir")
    r.add_argument("--pooling", default="mean", choices=["first", "last", "mean"])
    r.add_argument("--split-policy", default="repo", choices=["random", "function", "repo"])
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--C", type=float, default=1.0)
    r.add_argument("--min-class-count", type=int, default=20)
    r.add_argument("--allow-class-drop", action="store_true")
    r.add_argument("--control-task", action="store_true",
                   help="also run the Hewitt control task and report selectivity")
    r.add_argument("--occurrence-cap", type=int, default=2000,
                   help="PROTOCOL cap per (role, language); 0 disables")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--output", required=True)
    sub.add_parser("verify", help="synthetic self-check")
    args = ap.parse_args(argv)
    return cmd_verify(args) if args.cmd == "verify" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
