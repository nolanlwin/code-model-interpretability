"""Cross-lingual PROBE transfer: train on one language's activations, evaluate
on another's — on the same problems, against the same baseline.

The companion to `baselines.py transfer`. That one asks how far a character
n-gram model gets across languages; this asks how far the model's residual
stream gets, on identical cells, so the two can sit in one table.

WHY MATCHED PROBLEMS. `problem_id` is a hash of the problem description, so
it is the same string in every language implementing that problem. Restricting
both sides to shared ids holds the algorithm constant and varies only surface
form. Unmatched transfer confounds "roles do not transfer" with "different
problems", and only Python/JavaScript/PHP share ids at usable rates.

WHY THE FOLDS LINE UP. `three_way_split` hash-buckets each group
independently of which other groups are present, so a problem lands in the
same fold in both languages. Training on A's train fold and evaluating on B's
test fold therefore cannot leak: no problem appears on both sides.

    python scripts/crosslang.py run \
        --train-store outputs/activations_xlcost/python_train_qwen25coder15b \
        --test-store  outputs/activations_xlcost/javascript_train_qwen25coder15b \
        --role accumulator --output outputs/crosslang/probe_python_to_javascript_accumulator.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from bootstrap_ci import cluster_bootstrap_ci, macro_f1_stat  # noqa: E402
from probe import git_commit, load_records_from_store, three_way_split  # noqa: E402


def binarise(records: list[dict], role: str | None) -> np.ndarray:
    if role is None:
        return np.array([r["y"] for r in records])
    return np.array(["target" if r["y"] == role else "other" for r in records])


def resolution(y: np.ndarray, labels: list[str]) -> tuple[float | None, str | None, int]:
    """Macro-F1 movement from one occurrence of the smallest class flipping."""
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return None, None, 0
    small, big = min(counts, key=counts.get), max(counts, key=counts.get)
    flipped = y.copy()
    flipped[np.where(y == small)[0][0]] = big
    return (float(macro_f1_stat(y, y, labels) - macro_f1_stat(y, flipped, labels)),
            small, int(counts[small]))


def cmd_run(args: argparse.Namespace) -> int:
    tr_recs, tr_meta = load_records_from_store(Path(args.train_store))
    te_recs, te_meta = load_records_from_store(Path(args.test_store))
    # load_records_from_store returns run STATS, which carry model_id but not
    # label_field; that lives in the store's meta.json. Checking it matters:
    # a store built from `occurrence_type` and one built from `role` hold
    # different class schemes and must never be compared.
    metas = {name: json.loads((Path(sd) / "meta.json").read_text(encoding="utf-8"))
             for name, sd in (("train", args.train_store), ("test", args.test_store))}
    for name, meta in metas.items():
        lf = meta.get("label_field", "occurrence_type")
        if lf != args.expect_label_field:
            raise SystemExit(
                f"{name} store was built with --label-field {lf!r}, expected "
                f"{args.expect_label_field!r}. A store labelled from a different "
                "occurrence field cannot be compared against this one."
            )
    if tr_meta.get("model_id") != te_meta.get("model_id"):
        raise SystemExit(
            f"different models: train={tr_meta.get('model_id')} "
            f"test={te_meta.get('model_id')}. Transfer across models is not "
            "what this measures."
        )

    shared = {r["repo"] for r in tr_recs} & {r["repo"] for r in te_recs}
    if len(shared) < args.min_shared:
        raise SystemExit(
            f"only {len(shared)} shared problem ids (--min-shared {args.min_shared}). "
            "Matched transfer is unavailable for this pair."
        )
    tr_recs = [r for r in tr_recs if r["repo"] in shared]
    te_recs = [r for r in te_recs if r["repo"] in shared]

    y_tr, y_te = binarise(tr_recs, args.role), binarise(te_recs, args.role)
    labels = sorted(set(y_tr.tolist()) | set(y_te.tolist()))
    if len(set(y_tr.tolist())) < 2:
        raise SystemExit(f"train side has one class: {set(y_tr.tolist())}")

    X_tr = np.stack([r["X"] for r in tr_recs])   # [n, L+1, H]
    X_te = np.stack([r["X"] for r in te_recs])
    g_tr = np.array([r["repo"] for r in tr_recs])
    g_te = np.array([r["repo"] for r in te_recs])
    n_layers = X_tr.shape[1]

    per_seed = []
    for seed in args.seeds:
        a_tr, a_val, a_test = three_way_split(len(tr_recs), g_tr, y_tr, "repo", seed)
        _, _, b_te = three_way_split(len(te_recs), g_te, y_te, "repo", seed)

        def fit_eval(layer, fit_idx, eval_X, eval_y, eval_idx):
            sc = StandardScaler().fit(X_tr[fit_idx, layer])
            clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced",
                                     random_state=seed).fit(sc.transform(X_tr[fit_idx, layer]),
                                                            y_tr[fit_idx])
            pred = clf.predict(sc.transform(eval_X[eval_idx, layer]))
            return macro_f1_stat(eval_y[eval_idx], pred, labels), pred

        # Layer chosen on the SOURCE language's validation fold. Selecting it
        # on the target would be selecting on the thing being measured.
        val_scores = [fit_eval(ly, a_tr, X_tr, y_tr, a_val)[0] for ly in range(n_layers)]
        best = int(np.argmax(val_scores))
        f1_transfer, pred_te = fit_eval(best, a_tr, X_te, y_te, b_te)
        f1_indomain, _ = fit_eval(best, a_tr, X_tr, y_tr, a_test)

        # Shuffled source labels: anything left is chance plus class imbalance.
        rng = np.random.default_rng(seed)
        y_shuf = rng.permutation(y_tr)
        sc = StandardScaler().fit(X_tr[a_tr, best])
        clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced",
                                 random_state=seed).fit(sc.transform(X_tr[a_tr, best]),
                                                        y_shuf[a_tr])
        f1_shuf = macro_f1_stat(y_te[b_te], clf.predict(sc.transform(X_te[b_te, best])), labels)

        per_seed.append({"seed": seed, "selected_layer": best,
                         "indomain_macro_f1": f1_indomain,
                         "transfer_macro_f1": f1_transfer,
                         "shuffled_source_macro_f1": f1_shuf,
                         "_pred": pred_te, "_idx": b_te})

    idx0 = per_seed[0]["_idx"]
    ci = cluster_bootstrap_ci(y_te[idx0], per_seed[0]["_pred"], g_te[idx0], labels,
                              n_boot=args.n_boot, seed=0)
    rho, small, small_n = resolution(y_te[idx0], labels)
    maj = Counter(y_tr.tolist()).most_common(1)[0][0]

    result = {
        "protocol_version": "1.0", "git_commit": git_commit(),
        "kind": "crosslang_probe_transfer",
        "model_id": tr_meta.get("model_id"),
        "train_store": args.train_store, "test_store": args.test_store,
        "role": args.role, "labels": labels, "seeds": args.seeds,
        "n_shared_problems": len(shared),
        "n_train": len(tr_recs), "n_test": len(te_recs), "n_test_fold": int(len(idx0)),
        "layer_selected_on": "source-language validation fold",
        "selected_layers": [s["selected_layer"] for s in per_seed],
        "indomain_macro_f1_mean": float(np.mean([s["indomain_macro_f1"] for s in per_seed])),
        "transfer_macro_f1_mean": float(np.mean([s["transfer_macro_f1"] for s in per_seed])),
        "shuffled_source_macro_f1_mean": float(np.mean([s["shuffled_source_macro_f1"]
                                                        for s in per_seed])),
        "majority_macro_f1": float(macro_f1_stat(y_te[idx0],
                                                 np.full(len(idx0), maj), labels)),
        "transfer_ci": {k: ci[k] for k in ("ci_low", "ci_high", "method", "n_clusters")},
        "resolution_rho": rho,
        "smallest_test_class": {"label": small, "n": small_n},
        "test_predictions": [
            {"occurrence_id": te_recs[i].get("occurrence_id"), "seed": s["seed"],
             "y_true": str(y_te[i]), "y_pred": str(p), "cluster": str(g_te[i])}
            for s in per_seed for p, i in zip(s["_pred"], s["_idx"])
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.role}] {Path(args.train_store).name} -> {Path(args.test_store).name}")
    print(f"  {len(shared)} shared problems | layers {result['selected_layers']}")
    print(f"  in-domain      {result['indomain_macro_f1_mean']:.4f}")
    print(f"  TRANSFER       {result['transfer_macro_f1_mean']:.4f}  "
          f"CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    print(f"  shuffled src   {result['shuffled_source_macro_f1_mean']:.4f}")
    print(f"  majority       {result['majority_macro_f1']:.4f}")
    print(f"  rho            {rho if rho is None else round(rho, 5)} "
          f"(smallest test class {small} n={small_n})")
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--train-store", required=True)
    r.add_argument("--test-store", required=True)
    r.add_argument("--role", default=None,
                   help="binary target-vs-rest; omit for multiclass")
    r.add_argument("--expect-label-field", default="role",
                   help="refuse stores not built with this --label-field")
    r.add_argument("--min-shared", type=int, default=200)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--C", type=float, default=1.0)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--output", required=True)
    return cmd_run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
