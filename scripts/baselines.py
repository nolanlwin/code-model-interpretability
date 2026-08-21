"""Model-free baselines for occurrence_type probing (PROTOCOL.md §4).

Every probe number must be reported as a delta over the strongest of these.
Runs from either:

- a probing manifest (``--manifest``): majority, NAME-ONLY, and scalar
  covariates (token_len, function_len_chars, occurrence_frequency) — no source
  code needed; or
- an occurrence JSONL + canonical corpus (``--occurrences`` + ``--canonical``):
  adds the MASKED SOURCE LINE and ±k-char window baselines, which are the
  strong ones (masked line reached 0.948 acc / 0.777 macro F1 on CodeSearchNet
  Python — near the reported probe numbers, with no model).

Same grouped-split policies as scripts/probe.py so the numbers are comparable.

    uv run python scripts/baselines.py run --manifest outputs/activations_java/manifest.jsonl \
        --split-policy repo --output outputs/probe_results/java_repo_baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from bootstrap_ci import macro_f1_stat  # noqa: E402
from probe import three_way_split  # noqa: E402


def git_commit() -> str:
    """Commit that produced this result (empty if not a git checkout)."""
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_SCRIPT_DIR,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def load_from_manifest(path: Path) -> list[dict]:
    rows = [
        r
        for r in _read_jsonl(path)
        if r.get("occurrence_type") and r.get("variable")
    ]
    return [
        {
            "y": r["occurrence_type"],
            "variable": str(r["variable"]),
            "repo": str(r.get("repo") or "?"),
            "function": f'{r.get("repo")}::{r.get("path")}::{r.get("source_row")}::{r.get("function")}',
            "covariates": [
                float(r.get("token_len") or 0),
                float(r.get("function_len_chars") or 0),
                float(r.get("occurrence_frequency") or 0),
            ],
            "line_masked": None,
            "statement_masked": None,
            "window_masked": None,
        }
        for r in rows
    ]


def load_from_occurrences(occ_path: Path, canon_path: Path, window: int,
                          label_field: str = "occurrence_type") -> list[dict]:
    """Occurrence rows joined to their program text, ready for the baselines.

    ``label_field`` selects what is being predicted. The boolean workstream
    labels each site with ``occurrence_type``; ``role_occurrences.py`` labels
    it with ``role``, which is what cross-lingual transfer needs -- the
    question there is whether a probe that separates accumulator from other
    roles in one language still does so in another.
    """
    canon_rows = _read_jsonl(canon_path)
    # XLCoST canonical rows join by problem_id; CodeSearchNet by 1-based line index.
    by_problem = {r["problem_id"]: r.get("code", "") for r in canon_rows if "problem_id" in r}
    by_index = [r.get("code", "") for r in canon_rows]
    out = []
    for r in _read_jsonl(occ_path):
        if not r.get(label_field):
            continue
        if r.get("problem_id") is not None:
            code = by_problem.get(r["problem_id"], "")
        else:
            i = int(r.get("source_row", 0)) - 1
            code = by_index[i] if 0 <= i < len(by_index) else ""
        if not code:
            continue
        span = r.get("source_span") or [0, 0]
        var = str(r.get("variable", ""))
        s = code.rfind("\n", 0, span[0]) + 1
        e = code.find("\n", span[1])
        e = len(code) if e < 0 else e
        # Enclosing STATEMENT: the honest local-surface unit for brace
        # languages, whose detokenized XLCoST programs are single-line
        # (measured 0.11 newlines/program vs 22.3 for Python), making the
        # line baseline degenerate there (line == whole program).
        ss, se = statement_bounds(code, span[0], span[1])
        group = r.get("problem_id") or r.get("repo") or "?"
        out.append(
            {
                "occurrence_id": r.get("occurrence_id"),
                "y": r[label_field],
                "variable": var,
                "repo": str(group),
                "function": f'{group}::{r.get("function")}',
                "covariates": [float(len(code)), float(span[0]), 0.0],
                "line_masked": code[s:e].replace(var, " VAR "),
                "statement_masked": code[ss:se].replace(var, " VAR "),
                "window_masked": code[max(0, span[0] - window): span[1] + window].replace(var, " VAR "),
            }
        )
    return out


_TRIPLE_QUOTES = {'"""', "'''"}


def _code_mask(code: str) -> tuple[list[bool], list[int]]:
    """Per-character (is_code, bracket_depth).

    is_code is False inside string/char literals and comments, so delimiters
    there are not treated as syntax. This is a heuristic lexer, not a parser:
    it handles single, double, backtick, and triple-quoted literals, line and
    block comments, and backslash escapes. Known limit: raw literals
    (raw triple-quoted strings) still honor escapes, which can only extend a
    masked region, never end one early -- the conservative direction for a
    baseline. depth counts ONLY () and [] — not {} —
    so that the semicolons in a C-style ``for (init; cond; step)`` header sit
    at depth > 0 and never split a statement, including when the occurrence
    itself is inside that header (where a depth-relative test would accept
    them and yield a fragment like ``i < n;``).
    """
    n = len(code)
    is_code = [True] * n
    depth = [0] * n
    i = d = 0
    quote = None          # active string delimiter
    comment = None        # "line" | "block"
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if comment == "line":
            is_code[i] = False
            if ch == "\n":
                comment = None
                is_code[i] = True          # the newline itself is a boundary
        elif comment == "block":
            is_code[i] = False
            if ch == "*" and nxt == "/":
                is_code[i + 1] = False
                depth[i] = d
                i += 2
                comment = None
                continue
        elif quote:
            is_code[i] = False
            # Escapes apply inside triple-quoted literals too: without this,
            # an escaped delimiter run (\\""" ) closes the literal early and
            # its remaining newlines/semicolons become false boundaries.
            if ch == "\\":
                if i + 1 < n:
                    is_code[i + 1] = False
                depth[i] = d
                i += 2
                continue
            if code.startswith(quote, i):
                for k in range(i, min(i + len(quote), n)):
                    is_code[k] = False
                    depth[k] = d
                i += len(quote)
                quote = None
                continue
        elif code[i:i + 3] in _TRIPLE_QUOTES:
            # Triple-quoted string / Java text block. Must be detected BEFORE
            # the single-quote branch: that branch opens on the first
            # delimiter and closes on the second, leaving the body exposed as
            # code whenever the body contains an unbalanced quote character.
            quote = code[i:i + 3]
            for k in range(i, min(i + 3, n)):
                is_code[k] = False
                depth[k] = d
            i += 3
            continue
        elif ch in "\"'`":
            quote = ch
            is_code[i] = False
        elif ch == "/" and nxt == "/":
            comment = "line"
            is_code[i] = False
        elif ch == "#":
            comment = "line"
            is_code[i] = False
        elif ch == "/" and nxt == "*":
            comment = "block"
            is_code[i] = False
        elif ch in "([":
            d += 1
        elif ch in ")]":
            d = max(0, d - 1)
        depth[i] = d
        i += 1
    return is_code, depth


def statement_bounds(code: str, start: int, end: int) -> tuple[int, int]:
    """Enclosing statement: nearest code-level boundary on each side.

    Boundaries are ``{``, ``}``, ``;`` and newline — all only OUTSIDE
    parentheses/brackets, since a wrapped call or a multiline
    ``for (init; cond; step)`` header contains newlines and semicolons that
    are continuations rather than statement ends.
    Delimiters inside strings and comments are skipped. An occurrence inside
    a ``for (init; cond; step)`` header therefore expands to the whole
    header rather than to a fragment between its internal semicolons.
    """
    is_code, paren = _code_mask(code)

    def is_boundary(i: int) -> bool:
        if not is_code[i]:
            return False
        ch = code[i]
        # Inside () or [] nothing terminates a statement: a wrapped call or a
        # for-header split across lines carries newlines that are
        # continuations, and its semicolons belong to the header.
        if paren[i] > 0:
            return False
        return ch in ";{}\n"

    ss, se = 0, len(code)
    for i in range(min(start, len(code)) - 1, -1, -1):
        if is_boundary(i):
            ss = i + 1
            break
    for i in range(max(end, 0), len(code)):
        if is_boundary(i):
            se = i + 1
            break
    return ss, se


def _fit_text(texts: list[str], y: np.ndarray, tr: np.ndarray, te: np.ndarray,
              labels: list[str], seed: int) -> dict:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=60000)
    try:
        Xtr = vec.fit_transform([texts[i] for i in tr])
    except ValueError:
        return {"macro_f1": float("nan"), "acc": float("nan"),
                "note": "empty vocabulary", "_pred": None}
    Xte = vec.transform([texts[i] for i in te])
    clf = LogisticRegression(max_iter=2000, C=4.0, random_state=seed).fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    return {
        "macro_f1": macro_f1_stat(y[te], pred, labels),
        "acc": float(accuracy_score(y[te], pred)),
        "_pred": pred,
    }


def _fit_scalar(feats: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                labels: list[str], seed: int) -> dict:
    sc = StandardScaler().fit(feats[tr])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(feats[tr]), y[tr])
    pred = clf.predict(sc.transform(feats[te]))
    return {
        "macro_f1": macro_f1_stat(y[te], pred, labels),
        "acc": float(accuracy_score(y[te], pred)),
        "_pred": pred,
    }


def cmd_run(args: argparse.Namespace) -> int:
    if args.manifest:
        recs = load_from_manifest(Path(args.manifest))
        source = args.manifest
    else:
        recs = load_from_occurrences(Path(args.occurrences), Path(args.canonical), args.window)
        source = args.occurrences
    if not recs:
        raise SystemExit("no records")

    if args.sample_ids:
        wanted = set(json.loads(Path(args.sample_ids).read_text(encoding="utf-8")))
        before = len(recs)
        recs = [r for r in recs if r.get("occurrence_id") in wanted]
        matched = len(recs)
        print(f"sample filter: {matched}/{before} occurrences match "
              f"{len(wanted)} sampled ids ({args.sample_ids})")
        if matched < 0.9 * len(wanted):
            raise SystemExit(
                "fewer than 90% of sampled ids matched - wrong occurrence file "
                "or wrong sample file; refusing to report incomparable numbers"
            )

    counts = Counter(r["y"] for r in recs)
    keep = {k for k, v in counts.items() if v >= args.min_class_count}
    dropped = {k: v for k, v in counts.items() if k not in keep}
    recs = [r for r in recs if r["y"] in keep]
    y = np.array([r["y"] for r in recs])
    labels = sorted(keep)
    group_key = "repo" if args.split_policy == "repo" else "function"
    groups = np.array([r[group_key] for r in recs])
    names = [r["variable"] for r in recs]
    covs = np.array([r["covariates"] for r in recs])
    have_text = recs[0]["line_masked"] is not None

    per_seed = []
    for seed in args.seeds:
        tr, val, te = three_way_split(len(recs), groups, y, args.split_policy, seed)
        tr = np.concatenate([tr, val])  # baselines have no layer to select
        _maj = Counter(y[tr].tolist()).most_common(1)[0][0]
        _maj_pred = np.full(len(te), _maj)
        row = {
            "seed": seed,
            "_te": te,
            "majority": {
                "acc": float(np.mean(y[te] == _maj)),
                "macro_f1": macro_f1_stat(y[te], _maj_pred, labels),
                "_pred": _maj_pred,
            },
            "name_only": _fit_text(names, y, tr, te, labels, seed),
            "covariates_only": _fit_scalar(covs, y, tr, te, labels, seed),
        }
        if have_text:
            row["line_masked"] = _fit_text([r["line_masked"] for r in recs], y, tr, te, labels, seed)
            row["statement_masked"] = _fit_text([r["statement_masked"] for r in recs], y, tr, te, labels, seed)
            row["window_masked"] = _fit_text([r["window_masked"] for r in recs], y, tr, te, labels, seed)
        per_seed.append(row)

    def agg(key: str, metric: str) -> float:
        vals = [s[key][metric] for s in per_seed if key in s and np.isfinite(s[key].get(metric, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    keys = ["majority", "name_only", "covariates_only"] + (
        ["line_masked", "statement_masked", "window_masked"] if have_text else [])
    result = {
        "source": source,
        "git_commit": git_commit(),
        "sample_ids": args.sample_ids,
        "split_policy": args.split_policy,
        "n_records": len(recs),
        "classes_used": labels,
        "classes_dropped": dropped,
        "seeds": args.seeds,
        "aggregate": {k: {"macro_f1": agg(k, "macro_f1"), "acc": agg(k, "acc")} for k in keys},
        "strongest_baseline_macro_f1": max(agg(k, "macro_f1") for k in keys),
        "per_seed": per_seed,
    }
    # Per-occurrence predictions for the STRONGEST baseline, in the same shape
    # probe.py emits, so `bootstrap_ci.py delta <probe>.json <baselines>.json`
    # can compute a paired clustered CI on probe-minus-baseline. Without this
    # the strongest baseline is a point estimate with no interval, and a
    # probe-vs-baseline margin cannot be told apart from zero.
    best_key = max(keys, key=lambda k: (agg(k, "macro_f1"), k))
    occ_ids = [r.get("occurrence_id") for r in recs]
    clusters = [str(r["repo"]) for r in recs]  # problem-level, matching probe.py

    # Pairing is BY occurrence_id, so emitting rows without one is worse than
    # emitting nothing: bootstrap_ci.py keys predictions into a dict, so every
    # null id would collapse onto a single entry and the resulting "CI" would
    # be computed over one occurrence while looking perfectly well-formed.
    # --manifest records (load_from_manifest) carry no ids at all. Refuse, and
    # say why in the artifact rather than only on stdout.
    n_missing = sum(o is None for o in occ_ids)
    n_dupe = len(occ_ids) - len(set(occ_ids))
    pairable = n_missing == 0 and n_dupe == 0
    result["test_predictions_baseline"] = best_key if pairable else None
    if not pairable:
        why = (f"{n_missing} of {len(occ_ids)} records have no occurrence_id"
               if n_missing else f"{n_dupe} duplicate occurrence_ids")
        result["test_predictions"] = []
        result["test_predictions_skipped"] = (
            f"not emitted: {why}. Pairing in bootstrap_ci.py delta is by "
            "occurrence_id; --manifest input does not carry one, so use "
            "--occurrences/--canonical to get a probe-vs-baseline CI."
        )
        print(f"  NOTE: test_predictions not emitted - {why}")
    else:
        preds = []
        for srow in per_seed:
            te_idx, cell = srow["_te"], srow.get(best_key)
            if cell is None or cell.get("_pred") is None:
                continue
            for j, i in enumerate(te_idx):
                preds.append({
                    "occurrence_id": occ_ids[i],
                    "seed": srow["seed"],
                    "y_true": str(y[i]),
                    "y_pred": str(cell["_pred"][j]),
                    "cluster": clusters[i],
                })
        result["test_predictions"] = preds
    # Drop the private arrays before serialising.
    for srow in per_seed:
        srow.pop("_te", None)
        for k in keys:
            if isinstance(srow.get(k), dict):
                srow[k].pop("_pred", None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.split_policy}] n={len(recs)} classes={labels} dropped={dropped}")
    for k in keys:
        print(f"  {k:<16} macroF1={agg(k, 'macro_f1'):.4f}  acc={agg(k, 'acc'):.4f}")
    print(f"  STRONGEST baseline macroF1 = {result['strongest_baseline_macro_f1']:.4f}")
    print(f"wrote {out}")
    return 0


def _fit_predict_across(train_texts, y_tr, test_texts, labels, seed):
    """Fit the char n-gram model on one corpus, predict on another.

    The vectorizer is fitted on the TRAINING language only. That is the point:
    whatever character structure carries the label in language A has to be
    present in B for this to score above chance, which is exactly the
    surface-transfer hypothesis being tested.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                          max_features=60000)
    try:
        Xtr = vec.fit_transform(train_texts)
    except ValueError:
        return None, {"note": "empty vocabulary"}
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                             random_state=seed).fit(Xtr, y_tr)
    pred = clf.predict(vec.transform(test_texts))
    return pred, {}


def cmd_transfer(args: argparse.Namespace) -> int:
    """Cross-lingual surface baseline: fit on A, evaluate on B.

    This is the control the cross-lingual probe result has never had. If a
    character n-gram model trained on language A transfers to B as well as the
    probe does, then transfer is measuring surface regularity that the two
    languages share, not a language-universal role representation.
    """
    tr = load_from_occurrences(Path(args.train_occurrences), Path(args.train_canonical),
                              args.window, label_field=args.label_field)
    te = load_from_occurrences(Path(args.test_occurrences), Path(args.test_canonical),
                              args.window, label_field=args.label_field)
    if not tr or not te:
        raise SystemExit(f"empty corpus: train={len(tr)} test={len(te)}")

    matched_note = "unmatched (different problems on each side)"
    if args.matched:
        shared = {r["repo"] for r in tr} & {r["repo"] for r in te}
        if len(shared) < args.min_shared:
            raise SystemExit(
                f"only {len(shared)} shared problem ids between these languages "
                f"(--min-shared {args.min_shared}). Matched transfer is not "
                "available for this pair; rerun without --matched and label the "
                "result unmatched."
            )
        tr = [r for r in tr if r["repo"] in shared]
        te = [r for r in te if r["repo"] in shared]
        matched_note = f"matched on {len(shared)} shared problems"

    def binarise(rows):
        return np.array([("target" if r["y"] == args.role else "other") for r in rows]) \
            if args.role else np.array([r["y"] for r in rows])

    y_tr, y_te = binarise(tr), binarise(te)
    labels = sorted(set(y_tr.tolist()) | set(y_te.tolist()))
    if len(set(y_tr.tolist())) < 2:
        raise SystemExit(f"training side has one class only: {set(y_tr.tolist())}")

    def _ws(texts):
        """Collapse whitespace runs, when --normalize-whitespace is given.

        XLCoST draws Python from the formatted mirror and JavaScript and PHP
        from the tokenized one, so Python arrives with newlines and indentation
        while the others arrive flattened onto one line. That makes the
        typological boundary and the corpus-provenance boundary the same line.
        This switch equalises the two sides so the difference can be measured
        rather than argued about; see
        results/lp4fm/whitespace_normalisation_check.csv.
        """
        if not getattr(args, "normalize_whitespace", False):
            return texts
        return [None if t is None else re.sub(r"\s+", " ", t).strip() for t in texts]

    feats = {
        "statement_masked": (_ws([r["statement_masked"] for r in tr]),
                             _ws([r["statement_masked"] for r in te])),
        "line_masked": (_ws([r["line_masked"] for r in tr]),
                        _ws([r["line_masked"] for r in te])),
        "window_masked": (_ws([r["window_masked"] for r in tr]),
                          _ws([r["window_masked"] for r in te])),
        "name_only": ([r["variable"] for r in tr], [r["variable"] for r in te]),
    }

    rng = np.random.default_rng(args.seed)
    results, preds_for_ci = {}, None
    for name, (a, b) in feats.items():
        if a[0] is None:
            continue
        per_seed = []
        for sd in args.seeds:
            pred, note = _fit_predict_across(a, y_tr, b, labels, sd)
            if pred is None:
                per_seed.append({"seed": sd, **note})
                continue
            per_seed.append({
                "seed": sd,
                "macro_f1": macro_f1_stat(y_te, pred, labels),
                "acc": float(accuracy_score(y_te, pred)),
                "_pred": pred,
            })
        good = [x for x in per_seed if "macro_f1" in x]
        results[name] = {
            "macro_f1": float(np.mean([x["macro_f1"] for x in good])) if good else float("nan"),
            "acc": float(np.mean([x["acc"] for x in good])) if good else float("nan"),
        }
        if name == "statement_masked" and good:
            preds_for_ci = [
                {"occurrence_id": te[i].get("occurrence_id"), "seed": x["seed"],
                 "y_true": str(y_te[i]), "y_pred": str(x["_pred"][i]),
                 "cluster": str(te[i]["repo"])}
                for x in good for i in range(len(te))
            ]

    # Shuffled-label control: permute the TRAINING labels. Anything the model
    # still scores after that is chance plus class imbalance, not transfer.
    shuf_scores = []
    for sd in args.seeds:
        y_shuf = rng.permutation(y_tr)
        pred, note = _fit_predict_across(feats["statement_masked"][0], y_shuf,
                                         feats["statement_masked"][1], labels, sd)
        if pred is not None:
            shuf_scores.append(macro_f1_stat(y_te, pred, labels))
    majority = Counter(y_tr.tolist()).most_common(1)[0][0]
    result = {
        "protocol_version": "1.0",
        "git_commit": git_commit(),
        "kind": "crosslang_surface_baseline",
        "train": args.train_occurrences, "test": args.test_occurrences,
        "label_field": args.label_field, "role": args.role,
        "pairing": matched_note,
        "n_train": len(tr), "n_test": len(te),
        "labels": labels, "seeds": args.seeds,
        "class_balance_test": dict(Counter(y_te.tolist())),
        "majority_macro_f1": macro_f1_stat(y_te, np.full(len(y_te), majority), labels),
        "shuffled_label_control_macro_f1": (float(np.mean(shuf_scores)) if shuf_scores
                                            else float("nan")),
        "aggregate": results,
        "strongest_baseline_macro_f1": max(
            (v["macro_f1"] for v in results.values() if np.isfinite(v["macro_f1"])),
            default=float("nan")),
        "test_predictions": preds_for_ci or [],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.role or 'multiclass'}] {Path(args.train_occurrences).stem} -> "
          f"{Path(args.test_occurrences).stem}  ({matched_note})")
    print(f"  n_train={len(tr)} n_test={len(te)} classes={labels}")
    for k, v in results.items():
        print(f"    {k:<18} macroF1={v['macro_f1']:.4f}  acc={v['acc']:.4f}")
    print(f"    {'majority':<18} macroF1={result['majority_macro_f1']:.4f}")
    print(f"    {'shuffled labels':<18} macroF1={result['shuffled_label_control_macro_f1']:.4f}")
    print(f"  STRONGEST transferred baseline = {result['strongest_baseline_macro_f1']:.4f}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest")
    src.add_argument("--occurrences")
    r.add_argument("--canonical", help="canonical JSONL (required with --occurrences)")
    r.add_argument("--sample-ids",
                   help="probe.py <output>.sample_ids.json - restrict to the probe's exact occurrence set")
    r.add_argument("--window", type=int, default=120)
    r.add_argument("--split-policy", default="repo", choices=["random", "function", "repo"])
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--min-class-count", type=int, default=20)
    r.add_argument("--output", required=True)
    t = sub.add_parser("transfer", help="fit the surface baseline on one language, "
                                        "evaluate on another")
    t.add_argument("--normalize-whitespace", action="store_true",
                   help="collapse whitespace runs in the masked features on "
                        "both sides, to measure whether the formatting "
                        "difference between XLCoST's two source mirrors "
                        "explains cross-language transfer differences")
    t.add_argument("--train-occurrences", required=True)
    t.add_argument("--train-canonical", required=True)
    t.add_argument("--test-occurrences", required=True)
    t.add_argument("--test-canonical", required=True)
    t.add_argument("--label-field", default="role",
                   help="'role' for cross-lingual transfer, 'occurrence_type' for "
                        "the boolean workstream's labels")
    t.add_argument("--role", default=None,
                   help="binary target-vs-rest for this role; omit for multiclass")
    t.add_argument("--matched", action="store_true",
                   help="restrict both sides to shared problem ids, so the same "
                        "algorithms appear on each side")
    t.add_argument("--min-shared", type=int, default=200)
    t.add_argument("--window", type=int, default=120)
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--output", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "transfer":
        return cmd_transfer(args)
    if args.occurrences and not args.canonical:
        ap.error("--occurrences requires --canonical")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
