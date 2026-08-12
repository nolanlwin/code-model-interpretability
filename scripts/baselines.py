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


def load_from_occurrences(occ_path: Path, canon_path: Path, window: int) -> list[dict]:
    canon_rows = _read_jsonl(canon_path)
    # XLCoST canonical rows join by problem_id; CodeSearchNet by 1-based line index.
    by_problem = {r["problem_id"]: r.get("code", "") for r in canon_rows if "problem_id" in r}
    by_index = [r.get("code", "") for r in canon_rows]
    out = []
    for r in _read_jsonl(occ_path):
        if not r.get("occurrence_type"):
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
                "y": r["occurrence_type"],
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


def _code_mask(code: str) -> tuple[list[bool], list[int]]:
    """Per-character (is_code, bracket_depth).

    is_code is False inside string/char literals and comments, so delimiters
    there are not treated as syntax. depth counts ONLY () and [] — not {} —
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
            if ch == "\\":
                if i + 1 < n:
                    is_code[i + 1] = False
                depth[i] = d
                i += 2
                continue
            if ch == quote:
                quote = None
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
        return {"macro_f1": float("nan"), "acc": float("nan"), "note": "empty vocabulary"}
    Xte = vec.transform([texts[i] for i in te])
    clf = LogisticRegression(max_iter=2000, C=4.0, random_state=seed).fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    return {
        "macro_f1": macro_f1_stat(y[te], pred, labels),
        "acc": float(accuracy_score(y[te], pred)),
    }


def _fit_scalar(feats: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                labels: list[str], seed: int) -> dict:
    sc = StandardScaler().fit(feats[tr])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(feats[tr]), y[tr])
    pred = clf.predict(sc.transform(feats[te]))
    return {
        "macro_f1": macro_f1_stat(y[te], pred, labels),
        "acc": float(accuracy_score(y[te], pred)),
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
        row = {
            "seed": seed,
            "majority": {
                "acc": float(np.mean(y[te] == Counter(y[tr].tolist()).most_common(1)[0][0])),
                "macro_f1": macro_f1_stat(
                    y[te], np.full(len(te), Counter(y[tr].tolist()).most_common(1)[0][0]), labels
                ),
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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.split_policy}] n={len(recs)} classes={labels} dropped={dropped}")
    for k in keys:
        print(f"  {k:<16} macroF1={agg(k, 'macro_f1'):.4f}  acc={agg(k, 'acc'):.4f}")
    print(f"  STRONGEST baseline macroF1 = {result['strongest_baseline_macro_f1']:.4f}")
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
    args = ap.parse_args(argv)
    if args.occurrences and not args.canonical:
        ap.error("--occurrences requires --canonical")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
