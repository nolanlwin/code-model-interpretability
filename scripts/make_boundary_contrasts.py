"""Problem-clustered uncertainty for LP4FM boundary contrasts.

The estimator matches the paper's table: macro-F1 is computed per
role/source/target/seed, then averaged within the close or Python-transfer
group. Bootstrap draws resample the shared XLCoST problem identifiers once and
reuse that draw across every cell and condition.

The intervals are conditional on the committed model runs. In particular, the
random-network condition contains one weight initialization and two probe
seeds, so these intervals do not measure variation across random initializations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/lp4fm/masked_probe"
ROLES = ("accumulator", "index_key", "iterator")
PAIRS = (
    ("javascript", "php"),
    ("php", "javascript"),
    ("javascript", "python"),
    ("php", "python"),
    ("python", "javascript"),
    ("python", "php"),
)
DEFAULT_BASE = "qwen25coder15b"


def conditions_for(slug_base: str) -> dict[str, str]:
    """The three condition slugs a model's masked-probe run produces."""
    return {
        "span_trained": slug_base,
        "context_trained": f"{slug_base}poolcontext16",
        "context_untrained": f"{slug_base}randominits0poolcontext16",
    }


def _macro_f1_from_confusion(counts: np.ndarray) -> np.ndarray:
    """Binary macro-F1 from [..., tn, fp, fn, tp] confusion counts."""
    tn, fp, fn, tp = np.moveaxis(counts, -1, 0)
    f1_negative = np.divide(
        2 * tn,
        2 * tn + fp + fn,
        out=np.zeros_like(tn, dtype=float),
        where=(2 * tn + fp + fn) != 0,
    )
    f1_positive = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) != 0,
    )
    return (f1_negative + f1_positive) / 2


def _load(
    conditions: dict[str, str],
) -> tuple[list[str], dict[tuple[str, str, str, str, int], np.ndarray]]:
    raw: dict[tuple[str, str, str, str, int], list[dict]] = {}
    problems: set[str] = set()
    for condition, slug in conditions.items():
        for role in ROLES:
            for source, target in PAIRS:
                path = DATA / f"probe_{role}_{source}_to_{target}_{slug}.json"
                artifact = json.loads(path.read_text())
                for row in artifact["test_predictions"]:
                    seed = int(row["seed"])
                    key = (condition, role, source, target, seed)
                    raw.setdefault(key, []).append(row)
                    problems.add(str(row["cluster"]))

    problem_list = sorted(problems)
    problem_index = {problem: index for index, problem in enumerate(problem_list)}
    cells: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    for key, rows in raw.items():
        confusion = np.zeros((len(problem_list), 4), dtype=np.int64)
        for row in rows:
            true_positive = row["y_true"] == "target"
            pred_positive = row["y_pred"] == "target"
            if not true_positive and not pred_positive:
                bucket = 0  # true negative
            elif not true_positive and pred_positive:
                bucket = 1  # false positive
            elif true_positive and not pred_positive:
                bucket = 2  # false negative
            else:
                bucket = 3  # true positive
            confusion[problem_index[str(row["cluster"])], bucket] += 1
        cells[key] = confusion
    return problem_list, cells


def _group_scores(
    weights: np.ndarray,
    cells: dict[tuple[str, str, str, str, int], np.ndarray],
    condition: str,
    python_pairs: bool,
) -> np.ndarray:
    scores = []
    for (cell_condition, _role, source, target, _seed), confusion in cells.items():
        if cell_condition != condition:
            continue
        is_python = "python" in (source, target)
        if is_python != python_pairs:
            continue
        scores.append(_macro_f1_from_confusion(weights @ confusion))
    return np.mean(np.stack(scores), axis=0)


def main(n_boot: int = 5000, seed: int = 0, slug_base: str = DEFAULT_BASE) -> int:
    conditions = conditions_for(slug_base)
    out = DATA / ("boundary_contrasts.csv" if slug_base == DEFAULT_BASE
                  else f"boundary_contrasts_{slug_base}.csv")
    problems, cells = _load(conditions)
    n_problems = len(problems)
    rng = np.random.default_rng(seed)
    boot = rng.multinomial(
        n_problems,
        np.full(n_problems, 1 / n_problems),
        size=n_boot,
    )
    weights = np.vstack([np.ones(n_problems, dtype=np.int64), boot])

    scores = {
        (condition, group): _group_scores(
            weights, cells, condition, python_pairs=(group == "python")
        )
        for condition in conditions
        for group in ("close", "python")
    }
    estimates = {
        "span_trained_boundary": (
            scores["span_trained", "python"] - scores["span_trained", "close"]
        ),
        "context_trained_boundary": (
            scores["context_trained", "python"]
            - scores["context_trained", "close"]
        ),
        "context_untrained_boundary": (
            scores["context_untrained", "python"]
            - scores["context_untrained", "close"]
        ),
        "context_boundary_difference_in_differences": (
            scores["context_trained", "python"]
            - scores["context_trained", "close"]
            - scores["context_untrained", "python"]
            + scores["context_untrained", "close"]
        ),
        "context_trained_minus_untrained_close": (
            scores["context_trained", "close"]
            - scores["context_untrained", "close"]
        ),
        "context_trained_minus_untrained_python": (
            scores["context_trained", "python"]
            - scores["context_untrained", "python"]
        ),
        "boundary_shift_after_excluding_occurrence": (
            scores["context_trained", "python"]
            - scores["context_trained", "close"]
            - scores["span_trained", "python"]
            + scores["span_trained", "close"]
        ),
    }

    rows = []
    for estimand, draws in estimates.items():
        low, high = np.quantile(draws[1:], [0.025, 0.975])
        rows.append({
            "estimand": estimand,
            "estimate": round(float(draws[0]), 4),
            "ci_low": round(float(low), 4),
            "ci_high": round(float(high), 4),
            "n_problem_ids": n_problems,
            "n_boot": n_boot,
            "interval": "problem-clustered percentile",
            "scope": "conditional on one random weight initialization",
        })

    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} contrasts -> {out.relative_to(ROOT)}")
    for row in rows:
        print(
            f"  {row['estimand']}: {row['estimate']:+.4f} "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        )
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug-base", default=DEFAULT_BASE,
                    help="model slug the condition suffixes attach to; a "
                         "non-default base writes boundary_contrasts_<base>.csv")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(main(n_boot=args.n_boot, seed=args.seed, slug_base=args.slug_base))
