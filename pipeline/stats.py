"""Statistics for the unified pipeline: frozen grouped splits, cluster
bootstrap CIs, control-task labels, provenance.

Ported from the boolean workstream (PROTOCOL.md; scripts/probe.py and
scripts/bootstrap_ci.py), where this machinery went through four rounds of
review. The resampling unit is always the PROGRAM: tokens and occurrences
within a program share a forward pass and are not independent.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def hash_split(program_ids, seed, fractions=(0.7, 0.1, 0.2)):
    """Deterministic per-program fold assignment (train/val/test).

    A program lands in the same fold for a given seed REGARDLESS of which
    other programs are present, so perturbed corpora that drop a few
    programs still share test folds with baseline — the property paired
    comparisons depend on. (GroupShuffleSplit reshuffles wholesale when the
    group list changes; measured once at 76 shared of ~400.)
    """
    f_tr, f_val, f_te = fractions
    idx = np.arange(len(program_ids))

    def bucket(g: str) -> float:
        return int(hashlib.sha1(f"{g}:{seed}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    fr = np.array([bucket(str(g)) for g in program_ids])
    return idx[fr >= f_te + f_val], idx[(fr >= f_te) & (fr < f_te + f_val)], idx[fr < f_te]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _cluster_indices(clusters):
    out: dict = {}
    for i, c in enumerate(clusters):
        out.setdefault(c, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def cluster_bootstrap_ci(y_true, y_pred, clusters, n_boot=1000, seed=0, alpha=0.05):
    """BCa bootstrap CI on macro F1, resampling whole programs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cmap = _cluster_indices(np.asarray(clusters))
    keys = list(cmap)
    observed = macro_f1(y_true, y_pred)

    def stat(idx):
        return macro_f1(y_true[idx], y_pred[idx])

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        boot[b] = stat(np.concatenate([cmap[keys[p]] for p in picked]))
    jack = np.array([stat(np.concatenate([cmap[k] for k in keys if k != d])) for d in keys])
    prop = np.clip(np.mean(boot < observed), 1e-9, 1 - 1e-9)
    z0 = norm.ppf(prop)
    jm = jack.mean()
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else np.sum((jm - jack) ** 3) / den
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def adj(z):
        return float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo = float(np.quantile(boot, np.clip(adj(z_lo), 0, 1)))
    hi = float(np.quantile(boot, np.clip(adj(z_hi), 0, 1)))
    return {"point": observed, "ci_low": lo, "ci_high": hi,
            "n_clusters": len(cmap), "cluster_warning": len(cmap) < 30}


def permute_labels_within_programs(labels, program_ids, seed=1234):
    """Random-label control: shuffle labels inside each program, preserving
    each program's positive count. Selectivity = real F1 - control F1."""
    labels = np.asarray(labels).copy()
    rng = np.random.default_rng(seed)
    for idx in _cluster_indices(np.asarray(program_ids)).values():
        labels[idx] = rng.permutation(labels[idx])
    return labels
