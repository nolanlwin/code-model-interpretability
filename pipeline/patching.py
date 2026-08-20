"""Activation-patching library for Python class_struct (v1).

Independent of pipeline.run_experiment. Hugging Face hidden_states[k] is the
residual entering decoder block k (k=0 is embeddings / block-0 input; the
final index is the final-norm output).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .patching_prompts import (
    EVAL_SHA256,
    FALSE_COMPLETION,
    N_EVAL,
    SMOKE_SHA256,
    TRUE_COMPLETION,
    sha256_file,
)
EXPERIMENT = "class_struct_activation_patching_v1"
LANGUAGE = "Python"
RANDOM_SEED = 20260818
DATASET_REVISION = "912f6e468df675f11237f3c9b7635f09a6a95584"
MAX_PROMPT_TOKENS = 128
N_BOOT = 10_000
BLOCK_SIZE = 64
DRIFT_TAU_FLOOR = 1e-4
IDENTITY_DRIFT_MULT = 10
# fp16 True/False logit differences move in steps of 2^-6 = 0.015625 around
# typical magnitudes. A same-source rewrite can therefore wiggle by 1–2 ULPs
# even when the within-batch unpatched repeat is exact. This floor is only
# for identity/no-op checks; causal_gate keeps using drift_tau() so the
# 0.10-logit effect bar does not silently rise.
FP16_IDENTITY_TAU = 0.05

TRUE_TOKEN_IDS = {
    "Qwen/Qwen2.5-1.5B": 3007,
    "Qwen/Qwen2.5-Coder-1.5B": 3007,
    "bigcode/starcoder2-7b": 2969,
}
FALSE_TOKEN_IDS = {
    "Qwen/Qwen2.5-1.5B": 3557,
    "Qwen/Qwen2.5-Coder-1.5B": 3557,
    "bigcode/starcoder2-7b": 3208,
}

MODELS = {
    "Qwen/Qwen2.5-1.5B": {
        "revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "n_blocks": 28,
        "n_hidden": 29,
        "probe_index": 18,
        "microbatch": 32,
        "item_forward_ceiling": 90_000,
        "prior_f1": 0.9832,
        "prior_acc": 0.9990,
        "expanded": True,
        "fp32_gpu": "L4",
    },
    "Qwen/Qwen2.5-Coder-1.5B": {
        "revision": "df3ce67c0e24480f20468b6ef2894622d69eb73b",
        "n_blocks": 28,
        "n_hidden": 29,
        "probe_index": 8,
        "microbatch": 32,
        "item_forward_ceiling": 26_000,
        "prior_f1": 0.9820,
        "prior_acc": 0.9989,
        "expanded": False,
        "fp32_gpu": "L4",
    },
    "bigcode/starcoder2-7b": {
        "revision": "bb9afde76d7945da5745592525db122d4d729eb1",
        "n_blocks": 32,
        "n_hidden": 33,
        "probe_index": 5,
        "microbatch": 8,
        "item_forward_ceiling": 26_000,
        "prior_f1": 0.9828,
        "prior_acc": 0.9990,
        "expanded": False,
        "fp32_gpu": "L40S",
    },
}

DIRECTIONS = ("denoise", "noise")
NAME_SPANS = ("query_name", "declaration_name")
L4_USD_PER_HOUR = 0.80
L40S_USD_PER_HOUR = 1.95
HOOK_MATCH_ATOL = 1e-4
CONTROLLER_HARD_STOP_USD = 50.0
EVAL_PROMPT_SHA256 = EVAL_SHA256
SMOKE_PROMPT_SHA256 = SMOKE_SHA256


class PatchingError(ValueError):
    """Hard pre-GPU / protocol failure."""


def hash_split(program_ids, seed, fractions=(0.7, 0.1, 0.2)):
    """Deterministic program-group split without importing the probe stack."""
    f_tr, f_val, f_te = fractions
    if not np.isclose(f_tr + f_val + f_te, 1.0):
        raise PatchingError("split fractions must sum to one")
    idx = np.arange(len(program_ids))

    def bucket(group: str) -> float:
        raw = hashlib.sha1(f"{group}:{seed}".encode()).hexdigest()[:8]
        return int(raw, 16) / 0xFFFFFFFF

    fr = np.asarray([bucket(str(g)) for g in program_ids])
    return (
        idx[fr >= f_te + f_val],
        idx[(fr >= f_te) & (fr < f_te + f_val)],
        idx[fr < f_te],
    )


def macro_f1(y_true, y_pred) -> float:
    """Binary macro-F1 used by probe fitting, kept dependency-light."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for label in (0, 1):
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        den = 2 * tp + fp + fn
        scores.append(0.0 if den == 0 else 2 * tp / den)
    return float(np.mean(scores))


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_sha256(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(p).resolve() for p in paths):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def default_bundle_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        root / "pipeline" / "patching.py",
        root / "pipeline" / "patching_prompts.py",
        root / "pipeline" / "run_patching.py",
        root / "scripts" / "modal_patching.py",
        root / "data" / "patching" / "class_struct_python_v1.jsonl",
        root / "data" / "patching" / "class_struct_python_smoke_v1.jsonl",
    ]


# ---------------------------------------------------------------------------
# Token spans
# ---------------------------------------------------------------------------

def char_span_to_token_index(offset_mapping, span: Sequence[int]) -> int:
    s, e = int(span[0]), int(span[1])
    hits = [i for i, (a, b) in enumerate(offset_mapping) if a < e and s < b]
    if len(hits) != 1:
        raise PatchingError(
            f"span {list(span)!r} hit {len(hits)} tokens {hits}; need exactly one"
        )
    return hits[0]


def encode_offsets(tokenizer, text: str):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), list(enc["offset_mapping"])


def discover_completion_ids(tokenizer, model_id: str | None = None) -> tuple[int, int]:
    true_ids = tokenizer.encode(TRUE_COMPLETION, add_special_tokens=False)
    false_ids = tokenizer.encode(FALSE_COMPLETION, add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise PatchingError(
            f"{TRUE_COMPLETION!r}/{FALSE_COMPLETION!r} are not single tokens "
            f"(got {true_ids} / {false_ids})"
        )
    if true_ids[0] == false_ids[0]:
        raise PatchingError("True/False token ids are not distinct")
    if model_id:
        exp_t, exp_f = TRUE_TOKEN_IDS[model_id], FALSE_TOKEN_IDS[model_id]
        if true_ids[0] != exp_t or false_ids[0] != exp_f:
            raise PatchingError(
                f"{model_id} completion ids {true_ids[0]}/{false_ids[0]} "
                f"!= expected {exp_t}/{exp_f}"
            )
    return int(true_ids[0]), int(false_ids[0])


def pair_token_view(tokenizer, row: dict, side: str) -> dict:
    prompt = row["clean_prompt"] if side == "clean" else row["corrupt_prompt"]
    ids, offs = encode_offsets(tokenizer, prompt)
    spans = {}
    for key in ("keyword_char_span", "declaration_name_char_span",
                "query_name_char_span", "placebo_char_span"):
        spans[key] = char_span_to_token_index(offs, row[key][side])
    return {"ids": ids, "offsets": offs, "indices": spans, "n": len(ids)}


def validate_pair_tokenizer(tokenizer, row: dict, model_id: str | None = None) -> dict:
    clean = pair_token_view(tokenizer, row, "clean")
    corrupt = pair_token_view(tokenizer, row, "corrupt")
    if clean["n"] != corrupt["n"]:
        raise PatchingError(f"{row['pair_id']}: token counts {clean['n']} vs {corrupt['n']}")
    if clean["n"] > MAX_PROMPT_TOKENS:
        raise PatchingError(f"{row['pair_id']}: {clean['n']} tokens > {MAX_PROMPT_TOKENS}")
    diffs = [i for i, (a, b) in enumerate(zip(clean["ids"], corrupt["ids"])) if a != b]
    if diffs != [clean["indices"]["keyword_char_span"]]:
        raise PatchingError(
            f"{row['pair_id']}: id diffs at {diffs}, expected keyword index "
            f"{clean['indices']['keyword_char_span']}"
        )
    if corrupt["indices"]["keyword_char_span"] != clean["indices"]["keyword_char_span"]:
        raise PatchingError(f"{row['pair_id']}: keyword token index shifted")
    for key in ("declaration_name_char_span", "query_name_char_span", "placebo_char_span"):
        if clean["indices"][key] != corrupt["indices"][key]:
            raise PatchingError(f"{row['pair_id']}: {key} token index misaligned")
    discover_completion_ids(tokenizer, model_id)
    for prompt in (row["clean_prompt"], row["corrupt_prompt"]):
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        for comp in (TRUE_COMPLETION, FALSE_COMPLETION):
            full = tokenizer.encode(prompt + comp, add_special_tokens=False)
            if full[: len(prefix)] != prefix:
                raise PatchingError(
                    f"{row['pair_id']}: encoding prompt+{comp!r} does not preserve prefix"
                )
            if len(full) != len(prefix) + 1:
                raise PatchingError(
                    f"{row['pair_id']}: completion {comp!r} is not exactly one extra token"
                )
    return {
        "n_tokens": clean["n"],
        "keyword_index": clean["indices"]["keyword_char_span"],
        "declaration_index": clean["indices"]["declaration_name_char_span"],
        "query_index": clean["indices"]["query_name_char_span"],
        "placebo_index": clean["indices"]["placebo_char_span"],
    }


def left_pad_index(unpadded_index: int, pad_len: int) -> int:
    if unpadded_index < 0 or pad_len < 0:
        raise PatchingError("negative token/pad index")
    return int(unpadded_index) + int(pad_len)


def span_index(view: dict, span: str) -> int:
    mapping = {
        "query_name": "query_name_char_span",
        "declaration_name": "declaration_name_char_span",
        "placebo": "placebo_char_span",
        "keyword": "keyword_char_span",
    }
    return view["indices"][mapping[span]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def logit_diff(logits_last, true_id: int, false_id: int) -> float:
    arr = np.asarray(logits_last, dtype=np.float64)
    return float(arr[true_id] - arr[false_id])


def denoise_effect(d_function: float, d_patched_function: float) -> float:
    return float(d_patched_function - d_function)


def noise_effect(d_class: float, d_patched_class: float) -> float:
    return float(d_class - d_patched_class)


def class_function_gap(d_class: float, d_function: float) -> float:
    return float(d_class - d_function)


def signed_effect(direction: str, d_class: float, d_function: float, d_patched: float) -> float:
    if direction == "denoise":
        return denoise_effect(d_function, d_patched)
    if direction == "noise":
        return noise_effect(d_class, d_patched)
    raise PatchingError(f"unknown direction {direction}")


def ratio_of_means(effects: Sequence[float], gaps: Sequence[float]) -> float:
    eff = np.asarray(effects, dtype=np.float64)
    gap = np.asarray(gaps, dtype=np.float64)
    den = float(gap.mean())
    if den == 0:
        return float("nan")
    return float(eff.mean() / den)


def mean_of_ratios(effects: Sequence[float], gaps: Sequence[float]) -> float:
    eff = np.asarray(effects, dtype=np.float64)
    gap = np.asarray(gaps, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(eff / gap))


def _bca_from_boot(boot: np.ndarray, observed: float, jack: np.ndarray, alpha: float):
    from statistics import NormalDist

    norm = NormalDist()

    boot = boot[np.isfinite(boot)]
    if boot.size == 0 or jack.size < 2:
        return float("nan"), float("nan")
    prop = np.clip(np.mean(boot < observed), 1e-9, 1 - 1e-9)
    z0 = float(norm.inv_cdf(prop))
    jm = float(jack.mean())
    num = float(np.sum((jm - jack) ** 3))
    den = 6.0 * (float(np.sum((jm - jack) ** 2)) ** 1.5)
    a = 0.0 if den == 0 else num / den
    z_lo = float(norm.inv_cdf(alpha / 2))
    z_hi = float(norm.inv_cdf(1 - alpha / 2))

    def adj(z):
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))))

    lo = float(np.quantile(boot, np.clip(adj(z_lo), 0, 1)))
    hi = float(np.quantile(boot, np.clip(adj(z_hi), 0, 1)))
    return lo, hi


def clustered_mean_ci(values, clusters, n_boot: int = N_BOOT, seed: int = RANDOM_SEED,
                      alpha: float = 0.05) -> dict:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    cmap: dict[Any, list[int]] = {}
    for i, c in enumerate(clusters):
        cmap.setdefault(c, []).append(i)
    keys = list(cmap)
    observed = float(values.mean()) if values.size else float("nan")
    if len(keys) < 2:
        return {"point": observed, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_clusters": len(keys), "cluster_warning": True}

    def stat(idx):
        return float(values[np.asarray(idx)].mean())

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        boot[b] = stat(np.concatenate([cmap[keys[p]] for p in picked]))
    jack = np.array([
        stat(np.concatenate([cmap[k] for k in keys if k != d])) for d in keys
    ])
    lo, hi = _bca_from_boot(boot, observed, jack, alpha)
    return {"point": observed, "ci_low": lo, "ci_high": hi,
            "n_clusters": len(keys), "cluster_warning": len(keys) < 30}


def clustered_ratio_ci(effects, gaps, clusters, n_boot: int = N_BOOT,
                       seed: int = RANDOM_SEED, alpha: float = 0.05) -> dict:
    """BCa CI for ratio-of-means while resampling whole structural clusters."""
    effects = np.asarray(effects, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    clusters = np.asarray(clusters)
    if not (len(effects) == len(gaps) == len(clusters)):
        raise PatchingError("effects, gaps, and clusters must have equal lengths")
    cmap: dict[Any, list[int]] = {}
    for i, cluster in enumerate(clusters):
        cmap.setdefault(cluster, []).append(i)
    keys = list(cmap)
    observed = ratio_of_means(effects, gaps)
    if len(keys) < 2:
        return {
            "point": observed, "ci_low": float("nan"), "ci_high": float("nan"),
            "n_clusters": len(keys), "cluster_warning": True,
            "denominator_crosses_zero": True,
        }

    def stat(indices) -> float:
        idx = np.asarray(indices, dtype=np.int64)
        return ratio_of_means(effects[idx], gaps[idx])

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    boot_den = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([cmap[keys[p]] for p in picked])
        boot[b] = stat(idx)
        boot_den[b] = float(gaps[idx].mean())
    jack = np.asarray([
        stat(np.concatenate([cmap[k] for k in keys if k != dropped]))
        for dropped in keys
    ])
    lo, hi = _bca_from_boot(boot, observed, jack, alpha)
    denominator_crosses = bool(np.min(boot_den) <= 0 <= np.max(boot_den))
    return {
        "point": observed,
        "ci_low": lo,
        "ci_high": hi,
        "n_clusters": len(keys),
        "cluster_warning": len(keys) < 30,
        "denominator_crosses_zero": denominator_crosses,
    }


def clustered_spearman(x, y, clusters, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    clusters = np.asarray(clusters)

    def rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        cursor = 0
        while cursor < len(values):
            end = cursor + 1
            while end < len(values) and values[order[end]] == values[order[cursor]]:
                end += 1
            ranks[order[cursor:end]] = (cursor + end - 1) / 2 + 1
            cursor = end
        return ranks

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2:
            return float("nan")
        ra, rb = rankdata(a), rankdata(b)
        if np.std(ra) == 0 or np.std(rb) == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    observed = spearman(x, y)
    cmap: dict[Any, list[int]] = {}
    for i, c in enumerate(clusters):
        cmap.setdefault(c, []).append(i)
    keys = list(cmap)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([cmap[keys[p]] for p in picked])
        boot[b] = spearman(x[idx], y[idx])
    jack = np.array([
        spearman(
            x[np.concatenate([cmap[k] for k in keys if k != d])],
            y[np.concatenate([cmap[k] for k in keys if k != d])],
        )
        for d in keys
    ])
    lo, hi = _bca_from_boot(boot, observed, jack, 0.05)
    return {"point": observed, "ci_low": lo, "ci_high": hi}


def exact_pair_join(groups: dict[str, Sequence[dict]], *,
                    expected_pair_ids: Iterable[str] | None = None) -> list[dict[str, dict]]:
    """Join named result groups by pair_id, rejecting duplicates and missing rows."""
    indexed: dict[str, dict[str, dict]] = {}
    for label, rows in groups.items():
        bucket: dict[str, dict] = {}
        for row in rows:
            pair_id = str(row["pair_id"])
            if pair_id in bucket:
                raise PatchingError(f"duplicate {label} row for pair {pair_id}")
            bucket[pair_id] = row
        indexed[label] = bucket
    if expected_pair_ids is None:
        expected = set.intersection(*(set(v) for v in indexed.values())) if indexed else set()
    else:
        expected = {str(x) for x in expected_pair_ids}
    for label, bucket in indexed.items():
        missing = expected - set(bucket)
        extra = set(bucket) - expected
        if missing or extra:
            raise PatchingError(
                f"pair join mismatch for {label}: {len(missing)} missing, {len(extra)} extra"
            )
    return [{label: indexed[label][pid] for label in indexed} for pid in sorted(expected)]


def leave_one_name_out(values, names) -> dict:
    values = np.asarray(values, dtype=np.float64)
    names = np.asarray(names)
    out = {}
    for name in sorted(set(names.tolist())):
        mask = names != name
        out[str(name)] = float(values[mask].mean()) if mask.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# Random control
# ---------------------------------------------------------------------------

def cell_rng_key(pair_id: str, layer: int, span: str, direction: str,
                 control: str, random_seed: int = RANDOM_SEED) -> str:
    return stable_hash(pair_id, str(layer), span, direction, control, str(random_seed))


def random_control_noise(source: np.ndarray, dest: np.ndarray, cell_key: str,
                         random_seed: int = RANDOM_SEED) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    dest = np.asarray(dest, dtype=np.float32)
    delta = source - dest
    seed_int = int(stable_hash(str(random_seed), cell_key)[:16], 16) % (2 ** 32)
    rng = np.random.default_rng(seed_int)
    g = rng.standard_normal(delta.shape).astype(np.float32)
    dflat = delta.reshape(-1)
    gflat = g.reshape(-1)
    dn = float(np.dot(dflat, dflat))
    if dn < 1e-20:
        return np.zeros_like(delta)
    gflat = gflat - (float(np.dot(gflat, dflat)) / dn) * dflat
    gn = float(np.linalg.norm(gflat))
    target = float(np.linalg.norm(dflat))
    if gn < 1e-20:
        return np.zeros_like(delta)
    return (gflat * (target / gn)).reshape(delta.shape)


def inject_random(source, dest, cell_key: str, dtype=np.float16,
                  random_seed: int = RANDOM_SEED, rel_tol: float = 0.01):
    noise = random_control_noise(np.asarray(source), np.asarray(dest), cell_key, random_seed)
    dest_f = np.asarray(dest, dtype=np.float32)
    out32 = dest_f + noise
    cast = out32.astype(dtype, copy=False)
    n_src = float(np.linalg.norm(noise))
    n_cast = float(np.linalg.norm(np.asarray(cast, dtype=np.float32) - dest_f))
    if n_src > 1e-8 and abs(n_cast / n_src - 1.0) > rel_tol:
        raise PatchingError(
            f"random-control post-cast norm ratio {n_cast / n_src:.4f} exceeds {rel_tol}"
        )
    return cast


def numpy_dtype(dtype: str):
    normalized = str(dtype).lower()
    if normalized in ("float16", "fp16", "torch.float16"):
        return np.float16
    if normalized in ("float32", "fp32", "torch.float32"):
        return np.float32
    raise PatchingError(f"unsupported experiment dtype {dtype!r}")


# ---------------------------------------------------------------------------
# Probe coefficients
# ---------------------------------------------------------------------------

def raw_residual_params(scaler_mean, scaler_scale, coef, intercept) -> tuple[np.ndarray, float]:
    mean = np.asarray(scaler_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(scaler_scale, dtype=np.float64).reshape(-1)
    w = np.asarray(coef, dtype=np.float64).reshape(-1)
    b = float(np.asarray(intercept).reshape(-1)[0])
    w_raw = w / scale
    b_raw = b - float(np.dot(w_raw, mean))
    return w_raw, b_raw


def probe_margin(hidden, w_raw, b_raw) -> float:
    h = np.asarray(hidden, dtype=np.float64).reshape(-1)
    return float(np.dot(h, w_raw) + b_raw)


def save_probe_npz(path: Path, *, scaler_mean, scaler_scale, coef, intercept, classes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scaler_mean=np.asarray(scaler_mean, dtype=np.float64),
        scaler_scale=np.asarray(scaler_scale, dtype=np.float64),
        classifier_coef=np.asarray(coef, dtype=np.float64),
        classifier_intercept=np.asarray(intercept, dtype=np.float64),
        classifier_classes=np.asarray(classes),
    )
    return sha256_file(path)


def load_probe_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    required = {
        "scaler_mean", "scaler_scale", "classifier_coef",
        "classifier_intercept", "classifier_classes",
    }
    missing = required - set(data)
    if missing:
        raise PatchingError(f"probe {path} missing arrays {sorted(missing)}")
    classes = np.asarray(data["classifier_classes"]).reshape(-1).tolist()
    if classes != [0, 1]:
        raise PatchingError(f"probe {path} classes {classes} != [0, 1]")
    dimensions = {
        np.asarray(data["scaler_mean"]).size,
        np.asarray(data["scaler_scale"]).size,
        np.asarray(data["classifier_coef"]).size,
    }
    if len(dimensions) != 1:
        raise PatchingError(f"probe {path} parameter dimensions disagree: {dimensions}")
    if np.any(np.asarray(data["scaler_scale"]) <= 0):
        raise PatchingError(f"probe {path} has non-positive scaler scale")
    if not all(np.all(np.isfinite(np.asarray(data[k]))) for k in required):
        raise PatchingError(f"probe {path} contains non-finite parameters")
    w_raw, b_raw = raw_residual_params(
        data["scaler_mean"], data["scaler_scale"],
        data["classifier_coef"], data["classifier_intercept"],
    )
    data["w_raw"] = w_raw
    data["b_raw"] = b_raw
    return data


def load_probe_artifact(path: Path, model_id: str, *, require_metadata: bool = True,
                        expected_configuration_sha256: str | None = None,
                        expected_code_sha256: str | None = None) -> dict:
    """Load a portable probe and validate it against its model/revision/layer."""
    path = Path(path)
    data = load_probe_npz(path)
    metadata_path = path.parent / "probe_meta.json"
    if not metadata_path.is_file():
        if require_metadata:
            raise PatchingError(f"missing probe metadata {metadata_path}")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text())
    expected = MODELS[model_id]
    checks = {
        "model_id": model_id,
        "model_revision": expected["revision"],
        "layer": expected["probe_index"],
        "dataset_revision": DATASET_REVISION,
        "prompt_sha256": EVAL_PROMPT_SHA256,
        "smoke_prompt_sha256": SMOKE_PROMPT_SHA256,
        "model_dtype": "float16",
    }
    if expected_configuration_sha256 is not None:
        checks["configuration_sha256"] = expected_configuration_sha256
    if expected_code_sha256 is not None:
        checks["code_sha256"] = expected_code_sha256
    for key, value in checks.items():
        if key in metadata and metadata[key] != value:
            raise PatchingError(
                f"probe metadata {key}={metadata[key]!r} != expected {value!r}"
            )
        if require_metadata and key not in metadata:
            raise PatchingError(f"probe metadata missing {key}")
    if "artifact_sha256" in metadata:
        digest = sha256_file(path)
        if metadata["artifact_sha256"] != digest:
            raise PatchingError(
                f"probe checksum {digest} != metadata {metadata['artifact_sha256']}"
            )
    hidden_size = int(np.asarray(data["w_raw"]).size)
    if metadata.get("hidden_size") not in (None, hidden_size):
        raise PatchingError(
            f"probe hidden size {hidden_size} != metadata {metadata['hidden_size']}"
        )
    data["metadata"] = metadata
    data["layer"] = expected["probe_index"]
    data["model_id"] = model_id
    return data


def fit_probe_link(hidden, labels, program_ids, seeds=(0, 1, 2, 3, 4), C=1.0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(hidden, dtype=np.float32)
    y = np.asarray(labels)
    pids = np.asarray(program_ids)
    fits = []
    f1s, accs = [], []
    for seed in seeds:
        tr, val, te = hash_split(pids, seed)
        if min(len(tr), len(val), len(te)) == 0 or len(set(y[tr].tolist())) < 2:
            raise PatchingError(f"unusable split at seed {seed}")
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, C=C,
        )
        clf.fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[te]))
        f1s.append(macro_f1(y[te], pred))
        accs.append(float(accuracy_score(y[te], pred)))
        if list(clf.classes_) != [0, 1]:
            raise PatchingError(f"classes {clf.classes_} != [0, 1]")
        for v in (scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_):
            if not np.all(np.isfinite(v)):
                raise PatchingError("non-finite probe parameters")
        fits.append({
            "seed": seed,
            "scaler_mean": scaler.mean_.copy(),
            "scaler_scale": scaler.scale_.copy(),
            "coef": clf.coef_.copy(),
            "intercept": clf.intercept_.copy(),
            "classes": np.asarray(clf.classes_),
            "test_f1": f1s[-1],
            "test_acc": accs[-1],
        })
    return {
        "fits": fits,
        "primary": fits[0],
        "test_f1_mean": float(np.mean(f1s)),
        "test_acc_mean": float(np.mean(accs)),
    }


def probe_matches_prior(fit: dict, model_id: str, atol: float = 0.002) -> dict:
    meta = MODELS[model_id]
    checks = {
        "f1": abs(fit["test_f1_mean"] - meta["prior_f1"]) <= atol,
        "acc": abs(fit["test_acc_mean"] - meta["prior_acc"]) <= atol,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "got_f1": fit["test_f1_mean"], "got_acc": fit["test_acc_mean"],
            "prior_f1": meta["prior_f1"], "prior_acc": meta["prior_acc"]}


# ---------------------------------------------------------------------------
# Sweep schedule
# ---------------------------------------------------------------------------

def intervention_cells(model_id: str) -> list[tuple[int, str, str, str]]:
    """Complete preregistered fp16 schedule for one model."""
    meta = MODELS[model_id]
    n_hidden, probe, expanded = meta["n_hidden"], meta["probe_index"], meta["expanded"]
    cells: list[tuple[int, str, str, str]] = []

    def add(layer, span, direction, control):
        cells.append((int(layer), span, direction, control))

    for layer in range(n_hidden):
        for d in DIRECTIONS:
            add(layer, "query_name", d, "target")
    battery = list(range(n_hidden)) if expanded else [probe]
    for layer in battery:
        for span in ("declaration_name", "placebo"):
            for d in DIRECTIONS:
                add(layer, span, d, "target")
        for span in NAME_SPANS:
            for d in DIRECTIONS:
                add(layer, span, d, "random")
    for span in NAME_SPANS:
        for d in DIRECTIONS:
            add(probe, span, d, "same_source")
    return cells


def behavior_cells(model_id: str) -> list[tuple[int, str, str, str]]:
    """Behavioral gate needs only unpatched class/function baselines."""
    if model_id not in MODELS:
        raise PatchingError(f"unknown model {model_id}")
    return []


def primary_cells(model_id: str) -> list[tuple[int, str, str, str]]:
    """Small schedule sufficient for identity checks and the causal gate."""
    probe = MODELS[model_id]["probe_index"]
    cells: list[tuple[int, str, str, str]] = []
    for direction in DIRECTIONS:
        cells.extend([
            (probe, "query_name", direction, "target"),
            (probe, "placebo", direction, "target"),
            (probe, "query_name", direction, "random"),
            (probe, "query_name", direction, "same_source"),
            (0, "query_name", direction, "target"),
        ])
    return list(dict.fromkeys(cells))


def core_cells(model_id: str) -> list[tuple[int, str, str, str]]:
    """Cross-model all-layer curve: query-name target in both directions."""
    meta = MODELS[model_id]
    return [
        (layer, "query_name", direction, "target")
        for layer in range(meta["n_hidden"])
        for direction in DIRECTIONS
    ]


def expanded_cells(model_id: str) -> list[tuple[int, str, str, str]]:
    """Full-schedule cells absent from the primary and core phases."""
    already = set(primary_cells(model_id)) | set(core_cells(model_id))
    return [cell for cell in intervention_cells(model_id) if cell not in already]


def fp32_cells(layers: Sequence[int]) -> list[tuple[int, str, str, str]]:
    """Precision sensitivity battery at exactly the three frozen layers."""
    unique = list(dict.fromkeys(int(layer) for layer in layers))
    if len(unique) != 3:
        raise PatchingError(f"fp32 replication requires exactly 3 layers, got {unique}")
    cells = []
    for layer in unique:
        for direction in DIRECTIONS:
            cells.extend([
                (layer, "query_name", direction, "target"),
                (layer, "placebo", direction, "target"),
                (layer, "query_name", direction, "random"),
            ])
    return cells


def schedule_cells(model_id: str, schedule: str,
                   layers: Sequence[int] | None = None) -> list[tuple[int, str, str, str]]:
    schedules = {
        "behavior": behavior_cells,
        "primary": primary_cells,
        "core": core_cells,
        "expanded": expanded_cells,
        "full": intervention_cells,
    }
    if schedule == "fp32":
        if layers is None:
            raise PatchingError("fp32 schedule requires frozen --layers")
        for layer in layers:
            if not 0 <= int(layer) < MODELS[model_id]["n_hidden"]:
                raise PatchingError(f"fp32 layer {layer} out of range for {model_id}")
        return fp32_cells(layers)
    if schedule not in schedules:
        raise PatchingError(f"unknown schedule {schedule!r}")
    return schedules[schedule](model_id)


def select_fp32_layers(rows: Sequence[dict], model_id: str, *,
                       configuration_sha256: str | None = None,
                       source_dtype: str = "float16") -> list[int]:
    """Freeze probe layer plus the two strongest non-endpoint fp16 query cells."""
    meta = MODELS[model_id]
    filtered = [
        row for row in rows
        if row.get("model_id") == model_id
        and row.get("dtype") == source_dtype
        and row.get("span") == "query_name"
        and row.get("control") == "target"
        and (configuration_sha256 is None
             or row.get("configuration_sha256") == configuration_sha256)
    ]
    scores = []
    for layer in range(meta["n_hidden"]):
        if layer in (0, meta["n_hidden"] - 1, meta["probe_index"]):
            continue
        denoise = [r for r in filtered if r.get("layer") == layer and r.get("direction") == "denoise"]
        noise = [r for r in filtered if r.get("layer") == layer and r.get("direction") == "noise"]
        expected = {r["pair_id"] for r in denoise + noise}
        joined = exact_pair_join(
            {"denoise": denoise, "noise": noise}, expected_pair_ids=expected,
        )
        if not joined:
            continue
        sym = np.asarray([
            (item["denoise"]["signed_effect"] + item["noise"]["signed_effect"]) / 2
            for item in joined
        ], dtype=np.float64)
        scores.append((float(abs(sym.mean())), layer))
    if len(scores) < 2:
        raise PatchingError(f"need two complete non-endpoint layers for fp32 selection, got {scores}")
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [meta["probe_index"], scores[0][1], scores[1][1]]


def smoke_cells(probe_index: int) -> list[tuple[int, str, str, str]]:
    cells = []
    for d in DIRECTIONS:
        cells.append((probe_index, "query_name", d, "target"))
        cells.append((probe_index, "placebo", d, "target"))
        cells.append((probe_index, "query_name", d, "random"))
        cells.append((probe_index, "query_name", d, "same_source"))
        cells.append((0, "query_name", d, "target"))
    return cells


def n_source_forwards(n_pairs: int = N_EVAL) -> int:
    return n_pairs * 4


def estimate_forwards(model_id: str, n_pairs: int = N_EVAL) -> dict:
    cells = intervention_cells(model_id)
    n_cells = len(cells)
    item = n_pairs * n_cells + n_source_forwards(n_pairs)
    meta = MODELS[model_id]
    mb = meta["microbatch"]
    # clean/corrupt source forwards with hidden states, plus logits-only repeats
    full_batched = int(np.ceil(n_pairs / mb)) * 4
    full_batched += int(np.ceil(n_pairs / mb)) * n_cells
    gpu = "L4"
    rate = L40S_USD_PER_HOUR if gpu == "L40S" else L4_USD_PER_HOUR
    staged_sets = []
    already: set[tuple[int, str, str, str]] = set()
    staged_item = n_pairs * 4  # behavioral clean/function plus drift repeats
    staged_batched = int(np.ceil(n_pairs / mb)) * 4
    for schedule in (primary_cells(model_id), core_cells(model_id), expanded_cells(model_id)):
        delta = set(schedule) - already
        if delta:
            staged_item += n_pairs * (2 + len(delta))  # clean/function source activations
            staged_sets.append(len(delta))
            staged_batched += int(np.ceil(n_pairs / mb)) * 2
            for direction in DIRECTIONS:
                n_direction = sum(cell[2] == direction for cell in delta)
                staged_batched += int(np.ceil(n_pairs * n_direction / mb))
        already.update(schedule)
    fp32_top3_item = n_pairs * (4 + len(fp32_cells([0, 1, 2])))
    hours = max((staged_batched * 4.0) / 3600.0, staged_item / 8000.0)
    fp32_rate = L40S_USD_PER_HOUR if meta["fp32_gpu"] == "L40S" else L4_USD_PER_HOUR
    return {
        "model_id": model_id,
        "n_pairs": n_pairs,
        "n_layers": meta["n_hidden"],
        "n_intervention_cells": n_cells,
        "item_forwards": int(item),
        "staged_item_forwards": int(staged_item),
        "staged_new_cells": staged_sets,
        "fp32_top3_item_forwards": int(fp32_top3_item),
        "batched_forwards": int(full_batched),
        "staged_batched_forwards": int(staged_batched),
        "approx_padded_token_forwards": int(staged_item * MAX_PROMPT_TOKENS),
        "expected_keys": int(n_pairs * n_cells + n_pairs * 4),
        "ceiling": meta["item_forward_ceiling"],
        "gpu": gpu,
        "fp16_staged_gpu_only_conservative_hours": hours,
        "fp16_staged_gpu_only_conservative_usd": hours * rate,
        "fp32_top3_gpu": meta["fp32_gpu"],
        "fp32_top3_conservative_usd": (fp32_top3_item / 8000.0) * fp32_rate,
        "refused": staged_item > meta["item_forward_ceiling"],
    }


def estimate_all(n_pairs: int = N_EVAL) -> dict:
    per = {m: estimate_forwards(m, n_pairs) for m in MODELS}
    refused = [k for k, v in per.items() if v["refused"]]
    if refused:
        raise PatchingError(
            "forward-count ceiling exceeded: "
            + ", ".join(f"{k}={per[k]['staged_item_forwards']}" for k in refused)
        )
    return per


def projected_spend_usd(cost_ledger: dict, extra: float) -> float:
    return float(cost_ledger.get("spent_usd", 0.0)) + float(extra)


# ---------------------------------------------------------------------------
# Result rows / checkpoints
# ---------------------------------------------------------------------------

PRIMARY_KEY_FIELDS = (
    "prompt_sha256", "configuration_sha256", "model_id", "model_revision",
    "dtype", "pair_id", "layer", "span", "direction", "control", "random_seed",
)


def primary_key(row: dict) -> tuple:
    return tuple(row[k] for k in PRIMARY_KEY_FIELDS)


def make_result_row(**kwargs) -> dict:
    base = {k: None for k in (
        "run_id", "cluster_id", "name", "source_D", "destination_D", "patched_D",
        "signed_effect", "class_function_gap", "source_probe_margin",
        "destination_probe_margin", "patched_probe_margin",
        "source_probe_declaration_margin", "baseline_drift",
        "batch_size", "attempt_id", "timestamp",
    )}
    base.update({
        "random_seed": RANDOM_SEED,
        "dtype": "float16",
        "timestamp": time.time(),
        "attempt_id": 0,
    })
    base.update(kwargs)
    missing = [k for k in PRIMARY_KEY_FIELDS if base.get(k) is None]
    if missing:
        raise PatchingError(f"result row missing primary fields {missing}")
    return base


def atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def write_json_atomic(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def chunk_payload(rows: list[dict]) -> bytes:
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    return body.encode("utf-8")


def finalize_chunk(path: Path, rows: list[dict]) -> dict:
    keys = [primary_key(r) for r in rows]
    if len(keys) != len(set(keys)):
        raise PatchingError(f"duplicate keys in chunk {path}")
    payload = chunk_payload(rows)
    checksum = sha256_bytes(payload)
    atomic_write(path, payload)
    return {"path": str(path), "n_rows": len(rows), "sha256": checksum}


def read_chunk(path: Path, expected_sha: str | None = None) -> list[dict]:
    data = Path(path).read_bytes()
    digest = sha256_bytes(data)
    if expected_sha and digest != expected_sha:
        raise PatchingError(f"checksum mismatch {path}: {digest} != {expected_sha}")
    try:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchingError(f"invalid JSONL chunk {path}") from exc
    keys = [primary_key(r) for r in rows]
    if len(keys) != len(set(keys)):
        raise PatchingError(f"duplicate keys in {path}")
    return rows


@dataclass
class Lease:
    holder: str
    heartbeat: float
    function_id: str | None = None
    stale_after_s: float = 600.0
    released: bool = False

    def fresh(self, now: float | None = None) -> bool:
        if self.released:
            return False
        now = time.time() if now is None else now
        return (now - self.heartbeat) < self.stale_after_s


def load_lease(path: Path) -> Lease | None:
    p = Path(path)
    if not p.is_file():
        return None
    obj = json.loads(p.read_text())
    fields = {
        k: obj[k] for k in (
            "holder", "heartbeat", "function_id", "stale_after_s", "released",
        ) if k in obj
    }
    return Lease(**fields)


def acquire_lease(path: Path, holder: str, function_id: str | None = None,
                  now: float | None = None, allow_same: bool = True) -> Lease:
    now = time.time() if now is None else now
    existing = load_lease(path)
    if existing and existing.fresh(now) and not (allow_same and existing.holder == holder):
        raise PatchingError(
            f"lease held by {existing.holder} (age {now - existing.heartbeat:.1f}s)"
        )
    lease = Lease(
        holder=holder, heartbeat=now, function_id=function_id,
        stale_after_s=existing.stale_after_s if existing else 600.0,
    )
    write_json_atomic(path, asdict(lease))
    return lease


def heartbeat_lease(path: Path, holder: str, now: float | None = None,
                    stale_after_s: float | None = None) -> Lease:
    now = time.time() if now is None else now
    existing = load_lease(path)
    if existing is None:
        raise PatchingError(f"cannot heartbeat missing lease {path}")
    if existing.released:
        raise PatchingError("cannot heartbeat a released lease")
    if existing.holder != holder:
        raise PatchingError(
            f"controller lease changed (expected {holder!r}, found {existing.holder!r})"
        )
    existing.heartbeat = now
    if stale_after_s is not None:
        existing.stale_after_s = float(stale_after_s)
    blob = json.loads(Path(path).read_text()) if Path(path).is_file() else {}
    blob.update(asdict(existing))
    write_json_atomic(path, blob)
    return existing


def release_lease(path: Path, holder: str, now: float | None = None) -> Lease | None:
    now = time.time() if now is None else now
    existing = load_lease(path)
    if existing is None:
        return None
    if existing.fresh(now) and existing.holder != holder:
        raise PatchingError(
            f"cannot release lease held by {existing.holder!r}"
        )
    existing.released = True
    existing.heartbeat = now
    blob = json.loads(Path(path).read_text()) if Path(path).is_file() else {}
    blob.update({**asdict(existing), "released_at": now})
    write_json_atomic(path, blob)
    return existing


class ChunkStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.chunk_dir = self.run_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_component(value: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "--", str(value)).strip(".-")
        if not clean:
            raise PatchingError(f"unsafe empty path component from {value!r}")
        return clean

    def chunk_relative_path(self, phase: str, model_id: str, dtype: str,
                            config_sha: str, layer: int, block_id: int) -> Path:
        model = self._safe_component(model_id)
        dtype_part = self._safe_component(dtype)
        phase_part = self._safe_component(phase)
        config_part = self._safe_component(config_sha[:16])
        layer_part = "baseline" if int(layer) == -1 else f"L{int(layer):03d}"
        return Path("chunks") / model / dtype_part / config_part / phase_part / layer_part / f"B{int(block_id):04d}.jsonl"

    def _resolve_index_path(self, stored: str) -> Path:
        candidate = Path(stored)
        if candidate.is_absolute():
            # Legacy absolute paths are rebased by their chunks/ suffix after pulling.
            try:
                idx = candidate.parts.index("chunks")
            except ValueError as exc:
                raise PatchingError(f"non-portable chunk path {stored}") from exc
            candidate = Path(*candidate.parts[idx:])
        resolved = (self.run_dir / candidate).resolve()
        try:
            resolved.relative_to(self.run_dir)
        except ValueError as exc:
            raise PatchingError(f"chunk path escapes run directory: {stored}") from exc
        return resolved

    def write_block(self, phase: str, model_id: str, dtype: str, config_sha: str,
                    layer: int, block_id: int, rows: list[dict], *,
                    merge: bool = True, repair_corrupt: bool = True) -> dict:
        rel = self.chunk_relative_path(
            phase, model_id, dtype, config_sha, layer, block_id,
        )
        path = self.run_dir / rel
        index_path = self.run_dir / "chunk_index.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {"chunks": []}
        old_rec = next(
            (record for record in index.get("chunks", []) if record.get("path") == rel.as_posix()),
            None,
        )
        existing: list[dict] = []
        if merge and path.is_file():
            try:
                existing = read_chunk(path, old_rec.get("sha256") if old_rec else None)
            except PatchingError:
                if not repair_corrupt:
                    raise
                existing = []
        combined: dict[tuple, dict] = {primary_key(row): row for row in existing}
        for row in rows:
            if row.get("model_id") != model_id or row.get("dtype") != dtype:
                raise PatchingError("chunk row model/dtype does not match namespace")
            if row.get("configuration_sha256") != config_sha:
                raise PatchingError("chunk row configuration hash does not match namespace")
            if int(row.get("layer")) != int(layer):
                raise PatchingError("chunk row layer does not match namespace")
            key = primary_key(row)
            if key in combined and combined[key] != row:
                raise PatchingError(f"conflicting row for key {key}")
            combined[key] = row
        ordered = [combined[key] for key in sorted(combined, key=repr)]
        meta = finalize_chunk(path, ordered)
        meta["path"] = rel.as_posix()
        rec = {
            "phase": phase, "model_id": model_id, "dtype": dtype,
            "configuration_sha256": config_sha, "layer": int(layer),
            "block_id": int(block_id), **meta,
        }
        index["chunks"] = [c for c in index["chunks"] if c.get("path") != meta["path"]]
        index["chunks"].append(rec)
        index["chunks"].sort(key=lambda item: item["path"])
        write_json_atomic(index_path, index)
        return rec

    def load_valid_rows(self, *, model_id: str | None = None,
                        dtype: str | None = None,
                        configuration_sha256: str | None = None,
                        phases: Iterable[str] | None = None,
                        strict: bool = False) -> list[dict]:
        index_path = self.run_dir / "chunk_index.json"
        if not index_path.is_file():
            return []
        index = json.loads(index_path.read_text())
        allowed_phases = set(phases) if phases is not None else None
        rows, seen = [], set()
        for rec in index.get("chunks", []):
            if model_id is not None and rec.get("model_id") != model_id:
                continue
            if dtype is not None and rec.get("dtype") != dtype:
                continue
            if configuration_sha256 is not None and rec.get("configuration_sha256") != configuration_sha256:
                continue
            if allowed_phases is not None and rec.get("phase") not in allowed_phases:
                continue
            path = self._resolve_index_path(rec["path"])
            if not path.is_file() or ".tmp." in path.name:
                if strict:
                    raise PatchingError(f"indexed chunk missing: {path}")
                continue
            try:
                part = read_chunk(path, rec.get("sha256"))
            except PatchingError:
                if strict:
                    raise
                continue
            for r in part:
                k = primary_key(r)
                if k in seen:
                    raise PatchingError(f"duplicate key across chunks: {k}")
                seen.add(k)
                rows.append(r)
        return rows


def validate_or_create_manifest(run_dir: Path, *, run_id: str, config: dict,
                                metadata: dict | None = None) -> dict:
    """Create a minimal immutable manifest or validate the existing one."""
    run_dir = Path(run_dir)
    expected_sha = configuration_sha256(config)
    path = run_dir / "manifest.json"
    if path.is_file():
        current = json.loads(path.read_text())
        checks = {
            "run_id": run_id,
            "configuration_sha256": expected_sha,
            "prompt_sha256": config["prompt_sha256"],
            "smoke_prompt_sha256": config["smoke_prompt_sha256"],
            "code_sha256": config["code_sha256"],
        }
        mismatches = {
            key: (current.get(key), value)
            for key, value in checks.items()
            if current.get(key) != value
        }
        if mismatches:
            raise PatchingError(f"immutable manifest mismatch: {mismatches}")
        return current
    manifest = {
        **config,
        "run_id": run_id,
        "configuration_sha256": expected_sha,
        "created_at": time.time(),
        "metadata": metadata or {},
    }
    write_json_atomic(path, manifest)
    return manifest


def completeness_report(expected: Sequence[tuple], rows: Sequence[dict]) -> dict:
    present = [primary_key(r) for r in rows]
    pset, eset = set(present), set(expected)
    extra, missing = pset - eset, eset - pset
    duplicates = sum(count - 1 for count in Counter(present).values() if count > 1)
    expected_duplicates = sum(count - 1 for count in Counter(expected).values() if count > 1)
    finite = True
    probe_link_complete = True
    for r in rows:
        for k in (
            "source_D", "destination_D", "patched_D", "signed_effect",
            "class_function_gap", "baseline_drift", "source_probe_margin",
            "destination_probe_margin", "patched_probe_margin",
            "source_probe_declaration_margin",
        ):
            v = r.get(k)
            if v is not None and not np.isfinite(v):
                finite = False
        if r.get("prompt_sha256") == EVAL_PROMPT_SHA256 and r.get("model_id") in MODELS:
            probe_layer = MODELS[r["model_id"]]["probe_index"]
            if r.get("layer") == -1:
                required_probe = (
                    "source_probe_margin", "destination_probe_margin",
                    "patched_probe_margin", "source_probe_declaration_margin",
                )
                probe_link_complete &= all(r.get(key) is not None for key in required_probe)
            elif r.get("layer") == probe_layer:
                required_probe = (
                    "source_probe_margin", "destination_probe_margin", "patched_probe_margin",
                )
                probe_link_complete &= all(r.get(key) is not None for key in required_probe)
    return {
        "n_expected": len(expected),
        "n_present": len(present),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_duplicates": duplicates,
        "n_expected_duplicates": expected_duplicates,
        "finite": finite,
        "probe_link_complete": bool(probe_link_complete),
        "complete": (
            not extra and not missing and not duplicates and not expected_duplicates
            and finite and probe_link_complete and len(present) == len(eset)
        ),
        "missing_examples": [repr(key) for key in sorted(missing, key=repr)[:10]],
        "extra_examples": [repr(key) for key in sorted(extra, key=repr)[:10]],
    }


def expected_intervention_keys(model_id: str, pairs: list[dict], prompt_sha: str,
                               config_sha: str, dtype: str = "float16",
                               cells: list[tuple[int, str, str, str]] | None = None,
                               include_baselines: bool = True) -> list[tuple]:
    rev = MODELS[model_id]["revision"]
    if cells is None:
        cells = intervention_cells(model_id)
    keys = []
    for row in pairs:
        for layer, span, direction, control in cells:
            keys.append((
                prompt_sha, config_sha, model_id, rev, dtype, row["pair_id"],
                layer, span, direction, control, RANDOM_SEED,
            ))
        if include_baselines:
            for span, direction, control in (
                ("baseline_class", "none", "unpatched"),
                ("baseline_function", "none", "unpatched"),
                ("baseline_class", "none", "drift"),
                ("baseline_function", "none", "drift"),
            ):
                keys.append((
                    prompt_sha, config_sha, model_id, rev, dtype, row["pair_id"],
                    -1, span, direction, control, RANDOM_SEED,
                ))
    return keys


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def drift_tau(max_drift: float) -> float:
    return max(DRIFT_TAU_FLOOR, IDENTITY_DRIFT_MULT * float(max_drift or 0.0))


def identity_tau(max_drift: float, dtype: str = "float16") -> float:
    floor = DRIFT_TAU_FLOOR
    if str(dtype).lower() in ("float16", "fp16"):
        floor = max(floor, FP16_IDENTITY_TAU)
    return max(floor, IDENTITY_DRIFT_MULT * float(max_drift or 0.0))


def smoke_gate(rows: list[dict], tau: float | None = None,
               probe_index: int | None = None,
               expected_pair_ids: Iterable[str] | None = None) -> dict:
    drifts = [abs(r["baseline_drift"]) for r in rows if r.get("baseline_drift") is not None]
    max_drift = max(drifts) if drifts else 0.0
    dtypes = {str(row.get("dtype") or "float16") for row in rows}
    row_dtype = dtypes.pop() if len(dtypes) == 1 else "float16"
    if tau is None:
        tau = identity_tau(max_drift, row_dtype)

    if probe_index is None:
        candidate_layers = {
            int(r["layer"]) for r in rows
            if r.get("control") == "target" and r.get("span") == "placebo"
            and int(r.get("layer", -1)) > 0
        }
        if len(candidate_layers) != 1:
            raise PatchingError(
                f"smoke rows do not identify one probe layer: {sorted(candidate_layers)}"
            )
        probe_index = candidate_layers.pop()
    if expected_pair_ids is None:
        expected = {
            str(r["pair_id"]) for r in rows
            if r.get("layer") == -1 and r.get("control") == "unpatched"
        }
    else:
        expected = {str(pair_id) for pair_id in expected_pair_ids}
    if len(expected) != 8:
        return {
            "pass": False,
            "checks": {"exact_8_pair_ids": False},
            "error": f"smoke requires exactly 8 pair IDs, got {len(expected)}",
            "paper_evidence": False,
        }

    def pick(**criteria):
        return [
            row for row in rows
            if all(row.get(key) == value for key, value in criteria.items())
        ]

    groups = {
        "class": pick(layer=-1, span="baseline_class", direction="none", control="unpatched"),
        "function": pick(layer=-1, span="baseline_function", direction="none", control="unpatched"),
        "denoise_target": pick(layer=probe_index, span="query_name", direction="denoise", control="target"),
        "noise_target": pick(layer=probe_index, span="query_name", direction="noise", control="target"),
        "denoise_placebo": pick(layer=probe_index, span="placebo", direction="denoise", control="target"),
        "noise_placebo": pick(layer=probe_index, span="placebo", direction="noise", control="target"),
        "denoise_random": pick(layer=probe_index, span="query_name", direction="denoise", control="random"),
        "noise_random": pick(layer=probe_index, span="query_name", direction="noise", control="random"),
        "denoise_same": pick(layer=probe_index, span="query_name", direction="denoise", control="same_source"),
        "noise_same": pick(layer=probe_index, span="query_name", direction="noise", control="same_source"),
        "denoise_layer0": pick(layer=0, span="query_name", direction="denoise", control="target"),
        "noise_layer0": pick(layer=0, span="query_name", direction="noise", control="target"),
    }
    try:
        joined = exact_pair_join(groups, expected_pair_ids=expected)
    except PatchingError as exc:
        return {
            "pass": False,
            "checks": {"exact_row_cube": False},
            "error": str(exc),
            "paper_evidence": False,
        }
    d_class = [item["class"]["source_D"] for item in joined]
    d_fn = [item["function"]["source_D"] for item in joined]
    denoise_t = [item["denoise_target"]["signed_effect"] for item in joined]
    noise_t = [item["noise_target"]["signed_effect"] for item in joined]
    denoise_p = [item["denoise_placebo"]["signed_effect"] for item in joined]
    noise_p = [item["noise_placebo"]["signed_effect"] for item in joined]
    denoise_r = [item["denoise_random"]["signed_effect"] for item in joined]
    noise_r = [item["noise_random"]["signed_effect"] for item in joined]
    same = [
        value for item in joined
        for value in (item["denoise_same"]["signed_effect"], item["noise_same"]["signed_effect"])
    ]
    ident = [
        value for item in joined
        for value in (item["denoise_layer0"]["signed_effect"], item["noise_layer0"]["signed_effect"])
    ]
    checks = {
        "exact_8_pair_ids": True,
        "exact_row_cube": True,
        "class_D_positive_6_of_8": sum(v > 0 for v in d_class) >= 6,
        "function_below_class_6_of_8": sum(c > f for c, f in zip(d_class, d_fn)) >= 6,
        "mean_gap_positive": (_mean(d_class) - _mean(d_fn)) > 0,
        "mean_denoise_positive": _mean(denoise_t) > 0,
        "mean_noise_positive": _mean(noise_t) > 0,
        "target_gt_placebo": _mean(denoise_t) > _mean(denoise_p) and _mean(noise_t) > _mean(noise_p),
        "target_gt_random": _mean(denoise_t) > _mean(denoise_r) and _mean(noise_t) > _mean(noise_r),
        "max_drift_le_0_01": max_drift <= 0.01,
        "same_source_within_tau": bool(same) and all(abs(v) <= tau for v in same),
        "layer0_within_tau": bool(ident) and all(abs(v) <= tau for v in ident),
    }
    diagnostics = {
        "function_D_negative_6_of_8": sum(v < 0 for v in d_fn) >= 6,
        "denoise_sign_5_of_8": sum(v > 0 for v in denoise_t) >= 5,
        "noise_sign_5_of_8": sum(v > 0 for v in noise_t) >= 5,
    }
    return {
        "pass": all(checks.values()), "checks": checks, "diagnostics": diagnostics,
        "tau": tau, "max_drift": max_drift, "probe_index": probe_index,
        "n_class": len(d_class), "n_target_denoise": len(denoise_t),
        "n_target_noise": len(noise_t), "paper_evidence": False,
    }


def behavior_gate(d_class, d_function, clusters) -> dict:
    d_class = np.asarray(d_class, dtype=np.float64)
    d_function = np.asarray(d_function, dtype=np.float64)
    acc_c = (d_class > 0).astype(np.float64)
    acc_f = (d_function < 0).astype(np.float64)
    pair_gap = (d_class > d_function).astype(np.float64)
    gap = d_class - d_function
    ci_c = clustered_mean_ci(acc_c, clusters)
    ci_f = clustered_mean_ci(acc_f, clusters)
    ci_pg = clustered_mean_ci(pair_gap, clusters)
    ci_dc = clustered_mean_ci(d_class, clusters)
    ci_df = clustered_mean_ci(-d_function, clusters)
    ci_g = clustered_mean_ci(gap, clusters)
    checks = {
        "class_acc_ge_0.60": ci_c["point"] >= 0.60,
        "class_acc_ci_gt_0.50": ci_c["ci_low"] > 0.50,
        "class_D_ci_gt_0": ci_dc["ci_low"] > 0,
        "pair_gap_acc_ge_0.60": ci_pg["point"] >= 0.60,
        "pair_gap_acc_ci_gt_0.50": ci_pg["ci_low"] > 0.50,
        "gap_ci_gt_0": ci_g["ci_low"] > 0,
    }
    diagnostics = {
        "function_acc_ge_0.60": ci_f["point"] >= 0.60,
        "function_acc_ci_gt_0.50": ci_f["ci_low"] > 0.50,
        "neg_function_D_ci_gt_0": ci_df["ci_low"] > 0,
    }
    return {
        "pass": all(checks.values()), "checks": checks, "diagnostics": diagnostics,
        "class_acc": ci_c, "function_acc": ci_f, "class_D": ci_dc,
        "neg_function_D": ci_df, "gap": ci_g, "pair_gap_acc": ci_pg,
    }


def causal_gate(denoise, noise, denoise_placebo, noise_placebo,
                denoise_random, noise_random, gaps, clusters, tau: float) -> dict:
    threshold = max(0.10, 5 * tau)
    rec_d = ratio_of_means(denoise, gaps)
    rec_n = ratio_of_means(noise, gaps)
    td = np.asarray(denoise) - np.asarray(denoise_placebo)
    nd = np.asarray(noise) - np.asarray(noise_placebo)
    tr = np.asarray(denoise) - np.asarray(denoise_random)
    nr = np.asarray(noise) - np.asarray(noise_random)
    ci = {
        "denoise": clustered_mean_ci(denoise, clusters),
        "noise": clustered_mean_ci(noise, clusters),
        "denoise_minus_placebo": clustered_mean_ci(td, clusters),
        "noise_minus_placebo": clustered_mean_ci(nd, clusters),
        "denoise_minus_random": clustered_mean_ci(tr, clusters),
        "noise_minus_random": clustered_mean_ci(nr, clusters),
    }
    checks = {
        "denoise_ge_threshold": ci["denoise"]["point"] >= threshold,
        "noise_ge_threshold": ci["noise"]["point"] >= threshold,
        "denoise_ci_gt_0": ci["denoise"]["ci_low"] > 0,
        "noise_ci_gt_0": ci["noise"]["ci_low"] > 0,
        "denoise_placebo_ci_gt_0": ci["denoise_minus_placebo"]["ci_low"] > 0,
        "noise_placebo_ci_gt_0": ci["noise_minus_placebo"]["ci_low"] > 0,
        "denoise_random_ci_gt_0": ci["denoise_minus_random"]["ci_low"] > 0,
        "noise_random_ci_gt_0": ci["noise_minus_random"]["ci_low"] > 0,
        "recovery_denoise_ge_0.05": rec_d >= 0.05,
        "recovery_noise_ge_0.05": rec_n >= 0.05,
    }
    return {"pass": all(checks.values()), "checks": checks, "ci": ci,
            "threshold": threshold, "recovery_denoise": rec_d, "recovery_noise": rec_n}


def probe_ood_gate(class_margins, function_margins, clusters,
                   decl_auc: float, query_auc: float) -> dict:
    gap = np.asarray(class_margins, dtype=np.float64) - np.asarray(function_margins, dtype=np.float64)
    ci = clustered_mean_ci(gap, clusters)
    checks = {
        "margin_gap_ci_gt_0": ci["ci_low"] > 0,
        "declaration_auc_ge_0.70": decl_auc >= 0.70,
        "query_auc_ge_0.70": query_auc >= 0.70,
    }
    return {"pass": all(checks.values()), "checks": checks, "margin_gap": ci,
            "declaration_auc": decl_auc, "query_auc": query_auc}


def binary_auc(positive_scores, negative_scores) -> float:
    """Tie-aware Mann-Whitney AUC without importing sklearn."""
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    comparisons = pos[:, None] - neg[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def probe_ood_from_baselines(rows: Sequence[dict], *,
                             expected_pair_ids: Iterable[str] | None = None) -> dict:
    clean = [
        row for row in rows
        if row.get("layer") == -1 and row.get("span") == "baseline_class"
        and row.get("control") == "unpatched"
    ]
    function = [
        row for row in rows
        if row.get("layer") == -1 and row.get("span") == "baseline_function"
        and row.get("control") == "unpatched"
    ]
    joined = exact_pair_join(
        {"clean": clean, "function": function},
        expected_pair_ids=expected_pair_ids,
    )
    if not joined:
        raise PatchingError("no paired baseline rows for probe OOD gate")
    query_clean = [item["clean"].get("source_probe_margin") for item in joined]
    query_function = [item["function"].get("source_probe_margin") for item in joined]
    decl_clean = [item["clean"].get("source_probe_declaration_margin") for item in joined]
    decl_function = [item["function"].get("source_probe_declaration_margin") for item in joined]
    if any(value is None for value in query_clean + query_function + decl_clean + decl_function):
        raise PatchingError("baseline rows lack selected-layer probe margins")
    clusters = [item["clean"]["cluster_id"] for item in joined]
    return probe_ood_gate(
        query_clean, query_function, clusters,
        binary_auc(decl_clean, decl_function),
        binary_auc(query_clean, query_function),
    )


def evaluate_gates(rows: Sequence[dict], model_id: str, *, dtype: str = "float16",
                   configuration_sha256: str | None = None,
                   expected_pair_ids: Iterable[str] | None = None) -> dict:
    """Evaluate behavior and primary causal gates with exact pair alignment."""
    selected = [
        row for row in rows
        if row.get("model_id") == model_id and row.get("dtype") == dtype
        and (configuration_sha256 is None
             or row.get("configuration_sha256") == configuration_sha256)
    ]
    pair_ids = (
        {str(pair_id) for pair_id in expected_pair_ids}
        if expected_pair_ids is not None
        else {
            str(row["pair_id"]) for row in selected
            if row.get("layer") == -1 and row.get("control") == "unpatched"
        }
    )

    def pick(**criteria):
        return [
            row for row in selected
            if all(row.get(key) == value for key, value in criteria.items())
        ]

    clean = pick(layer=-1, span="baseline_class", direction="none", control="unpatched")
    function = pick(layer=-1, span="baseline_function", direction="none", control="unpatched")
    try:
        baselines = exact_pair_join(
            {"clean": clean, "function": function}, expected_pair_ids=pair_ids,
        )
    except PatchingError as exc:
        return {
            "behavior": {"pass": False, "error": str(exc)},
            "probe_ood": {"pass": False, "skipped": True, "reason": "behavior incomplete"},
            "causal": {"pass": False, "skipped": True, "reason": "behavior incomplete"},
        }
    d_class = [item["clean"]["source_D"] for item in baselines]
    d_function = [item["function"]["source_D"] for item in baselines]
    clusters = [item["clean"]["cluster_id"] for item in baselines]
    behavior = behavior_gate(d_class, d_function, clusters)
    try:
        probe_ood = probe_ood_from_baselines(
            selected, expected_pair_ids=pair_ids,
        )
    except PatchingError as exc:
        probe_ood = {"pass": False, "error": str(exc)}
    if not behavior["pass"]:
        return {
            "behavior": behavior,
            "probe_ood": probe_ood,
            "causal": {"pass": False, "skipped": True, "reason": "behavior gate failed"},
        }
    if not probe_ood["pass"]:
        return {
            "behavior": behavior,
            "probe_ood": probe_ood,
            "causal": {"pass": False, "skipped": True, "reason": "probe OOD gate failed"},
        }

    probe = MODELS[model_id]["probe_index"]
    selectors = {
        "denoise": dict(layer=probe, span="query_name", direction="denoise", control="target"),
        "noise": dict(layer=probe, span="query_name", direction="noise", control="target"),
        "denoise_placebo": dict(layer=probe, span="placebo", direction="denoise", control="target"),
        "noise_placebo": dict(layer=probe, span="placebo", direction="noise", control="target"),
        "denoise_random": dict(layer=probe, span="query_name", direction="denoise", control="random"),
        "noise_random": dict(layer=probe, span="query_name", direction="noise", control="random"),
        "denoise_same": dict(layer=probe, span="query_name", direction="denoise", control="same_source"),
        "noise_same": dict(layer=probe, span="query_name", direction="noise", control="same_source"),
        "denoise_layer0": dict(layer=0, span="query_name", direction="denoise", control="target"),
        "noise_layer0": dict(layer=0, span="query_name", direction="noise", control="target"),
    }
    try:
        primary = exact_pair_join(
            {name: pick(**criteria) for name, criteria in selectors.items()},
            expected_pair_ids=pair_ids,
        )
    except PatchingError as exc:
        return {
            "behavior": behavior,
            "probe_ood": probe_ood,
            "causal": {"pass": False, "skipped": True, "reason": str(exc)},
        }
    # Include current-vs-frozen baseline drift recorded in the separate primary
    # GPU phase, not just the original within-container repeat.
    drifts = [
        abs(float(row.get("baseline_drift") or 0.0))
        for row in selected if row.get("baseline_drift") is not None
    ]
    max_drift = max(drifts) if drifts else 0.0
    dtypes = {str(row.get("dtype") or "float16") for row in selected}
    row_dtype = dtypes.pop() if len(dtypes) == 1 else "float16"
    tau = drift_tau(max_drift)
    ident_tau = identity_tau(max_drift, row_dtype)
    gaps = [
        item["clean"]["source_D"] - item["function"]["source_D"]
        for item in baselines
    ]
    causal = causal_gate(
        [item["denoise"]["signed_effect"] for item in primary],
        [item["noise"]["signed_effect"] for item in primary],
        [item["denoise_placebo"]["signed_effect"] for item in primary],
        [item["noise_placebo"]["signed_effect"] for item in primary],
        [item["denoise_random"]["signed_effect"] for item in primary],
        [item["noise_random"]["signed_effect"] for item in primary],
        gaps,
        clusters,
        tau,
    )
    diagnostic_checks = {
        "max_cross_phase_drift_le_0_01": max_drift <= 0.01,
        "denoise_same_source_within_tau": all(
            abs(item["denoise_same"]["signed_effect"]) <= ident_tau for item in primary
        ),
        "noise_same_source_within_tau": all(
            abs(item["noise_same"]["signed_effect"]) <= ident_tau for item in primary
        ),
        "denoise_layer0_within_tau": all(
            abs(item["denoise_layer0"]["signed_effect"]) <= ident_tau for item in primary
        ),
        "noise_layer0_within_tau": all(
            abs(item["noise_layer0"]["signed_effect"]) <= ident_tau for item in primary
        ),
    }
    causal["diagnostic_checks"] = diagnostic_checks
    causal["pass"] = bool(causal["pass"] and all(diagnostic_checks.values()))
    return {
        "behavior": behavior, "probe_ood": probe_ood,
        "causal": causal, "tau": tau, "identity_tau": ident_tau,
        "max_cross_phase_drift": max_drift,
    }


# ---------------------------------------------------------------------------
# Torch adapters (imported lazily)
# ---------------------------------------------------------------------------

def load_causal_lm(model_id: str, device, dtype="float16", local_files_only: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    meta = MODELS[model_id]
    torch_dtype = torch.float16 if str(dtype) in ("float16", "fp16") else torch.float32
    tok = load_tokenizer_pinned(model_id, local_files_only=local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if local_files_only:
        # Transformers 5.8/Hugging Face Hub 1.27 can fail to resolve a full
        # commit through repository-ID cache lookup even when that exact
        # snapshot is present. The overnight run is deliberately offline, so
        # bypass Hub resolution. Provenance remains the pinned model_id and
        # revision in MODELS, the manifest, and every result row.
        model_source: str | Path = require_local_snapshot(model_id)
        model_kwargs = {"local_files_only": True}
    else:
        model_source = model_id
        model_kwargs = {"revision": meta["revision"], "local_files_only": False}
    model = AutoModelForCausalLM.from_pretrained(
        model_source, torch_dtype=torch_dtype, trust_remote_code=False,
        **model_kwargs,
    )
    model.eval()
    model.to(device)
    return tok, model, ArchitectureAdapter(model)


def local_snapshot_path(model_id: str, revision: str) -> Path | None:
    """Resolve an exact local HF snapshot without contacting the Hub."""
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        hub = Path(cache)
    else:
        hf_home = os.environ.get("HF_HOME")
        hub = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    repo = "models--" + model_id.replace("/", "--")
    snapshot = hub / repo / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def require_local_snapshot(model_id: str) -> Path:
    """Return the exact pinned snapshot or fail before requesting a GPU."""
    if model_id not in MODELS:
        raise PatchingError(f"unknown model {model_id}")
    revision = MODELS[model_id]["revision"]
    snapshot = local_snapshot_path(model_id, revision)
    if snapshot is None:
        raise PatchingError(
            f"missing exact local snapshot for {model_id}@{revision}; "
            "the Modal run is offline and will not download models"
        )
    return snapshot


def load_config_pinned(model_id: str, *, local_files_only: bool = True):
    """Load config from the same immutable source used for model weights."""
    from transformers import AutoConfig

    meta = MODELS[model_id]
    if local_files_only:
        return AutoConfig.from_pretrained(
            require_local_snapshot(model_id), local_files_only=True,
            trust_remote_code=False,
        )
    return AutoConfig.from_pretrained(
        model_id, revision=meta["revision"], local_files_only=False,
        trust_remote_code=False,
    )


def validate_local_model_snapshot(model_id: str):
    """Validate config and all referenced safetensors without reading weights."""
    snapshot = require_local_snapshot(model_id)
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise PatchingError(f"cached snapshot lacks config.json: {snapshot}")
    config = load_config_pinned(model_id, local_files_only=True)

    index_path = snapshot / "model.safetensors.index.json"
    single_path = snapshot / "model.safetensors"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PatchingError(f"invalid safetensors index {index_path}") from exc
        if (
            not isinstance(weight_map, dict) or not weight_map
            or not all(isinstance(name, str) and name for name in weight_map.values())
        ):
            raise PatchingError(f"empty safetensors weight map {index_path}")
        shards = sorted(set(weight_map.values()))
        unsafe = [name for name in shards if Path(name).is_absolute() or Path(name).name != name]
        if unsafe:
            raise PatchingError(f"unsafe safetensors shard names in {index_path}: {unsafe}")
        missing = [
            name for name in shards
            if not (snapshot / name).is_file() or (snapshot / name).stat().st_size <= 0
        ]
        if missing:
            raise PatchingError(
                f"cached snapshot has missing/empty safetensors shards: {missing}"
            )
    elif not single_path.is_file() or single_path.stat().st_size <= 0:
        raise PatchingError(
            f"cached snapshot has neither a non-empty model.safetensors nor "
            f"model.safetensors.index.json: {snapshot}"
        )
    return config


def load_tokenizer_pinned(model_id: str, *, local_files_only: bool = True):
    """Load a pinned tokenizer from its exact snapshot when running offline."""
    from transformers import AutoTokenizer

    meta = MODELS[model_id]
    if not local_files_only:
        return AutoTokenizer.from_pretrained(
            model_id, revision=meta["revision"], local_files_only=False,
            trust_remote_code=False,
        )
    return AutoTokenizer.from_pretrained(
        require_local_snapshot(model_id), local_files_only=True,
        trust_remote_code=False,
    )


class ArchitectureAdapter:
    def __init__(self, model):
        inner = model.model if hasattr(model, "model") else model
        if not hasattr(inner, "layers"):
            raise PatchingError("could not discover decoder block list at model.model.layers")
        if not hasattr(inner, "norm"):
            raise PatchingError("could not discover final norm at model.model.norm")
        self.model = model
        self.inner = inner
        self.blocks = list(inner.layers)
        self.norm = inner.norm
        self.n_blocks = len(self.blocks)
        cfg = getattr(model, "config", None)
        self.hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        self.n_hidden = self.n_blocks + 1
        if self.hidden_size <= 0:
            raise PatchingError("model config has no positive hidden_size")

    def assert_expected(self, model_id: str | None = None) -> None:
        if model_id and model_id in MODELS:
            exp = MODELS[model_id]
            if self.n_blocks != exp["n_blocks"] or self.n_hidden != exp["n_hidden"]:
                raise PatchingError(
                    f"{model_id}: blocks/hidden {self.n_blocks}/{self.n_hidden} "
                    f"!= {exp['n_blocks']}/{exp['n_hidden']}"
                )

    def _replace_rows(self, hidden, specs):
        import torch
        if hidden is None:
            return hidden
        new = hidden.clone()
        for row, pos, vec in specs:
            v = vec if torch.is_tensor(vec) else torch.as_tensor(
                vec, device=new.device, dtype=new.dtype)
            new[row, pos] = v.to(device=new.device, dtype=new.dtype)
        return new

    def _pre_hook(self, specs):
        def hook(module, args, kwargs=None):
            hidden = args[0] if isinstance(args, tuple) else args
            if isinstance(hidden, tuple):
                new_hidden = (self._replace_rows(hidden[0], specs),) + hidden[1:]
            else:
                new_hidden = self._replace_rows(hidden, specs)
            new_args = (new_hidden,) + tuple(args[1:]) if isinstance(args, tuple) else (new_hidden,)
            if kwargs is None:
                return new_args
            return new_args, kwargs
        return hook

    def _post_hook(self, specs):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (self._replace_rows(output[0], specs),) + output[1:]
            return self._replace_rows(output, specs)
        return hook

    def install(self, layer_index: int, specs):
        if layer_index < 0 or layer_index >= self.n_hidden:
            raise PatchingError(
                f"layer_index {layer_index} out of range 0..{self.n_hidden - 1}"
            )
        handles = []
        if layer_index < self.n_blocks:
            try:
                handles.append(self.blocks[layer_index].register_forward_pre_hook(
                    self._pre_hook(specs), with_kwargs=True))
            except TypeError:
                handles.append(self.blocks[layer_index].register_forward_pre_hook(
                    lambda m, a: self._pre_hook(specs)(m, a)))
        else:
            handles.append(self.norm.register_forward_hook(self._post_hook(specs)))
        return handles


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()


def capture_hidden_states(model, input_ids, attention_mask,
                          readout_ids: tuple[int, int] | None = None,
                          keep_layers: Sequence[int] | None = None):
    """Return last-token logits and a hidden-state tuple.

    Hugging Face still computes every layer when ``output_hidden_states=True``.
    ``keep_layers`` drops references to unused layers immediately after the
    forward so a 7B source pass does not retain a full 33-tensor tuple while
    copying three span vectors.
    """
    import torch

    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  use_cache=False, output_hidden_states=True)
    with torch.inference_mode():
        try:
            out = model(**kwargs, logits_to_keep=1)
        except TypeError:
            out = model(**kwargs)
    logits = out.logits[:, -1, :]
    n_hidden = len(out.hidden_states)
    expected = int(getattr(model.config, "num_hidden_layers", n_hidden - 1)) + 1
    if n_hidden != expected:
        raise PatchingError(f"hidden-state tuple length {n_hidden} != expected {expected}")
    if keep_layers is None:
        keep_set = set(range(n_hidden))
    else:
        keep_set = {int(layer) for layer in keep_layers}
        invalid = sorted(layer for layer in keep_set if not 0 <= layer < n_hidden)
        if invalid:
            raise PatchingError(f"keep_layers out of range: {invalid}")
    hs = tuple(
        out.hidden_states[i].detach() if i in keep_set else None
        for i in range(n_hidden)
    )
    del out
    if readout_ids is not None:
        logits = logits[:, list(readout_ids)]
    return logits, hs


def forward_logit_diffs(model, input_ids, attention_mask,
                        true_id: int, false_id: int):
    """Logits-only repeat; move only the two readout values to CPU."""
    import torch
    kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        use_cache=False, output_hidden_states=False,
    )
    with torch.inference_mode():
        try:
            out = model(**kwargs, logits_to_keep=1)
        except TypeError:
            out = model(**kwargs)
    last = out.logits[:, -1, :]
    diffs = last[:, true_id] - last[:, false_id]
    return diffs.detach().float().cpu().numpy()


def patched_forward(model, adapter: ArchitectureAdapter, input_ids, attention_mask,
                    layer_index: int, specs,
                    readout_ids: tuple[int, int] | None = None):
    import torch
    handles = adapter.install(layer_index, specs)
    try:
        with torch.inference_mode():
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, output_hidden_states=False)
            try:
                out = model(**kwargs, logits_to_keep=1)
            except TypeError:
                out = model(**kwargs)
        logits = out.logits[:, -1, :]
        if readout_ids is not None:
            logits = logits[:, list(readout_ids)]
        return logits
    finally:
        remove_hooks(handles)


def pad_batch(tokenizer, texts, device):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    pad_lens = (mask == 0).sum(dim=1).tolist()
    return input_ids, mask, pad_lens


def compare_hook_to_hidden_states(model, adapter, input_ids, attention_mask, atol=HOOK_MATCH_ATOL):
    """Unit/preflight: hook capture == output_hidden_states[k]."""
    import torch
    captured = {}

    def pre(k):
        def hook(module, args, kwargs=None):
            hidden = args[0] if isinstance(args, tuple) else args
            tensor = hidden[0] if isinstance(hidden, tuple) else hidden
            captured[k] = tensor.detach()
            if kwargs is None:
                return None
            return args, kwargs
        return hook

    def post_final(module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured[adapter.n_blocks] = tensor.detach()
        return output

    handles = []
    try:
        for k, block in enumerate(adapter.blocks):
            try:
                handles.append(block.register_forward_pre_hook(pre(k), with_kwargs=True))
            except TypeError:
                handles.append(block.register_forward_pre_hook(lambda m, a, kk=k: pre(kk)(m, a)))
        handles.append(adapter.norm.register_forward_hook(post_final))
        logits, hs = capture_hidden_states(model, input_ids, attention_mask)
    finally:
        remove_hooks(handles)
    diffs = {}
    if len(hs) != adapter.n_hidden or set(captured) != set(range(adapter.n_hidden)):
        raise PatchingError(
            f"hook/hidden-state coverage mismatch captured={sorted(captured)} "
            f"tuple_length={len(hs)} expected={adapter.n_hidden}"
        )
    for k, tensor in captured.items():
        ref = hs[k]
        diffs[k] = float((tensor.float() - ref.float()).abs().max().cpu())
        if diffs[k] > atol and not torch.equal(tensor, ref):
            raise PatchingError(
                f"hook hidden index {k} max abs diff {diffs[k]} > {atol}"
            )
    return diffs


def configuration_dict(prompt_sha: str, smoke_sha: str, code_sha: str) -> dict:
    return {
        "experiment": EXPERIMENT,
        "language": LANGUAGE,
        "prompt_sha256": prompt_sha,
        "smoke_prompt_sha256": smoke_sha,
        "code_sha256": code_sha,
        "models": {k: {"revision": v["revision"], "probe_index": v["probe_index"],
                       "n_hidden": v["n_hidden"]} for k, v in MODELS.items()},
        "dtype": "float16",
        "readout": {"true": TRUE_COMPLETION, "false": FALSE_COMPLETION,
                    "token_ids": {"true": TRUE_TOKEN_IDS, "false": FALSE_TOKEN_IDS}},
        "random_seed": RANDOM_SEED,
        "block_size": BLOCK_SIZE,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "dataset_revision": DATASET_REVISION,
        "ceilings": {k: v["item_forward_ceiling"] for k, v in MODELS.items()},
        "schedules": {
            model_id: {
                name: schedule_cells(model_id, name)
                for name in ("primary", "core", "expanded", "full")
            }
            for model_id in MODELS
        },
    }


def configuration_sha256(cfg: dict) -> str:
    return sha256_bytes(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())


def build_source_vectors(hs, pad_len: int, indices: dict) -> dict:
    """Keep only required span vectors at every layer; drop full tuples."""
    out = {}
    for layer, tensor in enumerate(hs):
        row = tensor[0]
        out[layer] = {
            span: row[left_pad_index(idx, pad_len)].detach().cpu().numpy()
            for span, idx in indices.items()
        }
    return out


def choose_source_dest(direction: str, control: str, clean_vec, corrupt_vec):
    """Return (source_vec, dest_vec, recipient_side) with recipient_side in {clean, corrupt}."""
    if control == "same_source":
        if direction == "denoise":
            return corrupt_vec, corrupt_vec, "corrupt"
        return clean_vec, clean_vec, "clean"
    if direction == "denoise":
        return clean_vec, corrupt_vec, "corrupt"
    if direction == "noise":
        return corrupt_vec, clean_vec, "clean"
    raise PatchingError(f"bad direction {direction}")


def patch_vector(control: str, source, dest, pair_id: str, layer: int, span: str,
                 direction: str, dtype: str = "float16") -> np.ndarray:
    if control in ("target", "same_source"):
        return np.asarray(source)
    if control == "random":
        key = cell_rng_key(pair_id, layer, span, direction, control)
        return inject_random(source, dest, key, dtype=numpy_dtype(dtype))
    raise PatchingError(f"control {control} has no patch vector")
