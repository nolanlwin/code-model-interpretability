"""Model-side machinery: token labeling, hidden-state extraction, probes.

Protocol (PROTOCOL.md; replaces the notebook-era version):
per-layer logistic regression on STANDARDIZED features, frozen
problem-hash 70/10/20 splits grouped by program, layer selected on the
VALIDATION fold, five seeds, macro F1 with program-clustered BCa CIs, a
random-label control task (selectivity), and provenance stamping.
Hidden states accumulate in float16 so 7B models fit Colab RAM.
"""

from collections import defaultdict
import gc
import json
import mmap
import os
import shutil
import tempfile

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .gates import load_tokenizer_gated
from .stats import (cluster_bootstrap_ci, git_commit, hash_split, macro_f1,
                    permute_labels_within_programs)

MAX_SEQ_LEN = 512
SEEDS = (0, 1, 2, 3, 4)
META = "_meta"


def layer_keys(results):
    """Integer layer keys of a train_probes result (skips the META entry)."""
    return sorted(k for k in results if isinstance(k, int))


def load_model(model_name, device, trust_remote_code=False):
    """trust_remote_code is now opt-in: none of the roster models need it."""
    from transformers import AutoModel

    tokenizer = load_tokenizer_gated(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True,
                                      trust_remote_code=trust_remote_code)
    model.eval().to(device)
    # BERT-style tokenizers prepend [CLS]; GPT-style prepend nothing.
    probe_ids = tokenizer("x", return_tensors="pt")["input_ids"][0].tolist()
    leading_special = len(probe_ids) - len(tokenizer.tokenize("x"))
    return tokenizer, model, leading_special


def label_tokens(code, target_names, tokenizer):
    """1 for tokens overlapping a whole-word occurrence of a target name.

    Span-based via offset mappings (the gate guarantees they are correct).
    KNOWN LIMITATION, stated in the paper: matches are textual, so
    occurrences inside strings/comments and same-named attributes are
    labeled too. Parser-derived spans are the post-deadline fix.
    """
    import re

    if not target_names:
        return [], []
    target_names = set(target_names)
    enc = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    spans = []
    for name in target_names:
        for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", code):
            spans.append((m.start(), m.end()))
    labels = [
        1 if any(ts < e and s < te for s, e in spans) else 0
        for ts, te in enc["offset_mapping"]
    ]
    return enc.tokens(), labels


def build_token_dataset(rows, role, tokenizer):
    """rows: dicts with 'code' and 'roles'. Keeps programs with >=1 positive token."""
    dataset, skipped = [], 0
    for row in rows:
        names = row["roles"].get(role, [])
        if not names:
            skipped += 1
            continue
        tokens, labels = label_tokens(row["code"], names, tokenizer)
        if not tokens or sum(labels) == 0:
            skipped += 1
            continue
        pid = next(
            (row[k] for k in ("program_id", "problem_id", "idx")
             if row.get(k) is not None),
            None,
        )
        if pid is None or str(pid) == "":
            raise ValueError(
                "row has no stable program identity (program_id / problem_id "
                "/ idx). A filtered-row-index fallback would let the same "
                "program land in different folds across strategies, silently "
                "invalidating paired comparisons — provide a stable id."
            )
        dataset.append({
            "code": row["code"], "labels": labels, "tokens": tokens,
            "program_id": str(pid),
        })
    return dataset, skipped


class DiskLayers(dict):
    """int -> path to a float16 memmap. Load one layer at a time, then close()."""

    def __init__(self, paths, n_used, cache_dir=None):
        super().__init__(paths)
        self.n_used = int(n_used)
        self.cache_dir = cache_dir

    def close(self, delete=True):
        self.clear()
        if delete and self.cache_dir and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir = None
        gc.collect()


def _release_mmap_pages(mm):
    """Flush and drop page-cache for a write memmap so RSS does not track file size."""
    mm.flush()
    buf = getattr(mm, "_mmap", None)
    if buf is None:
        return
    try:
        buf.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def dump_exists(dump_dir):
    return os.path.isfile(os.path.join(dump_dir, "meta.json")) and os.path.isfile(
        os.path.join(dump_dir, "labels.npy"))


def load_hidden_dump(dump_dir):
    """Reload a dump written by extract_hidden_states (layer_*.npy + sidecars)."""
    with open(os.path.join(dump_dir, "meta.json")) as f:
        meta = json.load(f)
    hidden = DiskLayers(
        {li: os.path.join(dump_dir, f"layer_{li}.npy") for li in range(meta["n_layers"])},
        n_used=meta["n_used"],
        cache_dir=dump_dir,
    )
    labels = np.load(os.path.join(dump_dir, "labels.npy"), allow_pickle=True)
    programs = np.load(os.path.join(dump_dir, "programs.npy"), allow_pickle=True)
    return hidden, labels, programs


def _write_dump_sidecar(cache_dir, n_layers, n_used, labels, programs):
    np.save(os.path.join(cache_dir, "labels.npy"), labels)
    np.save(os.path.join(cache_dir, "programs.npy"), programs)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"n_layers": int(n_layers), "n_used": int(n_used)}, f)


def load_layer_f32(hidden_by_layer, li):
    """Load a single layer as float32 and drop the mmap so RAM can be reclaimed."""
    src = hidden_by_layer[li]
    n_used = getattr(hidden_by_layer, "n_used", None)
    if isinstance(src, str):
        mm = np.load(src, mmap_mode="r")
        sl = mm[:n_used] if n_used is not None else mm
        X = np.array(sl, dtype=np.float32, copy=True)
        del mm, sl
        gc.collect()
        return X
    sl = src[:n_used] if n_used is not None else src
    X = np.array(sl, dtype=np.float32, copy=True)
    gc.collect()
    return X


def extract_hidden_states(dataset, tokenizer, model, leading_special, device,
                          out_dir=None):
    """Returns (hidden_by_layer fp16, labels, program_ids) — program identity
    is carried per token so splits can group by program downstream.

    Layer matrices are disk-backed memmaps so 400+ programs × 29 layers
    do not have to sit in RAM at once (Kaggle OOM'd the in-memory concat).
    """
    n_tokens = 0
    for sample in dataset:
        enc = tokenizer(sample["code"], return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN, padding=False)
        n_content = max(enc["input_ids"].shape[1] - leading_special, 0)
        n_tokens += min(len(sample["labels"]), n_content)
    if n_tokens == 0:
        raise ValueError("no tokens to extract")

    cache_root = os.environ.get("HIDDEN_CACHE_DIR", tempfile.gettempdir())
    os.makedirs(cache_root, exist_ok=True)
    cache_dir = out_dir or tempfile.mkdtemp(prefix="hidden_states_", dir=cache_root)
    os.makedirs(cache_dir, exist_ok=True)
    memmaps, all_labels, all_programs = None, [], []
    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="hidden states", leave=False)):
            enc = tokenizer(sample["code"], return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN, padding=False).to(device)
            n_content = enc["input_ids"].shape[1] - leading_special
            labels = sample["labels"][:n_content]
            n = len(labels)
            if n == 0:
                continue
            out = model(**enc).hidden_states
            if memmaps is None:
                dim = int(out[0].shape[-1])
                memmaps = {
                    li: np.lib.format.open_memmap(
                        os.path.join(cache_dir, f"layer_{li}.npy"),
                        mode="w+", dtype=np.float16, shape=(n_tokens, dim),
                    )
                    for li in range(len(out))
                }
            for li, hs in enumerate(out):
                content = hs[0, leading_special:leading_special + n].half().cpu().numpy()
                memmaps[li][offset:offset + n] = content
            all_labels.extend(labels)
            all_programs.extend([sample["program_id"]] * n)
            offset += n
            del out
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            if i % 20 == 19:
                for mm in memmaps.values():
                    _release_mmap_pages(mm)
            gc.collect()
    if memmaps is None:
        raise ValueError("no hidden states written")
    n_layers = len(memmaps)
    for mm in memmaps.values():
        _release_mmap_pages(mm)
        buf = getattr(mm, "_mmap", None)
        if buf is not None:
            try:
                buf.close()
            except BufferError:
                pass
    del memmaps
    gc.collect()
    hidden = DiskLayers(
        {li: os.path.join(cache_dir, f"layer_{li}.npy") for li in range(n_layers)},
        n_used=offset,
        cache_dir=cache_dir,
    )
    labels = np.array(all_labels)
    programs = np.array(all_programs)
    _write_dump_sidecar(cache_dir, n_layers, offset, labels, programs)
    return hidden, labels, programs


def train_probes(hidden_by_layer, labels, program_ids, seeds=SEEDS, C=1.0,
                 n_boot=1000):
    """Frozen problem-hash splits, validation-selected layer, multi-seed.

    Returns {layer: metrics averaged over seeds, plus a fitted (scaler,
    probe) for reuse} and results[META] with the headline numbers: the
    validation-selected layer per seed, test macro F1 mean/std at those
    layers, a program-clustered BCa CI on the pooled selected-layer test
    predictions, and provenance.
    """
    labels = np.asarray(labels)
    program_ids = np.asarray(program_ids)
    layers = sorted(hidden_by_layer)
    acc = {li: defaultdict(list) for li in layers}
    fitted = {}
    sel_layers, sel_f1, seeds_used = [], [], []
    pooled = {"y": [], "p": [], "g": []}

    for seed in seeds:
        tr, val, te = hash_split(program_ids, seed)
        if min(len(tr), len(val), len(te)) == 0 or len(set(labels[tr])) < 2:
            continue
        seeds_used.append(seed)
        store_fits = not fitted  # first VALID seed keeps the reusable probes
        val_f1, test_preds = [], []
        for li in layers:
            X = load_layer_f32(hidden_by_layer, li)
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=seed, C=C)
            X_tr = scaler.transform(X[tr])
            clf.fit(X_tr, labels[tr])
            p_tr = clf.predict(X_tr)
            p_va = clf.predict(scaler.transform(X[val]))
            p_te = clf.predict(scaler.transform(X[te]))
            del X, X_tr
            gc.collect()
            acc[li]["train_acc"].append(accuracy_score(labels[tr], p_tr))
            acc[li]["test_acc"].append(accuracy_score(labels[te], p_te))
            acc[li]["train_f1"].append(macro_f1(labels[tr], p_tr))
            acc[li]["test_f1"].append(macro_f1(labels[te], p_te))
            val_f1.append(macro_f1(labels[val], p_va))
            test_preds.append(p_te)
            if store_fits:
                fitted[li] = (scaler, clf)
        b = int(np.argmax(val_f1))
        sel_layers.append(layers[b])
        sel_f1.append(acc[layers[b]]["test_f1"][-1])
        pooled["y"].extend(labels[te])
        pooled["p"].extend(test_preds[b])
        pooled["g"].extend(program_ids[te])

    if not seeds_used:
        raise ValueError(
            "no seed produced a usable split (empty fold or single-class "
            "training data in every seed) — dataset too small or degenerate"
        )
    results = {}
    for li in layers:
        scaler, clf = fitted[li]
        results[li] = {
            "probe": clf, "scaler": scaler,
            **{k: float(np.mean(v)) for k, v in acc[li].items()},
        }
    ci = cluster_bootstrap_ci(pooled["y"], pooled["p"], pooled["g"], n_boot=n_boot)
    majority = np.zeros_like(np.asarray(pooled["y"]))
    results[META] = {
        "selected_layers": sel_layers,
        "seeds_used": seeds_used,
        "selected_layer": int(np.median(sel_layers)) if sel_layers else -1,
        "test_f1_mean": float(np.mean(sel_f1)) if sel_f1 else float("nan"),
        "test_f1_std": float(np.std(sel_f1)) if sel_f1 else float("nan"),
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "n_programs": int(len(set(program_ids.tolist()))),
        "cluster_warning": ci["cluster_warning"],
        "majority_f1": macro_f1(pooled["y"], majority),
        "seeds": list(seeds),
        "split": "hash_bucket_70_10_20_by_program",
        "layer_selected_on": "validation",
        "git_commit": git_commit(),
    }
    return results


def control_selectivity(hidden_by_layer, labels, program_ids, results,
                        seeds=SEEDS, C=1.0):
    """Random-label control (labels permuted within programs) evaluated with
    the same protocol at each seed's selected layer. Adds control_f1 and
    selectivity to results[META] and returns them."""
    labels_c = permute_labels_within_programs(labels, program_ids)
    program_ids = np.asarray(program_ids)
    sel = results[META]["selected_layers"]
    seeds_used = results[META].get("seeds_used", list(seeds)[: len(sel)])
    f1s = []
    for seed, li in zip(seeds_used, sel):
        tr, val, te = hash_split(program_ids, seed)
        if min(len(tr), len(te)) == 0 or len(set(labels_c[tr])) < 2:
            continue
        X = load_layer_f32(hidden_by_layer, li)
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed, C=C)
        clf.fit(scaler.transform(X[tr]), labels_c[tr])
        f1s.append(macro_f1(labels_c[te], clf.predict(scaler.transform(X[te]))))
        del X
        gc.collect()
    control = float(np.mean(f1s)) if f1s else float("nan")
    results[META]["control_f1"] = control
    results[META]["selectivity"] = results[META]["test_f1_mean"] - control
    return control, results[META]["selectivity"]


def best_layer(results):
    """Validation-selected layer (median across seeds). The old behaviour —
    argmax of TEST F1 — selected on the evaluation set."""
    return results[META]["selected_layer"]


def cross_evaluate(source_results, target_hidden, target_labels):
    """Apply each source-layer probe (with its scaler) to the target."""
    out = {}
    for li in layer_keys(source_results):
        if li not in target_hidden:
            continue
        scaler = source_results[li]["scaler"]
        X = load_layer_f32(target_hidden, li)
        pred = source_results[li]["probe"].predict(scaler.transform(X))
        del X
        gc.collect()
        out[li] = {
            "acc": accuracy_score(target_labels, pred),
            "f1": macro_f1(target_labels, pred),
        }
    return out


def probe_cosine(results_a, results_b):
    sims = {}
    for li in set(layer_keys(results_a)) & set(layer_keys(results_b)):
        wa = results_a[li]["probe"].coef_[0]
        wb = results_b[li]["probe"].coef_[0]
        sims[li] = float(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
    return sims
