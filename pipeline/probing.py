"""Model-side machinery: token labeling, hidden-state extraction, probes.

Same protocol as the notebooks: per-layer logistic regression
(class-balanced, C=1.0), 80/20 stratified split, macro F1, best layer
selected on test F1.
"""

from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

MAX_SEQ_LEN = 512


def load_model(model_name, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True,
                                      trust_remote_code=True)
    model.eval().to(device)
    # BERT-style tokenizers prepend [CLS]; GPT-style prepend nothing.
    probe_ids = tokenizer("x", return_tensors="pt")["input_ids"][0].tolist()
    leading_special = len(probe_ids) - len(tokenizer.tokenize("x"))
    return tokenizer, model, leading_special


def label_tokens(code, target_names, tokenizer):
    """1 for tokens overlapping a whole-word occurrence of a target name.

    Span-based via offset mappings, so multi-token names (e.g. 'v8' →
    ['Ġv', '8']) are labeled correctly for any tokenizer. Falls back to the
    notebooks' single-token string match for slow tokenizers.
    """
    import re

    if not target_names:
        return [], []
    target_names = set(target_names)
    try:
        enc = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    except (NotImplementedError, TypeError, ValueError):
        tokens, labels = tokenizer.tokenize(code), []
        for tok in tokens:
            is_continuation = tok.startswith("##")
            clean = tok.lstrip("Ġ▁Ā").lstrip("##")
            clean_inner = clean.strip("[]().,;:$")
            matched = (not is_continuation) and (clean in target_names or clean_inner in target_names)
            labels.append(1 if matched else 0)
        return tokens, labels

    spans = []
    for name in target_names:
        for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", code):
            spans.append((m.start(), m.end()))
    tokens = enc.tokens()
    labels = []
    for tok_start, tok_end in enc["offset_mapping"]:
        hit = any(tok_start < e and s < tok_end for s, e in spans)
        labels.append(1 if hit else 0)
    return tokens, labels


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
        dataset.append({"code": row["code"], "labels": labels, "tokens": tokens})
    return dataset, skipped


def extract_hidden_states(dataset, tokenizer, model, leading_special, device):
    all_hidden, all_labels = defaultdict(list), []
    with torch.no_grad():
        for sample in tqdm(dataset, desc="hidden states", leave=False):
            enc = tokenizer(sample["code"], return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN, padding=False).to(device)
            n_content = enc["input_ids"].shape[1] - leading_special
            labels = sample["labels"][:n_content]
            for li, hs in enumerate(model(**enc).hidden_states):
                content = hs[0, leading_special:leading_special + len(labels)]
                all_hidden[li].extend(content.float().cpu().numpy())
            all_labels.extend(labels)
    return ({li: np.stack(v) for li, v in all_hidden.items()}, np.array(all_labels))


def train_probes(hidden_by_layer, labels, test_size=0.2, random_state=42):
    n = len(labels)
    tr, te = train_test_split(np.arange(n), test_size=test_size,
                              random_state=random_state, stratify=labels)
    results = {}
    for li in tqdm(sorted(hidden_by_layer), desc="probes", leave=False):
        X = hidden_by_layer[li]
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=random_state, C=1.0)
        clf.fit(X[tr], labels[tr])
        pred_tr, pred_te = clf.predict(X[tr]), clf.predict(X[te])
        results[li] = {
            "probe": clf,
            "train_acc": accuracy_score(labels[tr], pred_tr),
            "test_acc": accuracy_score(labels[te], pred_te),
            "train_f1": f1_score(labels[tr], pred_tr, average="macro", zero_division=0),
            "test_f1": f1_score(labels[te], pred_te, average="macro", zero_division=0),
        }
    return results


def best_layer(results):
    return max(results, key=lambda li: results[li]["test_f1"])


def cross_evaluate(source_results, target_hidden, target_labels):
    """Apply each source-layer probe to the target's same layer."""
    out = {}
    for li in sorted(source_results):
        if li not in target_hidden:
            continue
        pred = source_results[li]["probe"].predict(target_hidden[li])
        out[li] = {
            "acc": accuracy_score(target_labels, pred),
            "f1": f1_score(target_labels, pred, average="macro", zero_division=0),
        }
    return out


def probe_cosine(results_a, results_b):
    sims = {}
    for li in sorted(set(results_a) & set(results_b)):
        wa = results_a[li]["probe"].coef_[0]
        wb = results_b[li]["probe"].coef_[0]
        sims[li] = float(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
    return sims
