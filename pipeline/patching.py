"""Activation patching with a trained probe as the readout.

Causal counterpart to the linear probes. Instead of asking whether role
information is *linearly decodable* at a layer, we ask whether that layer's
activation is *causally responsible* for the probe readout. For a matched
clean/corrupt program pair we cache the clean role-token activations, re-run
the model on the corrupt program while patching those activations in at one
layer, and measure how much the probe's positive-class probability on role
tokens is restored:

    recovery(L) = (M_patched(L) - M_corrupt) / (M_clean - M_corrupt)

M is the mean probe P(role) over role tokens; the readout is taken at the
probe's best layer. recovery ~1 => layer L carries the causal role signal;
~0 => it does not; the readout layer itself is 1.0 by construction (a sanity
anchor).

Pairs are token-aligned by role-occurrence order:
  perturbation — baseline (clean) vs a renamed strategy (corrupt)
  crosslang    — Python (clean) vs another language (corrupt)
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .probing import MAX_SEQ_LEN, label_tokens


def find_decoder_layers(model):
    """Locate the ModuleList of transformer blocks for any HF decoder.

    Picks the ModuleList whose length matches the configured layer count
    (`.layers` for Llama/Qwen, `.h` for GPT-2, etc.).
    """
    n = (getattr(model.config, "num_hidden_layers", None)
         or getattr(model.config, "n_layer", None))
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and (n is None or len(mod) == n):
            return mod
    raise RuntimeError("could not locate the decoder-layer ModuleList")


def role_token_positions(code, names, tokenizer, leading_special, seq_len):
    """Absolute input-sequence indices of role tokens, dropping truncated ones."""
    _, labels = label_tokens(code, names, tokenizer)
    return [leading_special + i for i, lab in enumerate(labels)
            if lab == 1 and leading_special + i < seq_len]


@torch.no_grad()
def _run(code, tokenizer, model, device):
    enc = tokenizer(code, return_tensors="pt", truncation=True,
                    max_length=MAX_SEQ_LEN, padding=False).to(device)
    hidden = model(**enc).hidden_states
    return enc, hidden


def _readout(hidden_layer, probe, positions):
    X = hidden_layer[0, positions].float().cpu().numpy()
    return float(probe.predict_proba(X)[:, 1].mean())


@torch.no_grad()
def patch_pair(clean_code, clean_names, corrupt_code, corrupt_names,
               tokenizer, model, layers, leading_special, device,
               probe, readout_layer):
    """Per-layer (M_clean, M_corrupt, M_patched) for one clean/corrupt pair.

    Returns None if either side has no usable role tokens. Patches hidden-state
    index L (block L-1's output) for L in 1..readout_layer.
    """
    enc_c, hs_c = _run(clean_code, tokenizer, model, device)
    enc_x, hs_x = _run(corrupt_code, tokenizer, model, device)
    seq_c, seq_x = enc_c["input_ids"].shape[1], enc_x["input_ids"].shape[1]

    pos_c = role_token_positions(clean_code, clean_names, tokenizer, leading_special, seq_c)
    pos_x = role_token_positions(corrupt_code, corrupt_names, tokenizer, leading_special, seq_x)
    if not pos_c or not pos_x:
        return None
    k = min(len(pos_c), len(pos_x))              # align on role-occurrence order
    pos_c, pos_x = pos_c[:k], pos_x[:k]

    readout_layer = max(1, readout_layer)
    m_clean = _readout(hs_c[readout_layer], probe, pos_c)
    m_corrupt = _readout(hs_x[readout_layer], probe, pos_x)

    out = {}
    for L in range(1, readout_layer + 1):
        clean_vec = hs_c[L][0, pos_c]            # (k, d), on device

        def hook(_mod, _inp, outp, cv=clean_vec, positions=pos_x):
            h = outp[0] if isinstance(outp, tuple) else outp
            h[0, positions] = cv.to(h.dtype)
            return outp

        handle = layers[L - 1].register_forward_hook(hook)
        try:
            hs_p = model(**enc_x).hidden_states
        finally:
            handle.remove()
        out[L] = (m_clean, m_corrupt, _readout(hs_p[readout_layer], probe, pos_x))
    return out


def patch_experiment(clean_by_pid, corrupt_by_pid, role, tokenizer, model, layers,
                     leading_special, device, probe, readout_layer,
                     max_pairs=None, min_gap=0.02):
    """Aggregate per-layer recovery over all matched clean/corrupt pairs.

    Only pairs where the corruption actually moves the readout by > min_gap are
    counted (otherwise the recovery ratio is degenerate). Returns
    (recovery_by_layer, n_pairs, mean_M_clean, mean_M_corrupt).
    """
    pids = [p for p in clean_by_pid if p in corrupt_by_pid]
    per_layer = defaultdict(list)
    m_cleans, m_corrupts, n_used = [], [], 0
    for pid in tqdm(pids, desc="patch pairs", leave=False):
        clean, corrupt = clean_by_pid[pid], corrupt_by_pid[pid]
        cn, xn = clean["roles"].get(role, []), corrupt["roles"].get(role, [])
        if not cn or not xn:
            continue
        res = patch_pair(clean["code"], cn, corrupt["code"], xn, tokenizer, model,
                         layers, leading_special, device, probe, readout_layer)
        if res is None:
            continue
        m_clean, m_corrupt, _ = next(iter(res.values()))
        if (m_clean - m_corrupt) <= min_gap:
            continue
        n_used += 1
        m_cleans.append(m_clean)
        m_corrupts.append(m_corrupt)
        for L, (mc, mx, mp) in res.items():
            per_layer[L].append((mp - mx) / (mc - mx))
        if max_pairs and n_used >= max_pairs:
            break

    recovery = {L: float(np.mean(v)) for L, v in per_layer.items()}
    return (recovery, n_used,
            float(np.mean(m_cleans)) if m_cleans else 0.0,
            float(np.mean(m_corrupts)) if m_corrupts else 0.0)
