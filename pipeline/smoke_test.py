"""End-to-end smoke test for the unified probing + patching pipeline.

Fast and offline (expects sshleifer/tiny-gpt2 cached). Three checks:

1. Patch mechanics — on a hand-built clean/corrupt pair, patching *at* the
   readout layer must recover the probe readout exactly (recovery == 1.0, the
   construction anchor), the readouts must be valid probabilities, and the
   forward hook must leave no residue (a plain forward after patching matches
   the original).

2. Pipeline — runs the real `perturbation` and `crosslang` experiments
   (probes + activation patching) on the tiny model and a handful of programs,
   then validates the emitted CSVs, including that patching.csv's readout-layer
   recovery column is ~1.0.

    python -m pipeline.smoke_test
    python -m pipeline.smoke_test --skip-pipeline   # mechanics check only
"""

import argparse
import csv
import math
import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np


def check_patch_mechanics(model_name):
    import torch
    from sklearn.linear_model import LogisticRegression

    from .patching import find_decoder_layers, patch_pair
    from .probing import extract_hidden_states, load_model

    device = "cpu"
    tokenizer, model, leading_special = load_model(model_name, device)
    layers = find_decoder_layers(model)

    clean = "def f ( a ) :\n    i = 0\n    while i < a :\n        i = i + 1\n    return i"
    corrupt = "def f ( a ) :\n    z = 0\n    while z < a :\n        z = z + 1\n    return z"
    names_c, names_x = {"i"}, {"z"}

    # A quick probe so patch_pair has a readout; label i/z tokens as positive.
    data = [{"code": clean, "labels": _lab(clean, names_c, tokenizer), "tokens": None},
            {"code": corrupt, "labels": _lab(corrupt, names_x, tokenizer), "tokens": None}]
    hidden, lab = extract_hidden_states(data, tokenizer, model, leading_special, device)
    readout_layer = max(1, len(model(**tokenizer(clean, return_tensors="pt")).hidden_states) - 2)
    probe = LogisticRegression(max_iter=1000, class_weight="balanced").fit(hidden[readout_layer], lab)

    # Snapshot a plain forward, then patch, then confirm no hook residue.
    enc = tokenizer(corrupt, return_tensors="pt")
    with torch.no_grad():
        before = model(**enc).hidden_states[readout_layer].clone()
    res = patch_pair(clean, names_c, corrupt, names_x, tokenizer, model, layers,
                     leading_special, device, probe, readout_layer)
    with torch.no_grad():
        after = model(**enc).hidden_states[readout_layer]

    assert res is not None, "patch_pair found no role tokens"
    m_clean, m_corrupt, _ = next(iter(res.values()))
    assert 0.0 <= m_clean <= 1.0 and 0.0 <= m_corrupt <= 1.0, "readout not a probability"
    _, _, m_patched_at_readout = res[readout_layer]
    recovery = (m_patched_at_readout - m_corrupt) / (m_clean - m_corrupt) \
        if m_clean != m_corrupt else 1.0
    assert abs(recovery - 1.0) < 1e-6, f"readout-layer recovery={recovery}, expected 1.0"
    assert torch.allclose(before, after, atol=1e-5), "forward hook left residue"
    print(f"[patch] readout_layer={readout_layer}  M_clean={m_clean:.3f} "
          f"M_corrupt={m_corrupt:.3f}  readout recovery={recovery:.4f}  hook clean  OK")


def _lab(code, names, tokenizer):
    from .probing import label_tokens
    return label_tokens(code, names, tokenizer)[1]


def _validate_probe_csv(path, key_col):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{path} has no rows"
    assert key_col in rows[0], f"{path} missing column {key_col}"
    print(f"[csv ] {os.path.relpath(path)}  {len(rows)} rows  OK")


def _validate_patching_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{path} has no rows"
    checked = 0
    for r in rows:
        if int(r["n_pairs"]) == 0:
            continue
        col = f"recovery_layer_{r['readout_layer']}"
        assert col in r, f"{path} missing {col}"
        val = float(r[col])
        assert math.isclose(val, 1.0, abs_tol=1e-3), \
            f"{path}: {r['condition']} readout-layer recovery={val}, expected ~1.0"
        checked += 1
    assert checked > 0, f"{path}: no conditions produced usable pairs"
    print(f"[csv ] {os.path.relpath(path)}  {len(rows)} rows  {checked} anchored  OK")


def run_pipeline(model, max_programs, dataset):
    import torch

    from .probing import load_model
    from .run_experiment import run_crosslang, run_perturbation

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[pipe] device={device}  model={model}")
    tokenizer, mdl, leading_special = load_model(model, device)

    with tempfile.TemporaryDirectory() as tmp:
        # min_gap negative so tiny-model pairs still count and the anchor fires.
        args = SimpleNamespace(role="index_key", model=model, dataset=dataset,
                               split="train", max_programs=max_programs, out=tmp,
                               patch=True, max_pairs=8, patch_min_gap=-1.0)

        pert = os.path.join(tmp, "perturbation")
        os.makedirs(pert, exist_ok=True)
        run_perturbation(args, tokenizer, mdl, leading_special, device, pert)
        _validate_probe_csv(os.path.join(pert, "per_layer.csv"), "strategy")
        _validate_probe_csv(os.path.join(pert, "summary.csv"), "best_layer")
        _validate_patching_csv(os.path.join(pert, "patching.csv"))

        xl = os.path.join(tmp, "crosslang")
        os.makedirs(xl, exist_ok=True)
        run_crosslang(args, tokenizer, mdl, leading_special, device, xl)
        _validate_probe_csv(os.path.join(xl, "crosslang.csv"), "language")
        _validate_patching_csv(os.path.join(xl, "patching.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--max-programs", type=int, default=25)
    ap.add_argument("--skip-pipeline", action="store_true")
    args = ap.parse_args()

    check_patch_mechanics(args.model)
    if not args.skip_pipeline:
        run_pipeline(args.model, args.max_programs, args.dataset)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
