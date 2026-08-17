# Unified variable-role probing pipeline

One dataset + one set of scripts replacing the per-role notebooks
(`probing_variable_roles`, `probing_accumulator*`, `probing_more_perturbations`,
and the per-role work on the team branches).

- **5 roles**: `index_key`, `accumulator`, `iterator`, `boolean`, `class_struct`
- **10 strategies**: `baseline`, `random_nouns`, `single_chars`, `all_same`,
  `numeric_vars`, `misleading_<role>` ×5
- **7 languages** (baseline labels): C++, Java, Python, C#, Javascript, PHP, C
- Protocol matches the notebooks: per-layer logistic regression
  (class-balanced, C=1.0), 80/20 stratified split, macro F1, best layer on test F1.

## 1. Build the dataset (CPU, minutes)

```bash
python -m pipeline.build_dataset --out dataset            # full XLCoST
python -m pipeline.build_dataset --out dataset --max-programs 500   # notebook-sized
```

Writes `dataset/python_perturbations/{train,valid,test}.jsonl`,
`dataset/multilingual_baseline/{train,valid,test}.jsonl`, and `dataset/stats.json`.
Rows are model-agnostic (code + role-name sets); token labels are derived per
tokenizer at experiment time.

## 2. Run experiments (GPU recommended)

```bash
# perturbation sweep for one role (Python)
python -m pipeline.run_experiment perturbation --role accumulator \
    --model Qwen/Qwen2.5-1.5B --max-programs 500

# cross-language transfer for one role
python -m pipeline.run_experiment crosslang --role index_key \
    --model Qwen/Qwen2.5-1.5B --max-programs 300
```

Outputs land in `results/unified/<model>/<role>/<mode>/`:
`per_layer.csv`, `summary.csv` (best layer + ΔF1 vs baseline),
`cosine_vs_baseline.csv`, `crosslang.csv`, and `patching.csv`.

To sweep everything:

```bash
for role in index_key accumulator iterator boolean class_struct; do
  python -m pipeline.run_experiment perturbation --role $role --max-programs 500
  python -m pipeline.run_experiment crosslang    --role $role --max-programs 300
done
```

### Activation patching

Both experiments also run **activation patching** with the trained probe as
the readout — the causal counterpart to the probes. For a matched clean/corrupt
program pair (perturbation: baseline vs a renamed strategy; crosslang: Python vs
another language, aligned on role-token order) it caches the clean role-token
activations, re-runs the corrupt program while patching them in one layer at a
time, and measures how much the probe's `P(role)` on role tokens is restored:

```
recovery(L) = (M_patched(L) − M_corrupt) / (M_clean − M_corrupt)
```

`recovery≈1` ⇒ layer L causally carries the role signal (the readout layer is
`1.0` by construction — a sanity anchor); `≈0` ⇒ it does not. Results go to
`patching.csv` (one row per condition: `readout_layer`, `n_pairs`, `m_clean`,
`m_corrupt`, and `recovery_layer_*`). Tunables: `--max-pairs` (default 150),
`--patch-min-gap` (default 0.02, min |M_clean−M_corrupt| for a pair to count),
and `--no-patch` to skip the stage.

### Smoke test (offline, seconds)

Sanity-check probes + patching end to end on the tiny cached model before a
full run — checks the patch mechanics (readout-layer recovery == 1.0, no hook
residue) and validates every CSV:

```bash
python -m pipeline.smoke_test                 # sshleifer/tiny-gpt2
python -m pipeline.smoke_test --skip-pipeline # mechanics check only
```

## 3. Publish the dataset

```bash
huggingface-cli login   # once
python -m pipeline.hf_upload --repo <user-or-org>/xlcost-variable-roles
```

The dataset card (`dataset_card/README.md`) credits XLCoST (Zhu et al., 2022,
arXiv:2206.08474, Apache-2.0) and documents fields, labeling, and limitations.

## Module map

| Module | Purpose |
|---|---|
| `xlcost.py` | token-list reconstruction, program loading |
| `roles.py` | role extractors (Python AST; regex for other languages) |
| `perturb.py` | naming strategies, role-parameterized misleading |
| `build_dataset.py` | materialize the two dataset configs |
| `probing.py` | token labeling, hidden-state extraction, per-layer probes |
| `patching.py` | activation patching with the probe as readout (recovery curves) |
| `run_experiment.py` | perturbation and cross-language experiment CLIs |
| `smoke_test.py` | offline end-to-end sanity check (patch mechanics + tiny-model run) |
| `hf_upload.py` | Hub upload of dataset + card |
