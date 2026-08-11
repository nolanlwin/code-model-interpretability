# Variable Roles in Transformer Code Models

Code for **"When Names Matter and When They Don't: Variable Roles and Language
Structure in Transformer Code Models"** — probing whether code LLMs represent
variables by their *program role* (boolean control, accumulator, index,
iterator, class/struct reference) or by surface token identity.

This repository holds the **boolean-control-variable** workstream and the
team-wide shared pipeline: corpus construction, occurrence extraction with
stable identities, activation extraction, protocol-grade probing (grouped
splits, control tasks, bootstrap CIs), model-free baselines, and the C1–C5
identifier-renaming experiment. The experimental contract every role follows
is frozen in [PROTOCOL.md](PROTOCOL.md).

## Environment

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), `transformers==5.8.0`
  (pinned — tokenizer behavior is gate-validated on this version)

```bash
uv sync
uv run python scripts/<script>.py --help
```

Optional `HF_TOKEN` in `.env` speeds up Hub downloads.

## Reproduce the results

Two commands per (language, model) pair — every step is idempotent or
resumable, and every number lands in `outputs/probe_results/*.json`:

```bash
bash scripts/run_language.sh Python Qwen/Qwen2.5-Coder-1.5B train   # corpus → probe → baselines
bash scripts/run_renaming.sh Python Qwen/Qwen2.5-Coder-1.5B train   # C1–C5 renaming experiment
```

On Colab, use [notebooks/colab_python_results.ipynb](notebooks/colab_python_results.ipynb)
(Runtime → GPU → Run all): it wraps the same two scripts, restores prior work
from Google Drive, sweeps multiple models, and prints the consolidated
results table.

## Pipeline

| Script | Role |
|---|---|
| `xlcost_data.py` | XLCoST acquisition from usable mirrors (the official release is TransCoder-tokenized and does not parse), detokenization, per-language parse validation, canonical JSONL keyed by cross-language `problem_id` |
| `xlcost_occurrences.py` | Boolean-flag occurrence extraction; stamps the protocol identity `problem:language:fN:bN:oN` and enforces the span-integrity gate |
| `tokenizer_gate.py` | Per-token offset verification for every model before extraction (a roundtrip check is not sufficient; DeepSeek-Coder requires the fast-tokenizer override under the pinned transformers) |
| `extract_activations.py` | One forward per program → mean-pooled residual streams in a float16 memmap store, resumable across sessions |
| `probe.py` | Layer-wise linear probing: problem-grouped frozen-hash splits, validation-selected layer, 5 seeds, explicit class handling, Hewitt control task (selectivity), cluster-BCa CIs, embedded per-item predictions |
| `baselines.py` | Model-free baselines (majority, name-only, masked source line, context window, scalar covariates) on the probe's exact occurrence sample |
| `rename_corpus.py` | Scope-correct C1–C5 identifier renaming (span edits, never `ast.unparse`); occurrence ids carried by span mapping so paired deltas join exactly |
| `bootstrap_ci.py` | Cluster bootstrap (percentile + BCa) and the paired clustered delta CI between conditions |
| `run_language.sh`, `run_renaming.sh` | The two entry points chaining everything above |

Supporting: `token_alignment.py` (char-span → token indices, invariant-tested),
`qwen_inference.py` (model loading / forwards), per-language occurrence
extractors (`variable_occurrences.py` for Python; `java_*`, `go_*`,
`javascript_*`, `php_*`, `ruby_*` via tree-sitter).

## Protocol highlights

- **Grouped, frozen splits.** Problems are assigned to train/val/test by a
  per-problem hash, so every renaming condition tests on the same fold and
  paired deltas use identical occurrence sets (`occurrence_id`-joined).
- **Baselines on the identical sample.** Probe numbers are reported against
  the strongest model-free baseline computed on the same occurrences and fold.
- **Gates before numbers.** Tokenizer offsets, span integrity, re-parse after
  renaming, and occurrence-id preservation are asserted, and failures are
  counted — never silently dropped.

See [PROTOCOL.md](PROTOCOL.md) for the frozen decisions and
[docs/dataset_comparison.md](docs/dataset_comparison.md) for the corpus
analysis (MuST-CoST vs XLCoST, per-language).

## Legacy pipeline

The pre-revision CodeSearchNet pipeline (`codesearchnet*.py`,
`boolean_flag_roles.py`, `clean_boolean_labels.py`, `dataset_v0.py`,
`activation_pipeline.py`, `notebooks/colab_activations_and_probing.ipynb`)
is retained for provenance of the original submission's Table 4 and the
frozen dataset v0 manifests. New work should use the XLCoST pipeline above.
