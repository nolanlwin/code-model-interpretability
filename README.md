# Probing Code LLMs — Variable Roles, Unified Pipeline

Investigates whether code LLMs encode **variable roles** in their hidden
states — using structural labels (AST/regex), never the variable's name —
across the 7 languages of [XLCoST](https://github.com/reddy-lab-code-research/XLCoST).

This branch (`unified-pipeline`) replaces the per-role notebooks with one
common dataset and one set of scripts:

- **5 roles**: `index_key`, `accumulator`, `iterator`, `boolean`, `class_struct`
- **10 naming strategies**: `baseline`, `random_nouns`, `single_chars`,
  `all_same`, `numeric_vars`, and `misleading_<role>` for each role
- **7 languages**: C++, Java, Python, C#, Javascript, PHP, C
- **Any model**: token labels are derived per tokenizer at experiment time

The dataset is published on the Hugging Face Hub as
[`dhyuti-n/xlcost-variable-roles`](https://huggingface.co/datasets/dhyuti-n/xlcost-variable-roles)
(see `dataset_card/README.md` for fields, labeling method, and XLCoST credits).

---

## 1. Environment

Python ≥ 3.10. Create a virtual environment and install the pipeline
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r pipeline/requirements.txt
```

GPU is recommended for the probing experiments (Colab works — the scripts
auto-select `cuda` / `mps` / `cpu`). Building the dataset is CPU-only.

The legacy root `requirements.txt` is the upstream XLCoST environment for the
original notebooks; the pipeline does not need it.

## 2. Get the data

**Option A — use the published dataset (no XLCoST download needed):**

```python
from datasets import load_dataset

perturb = load_dataset("dhyuti-n/xlcost-variable-roles", "python_perturbations")
multi   = load_dataset("dhyuti-n/xlcost-variable-roles", "multilingual_baseline")
```

To run the experiment CLIs against it, download the JSONL files into a local
`dataset/` directory:

```bash
hf download dhyuti-n/xlcost-variable-roles --repo-type dataset --local-dir dataset
```

**Option B — rebuild from XLCoST:** unzip `XLCoST_data.zip` into the repo root
(or set `XLCOST_ROOT=/path/to/XLCoST_data`), then:

```bash
python -m pipeline.build_dataset --out dataset                    # full corpus
python -m pipeline.build_dataset --out dataset --max-programs 500 # notebook-sized
```

Writes `dataset/python_perturbations/{train,valid,test}.jsonl`,
`dataset/multilingual_baseline/{train,valid,test}.jsonl`, and `dataset/stats.json`.

## 3. Run experiments

```bash
# perturbation sweep for one role (Python, all naming strategies)
python -m pipeline.run_experiment perturbation --role accumulator \
    --model Qwen/Qwen2.5-1.5B --split train --max-programs 500

# cross-language transfer for one role (Python-trained probe -> 6 languages)
python -m pipeline.run_experiment crosslang --role index_key \
    --model Qwen/Qwen2.5-1.5B --split train --max-programs 300
```

Sweep everything:

```bash
for role in index_key accumulator iterator boolean class_struct; do
  python -m pipeline.run_experiment perturbation --role $role --max-programs 500
  python -m pipeline.run_experiment crosslang    --role $role --max-programs 300
done
```

Results land in `results/unified/<model>/<role>/<mode>/`:

| File | Contents |
|---|---|
| `per_layer.csv` | train/test accuracy and macro F1 for every layer × strategy |
| `summary.csv` | best layer per strategy with ΔF1 vs baseline |
| `cosine_vs_baseline.csv` | probe-direction cosine similarity per layer |
| `crosslang.csv` | in-domain F1 + transfer accuracy/F1 per language |

Protocol (identical to the original notebooks): per-layer logistic regression
probes (class-balanced, C=1.0) on frozen hidden states, 80/20 stratified
split, macro F1, best layer selected on test F1. Any Hugging Face model works
via `--model`; swap in CodeBERT, RoBERTa, Qwen2.5-0.5B, DeepSeek-Coder, etc.

## 4. Publish / update the dataset

```bash
hf auth login    # once
python -m pipeline.hf_upload --repo <user-or-org>/xlcost-variable-roles [--private]
```

Uploads both configs plus the dataset card in `dataset_card/README.md`.

---

## Project structure

```
├── pipeline/                  # the unified pipeline (see pipeline/README.md)
│   ├── xlcost.py              #   XLCoST loading and token-list reconstruction
│   ├── roles.py               #   role extractors (Python AST; regex elsewhere)
│   ├── perturb.py             #   naming strategies incl. per-role misleading
│   ├── build_dataset.py       #   materialize the two dataset configs
│   ├── probing.py             #   token labeling, hidden states, probes
│   ├── run_experiment.py      #   perturbation + crosslang CLIs
│   ├── hf_upload.py           #   Hub upload
│   └── requirements.txt       #   minimal deps
├── dataset_card/README.md     # HF dataset card (XLCoST credits, fields, limits)
├── dataset/stats.json         # row counts of the published build (JSONL gitignored)
├── notebooks/                 # original per-role notebooks (superseded)
├── results/                   # experiment outputs
├── RESULTS.md, RESULTS_TABLE.md
└── XLCoST_data/               # source corpus (not tracked)
```

---

## Boolean workstream & protocol tooling (`scripts/`)

Alongside `pipeline/`, the repo carries the **boolean-control-variable**
workstream and the protocol-grade measurement tooling it was built with
(frozen in [PROTOCOL.md](PROTOCOL.md)):

```bash
bash scripts/run_language.sh Python Qwen/Qwen2.5-Coder-1.5B train   # corpus -> probe -> baselines
bash scripts/run_renaming.sh Python Qwen/Qwen2.5-Coder-1.5B train   # C1-C5 renaming deltas
```

Distinct properties of this path (see `docs/` for the analyses behind them):

- **Detokenized corpus**: `scripts/xlcost_data.py` pulls XLCoST from mirrors
  and detokenizes/validates it, so models see natural code rather than
  space-joined token streams.
- **Frozen problem-grouped splits** (per-problem hash), layer selected on a
  validation fold, 5 seeds, Hewitt control task, cluster-BCa CIs, and paired
  deltas joined on stable `occurrence_id`s.
- **Gates before numbers**: per-token tokenizer-offset checks per model
  (`scripts/tokenizer_gate.py`), span-integrity and re-parse gates in
  extraction and renaming; failures are counted, never silent.
- Colab entry point: `notebooks/colab_python_results.ipynb`
  (Runtime -> GPU -> Run all; resumable, checkpoints to Drive).

## Credits

Source programs come from **XLCoST** (Zhu et al., 2022,
[arXiv:2206.08474](https://arxiv.org/abs/2206.08474), Apache-2.0). This
repository adds structural role labels, renaming perturbations, and the
probing pipeline; full citation in `dataset_card/README.md`.
