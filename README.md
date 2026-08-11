# Mechanistic interpretability for variable roles in code LLMs

This repository builds a **boolean-control-variable** label set over Python from CodeSearchNet: names that tend to act as conditional signals (`if` / `while` tests, bool-related assignments, and related patterns). Everything is driven by small CLIs and JSONL on disk so you can reproduce or subset runs easily.

## Environment

- **Python** 3.11 or newer  
- **[uv](https://docs.astral.sh/uv/)** for environments and runs

```bash
uv sync
uv run python scripts/<script>.py ...
```

Optional: set **`HF_TOKEN`** when using the Hugging Face Hub so downloads are faster and less rate-limited.

## Layout

| Path | Purpose |
|------|--------|
| `scripts/` | Pipelines (data, labels, dataset v0, **CoST structure probing**) plus **Qwen inference** utilities |
| `data/codesearchnet_python/` | Canonical `python_{train,validation,test}.jsonl` (gitignored when large) |
| `outputs/` | AST parses, labels, dataset v0 shards, manifests (gitignored) |
| `notebooks/` | `state_probing_variable_roles.ipynb` (boolean), `structure_probing_cost.ipynb` (Colab syntactic-class + cross-lingual) |
| `docs/` | Dataset + experiment docs (`dataset_v0.md`, `structure_probing.md`, `colab_pro_setup.md`) |

## Scripts (in order)

| Script | Role |
|--------|------|
| `codesearchnet_python.py` | Stream CodeSearchNet Python from the Hub into canonical JSONL (`repo`, `path`, `code`). |
| `csn_function_ast.py` | Optional: parse each snippet to AST-shaped JSONL for analysis. |
| `boolean_flag_roles.py` | Heuristic **boolean_flag** labels; writes one JSON line per variable (and parse-error lines). |
| `clean_boolean_labels.py` | Join labels to the **same** canonical file, dedupe, filter noise, write cleaned JSONL + stats. |
| `dataset_v0.py` | **`split-repo`:** repository-level train/validation/test assignment. **`freeze`:** frozen split JSONL, `freeze_manifest.json`, and `docs/dataset_v0.md`. |
| `qwen_inference.py` | **`verify`:** load **Qwen2.5-1.5B**, forward text (default **`fixtures/minimal_forward.py`**, or **`--code-file` / `-` for stdin**) with hidden states and attentions; writes `outputs/qwen_inference/forward_verify.json`. |
| `token_alignment.py` | Map UTF-8 **`[start, end)`** spans to token indices via **`offset_mapping`** (same tokenizer settings as Qwen forward). **`verify`** checks `fixtures/token_alignment_cases.json`. **`align`** can default spans to **`stem_align_spans.json`** beside the code file; **`align-bundle`** reads one JSON body. |
| `variable_occurrences.py` | One JSONL row per **boolean-flag Name site** (`occurrence_type`, `source_span`, optional **`token_positions`** with **`--model-id`**). **`verify`** uses `fixtures/boolean_occurrence_sample.py`. **`extract`** accepts canonical JSONL or **`--code-file`**. |
| `activation_pipeline.py` | One forward per canonical row; **`.npz`** per occurrence (`first` / `last` / **`mean`** pooling, shape **`[num_layers+1, hidden_size]`**) plus **`manifest.jsonl`** (`activation_path`, `token_len`, `function_len_chars`, `occurrence_frequency`, …). **`verify`** runs one fixture row in a temp dir. |
| `cost_dataset.py` | **CoST** loader: read local `consolidated_data.csv`, keep the **1,417** problems with both **Java + Python**, write canonical `{problem_id, language, code}` JSONL. **`build`** / **`verify`**. |
| `structure_labels.py` | **tree-sitter** syntactic-class labels (shared Java/Python): one record per leaf with char `source_span`, `node_type`, and coarse `structural_class`. **`verify`** / **`extract`**. |

## End-to-end pipeline (train example)

1. **Canonical snippets** (same split for steps 2–5):

   ```bash
   uv run python scripts/codesearchnet_python.py download --split train
   ```

   Default: `data/codesearchnet_python/python_train.jsonl`. Use `--max-rows N` for a tiny sample.

2. **Boolean labels** — must use the **same** canonical path and order as in step 1 so each row gets a stable **`source_row`** (1-based line index). That index disambiguates duplicate `(repo, path)` rows in CodeSearchNet.

   ```bash
   uv run python scripts/boolean_flag_roles.py extract \
     --input data/codesearchnet_python/python_train.jsonl \
     --output outputs/labeled/boolean_flags.jsonl
   ```

   Successful rows include `variable`, `role`, `line`, `code`, `function`, `repo`, `path`, `source_row`. Failed parses are still emitted as one object per bad snippet with `parse_error` (and `source_row` when known).

3. **Clean** — canonical file must be the **identical** train JSONL as in step 1:

   ```bash
   uv run python scripts/clean_boolean_labels.py \
     --labels outputs/labeled/boolean_flags.jsonl \
     --canonical data/codesearchnet_python/python_train.jsonl \
     --output outputs/labeled/boolean_flags_clean.jsonl \
     --stats-output outputs/labeled/boolean_flags_clean.stats.json
   ```

   Use `--max-label-rows` / `--max-canonical-rows` only for quick experiments (truncated canonical makes many rows `missing_canonical` unless you built labels from that same prefix).

4. **Repository split** — each GitHub `repo` string is assigned to exactly one of **train** (80%), **validation** (10%), or **test** (10%) by default, using a seeded shuffle (not a per-row random split):

   ```bash
   uv run python scripts/dataset_v0.py split-repo \
     --labels outputs/labeled/boolean_flags_clean.jsonl \
     --output-dir outputs/dataset_v0 \
     --seed 42
   ```

   Writes `repo_split.jsonl` (`{"repo","split"}` per line) and `split_manifest.json`. Fractions are configurable (`--train-fraction`, `--val-fraction`, `--test-fraction`; must sum to 1).

5. **Freeze Dataset v0** — requires **`repo_split.jsonl` from step 4** (same `--output-dir` as `split-repo`). Rejoin cleaned labels to canonical `code`, attach sorted **`occurrences`**, and emit one JSONL per split plus reproducibility metadata:

   ```bash
   uv run python scripts/dataset_v0.py freeze \
     --labels outputs/labeled/boolean_flags_clean.jsonl \
     --canonical data/codesearchnet_python/python_train.jsonl \
     --repo-split outputs/dataset_v0/repo_split.jsonl \
     --output-dir outputs/dataset_v0
   ```

   Outputs `boolean_flags_v0_train.jsonl`, `boolean_flags_v0_validation.jsonl`, `boolean_flags_v0_test.jsonl`, and `freeze_manifest.json` (SHA-256 of inputs, outputs, and pipeline scripts; `git_revision` when available). Regenerates **`docs/dataset_v0.md`** with role rules, cleaning exclusions, split policy, and known limitations.

## What the cleaner removes

- `parse_error` rows and non–`boolean_flag` lines  
- **Duplicates** — `(source_row, function, variable)` when `source_row` is present; otherwise `(repo, path, function, variable)`  
- **Autogenerated**-looking snippets (regex on an early slice of source)  
- **Notebook-style** paths or obvious IPython hooks  
- **Short** snippets: fewer than five non-empty lines in joined `code`  
- **Invalid** Python on the canonical snippet  
- **Name not in function** — no top-level `def` / `async def` whose name matches `function`  
- **Rare names** — fewer than two occurrences of the variable in that function’s AST (`ast.Name` and `ast.arg`)

The **`.stats.json`** next to the clean file reports `input_label_lines_read`, `output_label_lines`, per-reason `dropped`, `role_counts`, and top-*K* variable and repository counts over **kept** rows.

### Reference scale (full Python train, one run)

Indexing **412,178** canonical lines and **197,444** label file lines (including **4,011** parse-error records) produced **188,597** cleaned labels. Drops included **3,550** low occurrence count, **920** autogenerated heuristic, **189** notebook-style, **177** short function, plus the parse-error lines. Your numbers will match this only when the same split and filters are used.

## AST parsing (optional)

```bash
uv run python scripts/csn_function_ast.py parse \
  --input data/codesearchnet_python/python_train.jsonl \
  --output outputs/ast_parsed/python_train_ast.jsonl
```

`--max-rows N` limits input lines.

## Dependencies

Listed in `pyproject.toml` (`datasets`, `transformers`, **`torch`**, `pandas`, `numpy`, `scikit-learn`, `tqdm`, `tree-sitter`, …). Parsing uses the standard library **`ast`** module unless you extend the project with Tree-sitter grammars.

## Qwen 2.5 inference (activation prep)

Load **`Qwen/Qwen2.5-1.5B`** with Transformers, run one causal forward with `output_hidden_states=True` and `output_attentions=True`, and save a JSON summary (input token ids, `convert_ids_to_tokens` pieces, per-layer **hidden** and **attention** shapes, optional **char offset** mapping from the fast tokenizer, and a check that `len(hidden_states) == num_hidden_layers + 1`).

```bash
uv run python scripts/qwen_inference.py verify
# same as:
uv run python scripts/qwen_inference.py verify --code-file fixtures/minimal_forward.py
# or pipe source:
echo 'x = 1' | uv run python scripts/qwen_inference.py verify --code-file -
```

Default **`--code-file`** is a tiny neutral **`fixtures/minimal_forward.py`**. Override with any UTF-8 path, or **`--code-file -`** to read stdin. Optional: `--device cuda|mps|cpu`, `--dtype bf16|fp16|fp32` (defaults favor GPU bf16 when supported; CPU uses fp32). First run downloads model weights from the Hub.

### Token alignment

Spans are **half-open** character indices in the same string you pass to the tokenizer (as in `ast`’s `end_col_offset`). **`token_alignment.py verify`** runs overlap + coverage checks on **`fixtures/token_alignment_cases.json`** (snake_case, camelCase, repeats, `def` params, attribute base span).

```bash
uv run python scripts/token_alignment.py verify
# default spans: fixtures/minimal_forward_align_spans.json (same dir/stem as code)
uv run python scripts/token_alignment.py align fixtures/minimal_forward.py -o outputs/align.json
uv run python scripts/token_alignment.py align fixtures/minimal_forward.py --spans fixtures/minimal_forward_align_spans.json -o outputs/align.json
# stdin for spans only (code from file):
echo '[{"variable":"f","source_span":[4,5]}]' | uv run python scripts/token_alignment.py align fixtures/minimal_forward.py --spans -
```

Use **`align-bundle`** with one JSON body `{"code":"...","occurrences":[...]}` if you need a single stdin pipe.

### Variable occurrences

Per-occurrence boolean-flag records (definition / assignment / conditional / loop / return, plus **`indexing_use`** when the name’s parent is a **`Subscript`**). **`--model-id`** fills **`token_positions`**; **`--no-tokens`** skips the Hub tokenizer.

```bash
uv run python scripts/variable_occurrences.py verify
uv run python scripts/variable_occurrences.py extract --code-file fixtures/boolean_occurrence_sample.py -o - --no-tokens
uv run python scripts/variable_occurrences.py extract --input data/codesearchnet_python/python_train.jsonl --output outputs/occurrences/boolean_flag_occurrences.jsonl --max-rows 100 --model-id Qwen/Qwen2.5-1.5B
```

### Activations

**`activation_pipeline.py extract`** reads canonical JSONL (`repo`, `path`, `code`), recomputes boolean occurrences with the Qwen tokenizer, runs **one** causal forward per input line (hidden states only), and writes **`--tensor-dir`** / `row_XXXXXXXX_occ_YYY.npz` plus **`--manifest`** JSONL. Use **`--max-rows 1000`** for a small validation slice. Each **`.npz`** holds float32 **`first`**, **`last`**, and **`mean`** activations of shape **`[num_layers+1, hidden_size]`** (full residual stack; slice a layer downstream). Manifest rows include **`activation_path`**, **`token_len`**, **`function_len_chars`**, **`occurrence_frequency`**, and **`layer`: null** (full tensor on disk).

```bash
uv run python scripts/activation_pipeline.py verify
uv run python scripts/activation_pipeline.py extract --input data/codesearchnet_python/python_train.jsonl \
  --manifest outputs/activations_v0/manifest.jsonl --tensor-dir outputs/activations_v0/npz \
  --max-rows 1000 --model-id Qwen/Qwen2.5-1.5B
```

`qwen_inference.py` exposes **`load_causal_lm_tokenizer`** and **`forward_hidden_cached`** for reuse (hidden-only forwards omit eager attention by default).

## Structure probing on CoST (Java + Python)

A second probing experiment asks **where in Qwen each code token's *syntactic class* (identifier / keyword / operator / string / number / punctuation / bool-null) is linearly decodable**, and whether a probe trained on Python transfers to Java (cross-lingual). It reuses the boolean probe's layer-by-layer `LogisticRegression` method but with multi-class tree-sitter labels over the parallel **CoST** corpus (`consolidated_data.csv`).

- **Notebook (primary, Colab-ready):** [`notebooks/structure_probing_cost.ipynb`](notebooks/structure_probing_cost.ipynb) — install deps, load `consolidated_data.csv`, load Qwen on a T4, build labels, extract per-layer activations (program cap + balanced token subsampling), train within- and cross-lingual probes, and plot macro-F1 vs layer + a confusion matrix.
- **Docs:** [`docs/structure_probing.md`](docs/structure_probing.md) (label space, protocol, limitations) and [`docs/colab_pro_setup.md`](docs/colab_pro_setup.md) (Colab billing/GPU walkthrough — a **T4 is sufficient**; Pay-As-You-Go recommended).

Headless reproduction of the canonical + label artifacts:

```bash
uv run python scripts/cost_dataset.py verify
uv run python scripts/cost_dataset.py build --output data/cost/java_python.jsonl
uv run python scripts/structure_labels.py verify
uv run python scripts/structure_labels.py extract \
  --input data/cost/java_python.jsonl -o outputs/structure_labels/leaves.jsonl

# Optional: headless per-token activations (.npz + manifest), generalizing activation_pipeline.py
uv run python scripts/structure_activations.py verify
uv run python scripts/structure_activations.py extract \
  --input data/cost/java_python.jsonl --max-rows 200 --model-id Qwen/Qwen2.5-1.5B
```
