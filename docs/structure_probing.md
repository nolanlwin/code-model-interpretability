# Structure probing — syntactic-class probes over CoST (Java + Python)

This experiment extends the repo's layer-by-layer linear-probe method (originally the
`boolean_flag` True/False probe) to a **multi-class syntactic-structure** probe, and adds a
**cross-lingual transfer** test using the parallel CoST corpus. The notebook runs the full
pipeline on **two models** for a controlled comparison: `Qwen/Qwen2.5-1.5B` (base) and
`Qwen/Qwen2.5-Coder-1.5B` (code-pretrained), loading and freeing each model in turn so only
one set of activations is in host RAM at a time.

Primary artifact: [`notebooks/structure_probing_cost.ipynb`](../notebooks/structure_probing_cost.ipynb)
(Colab-ready). Supporting CLIs: `scripts/cost_dataset.py`, `scripts/structure_labels.py`.

## Questions

1. For each token, predict its **syntactic class**. Where (which layer) is that class most
   linearly decodable, and does a probe trained on one language transfer to the other?
2. **Selectivity (Hewitt control task):** how much of the within-language score is the
   representation encoding structure vs. the probe memorizing token identity?
3. **Identifier roles:** can the model linearly separate `variable / function / type /
   parameter` — a task where token identity is *not* sufficient?
4. Does **code pretraining** (Coder vs base) change any of the above (transfer gap,
   selectivity, identifier-role depth)?

## Dataset

- **Source:** `consolidated_data.csv` (local) — the CoST / MuST-CoST corpus of GeeksforGeeks
  problems, one row per problem, one column per language
  (`Problem ID, Problem Title, C++, Java, Python, C#, Javascript, PHP, C`).
- **Scope:** Java + Python only. We keep the **1,417** problems that have a non-empty program
  in *both* languages, so the two languages stay aligned on `problem_id`.
- **Canonical record** (`scripts/cost_dataset.py build`): one JSON object per
  `(problem, language)`:
  - `problem_id`, `language` (`java` | `python`), `code`, `title`

## Label space (shared across Java and Python)

Labels come from **tree-sitter** (`scripts/structure_labels.py`): every **leaf** node of the
parse tree is mapped to one coarse `structural_class`. The notebook uses the 7-class space
(comments excluded by default):

| class | Python leaves | Java leaves |
|-------|---------------|-------------|
| `identifier` | `identifier` | `identifier`, `type_identifier`, `field_identifier`, `scoped_identifier` |
| `keyword` | `def`, `class`, `if`, `for`, `return`, `and`/`or`/`not`/`in`/`is`, ... | `class`, `int`, `if`, `return`, `instanceof`, ... |
| `operator` | `+ - * / % == < > = ...` | `+ - * / % == < > = ...` |
| `string` | `string`, `string_content`, `escape_sequence` | `string_literal`, `character_literal` |
| `number` | `integer`, `float` | `decimal_integer_literal`, `hex_integer_literal`, ... |
| `punctuation` | `( ) [ ] { } , ; : .` | `( ) [ ] { } , ; . ->` |
| `bool_null` | `True`, `False`, `None` | `true`, `false`, `null` |
| `comment` *(optional)* | `comment` | `line_comment`, `block_comment` |

Anonymous alphabetic tokens (e.g. `and`, `instanceof`) are classified as `keyword`; word
operators are therefore folded into `keyword` by design.

### Token assignment

- tree-sitter reports **byte** offsets; these are converted to **character** offsets so they
  align with the tokenizer `offset_mapping` (same contract as `scripts/token_alignment.py`).
- Each Qwen token inherits the class of the **first labeled character** it overlaps. Tokens
  that overlap no labeled leaf (whitespace-only, special tokens) are dropped from `X`/`y`.

## Probing protocol

- One forward per program (`AutoModel`, `output_hidden_states=True`) -> hidden-state tuple of
  length `num_hidden_layers + 1 = 29` (layer 0 is the embedding output), each `[1, seq, H]`,
  `H = 1536`.
- **Memory control:** structure labels are *dense* (every token). The notebook caps programs
  (`MAX_PROGRAMS_PER_LANG`, default 300) and **subsamples a per-class-balanced token pool**
  (`MAX_TOKENS_PER_LANG`, default 12,000) before stacking activations, stored as **float16**.
  Cost is `~ MAX_TOKENS_PER_LANG x 29 x 1536 x 2 bytes` (~1 GB/language at the defaults); only
  one model's activations are resident at a time.
- **Split:** by `problem_id` (default 80/20) so no program straddles train/test.
- **Probe:** `LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')`
  fit independently at each layer. Metric: **macro-F1** (robust to the identifier/punctuation
  imbalance) plus accuracy. **Majority-class baseline** reported for reference.
- **Within-language:** Python-only and Java-only curves (per-class F1 also tracked).
- **Cross-lingual:** fit on all of one language's tokens, evaluate on the other's at the same
  layer (`python->java` and `java->python`).
- **Control task (selectivity):** each token *type* (input id) is assigned a fixed random
  class sampled from the empirical class marginal; the same probe is retrained to predict
  those control labels. `selectivity = real macro-F1 − control macro-F1` per layer. High
  control F1 means the score is largely token-identity recall, not structural representation.
- **Identifier-role probe:** a second labeling pass splits `identifier` into
  `variable / function / type / parameter` from tree-sitter parent context, then re-runs the
  within + cross-lingual probes on the same model activations.

All of the above run identically for **both** models so results are directly comparable.

## Outputs (written to `output/` by the notebook)

- `structure_probe_macro_f1.png` — macro-F1 vs layer (within + cross), one panel per model.
- `structure_confusion_python.png` — best-layer Python confusion, one panel per model.
- `structure_selectivity.png` — selectivity and control F1 vs layer, per language/model.
- `structure_per_class_f1.png` — per-class F1 vs layer (Python), per model.
- `identifier_role_macro_f1.png` — identifier-role within + cross curves, per model.
- `structure_probe_results.json` — nested under `models.<name>` with per-layer macro-F1 /
  accuracy / per-class F1 for every condition, control, selectivity, and identifier roles.

## Reproducing the canonical/label artifacts (optional, headless)

```bash
uv run python scripts/cost_dataset.py verify
uv run python scripts/cost_dataset.py build --output data/cost/java_python.jsonl
uv run python scripts/structure_labels.py verify
uv run python scripts/structure_labels.py extract \
  --input data/cost/java_python.jsonl -o outputs/structure_labels/leaves.jsonl
```

## Known limitations

- **Leaf-class probing is shallow:** identifiers and punctuation dominate; `class_weight`
  and macro-F1 mitigate but do not remove the imbalance.
- **First-char token assignment:** a token spanning two leaves takes the first labeled
  class; rare for code (most tokens sit inside a single leaf).
- **Selectivity is now reported** via the Hewitt control task; note that for coarse
  structural class the control F1 is expected to be high (class ≈ token identity), so read
  the *identifier-role* selectivity for a stronger representation claim.
- **Identifier-role labels are heuristic:** parent-context rules cover the common cases
  (def/class names, calls, parameters, type identifiers) but edge cases exist (e.g. a Python
  constructor call `Stack()` is labeled `function`; Java `System.out` receivers are
  `variable`). Good enough for a coarse role probe, not a typed-AST ground truth.
- **Cross-lingual confound:** Python and Java tokenize differently (e.g. Python `string` is
  far more frequent); macro-F1 over a fixed class list keeps the comparison fair but absolute
  transfer numbers still reflect class-frequency differences.
- **Word operators as keywords:** `and`/`or`/`not`/`instanceof` are labeled `keyword`, not
  `operator`, because they are keyword tokens in both grammars.
