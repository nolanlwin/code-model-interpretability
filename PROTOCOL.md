# TEAM PROTOCOL v1.0 — renaming experiments across roles, languages, and models

**Frozen 2026-08-06.** Every member follows this exactly. Any deviation is re-frozen for *everyone*,
never for one role. Every `results.json` records `protocol_version: "1.0"`.

Owners: accumulator / index / iterator / **boolean (Naing)** / class-struct — one role each.

---

## 0. Three corrections before you start

### 0.1 MuST-CoST and XLCoST are two different datasets — pick deliberately

They are **not** the same corpus. Both come from the Reddy lab and both scrape GeeksforGeeks, but they
are separate releases:

| | **MuST-CoST** | **XLCoST** |
|---|---|---|
| Paper | Zhu, Suresh & Reddy, AAAI 2022 | Zhu et al., arXiv 2206.08474 |
| Problems | **1,625** *(verified — I counted the CSVs)* | 11,265 |
| Programs | 7,957 | **57,661** (7.2×) |
| Snippets | 71,033 | 509,091 |
| Languages | C++, Java, Python, C#, JS, PHP, C | identical 7 |
| Go | ❌ | ❌ |
| Distinguishing feature | fine-grained **snippet-level** alignment | program- and snippet-level, much larger |

**Naming — one thing to fix in the paper only.** We call it MuST-CoST internally, after the repo
([github.com/reddy-lab-code-research/MuST-CoST](https://github.com/reddy-lab-code-research/MuST-CoST)),
which is unambiguous and fine. But in the AAAI 2022 paper itself the *dataset* is named **CoST** and
**MuST** is the *pre-training method* (Multilingual Snippet Translation); the repo bundles both, hence
the combined name, and the data file inside is `CoST_data.zip`.

So: say "MuST-CoST" in Slack and in this document, but **cite it in the paper as
"CoST (Zhu, Suresh & Reddy, AAAI 2022)"** with the repo URL for the artifact. The current draft's
sentence "MuST-CoST extends this with additional multilingual snippet-level alignments" will read to an
informed reviewer as a misattribution, since MuST is not a dataset. Free fix.

**Recommendation: unify on XLCoST.** The reason is specific, not just size:

> MuST-CoST's distinguishing feature is snippet-level alignment — and that is precisely the thing that does
> not work for role extraction. **23 of 13,311 MuST-CoST Python snippets parse (0.2%)** because snippets are
> cut mid-function between comments. You must reassemble whole programs to run any AST extractor, at
> which point MuST-CoST is a 7.2× smaller XLCoST with no compensating advantage.

Two further points in XLCoST's favour: the XLCoST authors themselves note MuST-CoST "only has around 70
programs for testing and 50 for validation. Due to the small size…"; and **four of the five roles in
your paper are already on XLCoST**, so unifying there is nearly free for them and costs only the
boolean role a corpus change.

If the team still prefers MuST-CoST, everything else in this protocol applies unchanged — substitute the
corpus name and expect roughly 7× fewer occurrences per cell, which will widen every confidence
interval.

### 0.2 Verified corpus facts *(I measured these on the downloaded MuST-CoST release)*

- **Languages: C++, Java, Python, C#, JavaScript, PHP, C — 1,625 rows each. No Go.** Go is dropped, as
  decided.
- **Snippet-level data is unusable for role extraction: 23 of 13,311 Python snippets parse (0.2%).**
  Snippets are cut mid-function between comments. **Always reassemble whole programs** by concatenating
  a CSV row left-to-right. Never run an extractor on a snippet.
- Whole Python programs still need repair: **171 / 1,625 parse raw**; the dominant error is
  `unexpected indent` from comments re-inserted at column 0. Strip each snippet's leading comment before
  joining (this is what XLCoST already does). ~85 programs are Python 2 — drop them.
- Brace languages reassemble cleanly (C++ ~100%, Java/C#/JS ~99.7%, PHP/C ~99.3%).

### 0.3 The real threat to your headline result

Competitive-programming code **is already renamed**. Running the repo's own boolean extractor over
reassembled MuST-CoST Python, the most frequent "boolean" names are:

```
i (104)  count (98)  n (55)  root (49)  ans (46)  j (44)  sum (36)  res (32)  k (29)
```

Your dissociation (+0.079 accumulator / −0.272 index) is a delta from a baseline of **informative**
names. On this corpus the baseline already sits near your existing "single chars" condition
(F1 0.840 vs 0.893). **Expect both effects to shrink.** That is a construct-validity risk, and no
number of extra models fixes it.

Mitigation, and it is mandatory: report the **baseline name-informativeness** of every corpus slice —
mean identifier length, fraction of single-char names, fraction in a common-generic list — so the
shrinkage is explained rather than discovered by a reviewer.

---

## 1. Frozen shared decisions

| Item | Value |
|---|---|
| Corpus | **XLCoST**, program-level. **DECIDED 2026-08-07.** See §1.2 for how to obtain it — the official release is tokenized and must not be used raw |
| Languages | **Python, Java, JavaScript, PHP** — the CodeSearchNet ∩ XLCoST set |
| C / C++ / C# | **Cut.** No extractor exists, and 2 of 5 roles are undefined in C (no classes, no value-binding loop). State this once with the structural reason |
| Split unit | **problem_id**, grouped — never occurrence-level. The old 80/20 stratified split leaked (85.5% function overlap) |
| Split ratio | 70 / 10 / 20 train / val / test, seeded |
| Probe | `LogisticRegression`, L2, `StandardScaler` fitted on train only |
| Pooling | `mean` over the occurrence's token span |
| Layer selection | **Chosen on validation, reported on test.** Never argmax on test |
| Cross-model layers | Report at **normalized depth** `l/L`; models span 24–36 layers |
| Seeds | 5: `[0,1,2,3,4]` |
| transformers | **Pin `==5.8.0` in pyproject.toml**, and assert the version at runtime |
| Occurrence cap | **2,000 per (role, language)** — sampled ONCE from baseline, reused by every condition |

### 1.2 Corpus acquisition — do NOT use the official XLCoST download

The official release (Google Drive, `pair_data_tok_*`) is **TransCoder-tokenized**: `NEW_LINE` /
`INDENT` / `DEDENT` sentinels, spaces around all punctuation. **0 / 300 official Python programs
parse.** Use these instead — both are HuggingFace, no Drive fetch needed, and record counts match the
paper's Table 2, so they are the complete program-level dataset:

| Language | Source | Measured parse rate |
|---|---|---|
| Python | `giulio98/xlcost-formatted`, `data/Python-program-level/*.json` | 276/300 (92%) |
| C++ | `giulio98/xlcost-formatted` | detokenized |
| Java, C#, JS, PHP, C | `codeparrot/xlcost-text-to-code`, `data/<Lang>-program-level/*.json` | 268/300 (89%) **tokenized, parses as-is** |

Brace languages don't need detokenization — whitespace isn't semantic, so tree-sitter parses the
tokenized form directly. **Only Python does**, which is why the detokenized mirror matters there.

**Mandatory before extraction:** detokenize and run a per-language formatter (`black`,
`google-java-format`, `prettier`, `clang-format`). `System . out . println` is a very different BPE
sequence from `System.out.println`, and this paper is about how the model represents code — feeding it
unnaturally-spaced input is out-of-distribution and a reviewer can challenge every number. ~1 day.

**Verify first:** which form produced the paper's *existing* XLCoST results? If the tokenized release,
those numbers were computed on non-natural code and must be re-run.

### 1.1 The occurrence-set rule (violating this silently invalidates every CI)

Sample the occurrence set **once**, from the baseline condition, grouped by problem, seeded. **Every
renaming condition evaluates that same set**, joined by a stable `occurrence_id`:

```
occurrence_id = (problem_id, language, function_index, binding_index, occurrence_index)
```

Assign it at baseline extraction; the renamer emits it alongside the new span. **Do not join by
character span** — renaming shifts every downstream offset.

Before any paired delta, assert the two conditions' `occurrence_id` arrays are element-wise identical.
Fail the cell otherwise. The paired bootstrap is invalid without this.

---

## 2. Renaming conditions

Six conditions, identical for every role. **"Misleading" must mean the same *kind* of manipulation for
every role** or the dissociation compares apples to oranges.

| ID | Name | Construction rule |
|---|---|---|
| C0 | Baseline | Unchanged source |
| C1 | Neutral numeric | Every local identifier → `v1, v2, …` in declaration order |
| C2 | Single char | → `a, b, c, …` in declaration order |
| C3 | All-same | Every *target-role* variable → one identifier (`x`) |
| C4 | Random nouns | Sampled from a fixed noun list, seeded, disjoint from every role's name pool |
| C5 | **Misleading** | Target-role variable → a name sampled from the **empirical name distribution of a different role** in the same corpus |

**C5 is the load-bearing one.** Build one name pool per role from C0 (top-50 identifiers for that role,
by frequency, across all four languages). For role *R*, sample C5 names from `pool(R')` where `R'` is a
fixed partner role, declared in advance and identical across languages:

```
accumulator → index      index → accumulator      iterator → accumulator
boolean     → index      class/struct → variable-style names
```

Pools must be **disjoint** — assert no name appears in two pools before running.

### 2.1 Renaming requirements

- **Scope-correct.** Respect binding scope; never rename fields, methods, library names, or `self`/`this`.
- **Never rename inside strings or comments.**
- **Re-parse after renaming.** A program that fails to parse post-rename is dropped from *all*
  conditions, not just that one.
- **Report the token-count delta.** Renaming changes BPE token counts, which changes sequence length and
  every downstream position. Report mean |Δtokens| per condition; if it exceeds 15%, add a
  token-length-matched subset as a robustness row.

### 2.2 The layer-0 rule

Layer 0 is `embed_tokens(input_ids)` pooled over the identifier's own tokens — a **deterministic
function of the identifier string** (verified: 187 Java occurrences → 85 distinct layer-0 vectors; 20 of
28 repeated names bit-identical at layer 0, 0 of 28 at the final layer).

So **renaming changes layer 0 by construction.** Two consequences:

1. Never report a layer-0 delta as a finding. Label the axis `embed, 1…L`, never `0…L`.
2. **Use it as a construction-validity check:** under C1/C2, the name-only baseline must fall to chance.
   If it doesn't, the renamer leaked and the cell is void.

---

## 3. Models

| Model | Layers | Status |
|---|---|---|
| Qwen2.5-Coder-1.5B | 28 | clean |
| Qwen3-4B-Base | 36 | clean |
| Qwen2.5-Coder-7B | 28 | clean — **vocab 152064, untied**; re-verify, do not assume family uniformity |
| StarCoder2-7B | 32 | clean |
| Granite-3B-Code-Base | 32 | clean |
| DeepSeek-Coder-1.3B | 24 | **see below** |

### 3.1 DeepSeek-Coder is mandatory-fix

Under the pinned transformers 5.8.0, `AutoTokenizer` yields **11/11 wrong offsets** and deletes
whitespace (`"if is_valid:"` → `"ifis_valid:"`). Cause: `tokenizer_class=LlamaTokenizerFast` declared
over a ByteLevel `tokenizer.json`; transformers 5.x installs a Metaspace pre-tokenizer over it.

```python
tok = PreTrainedTokenizerFast.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")  # NOT AutoTokenizer
```

**Any DeepSeek activations already extracted must be re-run.** Sree's index-role numbers included.

### 3.2 The tokenizer gate — every model, before any extraction

A roundtrip check is **not sufficient**: Yi-Coder round-trips perfectly but has 8/17 wrong offsets.
Assert per token:

```python
for i, (s, e) in enumerate(offsets):
    assert src[s:e] == tok.decode([ids[i]])
```

Run it on a fixture containing non-ASCII, tabs, CRLF, and digits-in-identifiers. Also fix the
byte-vs-char bug: tree-sitter paths pass `node.start_byte/end_byte` into a character-indexed
`offset_mapping`.

---

## 4. Metrics and reporting

Reviewer QBg8 asked for **"confidence intervals or bootstrap estimates."** That is uncertainty on the
metric — not classifier confidence. **Do not ship a column headed "confidence" containing mean softmax;
it reads as evasion.**

Report per cell:

- `macro_f1` with a **95% BCa bootstrap CI, clustered at problem_id**, 10,000 resamples
- `delta_f1` vs C0 with a **paired** clustered BCa CI — baseline and renamed share occurrences, so the
  paired interval is substantially tighter and is the paper's central quantity
- `majority_baseline`, `name_only_baseline`, `selectivity` (probe F1 − Hewitt control-task F1)
- `n_occurrences`, `n_problems`, `n_clusters`, `max_cluster_share`
- per-class F1, support, confusion matrix

**Report the cluster count with every CI.** Clustered bootstrap is unreliable below ~30–40 clusters.

Selectivity is not optional: QBg8's probe-capacity complaint is *not* answered by CIs. The Hewitt
control task — random label per identifier type, drawn from the empirical marginal — must run for
**all five roles**, not just class/struct.

### 4.1 `results.json` schema (identical for every member)

```json
{"protocol_version":"1.0","role":"boolean","owner":"naing","corpus":"xlcost",
 "language":"python","model":"Qwen/Qwen2.5-Coder-1.5B","condition":"C5",
 "transformers_version":"5.8.0","seeds":[0,1,2,3,4],
 "split":{"unit":"problem_id","ratio":[0.7,0.1,0.2],"seed":0,
          "n_train":0,"n_val":0,"n_test":0,"n_clusters":0,"max_cluster_share":0.0},
 "selected_layer":14,"normalized_depth":0.50,"layer_selected_on":"validation",
 "macro_f1":0.0,"macro_f1_ci":[0.0,0.0],
 "delta_f1_vs_C0":0.0,"delta_f1_ci":[0.0,0.0],
 "majority_baseline":0.0,"name_only_baseline":0.0,"selectivity":0.0,
 "per_class_f1":{},"support":{},"confusion_matrix":[[]],
 "n_occurrences":0,"mean_token_delta":0.0,"occurrence_id_hash":"sha256:..."}
```

`occurrence_id_hash` is the hash of the sorted `occurrence_id` list — two conditions with different
hashes cannot be paired, and the evaluator must refuse.

---

## 5. Validation gates — a number does not count until all pass

1. Tokenizer gate green for that model (per-token offsets, non-ASCII fixture).
2. Every program parses post-rename; drop-set identical across conditions.
3. `occurrence_id` sets element-wise identical across all six conditions.
4. Name pools disjoint; C5 partner role matches the table in §2.
5. Layer selected on validation; test touched once.
6. `n_clusters` ≥ 30, reported.
7. Name-only baseline at chance under C1/C2.
8. `transformers.__version__ == "5.8.0"`.
9. Role predicate agreement: label 100 occurrences by hand, report precision; **and** diff against the
   old extractor on the same programs, so the new numbers are continuous with the rejected paper's.

---

## 6. Shared infrastructure — build once, everyone uses

| Component | Owner | Days |
|---|---|---|
| XLCoST loader + program reassembly + repair + parse validation | infra | 3 |
| Tokenizer registry + gate (6 models) | infra | 2 |
| Scope-correct renamer, 4 languages, with re-parse verification | infra | 6 |
| `probe.py` — grouped splits, seeds, all baselines, selectivity | stats | 3 |
| `bootstrap.py` — paired clustered BCa | stats | 2 |
| Results-cube serialization + resume | infra | 2 |

**Per member:** only the role predicate (4 languages), the name pool, and running the pipeline.

---

## 7. Honest scope warning

6 models × 5 roles × 6 conditions × 4 languages = **720 cells**. The binding constraint is
**person-days, not GPU-hours**. Adding a model is ~1 engineer-hour; adding a *language* is ~5
engineer-days.

Independent execution review of this plan concluded the corpus switch plus the renamer is roughly
48–65 person-days against 32–48 available before 29 Aug, and recommended shipping a
**methods-correction** paper first (grouped splits, paired CIs, selectivity on all five roles, offset
and tokenizer fixes, one or two extra models on the *existing* conditions — all re-analysis of cached
activations) with the unified corpus and standardized renamer going to ICLR on 25 Sep.

That advice is recorded here; the corpus decision is the team's. If 29 Aug is kept, cut to **2 models ×
2 languages** for the full condition grid and treat the remaining models as opportunistic extensions.

**Day 1, before anything else:** inventory what exists. For each role — who has the extractor code, do
the XLCoST activations still exist and on whose Drive, which models and pooling. This repo contains
**one** of the five role extractors (boolean) and zero lines of renaming code. Whether 29 Aug is a
re-analysis or a rebuild depends entirely on that answer.
