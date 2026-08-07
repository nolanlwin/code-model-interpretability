# Resubmission plan — *When Names Matter and When They Don't*

**Written 2026-08-06.** Supersedes the earlier draft of this file, which was written before I had the
paper and the reviews and which assumed the repo's boolean pipeline *was* the state of the work.

- Paper: [`docs/paper/when_names_matter_aiw2026.pdf`](paper/when_names_matter_aiw2026.pdf)
- Reviews: <https://openreview.net/forum?id=RWolTNOBxp> (AIW 2026 @ COLM, submission 117)
- Outcome: **Reject** (24 Jul 2026), on two ratings of **4 — weak accept**, both confidence 3.
- Scope of this plan: the whole resubmission, with the **boolean workstream (§4.4) called out
  separately** since that is your part.

Venue dates below were fetched from the official CFP pages on 2026-08-06. Numbers marked *(measured)*
were computed in this repository today and are reproducible.

---

## 1. What the reviews actually said

Two weak accepts and a reject means nothing was *wrong* — the paper was outscored. Both reviewers
liked the same thing and disliked overlapping things, so the revision target is unusually clear.

**Both reviewers' single favourite result is the accumulator/index dissociation** (accumulator
ΔF1 = +0.079 under misleading renaming vs index −0.272/−0.285). QBg8: "clean, falsifiable, and an
informative finding," "not reducible to prior syntax-probing work." esk2: "the strongest and most
original contribution." **That is the paper's spine. Everything else should be rebuilt to serve it.**

Converging asks, in the order they will decide the next outcome:

| # | Ask | Who | Notes |
|---|---|---|---|
| 1 | **Causal validation** — patching, ablation, or steering | esk2 (top con) | "linear probes can recover information the model does not causally use" |
| 2 | **Control task / probe-capacity check on the headline roles** | QBg8 | Sharp: the Hewitt control "is already used elsewhere in the paper (§4.5, §5)" but not where it matters |
| 3 | **Uncertainty quantification** — CIs or bootstrap | both | Currently every number is a bare point estimate |
| 4 | **One unified narrative** | esk2 | "reads as a collection of probing results"; §6 PCA "feels like two separate papers" |
| 5 | **Model coverage beyond the Qwen family** | both | DeepSeek-Coder-1.3B "appears once, only for index, with no stated rationale" |
| 6 | **Tables for results that exist only as figures** | esk2 | Cheap, do it |
| 7 | **"Many results are unsurprising"** | esk2 | Framing problem, not an experiment problem — see §3 |

Items 2, 3, 6 are days of work. Item 1 is the real cost. Item 4 is free and probably worth the most
per hour.

---

## 2. The boolean section is the weakest part of the paper — and that is the opportunity

§4.4 is **five sentences, one table, three figures**. Measured against the other four roles it is
missing every control the paper already knows how to run:

| | accumulator | index | iterator | **boolean** | class/struct |
|---|---|---|---|---|---|
| Renaming conditions | 6 | 3 | — | **none** | 3 |
| Cross-language transfer | ✓ | ✓ | — | **none** | ✓ |
| Token-identity control / selectivity | — | — | — | **none** | ✓ (0.19–0.21) |
| Second model | — | ✓ | ✓ | **none** | — |

Reviewer QBg8's "probe-capacity concerns go untested for the headline roles" lands squarely here.

### 2.1 Table 4's Go row is a null, and it is currently written as if it were a result

| Language | Layer-0 acc | Layer-0 F1 | Peak layer | Peak F1 | ΔF1 |
|---|---|---|---|---|---|
| Python | 0.702 | 0.428 | 14 | 0.980 | **+0.552** |
| Java | 0.908 | 0.652 | 17 | 0.971 | **+0.319** |
| Go | 0.992 | **0.886** | 18 | **0.890** | **+0.004** |

In Go, 28 transformer layers add **0.004 macro-F1** over the embedding layer. The abstract and
conclusion list "0.890 for Go" alongside Python and Java as if it were the same kind of number. It
isn't — it is a null, and a reviewer who reads the table will see the claim undercut by its own row.

Using layer 0 as the baseline was the right instinct and I verified *why* it is the right baseline:
in Qwen2, `hidden_states[0]` is `embed_tokens(input_ids)` — RoPE is applied inside attention and never
added to the residual — and `_pool_tokens` pools only over the occurrence's own token span. So layer 0
is a deterministic, context-free, position-free function of the identifier string. Confirmed on the
real Java cache *(measured)*: 187 occurrences produce only **85 distinct layer-0 vectors**; no layer-0
group ever contains two different variable names; of 28 names appearing at multiple sites, **20 are
bit-identical at layer 0** and **0 of 28** at the final layer.

**So Table 4's layer-0 column *is* a rigorous name-only baseline. Say that explicitly in the paper —
it is a methodological strength you are currently getting no credit for.**

### 2.2 But layer 0 controls the *name*, not the *context* — and that gap is the reviewable hole

`occurrence_type` is defined by the enclosing AST node, which is visible in the surrounding source
text. A probe at layer 14 sees a contextualised representation; beating a name-only baseline does not
show it is doing anything beyond re-encoding the adjacent `return` / `if` / `for` token.

I built a canonical corpus from the frozen validation shard (9,718 snippets → 28,554 occurrences) and
fit character-n-gram classifiers **with no model loaded at all** *(measured)*:

| Feature set (no LLM) | random split | grouped by function | grouped by repo |
|---|---|---|---|
| Majority class | 0.647 acc | 0.655 | 0.641 |
| Variable **name** only | 0.690 / F1 0.362 | 0.688 / 0.352 | 0.633 / 0.279 |
| ±120-char window, name masked | 0.835 / 0.576 | 0.848 / 0.578 | 0.844 / 0.520 |
| **Source line, name masked** | **0.948 / F1 0.777** | **0.961 / 0.813** | **0.960 / 0.793** |

**The good news, and it is genuinely good:** Python's peak probe macro-F1 of **0.980** still clears a
model-free context baseline of **≈0.78–0.81** by roughly **+0.17–0.20**. The boolean Python result
looks like it survives the control that is currently missing. Java (0.971) very likely does too.

**The bad news:** Go's 0.890 peak sits *below* this baseline range in a language where the layer-0
number is already 0.886. Go is lexical, end to end.

Caveat to check before quoting any of this: my occurrence set comes from the validation shard, so its
class balance (`conditional_use` 18,485 / `assignment` 5,323 / `return_use` 4,297 / `loop_use` 298 /
`indexing_use` 151) may differ from the paper's n = 7,981 Python probe set. Macro-F1 is very sensitive
to that. **Recompute on the exact probe set before it goes in a table.**

### 2.3 Two more things a reviewer will find

**The split is stratified, not grouped.** Table 4's caption says "80/20 stratified split." On this
corpus *(measured)* **85.5%** of test occurrences come from a function that also appears in train and
**79.1%** share a variable name with train. The repo-level split `dataset_v0.py` already builds is
never loaded by the probing path. Re-running under repo-grouped splits is a few hours and either
removes an objection or tells you something you need to know.

**The `boolean_flag` construct is loose.** `names_in_boolean_test` unconditionally accepts any bare
`Load` Name, so `if items:` labels a list and `return x` labels anything. Of 19,514 distinct
`(row, function, variable)` labels *(measured)*, **34.0%** have any strong boolean evidence
(bool-literal assignment, `return True/False`, `not name`) and **60.1%** rest only on the variable
appearing in a truthiness test. Most frequent labeled "boolean flags": `result`, `value`, `data`,
`response`, `self`, `i`, `x`. The paper says labels come from "lightweight static analysis plus manual
audit" — if the audit covered boolean, report its precision; if it didn't, this is a one-day fix
(§4, B6) that closes a hole a software-engineering reviewer will drive through.

---

## 3. The reframe: make it one paper

esk2's "reads as a collection of probing results" and "many results are unsurprising" are the same
complaint. The fix is not more experiments — it is one axis that every section reports against.

> **Axis: for each variable role, is the model reading the name or computing the structure?**

- **accumulator** → structural (improves under misleading names, +0.079)
- **index** → lexical (collapses, −0.272 / −0.285)
- **boolean** → *language-dependent*: structural in Python (+0.552 over the name-only baseline, and
  still ahead of a context baseline), **null in Go** (+0.004)
- **class/struct** → lexical-plus-contextual (selectivity 0.19–0.21, but raw F1 gains under renaming
  are identity-driven)
- **iterator** → currently unclassified, because it has no renaming condition (QBg8's complaint)

Under this framing the boolean role stops being the thinnest section and becomes **the third and most
interesting point on the axis: the one showing the lexical/structural split is not a property of the
role alone but of the role × language.** That is a genuinely non-obvious claim, it is not in any of the
prior work you cite, and it directly answers "results are unsurprising."

It also tells you what to do with §6: **the PCA language-identity analysis moves to an appendix, or it
becomes the mechanism section for the boolean finding** — if language identity is a low-dimensional
early subspace (r = +0.883 at layer 0, peaking layer 4), that is a candidate explanation for why the
same role is structural in one language and lexical in another. Wired in that way it stops being "two
separate papers." Wired in no way, cut it.

---

## 4. The boolean workstream — your part, prioritised

Everything here runs on **CPU against activations you already have**, except B5.

**B1 — Context baseline at every layer (½ day, highest value).**
Add two model-free rows to Table 4: masked-source-line char n-grams, and a bag-of-context-token
baseline over the ±k Qwen BPE token ids around the occurrence. Report probe F1 as a **delta over the
stronger of {layer 0, context}**, not over layer 0 alone. A prototype is at `scratchpad/baselines.py`;
promote it to `scripts/baselines.py`. This converts your section's biggest liability into a control
table.

**B2 — Hewitt control task and selectivity (½ day).**
Exactly what QBg8 asked for, and the paper already runs it in §4.5/§5, so it is a copy of existing
code: assign each identifier *type* a random label from the empirical `occurrence_type` marginal,
re-run the layer sweep, report selectivity = probe F1 − control F1. Without this, "0.980" is not
defensible against a probe-capacity objection.

**B3 — Grouped splits and bootstrap CIs (1 day).**
Repo-grouped and function-grouped splits alongside the current stratified one — report all three, the
divergence is itself informative. Five seeds, repo-clustered bootstrap CIs on every number. **Report
the cluster count and the max-repo share**; the local Java sample spans 8 repos with 62% from one, and
clustered CIs below ~30 clusters are not trustworthy. This closes ask #3 for your section.

**B4 — The renaming condition boolean is missing (1–2 days, needs one GPU pass).**
The paper's signature manipulation, applied to boolean for the first time: α-rename flag variables to
`v1, v2…` (neutral), to a single token (all-same), and to index-style names (misleading), holding the
AST fixed. Prediction worth writing down *before* you run it: **Python drops moderately, Go barely
moves** (it has nothing to lose — it is already at its lexical ceiling). If that holds, the boolean
row joins the dissociation table as a proper entry rather than a footnote.

**B5 — One causal experiment (3–5 days, the expensive one).**
Minimal pairs editing one line — `flag = True` vs `flag = 0` — token-count matched so the sequences
align index-for-index and the identifier is byte-identical across the pair. Readout is **logit
difference**, never probability (Zhang & Nanda, ICLR 2024). Single-layer residual patching, **both
directions**: denoising for sufficiency, noising for necessity. Controls: random-position patching,
random-direction patching at matched norm, and **never zero-ablation** — resample or mean ablation
against a stated reference distribution. Report dtype and logit determinism; patching effects on a
1.5B model in fp16 can sit inside numerical noise.
*If the team can only afford one causal experiment for the whole paper, boolean is the right role to
spend it on:* the readout is binary and unambiguous, the counterfactual is a one-line edit, and it
directly tests the Python-structural / Go-lexical claim.

**B6 — Label precision, scoped small (1 day).**
Don't promise per-pattern precision on 200 sites — 5 occurrence types × 6 detection patterns gives ~7
per cell. Draw **120 sites stratified 60/60** between the dominant pattern and everything else, two
annotators, pre-registered guideline, report Cohen's kappa and precision on the dominant pattern. Then
check whether the probe's **errors correlate with the human-adjudicated label errors** — that is the
cleanest evidence about whether the probe learned the labeler's mistakes, and it is free once the
sample exists.

**Do not** spend time extending boolean to six languages. Python + Java + Go is enough for the
role × language claim, and more languages add surface without adding argument.

---

## 5. Paper-level changes (not yours, but they gate the outcome)

- **Add one causal experiment.** Both reviewers' top con; the second reviewer names it first. B5 is
  the cheapest credible version.
- **Give iterator its renaming condition**, or drop it from the abstract's list of five. QBg8 counted.
- **Justify or drop the lone DeepSeek-Coder table.** One extra model on *one* role reads as an
  accident. Either run it on the two dissociation roles (accumulator + index) so it functions as a
  replication, or move it to an appendix and stop listing it as coverage.
- **Tables for every figure-only result.** Free.
- **Move PCA to an appendix** unless it is wired to the role findings as in §3.
- **State the layer-0-is-the-embedding-table argument explicitly**, and label every layer axis
  **"embed, 1…28"**, never "0…28." Your central baseline depends on that index being understood.

---

## 6. Venue and timeline

**There is no Mechanistic Interpretability workshop at NeurIPS 2026.** That series ran ICML 2024 →
NeurIPS 2025 → **ICML 2026** (Seoul, 10 July; deadline was 8 May). Verified live options:

| Venue | Deadline | Pages | Fit |
|---|---|---|---|
| **InterpScience** — *Interpretability as a Science* (NeurIPS, Sydney) | **28 Aug 2026 AoE** | 5 / 9 | Best topical fit: measurement validity, causal claims, falsifiability. **Prohibits papers under review at any other workshop.** |
| **Interp4Discovery** (NeurIPS, Atlanta) | **29 Aug 2026 AoE** | 5 (6 CR) | Double-blind; "failure cases and negative results are welcome"; concurrent ICLR/NeurIPS permitted |
| **XAI4Science** (NeurIPS, Sydney) | **29 Aug 2026 AoE** | 8 / 5 | Anonymity optional; concurrent submission to other venues explicitly allowed |
| **ICLR 2027** | abstract 18 Sep / paper **25 Sep 2026** | full | The real target once B5 exists |

NeurIPS-wide: mandatory workshop notification 29 Sep 2026. BlackboxNLP 2026 direct (17 Jul) and its
Reproducibility track (24 Jul) are closed; MSR 2027 Data & Tool Showcase (~10 Nov) remains as an
artifact venue for the dataset and the six-language extractors.

**Recommended path.** You have **22 days** to 28 Aug and a paper that already exists. That is enough
for B1+B2+B3+B6 and the paper-level tidy-up, but **not** for B5 done properly.

1. **28/29 Aug — resubmit the revised paper without new causal work.** Target **XAI4Science** (8 pages,
   concurrent submission allowed, so it does not block ICLR). The revision is: unified dissociation
   framing (§3), boolean promoted to a full role with B1–B4, controls and CIs everywhere, PCA to
   appendix, iterator fixed or dropped. That addresses asks 2–7 — five of the seven — and two weak
   accepts with those closed is a different conversation.
2. **25 Sep — ICLR 2027** with B5 added. Four weeks after the workshop, which is exactly the right
   amount of time for one patching experiment.
3. Do **not** submit to InterpScience if you want the XAI4Science option — it forbids concurrent
   workshop review. Pick one.

### 22-day sketch

| Days | Boolean (you) | Paper-level |
|---|---|---|
| 1–2 | B1 context baselines, B2 control task | Reframe outline; PCA → appendix |
| 3–4 | B3 grouped splits + bootstrap CIs | Iterator: renaming or removal decision |
| 5–9 | B4 renaming pass (one GPU run) | DeepSeek decision; figure→table sweep |
| 10–12 | B6 annotation; recompute Table 4 with all baselines | Rewrite §1–3 around the dissociation axis |
| 13–17 | Boolean section rewrite | Full draft |
| 18–20 | — | Red-team pass, limitations, anonymity scrub |
| 21–22 | Buffer | Submit |

**Register OpenReview accounts now if any co-author lacks one** — approval can take up to two weeks.

---

## 7. What would change the plan

- **B1 shows the context baseline at or above 0.98 on the paper's actual Python probe set** → the
  Python boolean result is surface-driven too, the role becomes uniformly lexical, and §4.4 turns into
  a negative result. Still publishable, but the framing in §3 must change rather than be forced.
- **B2 shows low selectivity** (control-task F1 close to probe F1) → the probe has capacity to
  memorise at this n; drop to a lower-capacity probe or report MDL codelength instead of accuracy.
- **B4 shows Python drops as hard as Go under misleading renaming** → the role × language claim dies;
  fall back to reporting boolean as a second lexical role alongside index, which still strengthens the
  dissociation table by adding a point.

---

## 7b. Causal work and multi-model expansion (added 2026-08-06)

### URGENT: the DeepSeek-Coder tokenizer is broken in this environment *(verified)*

Run in the repo venv (transformers 5.8.0), on `"if is_valid:\n    total += count\n    return found"`:

| Model | Loader → class | vocab | roundtrip exact | offsets wrong |
|---|---|---|---|---|
| Qwen2.5-1.5B | AutoTokenizer → `Qwen2Tokenizer` | 151643 | ✅ | 0/12 |
| Qwen2.5-Coder-1.5B | AutoTokenizer → `Qwen2Tokenizer` | 151643 | ✅ | 0/12 |
| Qwen3-1.7B-Base | AutoTokenizer → `Qwen2Tokenizer` | 151643 | ✅ | 0/12 |
| StarCoder2-3B | AutoTokenizer → `GPT2Tokenizer` | 49152 | ✅ | 0/12 |
| **DeepSeek-Coder-1.3B** | **AutoTokenizer → `LlamaTokenizer`** | 32000 | **❌** | **11/11** |
| DeepSeek-Coder-1.3B | `PreTrainedTokenizerFast` → `TokenizersBackend` | 32000 | ✅ | 0/14 |

`AutoTokenizer` on DeepSeek-Coder decodes `"if is_valid:\n    return found"` to
**`"ifis_valid:returnfound"`** — every space deleted — and **every** `offset_mapping` span is wrong
(token `'valid'` maps to source span `(2,5)` = `' is'`). Cause: the repo declares
`tokenizer_class=LlamaTokenizerFast` over a ByteLevel `tokenizer.json`, and transformers 5.x installs a
Metaspace pre-tokenizer over it. Fixed upstream in transformers ≥ 5.14.0 (merged 2026-07-15); this repo
is pinned at 5.8.0, inside the broken window.

**Consequence:** `scripts/token_alignment.py` builds every occurrence's token positions from
`offset_mapping`. If the paper's DeepSeek-Coder-1.3B index-role column (ΔF1 = −0.285) was produced in
this environment, **it was pooled from the wrong tokens and must be re-run.** Check which transformers
version produced it before anything else. Fix is one line: load via `PreTrainedTokenizerFast`, or
upgrade to ≥ 5.14.0 and re-verify.

This also corrects an earlier claim in this plan: DeepSeek-Coder *is* byte-level BPE, but that does not
make it safe here — the declared tokenizer class overrides the pre-tokenizer.

### There is no small dense Qwen3-Coder

Verified on the Hub: the Qwen3 code line is `Qwen3-Coder-30B-A3B-Instruct` and `Qwen3-Coder-Next` (80B),
both instruct-only MoE. No base model, nothing under 10B. The entire 2026 code-model wave is large
sparse MoE and out of Colab reach. `Qwen3.5-4B` is a **vision-language** model (248K vocab) — not a
drop-in. So the recency answer is the general line: **`Qwen3-1.7B-Base`**, which I verified uses the
*identical* tokenizer to Qwen2.5 (0/12 bad offsets, same vocab) — meaning it costs almost nothing in
pipeline work.

### The scoping insight that makes multi-model affordable

**Multi-model *probing* and multi-model *causal* work have completely different costs.**

- Probing requires re-running the whole correlational pipeline per model, because each tokenizer
  changes occurrence alignment: 8,000 occurrences × 3 languages × 6 renaming conditions × 4 models
  ≈ 576,000 extractions, plus per-model alignment re-verification. **This does not fit before 25 Sep.**
- The causal experiments use a **fixed ~400-item prompt set** that is model-independent. Running it on
  a second and third model is cheap.

So state the scope deliberately rather than letting it read as an oversight: *"probing results are
reported for the models of the original submission; causal results are reported for N models, because
the causal task uses a fixed prompt set requiring no per-model re-extraction."* That answers esk2's
generality complaint on the half where it is affordable.

### Recommended model set

| Model | Role in the argument | Marginal cost |
|---|---|---|
| Qwen2.5-1.5B + Qwen2.5-Coder-1.5B | Keep as anchor. Code-specialization contrast at fixed scale | zero |
| Qwen3-1.7B-Base | Recency answer; identical tokenizer | ~zero |
| StarCoder2-3B | Different family, tokenizer, corpus (The Stack v2); fully auditable training data | new tokenizer |
| DeepSeek-Coder-1.3B | Completes QBg8's "appears once" complaint — **only after the tokenizer fix** | new tokenizer + re-run |

Avoid CodeLlama and CodeGemma for now: CodeGemma is gated (401 unauthenticated), and CodeLlama shares
the Llama tokenizer-class hazard above. Describe the set honestly as **"two families plus a
specialization control and a generation control"** — not as four independent samples.

### The behavioral floor exists *(measured)*

288 counterbalanced prompts on Qwen2.5-1.5B, readout `assert <flag> is` → logit(` True`) − logit(` False`):
overall accuracy **0.767**, mean signed logit difference **+1.290**. On the 144 **conflict** trials where
the binding and the nearest boolean literal disagree, the model follows the **binding 73.6%** vs the
nearest literal 26.4%. Patching has a real target.

**Caveat that must be handled:** a 27-point class asymmetry (`True` 0.903 / `False` 0.632). Report
calibrated accuracy per ground-truth class, require both to clear the gate independently, and compute
the patching recovery ratio separately per flip direction with its own denominator.

### The trap: boolean value-binding is the wrong headline

Localizing a boolean's *bound value* is a domain replication of Feng & Steinhardt (ICLR 2024),
Wu/Geiger/Millière (ICML 2025) and Prakash et al. (ICLR 2024). It also tests a **different construct
than the paper claims** — value, not role — so esk2's "unsurprising" survives.

Boolean's correct roles are (a) the **de-risked causal substrate** — it is the only readout whose floor
has actually been measured — and (b) the **third point on the lexical/structural axis**. The headline
causal claim should test the accumulator/index dissociation the reviewers singled out.

### Three controls that are missing and would decide the outcome

1. **Per-item join between probe and behaviour.** Report the correlation between each item's probe-margin
   change and its behavioural change under the same manipulation, and apply the frozen probe to the
   *patched* residuals. Nearly free — no forwards beyond the patching sweep — and it is the single thing
   that converts "our probe reads X" into "the direction our probe reads is the direction that moves
   behaviour." Without it you have a probing paper and a patching paper stapled together, which is
   exactly the "collection of results" complaint.
2. **Layer-0 tautology bound on name-span patching.** At layer 0 the residual at a name token *is* that
   name's embedding (verified in §2.2), so patching name spans early re-inserts the name by construction.
   Report the layer-0 patch as an explicit upper bound; the claim survives only if mid-layer recovery
   exceeds it.
3. **Steering controls stronger than matched-norm random.** In 1536+ dimensions a random vector is
   near-orthogonal to everything, so that control is nearly free to pass. Report `cos(v, W_U[A]−W_U[B])`
   and `cos(v, E[A]−E[B])`, and train `v` on one name pair then evaluate on a disjoint pair.

### Tooling and compute

Use **raw HF forward hooks**, not TransformerLens — it supports almost none of the diverse code models
(no StarCoder2, no DeepSeek-Coder, no Granite). nnsight is the fallback if hooks get unwieldy.

Run the sweep in **fp16 with `use_cache=False`**, not fp32. A T4 has fp16 tensor cores but no fp32 ones
(~7.8× slower), so an fp32 mandate turns a 2.5-hour sweep into ~20 hours and won't fit 16 GB anyway.
Replicate only the top-3 (layer, span) cells in fp32 at batch 1 as a numerical check.

**Person-days, not GPU-hours, are the binding constraint.** Unbudgeted infrastructure — hook hardening
against real weights, correct padded batching (padding side differs per model; `pad_token` is None for
StarCoder2), per-model tokenizer registry, pair mining, results-cube serialization with resume,
bootstrap plumbing, figure pipeline, pytest from zero — totals 18–25 person-days. Buy Colab Pro on
day 0 (~$20); an L4 is ~2.6× a T4 and allocation stops being a coin flip.

### Revised triage

**29 Aug — one model, one language, one readout.** Patching in both directions with a placebo span and
bootstrap CIs, plus the CPU-only probe-hygiene package (Hewitt control on all five roles, grouped
splits, CIs, model-free baselines printed beside every probe number). The hygiene track needs no GPU,
no hooks, no re-extraction, and is the only deliverable guaranteed to land.

**25 Sep (ICLR) — add models and steering** with the full control battery.

**Day 1, before anything else:** verify the ~24,000-occurrence activation cache is intact on Drive and
pull it to local disk. `data/codesearchnet_*` are 0 bytes and the dataset_v0 train shard is missing; if
the Drive cache is also gone, the one guaranteed deliverable isn't free either.

---

## 8. Reproducing the measured numbers

```bash
uv run python scripts/variable_occurrences.py extract --input <canonical.jsonl> --output occ.jsonl --no-tokens
```

Baselines: `scratchpad/baselines.py` (promote to `scripts/baselines.py`). The layer-0 identity check
reads `outputs/npz_cache_java/*.npz` against `outputs/activations_java/manifest.jsonl`.

## 9. References worth adding

The reviewers did not complain about related work, so this is optional — but these are the citations
that make the controls above read as standard practice rather than as concessions:

Hewitt & Liang, *Control Tasks* (EMNLP 2019, [1909.03368](https://arxiv.org/abs/1909.03368)) — you
already use this, cite it by name in §4.4 too ·
Voita & Titov, *MDL probing* (EMNLP 2020) — the fallback if selectivity is weak ·
Hewitt et al., *Conditional probing* (EMNLP 2021, [2109.09234](https://arxiv.org/abs/2109.09234)) —
formalises "information beyond the identifier string," which is exactly your layer-0 delta ·
Belinkov, *Probing Classifiers* (CL 2022) — the "decodability ≠ use" citation esk2 is gesturing at ·
Zhang & Nanda, *Activation patching best practices* (ICLR 2024,
[2309.16042](https://arxiv.org/abs/2309.16042)) and Heimersheim & Nanda
([2404.15255](https://arxiv.org/abs/2404.15255)) — for B5 ·
Wu, Geiger & Millière, *How Do Transformers Learn Variable Binding in Symbolic Programs?* (ICML 2025,
[2505.20896](https://arxiv.org/abs/2505.20896)) — probed for program state, got 30.9%, concluded
dynamic routing; your high decodability needs one sentence reconciling with their null ·
Anand et al., *A Critical Study of What Code-LLMs (Do Not) Learn* (ACL Findings 2024,
[2406.11930](https://arxiv.org/abs/2406.11930)) — larger code models encode *less* structure; a
citable defence of working at 1.5B against reviewer ask #5 ·
Le et al., *When Names Disappear* ([2510.03178](https://arxiv.org/abs/2510.03178)) — the behavioural
version of your renaming claim; cite it and state that yours is the representational version.
