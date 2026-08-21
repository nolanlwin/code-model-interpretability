# Python `class_struct` Activation-Patching Experiment

## Summary and locked decisions

This is one new experiment with three model runs:

| Experiment | Language | Model runs |
|---|---|---:|
| Activation patching | Python only | 3 |

Models:

1. `Qwen/Qwen2.5-1.5B`
2. `Qwen/Qwen2.5-Coder-1.5B`
3. `bigcode/starcoder2-7b`

The existing perturbation and cross-language experiments will not be rerun. Patching will be a separate pipeline; [run_experiment.py](/Users/randylim/mech-interp-coding-llms/pipeline/run_experiment.py) will remain limited to `perturbation` and `crosslang`.

The causal question is:

> Does class-versus-function information in the residual stream at a later use of an identifier causally affect whether the model predicts that identifier is a type?

The overnight run will be one detached Modal controller, but it will invoke separate GPU functions sequentially. It stops replication if Qwen fails any mandatory probe, behavior, probe-OOD, causal, or infrastructure gate before its core completes. If Qwen reaches `core_complete`, it attempts Coder and StarCoder2 without requiring laptop involvement.

```text
Local tests
  → Modal CPU/tokenizer preflight
  → synchronous 8-pair Qwen smoke
  → detached controller
      → Qwen probe-prior gate → behavior/probe-OOD gates → primary causal gate → core sweep
      → Coder probe-prior gate → behavior/probe-OOD gates → primary causal gate → core sweep
      → Star probe-prior gate → behavior/probe-OOD gates → primary causal gate → core sweep
      → expanded controls for every core-complete model
      → fp32 robustness cells for every core-complete model
      → final CPU summaries and exact completeness reports
```

“Finished” means a complete, validated result cube or a preregistered null/failure record—not necessarily a positive effect.

## 1. Freeze the experimental protocol

### Prompt pair and readout

For each identifier, generate matched prompts differing at exactly one token:

```python
class Node():
    pass

assert isinstance(Node, type) is
```

```python
def Node():
    pass

assert isinstance(Node, type) is
```

Use:

- Class prompt as `clean`.
- Function prompt as `corrupt`.
- Primary patch span: `Node` inside `isinstance`.
- Secondary patch span: `Node` in the declaration.
- Placebo span: the unique `pass` in the target body.
- Readout: `D = logit(" True") - logit(" False")`.
- Raw base-model completion only.
- No chat template, few-shot examples, `generate()`, probability conversion, or patching of the `class`/`def` token.

The model revisions and preregistered hidden-state indices are:

| Model | Revision | Blocks / hidden indices | Probe index |
|---|---|---:|---:|
| Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | 28 / `0…28` | 18 |
| Qwen2.5-Coder-1.5B | `df3ce67c0e24480f20468b6ef2894622d69eb73b` | 28 / `0…28` | 8 |
| StarCoder2-7B | `bb9afde76d7945da5745592525db122d4d729eb1` | 32 / `0…32` | 5 |

The CSV layer number means Hugging Face `hidden_states[k]`, corresponding to the residual entering decoder block `k`. Index zero is the embedding/block-0 input; the final index is the final normalized hidden state.

### Frozen evaluation prompts

Create `data/patching/class_struct_python_v1.jsonl` containing exactly 288 pairs:

- 4 prefix variants.
- 3 target-body variants.
- 4 query-distance/gap variants.
- `4 × 3 × 4 = 48` structural clusters.
- 6 lexical variants per cluster.
- `48 × 6 = 288` evaluation pairs.

Prefix variants are exactly:

1. Empty.
2. `LIMIT = 4\nmode = "safe"\n\n`.
3. `def helper(value):\n    return value + 1\n\n`.
4. `class Helper():\n    marker = 1\n\n`.

Body variants are exactly:

1. `    pass\n`.
2. `    marker = 7\n    pass\n`.
3. `    label = "ready"\n    marker = len(label)\n    pass\n`.

Gap variants are exactly:

1. Empty.
2. `sentinel = 3\n\n`.
3. `left = 2\nright = left + 5\n\n`.
4. `values = [1, 2, 3]\ntotal = sum(values)\nstatus = total > 0\n\n`.

Use this exact 24-name bank:

```text
Node, Item, Point, Entry, Record, Token,
Buffer, Parser, Scanner, Packet, Frame, Block,
Tree, Graph, State, Value, Stack, Table,
Matrix, Vector, Widget, Element, Vertex, Edge
```

For structural cluster `c`, first recover its factor IDs in prefix-major, then body, then gap order. For lexical variant `j`:

```text
prefix_id = c // 12
body_id = (c % 12) // 4
gap_id = c % 4
group = (prefix_id + body_id + gap_id) mod 4
name_index = 6*group + j
```

This produces six distinct names per cluster, twelve uses of every name overall, and a balanced name distribution within every prefix, body, and gap level.

Each JSONL row contains:

```text
schema_version
pair_id
cluster_id
prefix_id
body_id
gap_id
lexical_variant
name
clean_prompt
corrupt_prompt
clean_expected
corrupt_expected
keyword_char_span
declaration_name_char_span
query_name_char_span
placebo_char_span
```

Create a separate engineering-only file, `data/patching/class_struct_python_smoke_v1.jsonl`, using:

```text
Cell, Column, Key, Index, Result, Context, Config, Model
```

paired with structural clusters:

```text
0, 7, 13, 18, 25, 31, 38, 47
```

These eight rows never enter the 288-pair analysis.

Canonicalization is sorted-key, compact, ASCII JSON with one row per LF-terminated line and a final newline. The frozen file hashes are:

| File | SHA-256 |
|---|---|
| `data/patching/class_struct_python_v1.jsonl` | `6077e73c158616cf5f9175e4cf49daa2e4b2016ebc8afd12edc708572e79bd7b` |
| `data/patching/class_struct_python_smoke_v1.jsonl` | `522f9b2be880a861af22b6d7948f8837a67fb00897e73e02c7de094676a27425` |

The generator refuses to write files whose generated bytes do not match these constants, and validation compares both the file bytes and regenerated rows. Once any behavioral result has been produced, a changed prompt hash requires a new experiment version and run ID.

### Pre-GPU validation

For all 296 rows and all three pinned tokenizers, require:

- `add_special_tokens=False`.
- No truncation.
- Clean and corrupt token counts are identical.
- Their token IDs differ exactly once, at `class` versus `def`.
- All token IDs after that keyword are identical.
- Declaration, query, and placebo token indices align across the pair.
- Each identifier occurrence is exactly one token.
- `pass` is exactly one token.
- `" True"` and `" False"` are distinct one-token suffixes.
- Encoding `prompt + completion` preserves the original prompt prefix.
- Prompt length is at most 128 tokens.
- Correctly completed programs parse and pass their assertion.
- The opposite completion raises `AssertionError`.

Expected answer-token IDs, which the code must rediscover and assert rather than silently hardcode:

| Tokenizer | `" True"` | `" False"` |
|---|---:|---:|
| Qwen/Qwen-Coder | 3007 | 3557 |
| StarCoder2 | 2969 | 3208 |

There will be no first-subtoken fallback. A tokenizer mismatch is a hard pre-GPU failure.

## 2. Build the independent patching pipeline

### Public interfaces

The independent implementation consists of:

- `pipeline/patching.py`: model adapters, hook management, activation caching, patch construction, metrics, controls, probe scoring, and statistics.
- `pipeline/patching_prompts.py`: frozen prompt generation plus strict schema, byte, semantic, and distribution validation.
- `pipeline/run_patching.py`: local/remote-neutral CLI with `generate-prompts`, `validate`, `estimate`, `smoke`, `sweep`, `evaluate`, `summarize`, `check-completeness`, `extract-probe`, and `fit-probe` subcommands.
- `scripts/modal_patching.py`: Modal functions and the detached gated controller.

The CLI configuration must explicitly contain:

```text
experiment = class_struct_activation_patching_v1
language = Python
prompt_sha256
smoke_prompt_sha256
model IDs and revisions
selected hidden-state indices
dtype
readout strings and discovered token IDs
layer schedule
span schedule
direction schedule
control schedule
random seed
primary-key fields and schedules from which the expected key set is derived
forward-count ceilings
```

Before a paid sweep, `estimate` prints:

- Prompt pairs.
- Layers.
- Intervention cells.
- Item-forwards.
- Batched forwards.
- Approximate padded token-forwards.
- Expected output keys.
- GPU-only modeled cost estimates and an explicit scope warning.

It must refuse configurations above:

- Qwen: `90,000` scientific-sweep item-forwards.
- Coder: `26,000`.
- StarCoder2: `26,000`.

The ceiling applies to staged fp16 scientific-sweep item-forwards. Probe extraction, probe fitting, behavioral gates, smoke, CPU/RAM, image startup, and preemption are outside the CLI estimate and are instead covered by the detached controller's end-to-end `$50` projected-spend stop.

### Model loading

Use:

- `AutoModelForCausalLM`, not `AutoModel`.
- Explicit `torch.float16` for the main experiment.
- `model.eval()`.
- `torch.inference_mode()`.
- `use_cache=False`.
- `trust_remote_code=False`.
- `local_files_only=True`.
- Exact pinned revisions.
- No automatic downloads.
- No chat template.
- No `device_map="auto"`.

Use `logits_to_keep=1` when supported. If a model class lacks that parameter, retain only the final-position logits immediately after the forward.

Batching uses left padding with `pad_token_id=eos_token_id` and a correct attention mask. Character/token spans are discovered before padding and shifted by the per-row left-padding amount. This allows the last tensor position to remain the final real prompt token for every row.

Initial recipient microbatches:

- Qwen and Coder: 32.
- StarCoder2: 8.
- StarCoder2 fp32 replication: 2.

On CUDA OOM:

1. Record the failed batch size.
2. Clear temporary tensors and the CUDA cache.
3. Halve the microbatch.
4. Retry the same deterministic keys.
5. Fail the phase if batch size one still OOMs.

No scientific condition may be dropped due to OOM.

### Residual hooks

Implement an architecture adapter that discovers and asserts:

- Decoder block list.
- Final normalization module.
- Number of blocks.
- Hidden size.
- Hidden-state tuple length.

Hook mapping:

- Hidden index `0`: input to decoder block 0.
- Internal hidden index `k`: input to decoder block `k`.
- Final hidden index: output of final normalization.

Before scientific execution, compare hook captures against `output_hidden_states=True` on real Qwen and StarCoder2 forwards. Exact equality is expected in fp16; otherwise the maximum absolute difference must be below a preregistered numerical tolerance.

Hooks must:

- Handle tensor and tuple/module-input structures.
- Replace only selected batch rows and token positions.
- Preserve dtype and device.
- Never mutate source-cache tensors.
- Always be removed in `finally`.
- Support multiple conditions packed into one recipient forward.

Per prompt block, perform clean and corrupt source forwards once, retain only required span vectors and baseline readout logits, then free full hidden-state tuples.

### Patch directions and controls

Define effects so positive always means movement toward the source state:

```text
gap_i = D(class_i) - D(function_i)

denoise_i =
    D(function_i patched from class_i) - D(function_i)

noise_i =
    D(class_i) - D(class_i patched from function_i)
```

Interventions:

1. Query-name full-residual replacement.
2. Declaration-name full-residual replacement.
3. Placebo replacement at `pass`.
4. Matched-norm random injection at the query name.
5. Matched-norm random injection at the declaration name.
6. Same-source no-op.
7. Repeated unpatched baseline for drift.
8. Layer-0 identity check.

For a random control:

```text
delta = source - destination
```

Generate deterministic Gaussian noise from seed `20260818` plus a SHA-256-derived cell key, project it orthogonally to `delta`, normalize it to the same L2 norm, then inject it into the destination. Construct in fp32, cast to model dtype, and require the post-cast norm ratio to remain within 1%.

Never use zero ablation.

### Raw result schema

Every intervention row must be uniquely keyed by:

```text
prompt_sha256
configuration_sha256
model_id
model_revision
dtype
pair_id
layer
span
direction
control
random_seed
```

Store at least:

```text
run_id
cluster_id
name
source_D
destination_D
patched_D
signed_effect
class_function_gap
source_probe_margin
destination_probe_margin
patched_probe_margin
baseline_drift
batch_size
attempt_id
timestamp
```

Probe margins may be null only in pre-probe engineering smoke rows.

## 3. Re-freeze and require the probe link

The original fitted probes cannot be reused because their scalers and classifier weights were never serialized and the hidden-state dumps were deleted. Recreate a new, frozen probe-link artifact per model before that model’s patching result is accepted.

The frozen probe source is the existing Modal-volume file:

```text
/data/dataset/python_perturbations/train.jsonl
SHA-256 1729bef9187b6f92a9d162c265c771a9294fffab8d406b1b696cd6234506a4f3
dataset revision 912f6e468df675f11237f3c9b7635f09a6a95584
```

For each model:

1. Preserve file order while filtering exactly the rows with `language == "Python"`, `strategy == "baseline"`, and a truthy `roles.class_struct`; require exactly 400 rows and 400 unique, non-empty program IDs.
2. Require `build_token_dataset` to keep all 400 rows with zero skips, so tokenizer-specific cohort drift is a hard failure.
3. Extract only the preregistered hidden-state index: 18, 8, or 5.
4. Store activations as float16, preserving token labels and program IDs.
5. Release the GPU.
6. Fit on a CPU-only Modal function.
7. Use the original grouped 70/10/20 split logic and seeds `0…4`.
8. Fit `StandardScaler` on training tokens only.
9. Fit `LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=seed)`.
10. Save all five probes; seed zero is primary.

Persist portable NPZ arrays rather than sklearn pickles:

```text
scaler_mean
scaler_scale
classifier_coef
classifier_intercept
classifier_classes
```

The accompanying JSON metadata records:

- All source/model/code revisions.
- Program-ID and split hashes.
- Dataset and prompt hashes.
- Layer index.
- Model dtype.
- Torch, Transformers, NumPy, SciPy, sklearn, and Modal versions.
- Token count.
- Fit metrics.
- Artifact SHA-256.
- The frozen source-JSONL SHA-256, ordered source-program-ID hash, token-program-ID hash, and checksums of `hidden.npy`, `labels.npy`, and `programs.npy`.

Keep the selected-layer extraction shards until the paper results are accepted; do not repeat the previous deletion that made the probes unrecoverable.

Probe acceptance:

- All five fits are finite.
- `classes == [0, 1]`.
- Class-name tokens are positive class 1.
- Mean held-out F1 and accuracy reproduce the prior selected-layer CSV within `0.002` absolute:

| Model | Prior F1 | Prior accuracy |
|---|---:|---:|
| Qwen | 0.9832 | 0.9990 |
| Coder | 0.9820 | 0.9989 |
| StarCoder2 | 0.9828 | 0.9990 |

On the 288 synthetic prompts:

- Mean class-minus-function probe margin is positive with clustered CI excluding zero.
- AUC is at least 0.70 at the declaration occurrence.
- AUC is at least 0.70 at the query occurrence.

Because probe linkage is required:

- Qwen probe failure stops the entire experiment.
- Coder or Star probe failure records a failed probe-link gate, skips that model’s patch sweep, and continues to the next independent model.

Convert standardized coefficients into a raw-residual margin correctly:

```text
w_raw = coef / scaler_scale
b_raw = intercept - dot(w_raw, scaler_mean)
margin(h) = dot(h, w_raw) + b_raw
```

Unit-test this against sklearn’s `decision_function`.

The probe-to-behavior analysis is:

```text
probe_gap_i = probe(class_i) - probe(function_i)

symmetric_behavior_i = (denoise_i + noise_i) / 2
```

Report clustered Spearman correlation between these quantities. Do not claim that patched probe movement alone is causal: full-residual replacement makes that score movement partly true by construction.

## 4. Gates, sweep schedule, and statistics

### Engineering smoke

Run synchronously on Qwen before launching anything detached:

- Eight smoke pairs.
- Layer 18.
- Query target, placebo, and random control.
- Denoising and noising.
- Layer-0 identity.
- Same-source no-op.
- Repeated baseline.

Pass only if:

- At least 6/8 class prompts have `D > 0`.
- At least 6/8 function prompts have `D < 0`.
- Mean class/function gap is positive.
- Mean target denoising and noising are positive.
- At least 5/8 item effects have the expected sign in each direction.
- Target mean exceeds placebo and random means.
- Maximum repeated-forward drift is at most `0.01` logit.
- Same-source and layer-0 effects are within:

```text
tau = max(1e-4, 10 × maximum repeated-forward drift)
```

Failure stops before the detached overnight launch. Smoke rows are engineering output, never paper evidence.

### Behavioral gate per model

On all 288 unpatched pairs require:

- Class two-choice accuracy at least 0.60.
- Function two-choice accuracy at least 0.60.
- Clustered 95% CI lower bound above 0.50 for each accuracy.
- CI lower bound for mean `D(class)` above zero.
- CI lower bound for mean `-D(function)` above zero.
- CI lower bound for the class/function gap above zero.

A failure records a valid behavioral null and skips that model’s patch sweep.

### Primary causal gate

At the preregistered layer and query span, require both directions to have:

- Mean raw effect at least `max(0.10 logit, 5*tau)`.
- Clustered 95% CI lower bound above zero.
- Target-minus-placebo CI lower bound above zero.
- Target-minus-random CI lower bound above zero.
- Ratio-of-means recovery of at least 0.05.

Qwen failure stops Coder and StarCoder2. Coder or Star failure remains a valid model-specific null and does not invalidate already completed models.

### Sweep matrix

Run the scientifically matched three-model core first so complete cross-model results are available as early as possible. Schedules are deltas on resume: cells already produced by an earlier phase are not repeated as result keys.

The ten-cell primary schedule for every model is, in both directions: query target at the probe layer, placebo target at the probe layer, query random at the probe layer, query same-source at the probe layer, and query target at layer 0. The core schedule is query target in both directions at every hidden index. The expanded schedule is the remaining delta needed to reach the model's full preregistered fp16 schedule.

| Model | Full fp16 intervention cells | New primary / core / expanded cells | Unique fp16 keys/item-forwards | Actual staged item-forwards | Staged batched forwards | Ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 294 | `10 / 54 / 230` | 85,824 | 87,552 | 2,736 | 90,000 |
| Coder | 70 | `10 / 54 / 6` | 21,312 | 23,040 | 720 | 26,000 |
| StarCoder2 | 78 | `10 / 62 / 6` | 23,616 | 25,344 | 3,168 | 26,000 |

Each unique-key count is `288 × (full intervention cells + 4 baseline/drift rows)`. Actual staged item-forwards are larger because every separately gated phase reloads clean/function source activations. Qwen's full battery covers declaration target, placebo target, query random, and declaration random in both directions at every hidden index, plus query/declaration same-source at layer 18. Coder and StarCoder2 apply that battery only at their probe layer; their six-cell expanded delta is declaration target, declaration random, and declaration same-source in both directions.

Execution order:

1. Qwen probe-prior, behavior, probe-OOD, and primary causal gates, then its matched core curve.
2. Coder gates and matched core.
3. StarCoder2 gates and matched core.
4. Expanded deltas for Qwen, Coder, and StarCoder2, but only for models whose core completed.
5. Precision replication for every core-complete model.
6. Final all-model CPU summary and exact completeness validation.

Any Qwen scientific or infrastructure failure before `core_complete` stops replication. A Coder or StarCoder2 null is recorded and skipped without erasing completed models; late-model infrastructure failures are recorded and the controller continues where safe. A Qwen expanded/fp32 infrastructure failure aborts, while a corresponding replica failure is isolated.

### fp32 replication

After fp16 curves are frozen, choose three query-layer cells per passing model:

1. The preregistered probe layer.
2. The two non-endpoint layers with the largest absolute symmetric fp16 target effect.

Rerun the 288 pairs in fp32 for:

- Four unpatched/repeated baseline rows per pair.
- Query target, placebo target, and query random in both directions at each of the three selected layers.

That is 18 intervention cells plus four baseline rows, or 6,336 fp32 item-forwards per passing model.

Hardware:

- Qwen and Coder fp32: L4 through the 1.5B sweep function, whose hard timeout is 10 hours.
- StarCoder2 fp32: L40S 48 GB, because 7B fp32 weights do not fit safely on a 24 GB L4.

This is a numerical sensitivity analysis, not a second layer-selection procedure. Report whether effect signs and primary significance conclusions change; do not silently replace fp16 results.

### Statistical outputs

Compute on CPU only:

- Mean raw denoising and noising.
- Target-minus-placebo paired effects.
- Target-minus-random paired effects.
- Ratio-of-means recovery:

```text
mean(effect_i) / mean(gap_i)
```

Never compute a mean of per-item ratios.

Use:

- 10,000 deterministic BCa bootstrap replicates.
- Seed `20260818`.
- Cluster resampling over 48 structural templates.
- All six lexical variants retained together when a cluster is sampled.
- Leave-one-name-out sensitivity across all 24 names.

If a bootstrapped recovery denominator crosses zero, retain the raw-effect inference and label the ratio interval unstable.

Produce after every completed model:

- Raw, uncompressed, checksummed `.jsonl` chunks plus `chunk_index.json`.
- Namespaced `summary.csv`, `summary.json`, `leave_one_name_out.json`, `probe_link.csv`, and `probe_link_summary.json` under `summaries/<model>/<dtype>/<configuration-prefix>/<eval-or-smoke>/`.
- Per-model/dtype gate diagnostics under `diagnostics/` and the controller-level `gate_report.json`.
- Exact schedule reports under `summaries/completeness/`.
- Final CPU-generated `overnight_report.json` and `overnight_report.md`.

## 5. Modal-only overnight execution

### Infrastructure

Create:

```bash
modal environment create patching
modal volume create --env patching class-struct-patching-results
```

Mount volumes as follows:

- Mount the existing `main`-environment volume `class-struct-data` exactly once, read-only, at `/data`. Its model cache is `/data/hf` (`HF_HOME=/data/hf`, `HF_HUB_CACHE=/data/hf/hub`) and its probe source is under `/data/dataset`.
- Mount the `patching`-environment volume `class-struct-patching-results` read-write at `/results`.

The single `/data` mount is required because the same Modal Volume cannot be mounted twice in one function.

The image and exact validated package pins are:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
torch==2.6.0
transformers==5.8.0
scipy==1.15.3
scikit-learn==1.7.2
numpy==2.2.6
tqdm==4.70.0
datasets==4.8.4
huggingface_hub==1.27.0
modal==1.5.4
```

Preflight hard-fails if installed versions differ. Do not expose `.git` remotely; instead inject and record the current base commit (`babd71a64b4032cbbd01496cf5b9e1dd3cf501a2` at this review point) plus a launch-time SHA-256 of the bundled patching code and prompt files. The bundle hash, not the base commit alone, covers the new uncommitted patching files.

Do not reuse the existing detached-plus-`.spawn()` launcher. Modal’s `--detach` keeps the app alive after the local process disconnects, so once server-side status is confirmed, the laptop can sleep or shut down; remaining plugged in is unnecessary. [Modal `run --detach` documentation](https://modal.com/docs/cli/latest/run)

### Functions and limits

| Function | Resource | Timeout |
|---|---|---:|
| `preflight` | CPU, 2 cores, 8 GiB | 15 min |
| `fit_probe` | CPU, 8 cores, 16 GiB | 60 min |
| `summarize` | CPU, 4 cores, 8 GiB | 60 min |
| `qwen_smoke` | L4, 2 cores, 16 GiB | 20 min |
| `extract_probe_1p5b` | L4, 2 cores, 16 GiB | 90 min |
| `extract_probe_7b` | L4, 4 cores, 32 GiB | 2 h |
| `sweep_1p5b` | L4, 2 cores, 16 GiB | 10 h per phase |
| `sweep_7b` | L4, 4 cores, 32 GiB | 8 h |
| `fp32_7b` | L40S, 4 cores, 48 GiB | 90 min |
| `run_all_gated` | CPU controller, 2 cores, 4 GiB | 24 h |

Every function explicitly uses:

```text
max_containers = 1
retries = 0
scaledown_window = 2
```

It relies on Modal's scale-to-zero defaults rather than setting warm/buffer containers, and uses the two-second minimum scale-down window so a paid accelerator becomes eligible for release promptly after its phase returns. There is no `.map()` over models, prompts, layers, or controls. The controller calls one remote function synchronously at a time. The controller timeout is 24 hours, Modal's maximum function timeout, so a longer interrupted run resumes from durable chunks rather than assuming a 30-hour invocation. [Modal timeout documentation](https://modal.com/docs/guide/timeouts)

The CLI's modeled, GPU-only fp16 staged estimates are `$8.7552` for Qwen, `$2.3040` for Coder, and `$2.8160` for StarCoder2 (`$13.8752` total). Its three-model fp32 top-three-layer estimate is `$2.8116` total. These figures use `$0.80/GPU-hour` for L4 and `$1.95/GPU-hour` for L40S, and deliberately exclude CPU, RAM, probe extraction/fitting, smoke, image startup, retries/preemption, and the controller.

The detached controller uses a separate end-to-end conservative ledger: `$0.0473/core-hour`, `$0.008/GiB-hour`, `$0.80/L4-hour`, and `$2.50/L40S-hour`, with 115% of elapsed time and a one-minute floor per tracked phase. These are accounting estimates, not a claim that Modal's invoice will match exactly. Current platform rates should still be checked before launch. [Modal pricing](https://modal.com/pricing)

Budget policy:

- `$25` projected end-to-end spend: write a warning to status but continue.
- `$50` projected end-to-end spend: refuse to start the next phase.
- Each GPU phase has its own timeout, preventing an unbounded single phase.
- No Workspace budget is changed.
- If the account happens to support an isolated environment budget, it may be set to `$50`, but the run does not rely on it; Modal documents environment budgets as a Team/Enterprise feature. [Modal budget documentation](https://modal.com/docs/guide/budgets)

Before each phase, projected spend is the conservative elapsed ledger plus the active controller estimate plus a fixed next-phase allowance. The fixed allowances are `$2` per 1.5B extraction, `$3` for 7B extraction, `$0.50` per probe fit, `$0.50/$0.75` for 1.5B/7B behavior, `$1/$2` for 1.5B/7B primary, `$4/$8` for 1.5B/7B core, `$9/$0.50/$1` for Qwen/Coder/Star expanded work, and `$3/$5` for 1.5B/7B fp32.

### Durable checkpoint contract

Use:

```text
/results/runs/class-struct-python-v1-20260818/
  manifest.json
  status.json
  gate_report.json
  cost_ledger.json
  lease.json
  smoke_receipt.json
  chunk_index.json
  fp32_selection.json
  model_failures.json
  probes/
  chunks/
  diagnostics/
  summaries/
  smoke/
  overnight_report.json
  overnight_report.md
```

The immutable manifest records:

- Prompt/configuration/code hashes.
- Model IDs and revisions.
- Model and software dtypes.
- Layer/span/control schedules.
- Seeds.
- Primary-key schema and schedules from which exact expected keys are deterministically derived.
- Expected row and forward counts.
- Resource and timeout configuration.
- Base Git commit, base image, exact requested and installed software versions, frozen probe-source SHA-256, and dataset revision.

Checkpoint unit:

```text
one phase × one layer × one 64-pair prompt block
```

For every chunk:

1. Write to a temporary file.
2. Flush and `fsync`.
3. Record row count and validate primary-key uniqueness.
4. Compute checksum.
5. Atomically rename with `os.replace`.
6. Atomically update `status.json`.
7. Explicitly commit the results volume.
8. Print the same progress event to Modal logs.

A reader or reused container calls `volume.reload()` before consuming another container’s results. Modal requires commits for cross-container visibility and reloads for already mounted readers. [Modal Volume documentation](https://modal.com/docs/guide/volumes)

Resume behavior:

- Ignore temporary chunks.
- Validate every finalized chunk’s checksum and exact keys.
- Recompute only missing keys.
- Reject the run ID if its manifest differs.
- Prevent duplicate controllers with a lease and heartbeat.
- Permit a same-function preemption attempt to resume immediately.
- Permit a new function call only after the old app has stopped or its heartbeat is stale for ten minutes.
- Mark complete only when the present key set exactly equals the expected key set.

### Launch checklist

Run in this order:

```bash
modal environment create patching
modal volume create --env patching class-struct-patching-results
```

Then:

1. Run all local unit tests.
2. Regenerate both frozen prompt files; generation itself refuses any bytes that do not match the pinned hashes.
3. Run strict Python/schema validation and all-three-tokenizer offline validation.
4. Run the forward/cost estimate and confirm no ceiling refusal.
5. Run the Modal CPU preflight synchronously.
6. Run the Qwen eight-pair L4 smoke synchronously.
7. Inspect the smoke gate report.
8. Confirm there is no existing patching app.
9. Launch the one detached controller.

Exact local commands from the repository root:

```bash
python -m pipeline.patching_prompts

python -m pipeline.run_patching validate

python -m pipeline.run_patching validate --tokenizers \
  Qwen/Qwen2.5-1.5B \
  Qwen/Qwen2.5-Coder-1.5B \
  bigcode/starcoder2-7b

python -m pipeline.run_patching estimate

pytest -q tests/test_patching.py tests/test_patching_prompts_strict.py
```

Tokenizer validation is offline by default and therefore requires all three pinned revisions to already exist in `/data/hf` for Modal preflight or in the configured local HF cache for a local run.

Commands:

```bash
modal run \
  --env patching \
  scripts/modal_patching.py::preflight \
  --run-id class-struct-python-v1-20260818
```

```bash
modal run \
  --env patching \
  scripts/modal_patching.py::qwen_smoke \
  --run-id class-struct-python-v1-20260818
```

The smoke command exits nonzero on a scientific no-go. Before detaching, inspect the durable receipt and require `"pass": true`:

```bash
modal volume get \
  --env patching \
  class-struct-patching-results \
  /runs/class-struct-python-v1-20260818/smoke_receipt.json \
  -
```

```bash
modal run \
  --detach \
  --env patching \
  --name csp-v1-20260818 \
  scripts/modal_patching.py::run_all_gated \
  --run-id class-struct-python-v1-20260818 \
  --resume
```

The detached `modal run` may remain attached to logs in its originating terminal; `--detach` means a local disconnect will not terminate the remote App. Launch it in one terminal and use a second terminal for the checks below. Before sleeping, confirm all three:

```bash
modal app list --env patching --json
```

```bash
modal app logs csp-v1-20260818 \
  --env patching \
  --tail 100 \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

```bash
modal volume get \
  --env patching \
  class-struct-patching-results \
  /runs/class-struct-python-v1-20260818/status.json \
  -
```

The status file must show `state: "running"`, a current phase, a fresh `heartbeat`, and `controller.function_call_id` before the laptop is closed. After that, the laptop may sleep or disconnect without ending the detached App.

### Morning and emergency commands

Morning status:

```bash
modal app list --env patching --json
```

```bash
modal app logs csp-v1-20260818 \
  --env patching \
  --tail 500 \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

```bash
modal volume get \
  --env patching \
  class-struct-patching-results \
  /runs/class-struct-python-v1-20260818/status.json \
  -
```

```bash
modal environment billing report patching \
  --for today \
  --resolution h \
  --tz local \
  --show-resources
```

Pull results:

```bash
modal volume get \
  --force \
  --env patching \
  class-struct-patching-results \
  /runs/class-struct-python-v1-20260818 \
  ./results/modal/patching/class-struct-python-v1-20260818
```

Emergency stop:

```bash
modal app stop csp-v1-20260818 --env patching --yes
```

Use `modal app stop`, not `modal container stop`; app stop permanently terminates the app and its running containers. [Modal app-management documentation](https://modal.com/docs/cli/latest/app)

Resume with the same scientific run ID and a new app name:

```bash
modal run \
  --detach \
  --env patching \
  --name csp-v1-20260818-resume1 \
  scripts/modal_patching.py::run_all_gated \
  --run-id class-struct-python-v1-20260818 \
  --resume
```

## 6. Tests and acceptance criteria

### Pure unit tests

Cover:

- Deterministic prompt generation and stable hashes.
- Exactly 288 evaluation and eight smoke pairs.
- Exactly 48 clusters, six names per cluster, and twelve uses per name.
- Smoke/evaluation disjointness.
- Correct character spans.
- Correct and incorrect Python completion semantics.
- Clean/corrupt token equality and one-token keyword difference.
- Hard failure for shifted, multi-token, truncated, or ambiguous spans.
- Left-padding index adjustment and last-position readout.
- True/False logit-difference sign.
- Both patch-direction sign conventions.
- Ratio-of-means versus mean-of-ratios.
- Clustered bootstrap resampling whole six-row clusters.
- Random-vector orthogonality, norm matching, determinism, and zero-delta handling.
- Hook replacement for batched rows and token positions.
- Tensor/tuple input handling.
- Hook removal after both success and exceptions.
- No mutation of source caches.
- Manifest mismatch rejection.
- Chunk checksum and duplicate-key rejection.
- Resume from missing/corrupt chunks.
- Lease and stale-heartbeat behavior.
- Forward-count and projected-cost refusal.
- Probe coefficient conversion matching sklearn.
- Probe class orientation, AUC, and correlation calculations.

### CPU integration tests

Using tiny randomly initialized Qwen2 and StarCoder2 configurations:

- Confirm adapter discovery.
- Confirm all hidden-state indices map to the intended hook points.
- Confirm a same-source patch is a no-op.
- Confirm layer-zero same-name replacement is a no-op.
- Confirm a deliberately changed source tensor alters a patched forward.
- Confirm clean, corrupt, patched, placebo, random, and resumed rows share the expected schema.

### Real-model preflight

Before the scientific sweep:

- Validate all 296 prompts with all three cached tokenizers.
- Verify pinned revisions exist locally.
- Refuse any network download.
- Compare real hook tensors with `hidden_states[k]`.
- Confirm layer-zero identity and mid-layer class/function divergence.
- Confirm no GPU-bearing function performs sklearn fitting, bootstrapping, plotting, or report generation.

### Final engineering acceptance

A model result is complete only if:

- Probe artifact passes all required gates.
- Behavioral gate is recorded.
- Every expected raw key appears exactly once.
- No unexpected keys appear.
- All required values are finite.
- Manifest and prompt/configuration hashes match.
- Completeness validator passes independently after pulling.
- CPU summaries can be rebuilt from raw rows alone.

The overall overnight job is complete when:

- Qwen failed a preregistered gate and the controller stopped cleanly; or
- Qwen passed and all three models were attempted, with each producing either a complete patching result or a recorded gate failure.
- No existing perturbation/cross-language result was modified or rerun.
- No unrelated dirty-worktree files were changed.
- GPU billing ended immediately after the last forward; all fitting, CIs, summaries, and reports used CPU-only functions.

## Assumptions

- Scope is exclusively Python `class_struct`; no Boolean experiment and no C++, JavaScript, C, Java, C#, or PHP.
- “Three runs” means one experiment repeated across three models.
- StarCoder2 means the previously used 7B model, not 3B.
- Completion by the morning is preferred but not a hard midnight cutoff; durable checkpoints take priority over rushing or weakening the protocol.
- The $50 figure is a conservative experiment-spend ceiling, not a platform hard budget.
- Qwen must pass before replication spend is authorized.
- Probe linkage is mandatory because the user selected the stronger probe-plus-causal interpretation.
- The existing `.DS_Store` change and untracked Kaggle configuration belong to the user and must remain untouched.
