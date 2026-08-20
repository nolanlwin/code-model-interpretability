# LP4FM plan — cross-lingual transfer of variable roles

Target: **LP4FM @ NeurIPS 2026**, submission **29 Aug 2026**, non-archival,
double-blind, 4-page short or 9-page full.

> **Dual-submission constraint.** InterpScience (28 Aug) forbids submissions
> under review at another NeurIPS workshop. Whatever goes there must be a
> *different paper*, not this one reframed. LP4FM has no such rule, so the
> constraint runs one way only.

## The claim

Identifier names are the lexical medium of code; variable roles are the
semantic content. XLCoST implements the same algorithm in several languages,
so we can hold meaning constant and vary surface form — the manipulation
LP4FM's typology topic (iii) asks for.

**The result we expect to report, and the reason the paper is worth writing:**
`probing_directional_analysis.ipynb` already shows Python-trained probes
transferring to Java/C++/C# at **97–98%**, with probe-weight cosine of only
0.15–0.30. That was measured with **no model-free baseline**. Within a single
language, our probes match a masked-statement n-gram classifier in every
language tested; six of seven XLCoST languages are C-family with near-identical
statement syntax. So the leading hypothesis is:

> Cross-lingual probe transfer in code measures **surface regularity**, not
> language-universal role encoding.

If the transferring baseline matches the probe, that is the paper. If the probe
beats it, we have genuine universality — also a paper. Either way the
experiment is decisive, which is why it is first.

It also dissolves the notebook's "paradox": high accuracy with low cosine is
what a near-trivially separable task produces, since many hyperplanes work.
That is the notebook's own H1.

## Corpus constraint — read before designing anything

`problem_id` is a hash of the problem description, stable across languages.
Shared ids per pair:

| | python | javascript | php | java | cpp | csharp | c |
|---|---|---|---|---|---|---|---|
| python | — | **2953** | **1145** | 66 | 74 | 62 | 14 |
| javascript | 2953 | — | **1529** | 85 | 75 | 75 | 11 |
| php | 1145 | 1529 | — | 11 | 17 | 14 | 1 |
| java | 66 | 85 | 11 | — | 122 | 175 | 30 |

**Matched-problem transfer is only available for Python / JavaScript / PHP.**
Everything else can do unmatched transfer only, where a drop confounds "roles
do not transfer" with "different problems". Report the matched triangle as the
result; unmatched cells are secondary and must be labelled as such.

## What to BUILD

Nothing here needs a GPU; all of it is testable against fakes.

**B1. Transferring surface baseline** — `scripts/baselines.py transfer`
Fit the masked-statement TF-IDF + logistic model on language A's occurrences,
predict on language B's. `_fit_text` already does fit/predict; what is missing
is a cross-corpus path — `cmd_run` only ever splits one corpus. Emits
`test_predictions` so `bootstrap_ci.py delta` works on it.
*This is the decisive piece. Build it first.*

**B2. Cross-lingual probe transfer** — `scripts/crosslang.py`
Train the layerwise probe on store A, evaluate on store B. Matched-problem
pairing on `problem_id` when both sides have it; otherwise unmatched, flagged.
Reuses `probe.py`'s layer selection and `bootstrap_ci`'s clustered CIs.
Shuffled-label control (permute role labels within B) alongside.

**B3. ρ per cell** — reuse `export_probe.resolution()` verbatim. Every cell of
the matrix carries its own resolution or it cannot be read.

**B4. Runner + export** — `scripts/run_crosslang.sh` mirroring
`run_language.sh`, and a `results/lp4fm/` exporter mirroring
`export_probe.py`.

## What to RUN

Roles: **accumulator** and **iterator** only. Languages: **Python, JavaScript,
PHP**.

Existing activation stores are **boolean-only** — they were built from
`outputs/xlcost_occ/`, which the boolean extractor writes. Accumulator and
iterator occurrences come from `role_occurrences.py`, so new stores are needed.
One store per (language, model) covers every role in the file.

| step | what | cost |
|---|---|---|
| R1 | `role_occurrences.py extract --role all` for Python, JS, PHP | CPU, minutes |
| R2 | cap to ~3,000 occurrences per role per language, sampled by whole problems (same rule as `probe.py`) | CPU, seconds |
| R3 | `extract_activations.py` — 3 languages × 1 model | **GPU, the expensive step** |
| R4 | transfer matrix: 3×3 per role, probe + baseline + shuffled control | CPU once R3 exists |
| R5 | add the other two models only if the R4 result holds | GPU |

R2 matters: uncapped, Python alone yields ~34k accumulator+iterator
occurrences, and the probe caps at 2,000 anyway. Capping by whole problems
keeps the probe's own cap meaningful and bounds GPU cost.

Start with **one model** (Qwen2.5-Coder-1.5B). If the baseline transfers as
well as the probe, more models will not change the conclusion and the GPU is
better spent elsewhere.

## Typology — scoped down deliberately

**Do not** correlate transfer accuracy against a hand-coded typological feature
table. 21 pairs, hand-chosen features, no pre-registration: low power and high
researcher degrees of freedom, and XLCoST's languages are typologically narrow
(all imperative, six of seven C-family).

Instead, **one contrast with a stated prediction**: PHP marks every variable
with an obligatory `$` sigil; Python and JavaScript do not. Prediction — if
role encoding is lexical, the morphological marker changes transfer into and
out of PHP asymmetrically; if it is structural, it does not. PHP is already the
outlier in our probing table (the only language whose probes clear their own
resolution), so there is a prior. That is topic iii directly, and n=3 with a
prediction beats n=21 fitted after the fact.

## Explicitly NOT doing

- **Snippet-level "uses an accumulator vs not" binary.** A second task
  definition costs a week and produces numbers that compare to nothing else we
  have run. Use the existing variable-level role labels.
- **The full 7×7 matrix as the headline.** Four of the seven languages cannot
  be matched; a 7×7 of 97% numbers with no baseline repeats the mistake §4.4
  already made once.
- **Anything depending on the static/dynamic typing axis or the "34-combo
  control"** until someone produces them. Neither exists on main or in any of
  the 35 remote branches.

## Kill criteria

Stop and rewrite the claim if:
- the transferring baseline matches the probe within ρ → the paper is the
  negative result, not the transfer result (this is the expected outcome);
- matched and unmatched transfer disagree → the unmatched cells are measuring
  problem difficulty, and only the triangle can be reported;
- ρ exceeds the transfer gaps → say so and report the resolution, as §4.4 does.
