# Proposal: interp-science-cfp-calibration-edits

## Why

A five-seat review panel scored `interp_science_short` against the published
Interpretability as a Science call for papers. Every seat returned Accept, and
no seat raised a blocking objection. The same four issues were nevertheless
raised independently by multiple seats, and all four are correctable by editing
text rather than by running new experiments. Three of them are accuracy defects
that a paper about reporting discipline should not carry.

The submission deadline is September 01, 2026 AoE, so the work must stay within
the existing five-page main-text budget and must not disturb any committed
artifact.

## What Changes

- Ground the reporting-matrix contribution in prior reporting-standard and
  falsifiability literature. The manuscript currently borrows the vocabulary of
  estimands, comparators, and preregistration without citing the fields it
  borrows from, which also leaves a listed workshop topic unaddressed.
- Report the boolean comparison honestly across languages and comparators. The
  body currently states only the Python result under the single most favourable
  surface comparator. Probe minus best baseline is `+0.021` for PHP with
  StarCoder2-7B, and against the next-best Python comparator the differences are
  `+0.010` to `+0.017`.
- Reframe the patching null. Recovery divides by the matched gap, so a
  compressed decision range cannot explain a recovery of `0.009` to `0.020`.
  The current "insensitive readout" attribution is not supported by the
  manuscript's own estimator.
- Correct three verified factual defects:
  - the paired boolean sample reduction is attributed to a predictor-overlap
    requirement, but `min_seed_coverage` is `1.0`, so overlap excluded nothing;
  - "all 4,032 primary rows" sums the primary and behavior schedules, where
    primary alone is 2,880;
  - a rename effect is quoted as `0.073` where the largest tabulated magnitude
    is `0.072`.

## Capabilities

### New Capabilities

- `interp-science-claim-calibration` — requirements binding each reported claim
  in the Interpretability as a Science submission to the comparator, language
  population, and estimator that actually support it, while preserving the
  existing five-page main-text budget.

### Modified Capabilities

<!-- none -->



## Verification scope

`verified: false`. The change edits manuscript prose, bibliography entries, and
the string-matching regression checks that bind prose to committed artifacts.
There is no precise logic to prove. Verification is by scenario check: the
existing `tests/test_interp4d_claims.py` audit must pass with checks updated to
the revised wording, the appendix must still regenerate unchanged, and the
rebuilt PDF must keep the main text within five pages.

## Impact

- `interp_science_short/main.tex` — abstract, introduction, sections 3.1 to 3.3,
  limitations, reproducibility appendix.
- `interp_science_short/refs.bib` — four new entries, each verified against a
  live arXiv or Crossref record.
- `tests/test_interp4d_claims.py` — checks whose expected strings change.
- No change to any file under `results/`, and no change to the appendix
  generator or its output.
