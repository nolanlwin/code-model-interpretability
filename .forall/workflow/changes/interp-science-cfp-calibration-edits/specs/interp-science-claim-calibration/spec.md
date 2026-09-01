# Delta spec: interp-science-claim-calibration

## ADDED Requirements

### Requirement: Comparator-scoped boolean reporting

The main text SHALL NOT present the boolean surface-sufficiency bound as if it
held under a single comparator on a single language. It SHALL state the range of
probe-minus-best-baseline differences across the committed languages and the
differences obtained under the next-best Python comparator.

#### Scenario: Body reports the full comparison

- **WHEN** a reader reads section 3.2 without consulting the appendix
- **THEN** the text states that probe minus best baseline reaches `+0.021` on PHP
- **AND** the text states that against the masked enclosing statement the Python differences are `+0.010` to `+0.017`
- **AND** the text states that the masked-line comparator is degenerate outside Python

### Requirement: Estimator-consistent causal interpretation

The manuscript SHALL NOT attribute the failed patching gate to a compressed
behavioral decision range, because recovery is a ratio normalized by the matched
class-minus-function gap and is therefore invariant to that compression.

#### Scenario: Patching null is described

- **WHEN** the abstract, section 3.3, the conclusion, or the limitations describe the failed gate
- **THEN** the description reads it as a bounded negative at the tested site, layer, and span
- **AND** no passage claims that an insensitive readout explains the observed recovery of `0.009` to `0.020`

### Requirement: Artifact-accurate provenance statements

Every provenance and sample-construction statement in the manuscript SHALL match
the committed artifact it describes.

#### Scenario: Paired boolean sample is described

- **WHEN** section 3.2 explains the reduction from the 1,301-problem source sample
- **THEN** the reduction is attributed to pooling the five seed test folds
- **AND** the text does not attribute the reduction to a predictor-overlap requirement, since `min_seed_coverage` is `1.0`

#### Scenario: Patching row count is described

- **WHEN** the reproducibility appendix states the number of recorded rows
- **THEN** the figure `4,032` is described as covering the primary and behavior schedules together

#### Scenario: Rename effect range is described

- **WHEN** section 3.1 states the effect of single-character and numeric renaming
- **THEN** the stated range does not exceed the largest tabulated magnitude of `0.072`

### Requirement: Verified bibliography entries

Every bibliography entry added by this change SHALL correspond to a real record
confirmed against a live arXiv or Crossref lookup, because the venue desk-rejects
submissions containing fabricated citations.

#### Scenario: New citation is added

- **WHEN** an entry is added to `refs.bib`
- **THEN** its title, authors, and identifier match a record retrieved from arXiv or Crossref

### Requirement: Preserved submission budget

The revised manuscript SHALL continue to satisfy the venue's five-page main-text
limit, and its committed artifacts SHALL remain unmodified.

#### Scenario: Manuscript is rebuilt

- **WHEN** the manuscript is compiled using the repository build
- **THEN** all main-text sections, including limitations, finish by page five and references begin on page six or later
- **AND** the generated appendix regenerates unchanged from the committed artifacts
- **AND** the manuscript-to-artifact audit passes

## MODIFIED Requirements

None.

## REMOVED Requirements

None.
