# Delta spec: interp-science-title

## ADDED Requirements

### Requirement: Title states the paper's finding

The submission title SHALL lead with the empirical finding rather than with a
classification of claim types, so that the first thing a reviewer reads points
at the paper's evidence.

#### Scenario: Title is rendered

- **WHEN** the manuscript is compiled
- **THEN** the title reads "Same Score, Different Evidence: Decodability, Surface Sufficiency, and Causal Relevance in Code Models"
- **AND** no file in the repository still refers to the retired title

### Requirement: Title is supported by the reported results

The title SHALL be defensible from the manuscript's own evidence, because the
paper's subject is the discipline of matching claims to evidence.

#### Scenario: Claim of equal scores with differing evidence

- **WHEN** a reader checks the title against Section 3
- **THEN** the three cases report comparable probe scores near `0.98`
- **AND** each case licenses a different conclusion, namely a surface-sufficiency bound, a confounded control, and a bounded site-state effect

### Requirement: Preserved submission budget

The retitled manuscript SHALL continue to satisfy the venue's five-page
main-text limit and its manuscript-to-artifact audit.

#### Scenario: Manuscript is rebuilt

- **WHEN** the manuscript is compiled using the repository build
- **THEN** all main-text sections finish by page five and references begin on page six or later
- **AND** the claim audit passes

## MODIFIED Requirements

None.

## REMOVED Requirements

None.
