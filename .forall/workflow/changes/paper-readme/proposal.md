# Proposal: paper-readme

## Why

The README is still a pipeline how-to. Famous paper repositories lead with the
paper, a teaser, the claim, a results table, then reproduction commands, then
citation. A visitor who lands here after de-anonymization should meet the
scientific object first.

## What Changes

Rewrite `README.md` in that genre. Keep the working setup and experiment
commands, but demote them below the papers, the central finding, and
citation. Use an existing figure as the teaser. Do not invent venue IDs,
authors, or arXiv handles. The manuscripts are still anonymous.

## Capabilities

### New Capabilities

- `paper-readme` — requirements on how the repository presents the papers
  to a first-time visitor.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. This is presentation copy. Verification is by scenario
check: both paper titles appear, no fabricated identifiers, the teaser path
exists, and the install and experiment commands from the previous README
remain runnable as written.

## Impact

- `README.md` only.
- No manuscript, pipeline, or artifact change.
