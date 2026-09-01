# Proposal: rehome-papers-and-data

## Why

The two manuscripts sit at the repository root next to pipeline code, and the
published dataset is split across `dataset/`, `dataset_card/`, and
`data/patching/`. A public paper repository should keep manuscripts in one
place and data in one place.

## What Changes

- Move `interp_science_short/` and `lp4fm_short/` under `paper/`.
- Move `dataset/` and `dataset_card/README.md` under `data/`, next to
  `data/patching/`. The Hugging Face card becomes `data/dataset/README.md`.
- Update defaults, ignore rules, README links, tests, and generators so they
  follow the new paths.

## Capabilities

### New Capabilities

- `repo-layout` — requirements on where manuscripts and data live.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. This is a path move. Verification is by scenario check: the
old roots are gone, the new roots exist, and the manuscript-to-artifact audit
still passes.

## Impact

- `paper/interp_science_short/`, `paper/lp4fm_short/`.
- `data/dataset/` (includes the former dataset card).
- `data/patching/` stays where it is.
- Callers that defaulted to `dataset/` or `dataset_card/` are updated.
