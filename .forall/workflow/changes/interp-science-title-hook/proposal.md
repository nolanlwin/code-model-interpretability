# Proposal: interp-science-title-hook

## Why

The submission title is currently a single technical noun phrase with no hook.
Recent interpretability papers reporting decodable-but-not-used results use a
two-part title whose first half states the finding and whose second half
describes the study, for example "Represented Is Not Computed", "Observable
Patterns Are Not Explanations", and "Located but Not Releasable".

The existing title also sets reader expectation on the paper's weakest
component. It reads as a taxonomy of claim types, which is the contribution
reviewers judged thinnest, and it sits in tension with the introduction's
statement that the paper does not propose another taxonomy. A hook stating the
empirical finding points readers at the evidence instead.

## What Changes

- Replace the title with "Same Score, Different Evidence: Decodability, Surface
  Sufficiency, and Causal Relevance in Code Models".
- Nothing else. The title appears in exactly one place in the repository.

## Capabilities

### New Capabilities

- `interp-science-title` — requirements on the submitted title's accuracy and on
  the manuscript remaining consistent and within its page budget after the
  change.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. This edits a single manuscript title line. There is no
precise logic to prove. Verification is by scenario check: the rebuilt PDF must
keep the main text within five pages, the manuscript-to-artifact audit must
pass, and no other file may reference the retired title.

## Impact

- `interp_science_short/main.tex`, line 24 only.
- No change to the abstract, body, bibliography, tests, or any committed
  artifact.
