# Proposal: checklist-reproducibility-yes

## Why

NeurIPS checklist items 4 and 5 were marked No because an anonymous review URL
was not attached to the PDF. Item 4 asks whether the paper discloses enough to
reproduce the claims, which both manuscripts do in the setup and
reproducibility appendix. Item 5 asks whether data and code are available.
Scripts, committed result files, and the labeled dataset exist. The named
repository is omitted from the PDF for double-blind review, which does not
make the answer No.

## What Changes

Mark items 4 and 5 as Yes in both workshop checklists and rewrite the
justifications. Update the Interp claim audit that currently requires those
No answers.

## Capabilities

### New Capabilities

- `checklist-reproducibility` — requirements on checklist items 4 and 5.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. This is a checklist answer change. Verification is by
scenario check: both checklists answer Yes on items 4 and 5, and the claim
audit still passes.

## Impact

- `paper/interp_science_short/checklist.tex`
- `paper/lp4fm_short/checklist.tex`
- `tests/test_interp4d_claims.py`
- rebuilt PDFs
