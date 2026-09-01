# Proposal: trim-interp-science-page-limit

## Why

The updated Interpretability as a Science manuscript places main-text limitations on page six, exceeding the five-page main-text limit. The paper must regain compliance without removing the newly added iterator evidence or changing the calibrated scientific claims.

## What Changes

- Condense redundant main-text discussion and status reporting.
- Keep all core findings, uncertainty statements, limitations, and the five-part control matrix.
- Rebuild the submission PDF and confirm that references begin after five pages of main text.

## Capabilities

### New Capabilities

- `interp-science-page-limit`

### Modified Capabilities

None.

## Verification scope

No formal logic is introduced. The layout requirement is spec-tracked and verified by rebuilding and inspecting the PDF.

## Impact

The change affects `interp_science_short/main.tex` and its generated PDF. It changes no APIs or dependencies.
