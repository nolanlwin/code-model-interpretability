# Proposal: checklist-impacts-licenses-assets

## Why

Checklist items 10, 12, and 13 were NA or No because the manuscripts did not
discuss broader impacts, name asset licenses, or document new assets. Marking
Yes without that text would be false. The papers need those paragraphs, then
the checklist can answer Yes.

## What Changes

Add appendix sections on broader impacts, existing-asset licenses, and new
assets to both workshop papers. Name XLCoST and Qwen Apache 2.0 and StarCoder2
BigCode OpenRAIL-M. Point the checklist justifications at those sections.

## Capabilities

### New Capabilities

- `checklist-impacts-licenses-assets` — items 10, 12, and 13 are Yes only
  after the manuscripts contain the matching discussion.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. Presentation plus license names checked against public
records. Scenario check that both papers discuss impacts, name licenses, and
document new assets, and that both checklists answer Yes.

## Impact

- Both `main.tex` files and both `checklist.tex` files under `paper/`.
- Rebuilt PDFs. Main-text page limits are unchanged because the new text is
  in the appendix.
