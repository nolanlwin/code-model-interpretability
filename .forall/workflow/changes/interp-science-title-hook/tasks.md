# Tasks: interp-science-title-hook

## 1. Specs & mapping

- [x] 1.1 Confirm specs cover the proposal capability
- [x] 1.2 Record the requirement mappings as spec-tracked in `mapping.delta.yaml`

## 2. Implementation

- [x] 2.1 Replace the `\title{...}` line in `interp_science_short/main.tex`
- [x] 2.2 Confirm no other file references the retired title

## 3. Verify

- [x] 3.1 Rebuild the PDF and confirm the main text still ends by page five
- [x] 3.2 Run the manuscript-to-artifact audit
- [x] 3.3 Run `forall check --change interp-science-title-hook` and resolve CRITICAL issues
