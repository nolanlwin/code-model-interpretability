# Delta spec: workshop-page-limit-compliance

## ADDED Requirements

### Requirement: LP4FM page limit

The LP4FM manuscript MUST compile successfully with no more than four pages of main text, with references and appendices beginning afterward as permitted by the venue.

#### Scenario: LP4FM PDF is rebuilt

- **WHEN** `lp4fm_short/main.tex` is compiled with its workshop template
- **THEN** the Discussion or final main-text section ends on or before page four and the generated PDF has resolved references

### Requirement: Interp as a Science page limit

The Interp as a Science manuscript MUST compile successfully with no more than five pages of main text, with references and appendices beginning afterward as permitted by the venue.

#### Scenario: Interp as a Science PDF is rebuilt

- **WHEN** `interp4d_short/main.tex` is compiled with its workshop template
- **THEN** the Discussion or final main-text section ends on or before page five and the generated PDF has resolved references

### Requirement: Self-contained main text

Each manuscript SHALL keep the evidence necessary for its central claim in the main text rather than relying exclusively on an appendix.

#### Scenario: Central evidence is reviewed

- **WHEN** the revised argument is assessed against the venue page limit
- **THEN** the main text contains the research question, essential method, primary quantitative result, principal limitation, and scoped conclusion

## MODIFIED Requirements

## REMOVED Requirements
