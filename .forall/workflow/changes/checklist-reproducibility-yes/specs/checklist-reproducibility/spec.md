# Delta spec: checklist-reproducibility

## ADDED Requirements

### Requirement: Item 4 is Yes

Checklist item 4, experimental result reproducibility, SHALL be answered Yes
in both workshop papers.

#### Scenario: Checklists are read

- **WHEN** either `checklist.tex` is read at the experimental-reproducibility item
- **THEN** the answer is `\answerYes{}`

### Requirement: Item 5 is Yes

Checklist item 5, open access to data and code, SHALL be answered Yes in both
workshop papers.

#### Scenario: Checklists are read at open access

- **WHEN** either `checklist.tex` is read at the open-access item
- **THEN** the answer is `\answerYes{}`
- **AND** the justification does not name a GitHub URL
