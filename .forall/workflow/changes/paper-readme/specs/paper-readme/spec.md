# Delta spec: paper-readme

## ADDED Requirements

### Requirement: README leads with the papers

The README SHALL present both workshop papers before pipeline instructions,
in the style of an official paper repository.

#### Scenario: A visitor opens the repository

- **WHEN** `README.md` is read from the top
- **THEN** both paper titles appear before any install command
- **AND** each title is linked to its manuscript PDF in the repository

### Requirement: Claims stay inside the manuscripts

The README SHALL not invent authors, arXiv identifiers, or acceptance status,
because the submissions are anonymous and under review.

#### Scenario: Citation and badges are read

- **WHEN** the citation block and header links are read
- **THEN** authors are given as Anonymous
- **AND** no arXiv identifier appears

### Requirement: Reproduction commands survive the rewrite

The README SHALL still contain the install, dataset, and experiment commands
that the previous README documented, so the rewrite does not drop a working
entry point.

#### Scenario: A user follows Setup and Experiments

- **WHEN** the Setup and Experiments sections are read
- **THEN** they include `pip install -r pipeline/requirements.txt`
- **AND** they include `pipeline.run_experiment` for perturbation and crosslang
- **AND** they include the published dataset identifier `dhyuti-n/xlcost-variable-roles`
