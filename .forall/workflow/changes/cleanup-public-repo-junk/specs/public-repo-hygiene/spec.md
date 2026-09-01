# Delta spec: public-repo-hygiene

## ADDED Requirements

### Requirement: Named junk files are absent

The files the owner listed for deletion SHALL not be present in the working
tree or the git index.

#### Scenario: Repository root is listed

- **WHEN** the repository root is listed
- **THEN** it does not contain `AGENTS.md`, `PROTOCOL.md`, `RESULTS.md`,
  `RESULTS_TABLE.md`, `WORK_UPDATE_ACCUMULATOR.md`, `pachingPlanAlgoverse.md`,
  `go_output.png`, or `java_output.png`

### Requirement: Superseded drafts are absent

Internal planning notes and superseded paper drafts SHALL not ship in the
public tree.

#### Scenario: Draft directories are inspected

- **WHEN** the working tree is searched for `sree_paper_ready/`, `lp4fm_paper/`,
  and `docs/`
- **THEN** those paths are gone

### Requirement: Manuscripts and entry points remain

Cleanup SHALL not delete the two workshop manuscripts, the dataset card, or
the README.

#### Scenario: Visitor opens the repository

- **WHEN** the repository is cloned
- **THEN** `README.md`, `interp_science_short/main.tex`, `lp4fm_short/main.tex`,
  and `dataset_card/README.md` exist
- **AND** `README.md` does not link to `PROTOCOL.md`
