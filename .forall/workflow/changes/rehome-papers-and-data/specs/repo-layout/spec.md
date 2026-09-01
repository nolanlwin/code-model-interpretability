# Delta spec: repo-layout

## ADDED Requirements

### Requirement: Manuscripts live under paper/

Both workshop manuscripts SHALL live under a single `paper/` directory.

#### Scenario: Repository is listed

- **WHEN** the working tree is listed
- **THEN** `paper/interp_science_short/main.tex` and `paper/lp4fm_short/main.tex` exist
- **AND** the old root directories `interp_science_short/` and `lp4fm_short/` do not exist

### Requirement: Dataset files live under data/

The published dataset, its card, and the patching inputs SHALL share one parent
directory.

#### Scenario: Data directory is listed

- **WHEN** `data/` is listed
- **THEN** it contains `patching/` and `dataset/`
- **AND** `data/dataset/README.md` and `data/dataset/stats.json` exist
- **AND** the old roots `dataset/` and `dataset_card/` do not exist

### Requirement: Callers follow the new paths

Scripts, tests, ignore rules, and the README SHALL address the new locations.

#### Scenario: Tests and README are read

- **WHEN** the claim audit and the README are read
- **THEN** they resolve manuscripts under `paper/`
- **AND** the README's dataset commands use `data/dataset`
