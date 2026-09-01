# Delta spec: repository-references

## ADDED Requirements

### Requirement: Notebooks name the repository directly

Every hard-coded repository URL in an executable notebook or script SHALL use
the current repository name, so that cloning and pushing do not depend on a
GitHub rename redirect.

#### Scenario: Repository is searched for retired URLs

- **WHEN** the working tree is searched for `github.com/nolanlwin/mech-interp`
- **THEN** no match is found outside git metadata

#### Scenario: Notebook clones the repository

- **WHEN** a notebook's clone cell is read
- **THEN** the URL is `https://github.com/nolanlwin/code-model-interpretability.git`

### Requirement: Artifact paths are preserved

Renaming the repository SHALL NOT move any path that names user storage rather
than the repository, because those paths locate real artifacts that the rename
does not touch.

#### Scenario: Drive artifact roots are inspected

- **WHEN** notebooks referencing `/content/drive/MyDrive/mech-interp/` are read
- **THEN** those paths are unchanged by this change

### Requirement: Edited notebooks remain loadable

Notebooks SHALL remain valid JSON after the substitution, since a malformed
notebook fails only at execution time on a remote runtime.

#### Scenario: Notebooks are parsed

- **WHEN** every edited notebook is parsed as JSON
- **THEN** each parses without error
- **AND** each retains its original cell count

## MODIFIED Requirements

None.

## REMOVED Requirements

None.
