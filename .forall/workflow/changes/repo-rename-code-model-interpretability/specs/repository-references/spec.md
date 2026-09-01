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

### Requirement: Checkout discovery survives the rename

A notebook that searches Drive for an existing repository checkout SHALL accept
both the current and the retired names, because renaming a repository on GitHub
does not rename a folder already present in a user's Drive.

#### Scenario: Drive holds a checkout under a retired name

- **WHEN** `colab_activations_and_probing.ipynb` searches for a repository root
- **THEN** the candidate list includes `code-model-interpretability`
- **AND** the previously accepted retired names are still present

#### Scenario: Scratch checkout directory is read

- **WHEN** a notebook's ephemeral clone directory under `/content` is read
- **THEN** it does not name a repository that no longer exists

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
