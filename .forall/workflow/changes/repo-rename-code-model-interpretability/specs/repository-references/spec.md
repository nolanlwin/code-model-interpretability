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

### Requirement: Artifact root follows the renamed Drive folder

The Drive artifact root SHALL name the folder as it now exists in Drive, since
a stale root fails silently rather than raising.

#### Scenario: Notebook resolves the artifact root

- **WHEN** a notebook computes its Drive artifact root
- **THEN** the path is `/content/drive/MyDrive/code-model-interpretability`
- **AND** the subfolders `xlcost`, `masked`, `causal`, `crosslang`, and `activations_java` are addressed beneath it

#### Scenario: Laptop addresses the same folder over Drive sync

- **WHEN** the macOS Google Drive sync candidates are read
- **THEN** they name the same folder as the Colab mount path

#### Scenario: No read path retains the retired root

- **WHEN** the notebooks and scripts are searched for a retired artifact root
- **THEN** the only surviving retired names are repository checkout fallbacks, never an artifact root

### Requirement: Browse links name the current repository

Links that address the repository through a third-party viewer SHALL use the
current name, including forms that embed the owner and repository without the
`github.com` host.

#### Scenario: Colab badge is followed

- **WHEN** the `colab.research.google.com/github/...` link in `README.md` is read
- **THEN** it names `nolanlwin/code-model-interpretability`

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
