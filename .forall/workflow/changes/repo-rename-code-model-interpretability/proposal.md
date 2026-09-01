# Proposal: repo-rename-code-model-interpretability

## Why

The GitHub repository is being renamed from `mech-interp` to
`code-model-interpretability`. The old name asserts mechanism, which conflicts
with the manuscripts' central argument that probe accuracy rarely licenses a
mechanistic conclusion. The new name states the field without the claim.

Eleven Colab and Kaggle notebooks clone the repository by hard-coded URL, and
one of them pushes results back to it. GitHub redirects renamed repositories,
so these keep working, but the redirect lapses the moment any account claims
the retired name. Notebooks are executed by collaborators on remote runtimes
where a silent clone failure is expensive to diagnose, so the URLs should name
the repository directly rather than depend on a redirect.

## What Changes

- Rewrite every hard-coded `github.com/nolanlwin/mech-interp` clone, push, and
  browse URL to the new repository name.
- Leave Google Drive artifact paths under `/content/drive/MyDrive/mech-interp/`
  unchanged. Those name folders in the user's Drive, not the repository, and
  rewriting them would break notebooks that read committed artifacts.
- Leave ephemeral checkout directories such as `/content/mech-interp` and
  `/root/mech-interp` unchanged. They are scratch paths inside disposable
  runtimes and carry no dependency on the repository name.

## Capabilities

### New Capabilities

- `repository-references` — requirements on how the repository refers to itself
  from executable notebooks and scripts.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. This is a mechanical string substitution with no logic to
prove. Verification is by scenario check: no retired repository URL survives,
every edited notebook still parses as JSON, and no Drive artifact path moves.

## Impact

- Eleven notebooks under `notebooks/`.
- No change to manuscripts, pipeline code, or committed result artifacts.
- The local git remote and the GitHub rename itself are performed outside this
  change, since neither is a repository file.
