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
- Rename the ephemeral checkout directories in the two notebooks that named the
  repository after a retired name. Both pass an explicit target directory to
  `git clone`, so neither was broken by the rename, but a scratch directory
  named after a repository that no longer exists misleads whoever reads the
  cell next.
- Extend, rather than replace, the Drive fallback list that
  `colab_activations_and_probing.ipynb` searches for a repository checkout.
  A repository checkout already sitting in Drive under a retired name keeps
  that name, so the retired entries must keep resolving while a fresh checkout
  under the current name also resolves.
- Rewrite the Drive artifact root, now that the owner has renamed the Drive
  folder holding `xlcost`, `masked`, `causal`, `crosslang`, and
  `activations_java` to match the repository. This covers both the Colab
  mount path and the macOS Google Drive sync paths that address the same
  folder from a laptop.
- Rewrite the Colab badge in `README.md`. It uses the
  `colab.research.google.com/github/<owner>/<repo>` form, so it did not appear
  in a search for `github.com/<owner>`.
- Rename the remaining ephemeral checkout directories under `/content` and
  `/root` so the whole tree agrees on one name.

The Drive rewrite is the one edit here that can fail silently. A wrong artifact
root does not raise: the restore step uses `cp -n`, which succeeds against a
missing source, and probes would then run on absent stores. The root is
therefore rewritten to a single confirmed name rather than guessed, and the
retired name is left in no read path.

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
