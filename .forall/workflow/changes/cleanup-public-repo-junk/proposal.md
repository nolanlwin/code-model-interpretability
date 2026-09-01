# Proposal: cleanup-public-repo-junk

## Why

The public repository still carries internal planning notes, superseded paper
drafts, screenshot dumps, and protocol logs. None of these is part of either
workshop manuscript, the pipeline, or the published dataset. They make the
landing page look like a working folder rather than a paper repository.

## What Changes

Remove the named junk files and other tracked artifacts that are not needed to
read or reproduce the two papers. Update the README so it no longer links to
`PROTOCOL.md`. Leave pipeline code, tests, notebooks, results, manuscripts,
and the dataset card in place.

## Capabilities

### New Capabilities

- `public-repo-hygiene` — requirements on what a public visitor should not
  find at the repository root.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. This is a file-removal change. Verification is by scenario
check: the named junk paths are gone, the README no longer links to them, and
the two manuscript directories remain.

## Impact

- Deleted planning and draft files listed in the change.
- `README.md` loses its `PROTOCOL.md` hyperlink. The protocol summary stays.
