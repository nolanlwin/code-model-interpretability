# Proposal: anon-4open-single-paper-mirrors

## Why

Checklist items 5 and 13 stay Yes only if a reviewer can actually get the
code and new assets. A named GitHub URL would break double-blind review and
would show both workshop papers in one place. Each paper needs its own
anonymous.4open.science snapshot.

## What Changes

Create two private, identity-stripped GitHub snapshots and anonymize them on
anonymous.4open.science. Each snapshot contains only one paper. Neither
contains `nolanlwin`, a Hugging Face username, or any mention of the other
workshop. Point both checklists and both new-assets appendix paragraphs at
the matching anonymous URL.

## Capabilities

### New Capabilities

- `anon-4open-single-paper-mirrors` — each workshop PDF links a
  single-paper anonymous.4open.science URL, with no named GitHub and no
  cross-workshop mention.

### Modified Capabilities

<!-- none -->

## Verification scope

`verified: false`. Copy and URL presence, not formal logic. Scenario checks
that both checklists cite `anonymous.4open.science`, that `github.com` is
absent from both manuscripts, and that the snapshots contain no author handle
and no other-workshop title.

## Impact

- Both `main.tex` files and both `checklist.tex` files under `paper/`.
- Claim tests that currently require the phrase about omitting a named URL.
- Two private GitHub repositories used only as 4open sources, not cited in
  the PDFs.
