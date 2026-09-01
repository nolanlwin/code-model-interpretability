# Tasks: anon-4open-single-paper-mirrors

## 1. Specs & mapping

- [x] 1.1 Confirm specs cover the proposal capability
- [x] 1.2 Record the requirement mappings as spec-tracked in `mapping.delta.yaml`

## 2. Implementation

- [x] 2.1 Pack two identity-stripped snapshots (one paper each) and push them as private GitHub repositories
- [x] 2.2 Create anonymous.4open.science mirrors with no `nolanlwin` and no other-workshop mention
- [x] 2.3 Point both checklists (items 5 and 13) and both new-assets paragraphs at the matching anonymous URL
- [x] 2.4 Update claim tests to require the anonymous URL and still forbid `github.com`
- [x] 2.5 Rebuild both PDFs and confirm main-text page limits

## 3. Verify

- [x] 3.1 Run both paper audits
- [x] 3.2 Run `forall check --change anon-4open-single-paper-mirrors`
