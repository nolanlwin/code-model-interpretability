# Tasks: repo-rename-code-model-interpretability

## 1. Specs & mapping

- [x] 1.1 Confirm specs cover the proposal capability
- [x] 1.2 Record the requirement mappings as spec-tracked in `mapping.delta.yaml`

## 2. Implementation

- [x] 2.1 Record each notebook's cell count before editing
- [x] 2.2 Rewrite hard-coded repository URLs to the new name
- [x] 2.3 Confirm Drive artifact paths and ephemeral checkout paths are untouched

## 3. Verify

- [x] 3.1 Confirm no retired repository URL survives outside git metadata
- [x] 3.2 Parse every edited notebook as JSON and compare cell counts
- [x] 3.3 Run `forall check --change repo-rename-code-model-interpretability`
