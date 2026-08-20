---
name: forall-propose
description: Propose a new Forall change with all spec artifacts through verification plan. Use when the user wants to describe what to build and get proposal, specs, design, verification, and tasks ready for implementation.
license: MIT
compatibility: Requires forall CLI.
metadata:
  author: forall
  version: "1.0"
---

Propose a new change — create the change and generate all planning artifacts in sequence.

Artifacts (forall workflow):
- `proposal.md` — what & why
- `specs/**/*.md` — requirements & scenarios
- `design.md` — how
- `verification.md` — proof plan + `mapping.delta.yaml`
- `tasks.md` — implementation checklist

When ready to implement, run `/forall:apply`.

**Input**: Change name (kebab-case) OR a description of what to build.

**Steps**

1. **If input is unclear, ask what to build** (not what to name it)

   Ask about scope or requirements if needed. Derive a kebab-case name yourself (e.g. "add session timeout" → `add-session-timeout`).

2. **Create the change**

   ```bash
   forall propose "<name>"
   ```

   Creates `.forall/changes/<name>/` with `proposal.md` and `mapping.delta.yaml`.

3. **Get artifact status**

   ```bash
   forall status --change "<name>" --json
   ```

   Parse JSON:
   - `artifacts[]` — id, status (`done` | `ready` | `blocked`), blockedBy
   - `applyReady` — true when all apply prerequisites exist
   - `changeDir` — path to the change folder

4. **Create artifacts in dependency order until apply-ready**

   Loop until `applyReady` is true:

   a. Find the first artifact with `status: "ready"`.

   b. Get instructions:
      ```bash
      forall instructions <artifact-id> --change "<name>"
      ```
      JSON fields: `instruction`, `template`, `resolvedOutputPath`, `context`, `rules`, `blockedBy`.

   c. Read completed dependency files from the change directory for context.

   d. Write the artifact using `template` as structure. Apply `context` and `rules` as constraints — do NOT copy them into the file.

   e. For **specs**: create one file per capability under `.forall/changes/<name>/specs/<capability>/spec.md`.

   f. For **verification**: update `mapping.delta.yaml` with verified requirements (id, file, symbols, contract).

   f. Re-run `forall status --change "<name>" --json` after each artifact.

5. **Show final status**

   ```bash
   forall status --change "<name>"
   ```

**Output**

Summarize:
- Change name and path
- Artifacts created
- Which requirements are `verified: true` in mapping delta
- "Ready for implementation — run `/forall:apply`"

**Guardrails**
- Create ALL artifacts required before apply (through `tasks`)
- For verified requirements: assign TS file + symbols in `mapping.delta.yaml`
- If change already exists, ask to continue (`/forall:continue`) or pick a new name
- Verify each file exists before proceeding to the next artifact
