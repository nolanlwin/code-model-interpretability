---
name: forall-continue
description: Continue a Forall change by creating the next ready artifact. Use when the user wants to progress planning — specs, design, verification, or tasks.
license: MIT
compatibility: Requires forall CLI.
metadata:
  author: forall
  version: "1.0"
---

Continue working on a change by creating the **next** ready artifact.

**Input**: Optionally a change name. If omitted, infer from context or ask.

**Steps**

1. **Select the change**

   ```bash
   forall list --json
   ```

   If ambiguous, ask the user to pick. Prefer the most recently touched change.

2. **Check status**

   ```bash
   forall status --change "<name>" --json
   ```

   If `applyReady` is true: congratulate and suggest `/forall:apply` or `/forall:archive`.

   If no artifact is `ready`: show blocked artifacts and what's missing.

3. **Create ONE artifact**

   Pick the first artifact with `status: "ready"`.

   ```bash
   forall instructions <artifact-id> --change "<name>"
   ```

   Read dependency files, fill `template`, write to the path implied by `generates` / `resolvedOutputPath`.

   Special cases:
   - **specs** → `.forall/changes/<name>/specs/<capability>/spec.md`
   - **verification** → `verification.md` + update `mapping.delta.yaml`
   - **tasks** → include proof tasks (`forall check`) for verified requirements

4. **Show progress**

   ```bash
   forall status --change "<name>"
   ```

**Guardrails**
- Create **one** artifact per invocation
- Never skip the dependency order
- Ask before guessing on ambiguous requirements
