---
name: forall-apply
description: Implement tasks from a Forall change — code, //@ annotations, and proofs. Use when the user wants to start or continue implementation.
license: MIT
compatibility: Requires forall CLI.
metadata:
  author: forall
  version: "1.0"
---

Implement tasks from a Forall change.

**Input**: Optionally a change name (e.g. `/forall:apply add-session-timeout`).

**Steps**

1. **Select the change**

   ```bash
   forall list --json
   ```

   Announce: "Using change: <name>".

2. **Check apply readiness**

   ```bash
   forall status --change "<name>" --json
   ```

   If `applyReady` is false: list blocked artifacts, suggest `/forall:continue`.

3. **Get apply instructions**

   ```bash
   forall instructions apply --change "<name>"
   ```

   Read all context files from the change directory:
   - `proposal.md`, `specs/**/*.md`, `design.md`, `verification.md`, `tasks.md`
   - `mapping.delta.yaml` and `.forall/verified/mapping.yaml`

4. **Show progress**

   Parse `tasks.md` checkboxes: N/M complete.

5. **Implement tasks (loop until done or blocked)**

   For each pending task:
   - Make focused code changes
   - For **verified** requirements (see mapping):
     - Add `//@ requires`, `//@ ensures`, `//@ contract` on mapped symbols
     - Update `.dfy` proof files (additions only)
     - Run `lsc regen` / `lsc check` on touched `.ts` files
   - Mark task complete: `- [ ]` → `- [x]`
   - After app-code edits: `forall sync --change "<name>"` (annotations) then `forall check --change "<name>"` (hot loop)
   - For **property_tested** requirements: add `.forall/scenarios/<id>.property.ts` and set `property_tested: true` in mapping.delta
   - Run `forall check --change "<name>"` after proof-related or property-test tasks

   **Pause if**: task unclear, design conflict, proof failure, or user interrupts.

6. **On completion**

   ```bash
   forall check --change "<name>"
   forall status --change "<name>"
   ```

   If all tasks done and check passes: suggest `/forall:verify` then `/forall:archive`.

**Guardrails**
- Read specs and mapping before editing code
- Verified requirements MUST pass `forall check` before archive
- Keep changes minimal per task
- Do not use `//@ assume` to cheat proofs — fix the proof or spec
