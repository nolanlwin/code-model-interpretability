---
name: forall-archive
description: Archive a completed Forall change after verification. Merges mapping and moves the change to archive.
license: MIT
compatibility: Requires forall CLI.
metadata:
  author: forall
  version: "1.0"
---

Archive a completed change.

**Input**: Optionally a change name.

**Steps**

1. **Select the change**

   ```bash
   forall list --json
   ```

   Ask if ambiguous.

2. **Check status**

   ```bash
   forall status --change "<name>" --json
   ```

   Warn if artifacts incomplete or tasks remain unchecked. Confirm with user before proceeding.

3. **Run verification**

   ```bash
   forall check --change "<name>"
   ```

   If exit code ≠ 0: report CRITICAL issues and **stop** unless user explicitly overrides.

4. **Archive**

   ```bash
   forall archive "<name>"
   ```

   This runs check, merges `mapping.delta.yaml` into `.forall/verified/mapping.yaml`, and moves the change to `.forall/changes/archive/YYYY-MM-DD-<name>/`.

5. **Summarize**

   ```
   ## Archive Complete

   **Change:** <name>
   **Archived to:** .forall/changes/archive/YYYY-MM-DD-<name>/
   **Mapping:** merged into .forall/verified/mapping.yaml
   **Formal verify:** passed
   ```

**Guardrails**
- Do not archive if `forall check` fails without explicit user override
- Do not skip mapping merge — traceability lives in `mapping.yaml`
- Confirm when tasks or artifacts are incomplete
