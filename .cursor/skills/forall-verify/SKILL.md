---
name: forall-verify
description: Verify a Forall change — spec completeness plus formal proofs. Use before archive to validate planning artifacts, implementation, and machine-checked verification.
license: MIT
compatibility: Requires forall CLI.
metadata:
  author: forall
  version: "1.0"
---

Verify that a change is ready to archive.

Forall verification has two layers:
1. **Planning** — tasks complete, specs cover requirements (heuristic)
2. **Formal** — `forall check` (mapping + Dafny proofs)

**Input**: Optionally a change name.

**Steps**

1. **Select the change**

   ```bash
   forall list --json
   ```

   Ask the user if ambiguous.

2. **Load artifacts**

   Read from `.forall/changes/<name>/`:
   - `proposal.md`, `specs/**/*.md`, `design.md`, `verification.md`, `tasks.md`
   - `mapping.delta.yaml`

3. **Verify completeness (heuristic)**

   - Parse `tasks.md`: flag incomplete `- [ ]` as CRITICAL
   - For each `### Requirement:` in delta specs, search codebase for implementation evidence
   - Flag missing implementations as CRITICAL or WARNING

4. **Verify coherence (heuristic)**

   - Compare implementation to `design.md` decisions
   - Note divergences as WARNING

5. **Run formal verification gate**

   ```bash
   forall check --change "<name>"
   ```

   For `property_tested: true` requirements, check runs the **property-tests** phase (`.forall/scenarios/*.property.ts`). On failure, read `counterexample` and `pbt.seed` from the JSON report and fix before re-running. Reproduce with:

   ```bash
   forall check --change "<name>" --pbt-seed 42 --pbt-examples 100
   ```

   Read `.forall/verified/reports/<name>-verify.json` if present.

   Map CRITICAL proof/mapping failures to the report. These **block archive**.

6. **Generate report**

   ```
   ## Verification Report: <name>

   ### Summary
   | Layer        | Status        |
   |--------------|---------------|
   | Completeness | X/Y tasks     |
   | Coherence    | notes         |
   | Formal       | PASS/FAIL     |

   ### CRITICAL (must fix)
   - ...

   ### WARNING
   - ...

   ### Assessment
   Ready for archive / N critical issues remain
   ```

**Final rule**

- Any CRITICAL from `forall check` → **not ready for archive**
- Heuristic WARNINGs alone → may archive with user confirmation
- If formal check passes and tasks complete → "Ready for `/forall:archive`"

**Guardrails**
- Always run `forall check` — do not skip formal verification for `verified: true` requirements
- Prefer actionable findings with file paths
- When uncertain, downgrade severity (SUGGESTION over WARNING)
