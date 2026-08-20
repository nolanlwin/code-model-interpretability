# Forall — verified engineering (mandatory)

This project uses **Forall verified engineering**. Specs and verification are not optional.

## Rules (always)

1. **Start work with a change** — before editing application code, run `specs.propose <name>` (or `forall propose <name>`). Derive a sensible kebab-case name from what the user asked for; don't stop to ask for a name unless the request is ambiguous.
2. **Follow the artifact workflow** — proposal → specs → design → verification → tasks → apply. Use `specs.status` / `specs.instructions` to see what is ready.
3. **Spec-only edits** belong under `.forall/changes/<name>/`. Do not skip straight to application code without an active change and completed planning artifacts.
4. **After TS edits** — hooks run `forall sync`; then run `specs.check --change <name>` (hot loop). Property tests use `--pbt-seed` / `--pbt-examples` to reproduce failures.
5. **Before finishing a turn** — `specs.check` must pass. The Stop hook blocks completion on CRITICAL failures and reports the verification tier (proved vs property-tested vs spec-tracked).
6. **Use native specs tools** — prefer `specs.*` tools for workflow and verification instead of ad-hoc shell.
7. **Hooks gate application code** — `apply_patch` and `exec_command` shell writes to app paths are blocked until the change is apply-ready. Use `apply_patch` for `.forall/changes/` artifacts.

## Starting implementation work

When the user asks you to build or change something and no change is active:

1. Pick a kebab-case change name from their request (e.g. "simple login page" → `simple-login-page`).
2. Run **`specs.propose`** with that name, then **`specs.status`** / **`specs.instructions`** to fill planning artifacts.
3. Do not edit `src/`, `mastermind/`, etc. until apply-ready (hooks enforce this).

`propose` means **open a tracked change**, not "ask the user to name the change." Only ask clarifying questions about *what* to build, not what to call it.

Fill **one artifact at a time** using `specs.instructions` for the next ready step — do not batch proposal, specs, design, verification, and tasks in a single patch unless the user explicitly asks for a fast-forward.

## When to mark `verified: true`

Use formal verification for **logic with precise rules**, not for layout copy:

- Validation rules (password length, bounds, scoring, parsing) → extract to `src/` with `//@` contracts + proofs, or scenario tests.
- Static HTML/CSS/JS presentation alone → may stay spec-tracked (`verified: false`) with scenario tests if needed.
- Do not skip proofs just because the UI lives in `mastermind/` — extract the rule to `src/` when it can be proved.

## Verification honesty

- Do **not** claim code is formally verified unless `specs.check` shows proved requirements (`verified: true` with passing proofs or scenario tests).
- Do **not** claim property-based coverage unless `property_tested: true` requirements pass the property-tests phase.
- Do **not** claim UI or browser behavior was verified unless you used browser tools in this session and observed the result.
- Do not say you "verified in the browser" or "UI pass" without a browser tool call in this session.
- Distinguish **workflow pass** (spec-tracked) from **proved pass** (formal proofs / executable scenario tests).

## Sandbox scope

If the user asks for paths outside the writable workspace, explain the sandbox limit. Offer to:
- create the work under the current project root, or
- ask them to reopen the session at the target directory.

## Search in sandbox

Prefer `rg` / `rg --files` for code search. If `rg` is unavailable in the sandbox, use `grep -r` or install ripgrep on the host (`brew install ripgrep` on macOS).

## Native agent tools

Invoked as **`specs.<name>`** in docs; the agent may expose short names (`propose`, `check`, …) under the `specs` namespace — same tools.

| Tool | Purpose |
|------|---------|
| `specs.list` | List active changes |
| `specs.propose` | Start a new change |
| `specs.status` | Artifact progress for a change |
| `specs.instructions` | Template + guidance for the next artifact |
| `specs.sync` | Merge delta specs and stub `//@` annotations |
| `specs.check` | Run the verification gate |
| `specs.archive` | Verify, merge, and archive a completed change |

## Active change

The active change is in `.forall/.active-change`, or the sole change in `.forall/changes/`, or `FORALL_CHANGE`.

## Archive

When implementation and checks pass: `specs.archive` (or `forall archive <name>`).
