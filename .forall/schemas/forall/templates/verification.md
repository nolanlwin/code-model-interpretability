# Verification plan: {{CHANGE_NAME}}

## Verified requirements

| Requirement id | Capability | File | Symbols | Notes |
|----------------|------------|------|---------|-------|
| | | | | |

## Property-based tests

For requirements marked `property_tested: true` in mapping:

| Requirement id | Property file | Symbol |
|----------------|---------------|--------|
| | `.forall/scenarios/<id>.property.ts` | |

Each `.property.ts` must default-export `runPropertyTests()` returning `{ ok, counterexample?, seed?, examplesRun? }`.

## Proof plan

<!-- Which .ts files get //@ annotations, which .dfy files need updates -->

## Mapping delta

See `mapping.delta.yaml` in this change directory.
