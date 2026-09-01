# Verification plan: trim-interp-science-page-limit

## Machine-checked requirements

No requirement introduces executable logic suitable for formal verification. Both requirements remain spec-tracked.

## Property-based tests

None.

## Spec-tracked checks

- Compile `interp_science_short/main.tex` with the repository LaTeX build.
- Inspect the generated PDF and confirm that the Limitations section ends on page five and the references start on page six.
- Run `tests/test_interp4d_claims.py` to check the retained headline claims against committed evidence.

## Mapping delta

See `mapping.delta.yaml` in this change directory.
