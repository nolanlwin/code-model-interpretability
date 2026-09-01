# Tasks: interp-science-cfp-calibration-edits

## 1. Specs & mapping

- [ ] 1.1 Confirm specs cover all proposal capabilities
- [ ] 1.2 Record requirement mappings in `mapping.delta.yaml` as spec-tracked, since the change edits manuscript prose rather than precise logic

## 2. Implementation

- [ ] 2.1 Add four verified bibliography entries to `interp_science_short/refs.bib` covering reporting standards, falsifiability, causal mediation, and the subspace patching illusion
- [ ] 2.2 Cite that literature in the introduction so the reporting-matrix contribution is positioned against prior reporting standards
- [ ] 2.3 Add the per-language difference range and the second-comparator sensitivity to section 3.2
- [ ] 2.4 Note that the masked-line comparator is degenerate outside Python, where XLCoST programs arrive on a single line
- [ ] 2.5 Reframe the patching null as a bounded negative in the abstract, section 3.3, the conclusion, and the limitations, removing the readout-scale attribution
- [ ] 2.6 Correct the paired boolean sample-reduction sentence in section 3.2
- [ ] 2.7 Correct the row-count statement in the reproducibility appendix
- [ ] 2.8 Correct the rename effect range in section 3.1 to `0.046`--`0.072`
- [ ] 2.9 Trim redundant main text as needed to hold the five-page budget

## 3. Verify

- [ ] 3.1 Update `tests/test_interp4d_claims.py` for the revised wording and add checks for the new claims
- [ ] 3.2 Confirm the appendix regenerates unchanged and the claim audit passes
- [ ] 3.3 Rebuild the PDF and confirm the main text ends by page five with references on page six
- [ ] 3.4 Run `forall check --change interp-science-cfp-calibration-edits` and resolve CRITICAL issues
