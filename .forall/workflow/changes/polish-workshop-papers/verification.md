# Verification plan: polish-workshop-papers

## Machine-checked requirements

No manuscript requirement is suitable for formal proof. Academic prose quality, venue fit, citation interpretation, and empirical claim scope require source-backed scholarly review. Page limits and successful compilation are executable finite checks rather than program contracts.

## Property-based tests

No property-based tests are planned because the user did not request them and the requirements do not quantify over an infinite input domain.

## Executable and source-backed checks

- Compile each manuscript through LaTeX and BibTeX passes with fatal errors enabled.
- Check logs for undefined citations, undefined references, and fatal LaTeX errors.
- Inspect PDF page boundaries to confirm four LP4FM main-text pages and five Interp as a Science main-text pages.
- Compare numerical claims in prose with committed tables, generated appendices, figures, and result files.
- Resolve every in-text citation key against the corresponding bibliography.
- Check literature and venue claims against primary papers, DOI records, official model or dataset documentation, and official workshop calls.
- Review anonymity and ensure no affiliation or credential is invented.

## Mapping delta

See `mapping.delta.yaml` in this change directory. All requirements are explicitly spec-tracked with `verified: false`.
