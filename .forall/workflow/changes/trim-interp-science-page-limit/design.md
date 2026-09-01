# Design: trim-interp-science-page-limit

## Context

The latest main branch adds modern iterator evidence. The resulting PDF places the Limitations section on page six, while the workshop allows five pages of main text.

## Goals / Non-Goals

### Goals

- Move the complete main text, including limitations, into five pages.
- Preserve the new three-model iterator result and every claim required by the control matrix.
- Keep references and supplementary evidence outside the main-text limit.

### Non-Goals

- Remove appendix evidence.
- Change numerical findings or broaden scientific claims.
- Alter the document class, font size, or margins.

## Decisions

- Remove the short standalone identifiability section because it repeats conclusions already stated in the matrix and evidence sections.
- Shorten nearby explanatory prose and table text where the same qualification appears elsewhere.
- Preserve the explicit limitations paragraph rather than hiding limitations in the appendix.
- Rebuild with the existing LaTeX toolchain and inspect page boundaries in the produced PDF.

## Risks / Trade-offs

Compression can make the argument harder to follow or accidentally drop a qualification. Regression checks and a source comparison will confirm that the five claims, key numerical results, and limitations remain present.

## Migration Plan

Edit the main manuscript, rebuild the PDF, run the manuscript regression test, inspect the page containing the final main-text section, and run the Forall verification gate.

## Open Questions

None.
