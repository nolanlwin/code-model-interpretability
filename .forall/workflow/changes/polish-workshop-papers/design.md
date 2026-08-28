# Design: polish-workshop-papers

## Context

Two existing anonymous NeurIPS workshop submissions require editorial revision under different venue scopes and page budgets. Their empirical content and bibliography already exist, so the work is a constrained scholarly revision rather than new research generation.

## Goals / Non-Goals

### Goals

- Align each manuscript with its official workshop call and audience.
- Improve argument structure, precision, qualification, and readability.
- Verify external claims and citations against authoritative sources.
- Verify reported numerical claims against committed tables, figures, and result files.
- Produce successfully compiled PDFs whose main texts respect the four-page and five-page limits.

### Non-Goals

- Inventing experiments, results, affiliations, credentials, institutional endorsement, or citations.
- Altering generated result tables by hand.
- Claiming formal verification of prose or empirical conclusions.
- Expanding either paper beyond its existing scientific scope.

## Decisions

Each paper will be revised independently because the venues reward different contributions. The LP4FM paper will foreground linguistic medium, cross-language transfer, and the distinction between surface and internal representations. The Interp as a Science paper will foreground measurement validity, intervention evidence, falsifiability, and implications for mechanistic interpretability methodology.

Fact-checking will use the official venue calls, primary papers or DOI records, model and dataset documentation, and committed experimental outputs. Unsupported claims will be narrowed or removed. New citations will be added only when a primary source is verified and directly relevant.

Editing will preserve LaTeX structure and generated appendix inputs. Main-text space will be reallocated toward the central evidence and limitations. Both papers will be compiled through full bibliography passes, inspected for unresolved references, and checked page by page to locate the main-text and reference boundary.

## Risks / Trade-offs

Stronger qualification may reduce rhetorical force but improves credibility. Adding missing methodological detail may pressure the page limits, so secondary analyses may move to appendices. The available experimental design may not support every desired claim, in which case the prose must remain narrower than the motivating hypothesis.

## Migration Plan

No runtime migration is required. Existing `.tex` files are revised in place, generated appendix files remain untouched unless their generators are intentionally updated, and rebuilt PDFs replace the prior compiled outputs.

## Open Questions

Any factual claim that cannot be verified from an authoritative source or committed result will be reported to the user rather than guessed.
