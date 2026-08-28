# Delta spec: verified-academic-claims

## ADDED Requirements

### Requirement: Source-backed factual claims

The revised manuscripts SHALL retain or introduce factual claims only when supported by the papers' reported evidence or by a verified primary or authoritative external source.

#### Scenario: A factual claim is revised

- **WHEN** a model, dataset, prior result, venue requirement, or literature claim is changed
- **THEN** the claim is checked against a primary paper, official documentation, the venue call, or committed experimental results

### Requirement: Citation integrity

Every in-text citation MUST resolve to a bibliography entry, and every material literature claim MUST accurately represent the cited source.

#### Scenario: Bibliographies are audited

- **WHEN** citation keys and cited claims are checked
- **THEN** missing, orphaned, unverifiable, or materially mischaracterized citations are corrected or explicitly reported as unresolved

### Requirement: Honest uncertainty

The revised manuscripts MUST distinguish measured results from interpretation and MUST qualify causal, mechanistic, universal, and novelty claims to match the available evidence.

#### Scenario: Evidence is limited

- **WHEN** a conclusion rests on a small sample, correlational probe, heuristic labeler, or restricted model family
- **THEN** the manuscript states the corresponding limitation and avoids stronger causal or general claims

## MODIFIED Requirements

## REMOVED Requirements
