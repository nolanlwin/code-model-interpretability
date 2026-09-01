# Delta spec: checklist-impacts-licenses-assets

## ADDED Requirements

### Requirement: Broader impacts are discussed

Both manuscripts SHALL discuss a positive and a negative societal effect of
the work, so checklist item 10 can be answered Yes.

#### Scenario: Appendix is read

- **WHEN** either manuscript's appendix is read
- **THEN** it contains a broader-impacts section that states a scientific
  benefit and a misuse path

### Requirement: Existing asset licenses are named

Both manuscripts SHALL name the license of each third-party corpus and model
they use.

#### Scenario: License paragraph is read

- **WHEN** the assets section is read
- **THEN** XLCoST and the Qwen 1.5B releases are named as Apache 2.0
- **AND** StarCoder2-7B is named as BigCode OpenRAIL-M

### Requirement: New assets are documented

Both manuscripts SHALL describe the labeled dataset and analysis artifacts
and point to the dataset card.

#### Scenario: New-assets paragraph is read

- **WHEN** the new-assets section is read
- **THEN** it mentions the labeled XLCoST variable-role data, the dataset
  card, and committed result files
- **AND** it does not name a GitHub URL
