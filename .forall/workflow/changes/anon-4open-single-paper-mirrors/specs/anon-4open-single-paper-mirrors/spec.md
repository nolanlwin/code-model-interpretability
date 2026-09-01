# Delta spec: anon-4open-single-paper-mirrors

## ADDED Requirements

### Requirement: Each paper cites an anonymous code URL

Both workshop manuscripts SHALL point checklist items 5 and 13 at an
`anonymous.4open.science` URL. They MUST NOT name a GitHub owner, a GitHub
repository, or a Hugging Face username.

#### Scenario: Checklist justifications are read

- **WHEN** either paper's checklist is read
- **THEN** items 5 and 13 remain Yes
- **AND** both justifications contain `anonymous.4open.science`
- **AND** neither `github.com` nor `nolanlwin` appears in the manuscript or
  checklist

### Requirement: Snapshots are single-paper

Each anonymous snapshot SHALL contain only the paper it supports. It MUST NOT
mention the other workshop, the other title, or the author GitHub handle.

#### Scenario: Interp snapshot is searched

- **WHEN** the Interp as a Science snapshot is searched
- **THEN** it does not contain `nolanlwin`, `LP4FM`, or
  `Cross-Language Probe Invariance`

#### Scenario: LP4FM snapshot is searched

- **WHEN** the LP4FM snapshot is searched
- **THEN** it does not contain `nolanlwin`, `Interpretability as a Science`,
  or `Same Score, Different Evidence`
