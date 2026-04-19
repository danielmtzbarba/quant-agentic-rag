# Phase 1 Grounding Hardening

Last updated: 2026-04-18

## Purpose

Phase 1 improves thesis grounding before adding more sophistication to synthesis or retrieval.

The immediate goal is to enforce a narrow, testable citation contract so the verifier and thesis
writer use the same rules.

## Scope

Phase 1 covers:

- exact inline citation syntax for thesis output
- deterministic rejection of malformed citation formats
- deterministic rejection of uncited numeric claims
- deterministic rejection of placeholder grounding text such as `evidence not provided`

Phase 1 does not yet cover:

- repair loops
- section-specific rewrite flows
- verifier-driven regeneration
- end-to-end grounding repair tests

## Citation Contract

All thesis citations must use this exact format:

```text
[source:<id>]
```

Examples:

- valid: `[source:filing-1]`
- valid: `[source:nvda-10-k-item_7]`
- invalid: `(source:filing-1)`
- invalid: `source:filing-1`
- invalid: footnotes or alternate citation wrappers

This is an intentionally strict contract.

The verifier extracts citations using exact bracket syntax. Any looser format creates a mismatch
between what the thesis appears to cite and what the verifier can actually ground.

## Prompting Rules

The thesis prompt now requires:

- exact `[source:<id>]` syntax only
- no alternate citation format
- inline citations for every material claim
- inline citations for every numeric claim
- omission of unsupported claims instead of placeholders

The prompt also explicitly instructs the model to delete any sentence or bullet that cannot be
grounded with exact citations.

## Deterministic Validator

The Phase 1 validator runs after thesis generation.

It rejects reports when any of the following are found:

### 1. Prohibited placeholder text

Current blocked phrase:

- `evidence not provided`

Reason:

- this phrase encodes a known synthesis failure and should not appear in final output

### 2. Malformed source references

Current malformed forms include:

- `(source:<id>)`
- bare `source:<id>`

Reason:

- the verifier only recognizes `[source:<id>]`
- malformed references create false appearance of grounding

### 3. Uncited numeric claims

Any line containing a numeric claim without an exact inline citation is rejected.

This includes examples like:

- percentages
- ratios
- dollar values
- counts

Reason:

- numeric claims are high-risk for hallucination or unsupported synthesis
- they should be easy to validate mechanically

## Implementation Notes

Current implementation points:

- thesis prompt:
  [prompts.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/prompts.py)
- deterministic validator:
  [workflow.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/workflow.py)
- unit tests:
  [test_verifier_grounding.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/tests/test_verifier_grounding.py)

The validator is intentionally narrow.

It is not trying to prove semantic truth in Phase 1. It is enforcing syntax and minimum grounding
discipline so later repair and rewrite phases operate on a cleaner contract.

## Expected Workflow

1. Thesis writer generates the report.
2. Deterministic validator checks syntax and grounding hygiene.
3. If validation fails, the workflow raises a synthesis error.
4. Later phases will replace this hard fail with a controlled repair path.

## Testing Strategy

Phase 1 should keep the following tests green:

- exact citation extraction works
- malformed `source:` references are rejected
- uncited numeric claims are rejected
- exact bracketed numeric citations are accepted
- placeholder grounding text is rejected

## Next Phases

Phase 2 should add:

- section-by-section grounding packets with evidence snippets
- less reliance on raw JSON input for the thesis writer

Phase 3 should add:

- one verifier-driven repair pass
- state fields for repair attempts and outcomes

Phase 4 should add:

- end-to-end canned regression tests asserting final verifier pass
