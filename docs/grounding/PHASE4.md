# Phase 4 Canned End-to-End Grounding Regressions

Last updated: 2026-04-18

## Purpose

Phase 4 adds a small canned end-to-end regression suite for grounding.

The suite exists to catch workflow regressions that unit tests alone will miss.

Typical failure modes:

- thesis prompt changes break citation behavior
- grounding packet changes break synthesis fidelity
- verifier threshold or logic changes alter pass/fail behavior
- repair logic changes stop recovering simple citation omissions

## What "Canned" Means

These tests use fixed local fixtures only.

They do not depend on:

- network calls
- live market data
- mutable corpora
- changing timestamps
- external APIs

The suite uses deterministic stubs for:

- planner / thesis / repair / verifier model calls
- analyst structured outputs
- fundamentals snapshot retrieval
- local corpus retrieval

## What "End-to-End" Means

The tests run the full workflow graph, not isolated helpers.

Covered stages:

1. planner
2. retrieval helpers
3. analyst outputs
4. thesis generation
5. verifier
6. optional single repair pass

The pass condition is the final workflow result, not an intermediate helper return.

## Current Fixture Set

### 1. Clean-pass fixture

Purpose:

- prove a stable grounded thesis passes the verifier without invoking repair

Assertions:

- final `verification_status == "pass"`
- no repair attempted
- exact `[source:<id>]` citations present
- `unsupported_findings == 0`
- `partially_grounded_findings == 0`

### 2. Repair fixture

Purpose:

- prove a stable under-cited thesis can be repaired into a passing result

Assertions:

- final `verification_status == "pass"`
- exactly one repair attempt occurred
- `initial_report` preserves the first thesis
- repaired report contains the missing source id
- final unsupported and partially grounded findings are zero

## Current Test Location

Implementation:

- [test_grounding_e2e.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/tests/test_grounding_e2e.py)

This suite is intentionally small.

Its job is not broad evaluation. Its job is to provide a stable regression gate for the grounding
contract.

## Fixture Design Rules

Stable fixtures should:

- use exact local `EvidenceRecord` instances
- use deterministic `AnalystOutput` objects
- avoid current-date dependence
- avoid any non-deterministic model behavior
- assert final verifier pass explicitly

Fixtures should not:

- fetch real filings or news
- depend on live embeddings
- depend on provider response formats
- mix multiple independent failure modes in one scenario

## Success Criteria

The regression suite is working if:

- a fully grounded canned thesis passes without repair
- a simple citation omission is recovered by one repair pass
- future grounding regressions fail these tests quickly

## Future Expansion

The suite can later grow with a small number of additional fixtures for:

- malformed citation syntax
- unsupported numeric claims
- contradiction-heavy synthesis
- evidence-gap-heavy outputs

The suite should remain small and deterministic.
