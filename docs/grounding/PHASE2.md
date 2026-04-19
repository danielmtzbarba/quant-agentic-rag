# Phase 2 Grounding Packet

Last updated: 2026-04-18

## Purpose

Phase 2 changes the thesis writer's primary input contract.

The thesis writer should not be forced to reconstruct grounded sections from raw serialized state.
Instead, it should receive a section-by-section grounding packet that already contains:

- section identity
- section objective
- findings assigned to the section
- allowed evidence ids
- compact evidence snippets tied to those ids

This reduces the gap between structured analyst output and final cited prose.

## Problem With Raw JSON

Passing a raw `model_dump_json()` payload into the thesis writer creates two avoidable problems:

1. the model must infer which findings belong in which section
2. the model must map evidence ids back to supporting snippets with weak local context

That makes citation drift more likely.

The thesis writer can still produce fluent prose, but it is more likely to:

- cite the wrong id
- drop evidence ids entirely
- paraphrase unsupported findings
- overfit to structure rather than evidence

## Phase 2 Contract

The thesis writer now uses a section grounding packet as the primary synthesis input.

The packet includes, for each section:

- `section_id`
- `title`
- `objective`
- section-level allowed evidence ids
- ordered findings

For each finding:

- `finding`
- `analyst`
- `confidence`
- `finding_type` when present
- `missing_data` when present
- allowed evidence ids for that finding
- up to two supporting snippets, each tied to an exact `[source:<id>]`

## Intended Behavior

The writer should:

- follow the section order in the packet
- use only evidence ids present in the packet
- cite only ids shown in the packet
- omit claims that cannot be grounded from those findings and snippets

The writer should not:

- invent evidence ids
- reconstruct unsupported facts from memory
- rely on raw JSON assumptions

## Implementation Notes

Current implementation points:

- packet renderer:
  [workflow.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/workflow.py)
- thesis prompt:
  [prompts.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/prompts.py)
- packet shape test:
  [test_thesis_preparation.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/tests/test_thesis_preparation.py)

The current packet is plain text, not JSON.

That is intentional for Phase 2. The goal is to give the thesis writer a compact, human-readable,
evidence-first context block without forcing the model to navigate nested serialized objects.

## Packet Shape

Conceptually, the packet looks like:

```text
Section ID: executive_summary
Section: Executive Summary
Objective: Summarize the highest-signal conclusions across analysts.
Allowed section evidence ids: [source:filing-1], [source:risk-1]
Findings:
1. finding=Revenue growth remains strong
   analyst=fundamentals
   confidence=0.90
   allowed_evidence_ids=[source:filing-1]
   snippet [source:filing-1] filing | 10-Q MDA | Revenue growth accelerated ...
```

This is a synthesis packet, not an audit artifact.

It is optimized for:

- local grounding
- citation compliance
- bounded prompt size

## Why Snippets Matter

Listing evidence ids alone is not enough.

The writer needs compact local evidence so it can:

- choose the right citation
- avoid merging unrelated findings
- avoid vague uncited restatement

Using short snippets also reduces the chance that the model will cite an id without actually using
the evidence associated with it.

## Testing Strategy

Phase 2 should keep the following guarantees under test:

- packet contains section identity and objective
- packet contains allowed evidence ids
- packet contains finding-level evidence ids
- packet contains snippets tied to exact source ids
- packet includes missing-data annotations when present

These tests do not prove final thesis quality.

They verify that the synthesis input contract remains grounded and structured.

## Relationship To Later Phases

Phase 3 should add:

- verifier-driven repair using the same grounding packet
- one bounded rewrite attempt on failed grounding

Phase 4 should add:

- end-to-end regression tests that assert final verifier pass on a canned run

## Design Constraint

The packet should remain bounded.

If it grows too large, the model will lose the benefit of localized evidence.

Current guidance:

- keep section-level packet text compact
- include at most a small number of snippets per finding
- prefer the highest-signal snippets over exhaustive evidence dumps
