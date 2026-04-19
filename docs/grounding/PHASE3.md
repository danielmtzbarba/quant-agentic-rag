# Phase 3 Grounding Repair Loop

Last updated: 2026-04-18

## Purpose

Phase 3 adds one verifier-driven repair pass to the workflow.

The goal is straightforward:

- let the thesis writer make one constrained rewrite when deterministic grounding fails
- preserve fail-closed behavior if the repaired thesis still fails
- prevent unbounded rewrite loops

## Trigger

The repair loop runs only after the first verifier pass finishes and sets:

- `verification_status = fail`
- `repair_attempted = false`

If both conditions hold, the verifier performs one repair pass and then re-verifies the repaired
report.

If the verifier still fails after repair, the workflow stops.

## Repair Inputs

The repair node receives:

- ticker
- question
- plan
- contradiction summary
- the same section grounding packet used in Phase 2
- the prior report
- verifier summary
- unsupported finding labels when present

This keeps the repair pass grounded in the same evidence contract as the original thesis writer.

## Repair Prompt Contract

The repair prompt requires:

- preserve supported claims when possible
- remove unsupported or weakly grounded claims
- use only exact `[source:<id>]` citations
- use only evidence ids present in the grounding packet
- delete any claim that cannot be repaired with exact citations
- produce a complete rewritten thesis, not a diff

## State Additions

Phase 3 adds these workflow state fields:

- `initial_report`
- `repair_attempted`
- `repair_reason`
- `repair_summary`

Purpose:

- `initial_report`
  preserves the first thesis before repair overwrites `report`
- `repair_attempted`
  enforces bounded retry behavior
- `repair_reason`
  captures the verifier summary that triggered repair
- `repair_summary`
  records that a repair pass was applied

## Workflow Behavior

Current control flow:

1. `thesis`
2. `verifier` pass 1
3. if verifier passes:
   - end
4. if verifier fails and no repair has been attempted:
   - run one repair helper using verifier feedback
   - `verifier` pass 2 on the repaired report
5. after repaired verifier result:
   - end, whether pass or fail

This is intentionally one-pass only.

## Why One Pass

More than one rewrite pass creates two problems:

- it can hide upstream grounding failures behind repeated retries
- it increases cost and latency without a clear guarantee of convergence

One repair pass is a reasonable compromise:

- enough to fix citation omissions or section-drift errors
- not enough to let the system churn indefinitely

## Validation

The repaired thesis still goes through the same deterministic validator used in Phase 1.

That means the repair pass can still fail before verifier scoring if it contains:

- malformed citations
- uncited numeric claims
- prohibited placeholder text

Repair is not an escape hatch around grounding rules.

## Testing Strategy

Phase 3 should keep these behaviors under test:

- one failed thesis can be repaired into a passing result
- the initial report is preserved
- verifier runs a second time after repair
- repair runs at most once
- a failed post-repair verifier result ends the workflow instead of looping

Current implementation points:

- verifier-contained repair flow:
  [workflow.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/workflow.py)
- repair prompt:
  [prompts.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/prompts.py)
- workflow tests:
  [test_grounding_repair.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/tests/test_grounding_repair.py)

## Next Phase

Phase 4 should add a small canned end-to-end regression test suite that asserts final verifier
pass on stable fixtures.
