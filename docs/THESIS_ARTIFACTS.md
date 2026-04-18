# Thesis Artifacts

## Purpose

Thesis artifacts are the durable output layer for the research system.

They preserve the final thesis in a human-readable format and make it possible to:

- audit what was generated for a research run
- compare repeated runs over time
- review verifier outcomes and contradictions
- support evaluation and regression analysis

## High-Level Model

The system should persist thesis outputs in two layers:

- object storage for the full Markdown artifact
- Postgres / Supabase for structured metadata about the artifact

This keeps the full thesis readable and portable while keeping metadata queryable for analytics,
evaluation, and audit workflows.

## Artifact Contents

A thesis artifact should contain:

- ticker
- question
- plan
- final thesis
- verification summary
- contradiction summary
- compact retrieval / run metadata

Recommended format:

- Markdown for the full persisted artifact

## Artifact Metadata

Each thesis artifact should be linked to a research run and store metadata such as:

- thesis identifier
- research run identifier
- ticker
- question
- creation timestamp
- verification status
- model name
- embedding model
- source counts
- citation coverage
- contradiction count
- estimated cost
- latency
- checksums / content hash

## Storage Strategy

The recommended production pattern is:

- full thesis artifact in object storage
- thesis metadata in `rag.thesis_artifacts`

This allows:

- durable storage of Markdown artifacts
- efficient database queries over artifact metadata
- later support for review queues, evaluation jobs, and cache reuse logic

## Artifact Location

Artifacts should use a deterministic object key pattern such as:

```text
theses/{ticker}/{yyyy}/{mm}/{run_id}.md
```

This keeps storage organized and makes repeated-run inspection easier.

## Why A Separate Artifact Layer

`research_runs` should remain the workflow execution log.

`thesis_artifacts` should become the persisted output layer.

That separation makes it easier to:

- keep runtime telemetry separate from output artifacts
- support retries and repeated runs cleanly
- add future human-review or publish states

## Relationship To Evaluation

Persisted thesis artifacts are useful for:

- auditing generated investment theses
- sampling failed verifier runs
- comparing repeated outputs for the same ticker/question
- tracking contradiction behavior over time
- measuring quality regressions across model or retrieval changes

## Notes

This document is intentionally high-level and focuses on the role of thesis artifacts in the
system.

Implementation details, storage backend decisions, caching strategy, rollout phases, and developer
operating notes are tracked under `docs/internal/`.
