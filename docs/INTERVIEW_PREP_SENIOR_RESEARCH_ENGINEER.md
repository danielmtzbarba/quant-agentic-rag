# Interview Prep: Senior Research Engineer – Generative AI, Agentic Systems

## Why this project is relevant

This repository is a credible example of applied LLM systems engineering, not just prompt chaining.
It shows:

- a typed multi-stage LangGraph workflow
- specialist analysis roles over different evidence profiles
- hybrid retrieval over financial documents
- deterministic grounding checks and a repair loop
- telemetry, audit persistence, and offline release-gate evaluation

That maps well to the role description:

- "designing and building LLM based systems and agentic workflows"
- "developing multi step pipelines, including RAG and multi agent architectures"
- "turning prototypes into systems that are reliable, scalable, and measurable"
- "evaluating model performance using real metrics"

The strongest framing is:

> I built an applied financial research system that separates ingestion, hybrid retrieval, specialist analysis, synthesis, verification, and audit persistence. The main engineering focus was grounding, retrieval quality, and making the workflow measurable and repairable rather than relying on raw LLM intuition.

## Executive summary

This is best described as a production-oriented agentic RAG pipeline for equity research.

It is not a fully autonomous agent that freely decides tools and plans at runtime. It is closer to a controlled multi-stage agent system:

- one planner
- three specialist analyst nodes
- one synthesis node
- one verifier
- one single-pass repair loop

That is actually a strong interview position because it shows system design discipline. You can explain that fully open-ended agents were intentionally constrained because financial analysis is a high-hallucination domain and traceability matters.

## Architecture at a glance

### Main layers

1. Source ingestion
   - SEC filings
   - earnings transcripts
   - news
   - market/fundamentals snapshot from `yfinance`

2. Normalization and chunking
   - documents normalized into typed records
   - chunks preserve provenance and domain metadata
   - filings chunked by section
   - transcripts chunked by speaker turn
   - news chunked at compact article level

3. Indexing
   - chunk embeddings generated in a separate pipeline
   - embeddings persisted in Postgres / Supabase-backed tables
   - pgvector path available for native vector search

4. Retrieval
   - analyst-specific retrieval profiles
   - query rewriting and multi-query decomposition
   - lexical retrieval + semantic retrieval
   - reciprocal rank fusion
   - reranking
   - freshness scoring
   - diversity control
   - optional neighbor chunk expansion

5. Reasoning and synthesis
   - planner generates execution plan
   - fundamentals analyst
   - sentiment analyst
   - risk analyst
   - contradiction detection and review
   - thesis preparation layer maps findings into sections
   - thesis writer produces final report

6. Reliability and governance
   - deterministic citation and numeric-claim checks
   - structured grounding validation against evidence ids
   - verifier pass/fail
   - one repair pass if grounding fails
   - telemetry, audit persistence, and release-gate evaluation

### Runtime graph

```text
User question
  -> planner
  -> parallel retrieval
     - fundamentals snapshot
     - fundamentals corpus retrieval
     - sentiment corpus retrieval
     - risk corpus retrieval
  -> specialist analysts
     - fundamentals analyst
     - sentiment analyst
     - risk analyst
  -> contradiction check
  -> contradiction review
  -> thesis preparation
  -> thesis writer
  -> verifier
  -> optional thesis repair
  -> final report + metrics + audit artifact
```

## End-to-end information flow

### 1. Request entry

The system starts from a `ResearchRequest` with:

- `ticker`
- `question`

The request can come from:

- CLI
- FastAPI `POST /v1/research`

The service layer wraps workflow execution and adds:

- total latency
- token usage aggregation
- model metadata
- retrieval metrics
- audit persistence
- thesis artifact persistence

### 2. Planning

The planner node turns the user question into a short execution plan.
This plan is not the final answer. It is a control artifact passed to downstream analysts.

Design value:

- keeps analysts aligned on scope
- makes chain behavior more inspectable
- creates a record of intent for debugging

### 3. Parallel evidence acquisition

The workflow then splits into four retrieval branches.

#### A. Fundamentals snapshot retrieval

Gets quick structured market data from `yfinance`.
This is a prototype convenience layer, not a gold-standard financial data source.

#### B. Fundamentals corpus retrieval

Queries filing-heavy evidence for accounting and business fundamentals.

#### C. Sentiment corpus retrieval

Queries transcript plus fresh-news evidence for management tone, guidance, and market narrative.

#### D. Risk corpus retrieval

Queries filing, transcript, and news evidence for balance-sheet, regulation, legal, and execution risks.

This is a strong design choice because not all analytical tasks should retrieve from the same corpus slice.

### 4. Hybrid retrieval internals

Each retrieval branch applies a profile with different constraints.

Core retrieval flow:

1. Query planner rewrites the request for retrieval.
2. The query is decomposed into multiple finance-specific subqueries.
3. Metadata filters restrict the search space.
4. Lexical search runs over filtered chunks.
5. Semantic search runs over chunk embeddings.
6. Ranked lists are fused with reciprocal rank fusion.
7. Top fused candidates are reranked.
8. Diversity policies prevent one source type from dominating.
9. Neighbor chunks may be attached for local context.
10. Returned records retain provenance-rich metadata.

Metadata used in retrieval includes:

- ticker
- document type
- form type
- filing section
- published timestamp
- publisher
- speaker
- speaker role
- article relevance / sentiment metadata

This is much better than naive "embed everything and top-k cosine similarity".

### 5. Evidence aggregation

The system merges:

- fundamentals snapshot evidence
- fundamentals evidence
- sentiment evidence
- risk evidence

Deduplication is by `source_id`, keeping the highest-scoring copy.

This merged pool is used later for:

- thesis synthesis
- report-level grounding checks
- retrieval metrics

### 6. Specialist reasoning

There are three specialist analyst nodes:

- fundamentals analyst
- sentiment analyst
- risk analyst

Each returns structured `AnalystOutput`:

- `summary`
- `findings`
- `evidence_gaps`
- `overall_confidence`

Each finding contains:

- `finding`
- `evidence_ids`
- `confidence`
- `missing_data`
- optional `finding_type`

This is one of the strongest parts of the design.
It moves the system away from opaque prose and toward typed intermediate reasoning artifacts.

### 7. Contradiction handling

The workflow then checks for cross-analyst contradictions.

There are two stages:

1. heuristic contradiction shortlist
2. LLM-based contradiction review with fallback heuristic review

Contradictions are normalized into fields such as:

- topic
- contradiction kind
- severity
- resolution status
- evidence ids on both sides

This is a good interview talking point because it shows you understand that multi-agent systems create disagreement, and disagreement must be surfaced rather than averaged away.

### 8. Thesis preparation

Before generating the final report, the system deterministically maps findings into report sections:

- executive summary
- bull case
- bear case
- key risks
- evidence gaps

This section-preparation step is important.
It reduces synthesis chaos and gives the writer model a constrained evidence contract.

### 9. Thesis writing

The thesis node generates the final investment thesis.
The prompt enforces:

- exact inline citation syntax: `[source:<id>]`
- no unsupported evidence ids
- no uncited numeric claims
- no placeholder text like "evidence not provided"

This is another strong design choice: the writer is explicitly constrained by a grounding packet instead of seeing only raw retrieval blobs.

### 10. Verification and repair

Verification has deterministic and LLM-assisted components.

Deterministic checks include:

- malformed citations
- uncited numeric claims
- prohibited placeholder text
- unsupported findings
- partially grounded findings

If the report fails and repair has not yet been attempted:

1. the system runs one repair pass
2. regenerates the thesis under stricter repair instructions
3. reruns verification

This "fail closed + one repair loop" pattern is exactly the kind of systems thinking interviewers want to hear.

### 11. Persistence and observability

The service persists:

- research run audit records
- thesis artifacts
- node metrics
- retrieval metrics
- verification metrics
- token and cost estimates

This is what makes the project look like an applied AI system instead of a notebook demo.

## Data model and storage design

### Core entities

- `DocumentRecord`
  - normalized source document
- `EvidenceChunk`
  - retrieval unit with provenance
- `ChunkEmbeddingORM`
  - embedding index row
- `ResearchRunORM`
  - workflow audit record
- `ThesisArtifactORM`
  - persisted thesis output and metadata

### Storage split

The design separates:

- raw source payloads for reproducibility
- normalized documents for source-agnostic processing
- chunks for retrieval
- embeddings for semantic search
- run/audit tables for operations and evaluation

That separation is a good systems answer when asked how you move from prototype to production.

## Why this is "agentic" and where it is not

### What is agentic here

- multiple role-specialized reasoning nodes
- a planner that produces workflow guidance
- tool-mediated retrieval rather than single prompt stuffing
- intermediate state passed across nodes
- a verifier that can trigger a repair action
- explicit contradiction management across analysts

### What is not truly agentic

- the graph is fixed, not dynamically routed
- there is no open-ended tool selection at runtime
- there is no long-lived memory across runs
- there is no autonomous task decomposition beyond controlled query planning
- there is no adaptive budget manager or policy model deciding when to stop

The honest framing is:

> This is a controlled agentic workflow, not a free-form autonomous agent. That was deliberate because the domain rewards reliability, auditability, and evidence control over autonomy.

## Retrieval system design talking points

If they ask "Why is your retrieval design good?" answer with these points:

### Retrieval is profile-aware

Different analytical tasks need different evidence.
Using the same top-k for all agents would blur business fundamentals, management tone, and risk disclosures.

### Retrieval is hybrid

Lexical search catches exact terms and filings language.
Semantic search catches paraphrases and concept matches.
RRF gives robustness across ranking modes.

### Retrieval is metadata-aware

A strong retrieval system for finance is not just similarity search.
Metadata matters:

- recent news should decay quickly
- 10-K and 10-Q sections have different value
- speaker role matters in transcripts
- off-ticker news must be filtered aggressively

### Retrieval is diversity-aware

Without diversity control, one Reuters cluster or one transcript can dominate evidence selection.
This system tries to preserve multiple source types.

### Retrieval preserves provenance

Every returned record keeps source ids and metadata so later stages can ground and audit claims.

## Evaluation design talking points

This repo has a better evaluation story than many candidate projects.

### What is implemented

- deterministic regression tests for grounding behavior
- offline golden-set evaluation
- release gates with explicit thresholds
- retrieval precision@k and recall@k support
- contradiction surfacing metric
- off-ticker evidence rate
- unsupported numeric claim rate
- pass rate after repair

### Why that matters

This shows you are evaluating system behavior, not only model outputs in isolation.
It also shows you understand that retrieval quality, grounding quality, and repair effectiveness are separate dimensions.

### Good interview framing

> I treated evaluation as a system-level contract. The goal was not just "did the model sound smart?" but "did the pipeline retrieve the right evidence, cite it correctly, avoid unsupported numeric claims, surface contradictions, and recover from failure with a repair pass?"

## Strong points of the project

These are the strongest things to emphasize.

### 1. Good decomposition

The pipeline separates ingestion, indexing, retrieval, reasoning, synthesis, and verification.
That is how serious applied AI systems should be built.

### 2. Structured intermediate artifacts

Typed analyst findings with evidence ids are much better than free-text handoffs.

### 3. Retrieval quality is taken seriously

This is not "vector DB + prompt".
The retrieval stack includes:

- query planning
- hybrid search
- reranking
- freshness
- diversity
- neighbor context

### 4. Grounding discipline

The deterministic citation and numeric-claim checks are strong.
Many projects claim "grounded generation" but do not enforce it.

### 5. Measurability

Telemetry, cost estimation, retrieval metrics, and release gates show engineering maturity.

### 6. Domain-aware design

The chunking and retrieval strategy respects financial document structure.
That matters a lot.

## Honest weaknesses and risks

You should be ready to discuss these without sounding defensive.

### 1. It is still partly prototype-grade on data sources

`yfinance` and Alpha Vantage are convenient, but not institution-grade market data.
For a production financial system, you would want more reliable and contractually supported providers.

### 2. The graph is fixed

The workflow is well controlled, but not dynamically adaptive.
A stronger next step would be conditional routing based on retrieval confidence, missing data, or verifier failures.

### 3. Evaluation is good but not complete

Gaps still include:

- human preference evaluation on final thesis quality
- repeated-run consistency under stochastic variation
- latency/cost quality frontiers
- online evaluation tied to user outcomes
- calibrated confidence evaluation

### 4. No explicit caching or budget optimization layer

There is telemetry and cost estimation, but not yet a strong cost-control policy such as:

- retrieval cache
- embedding cache
- planner bypass
- model tiering by task difficulty
- dynamic top-k or rerank budget

### 5. Limited true production hardening

The repo has API, Docker, and persistence, but I would not oversell it as fully productionized without:

- job orchestration / queues
- stronger retries and idempotency controls
- backpressure handling
- access control and secrets hardening
- SLOs and alerting
- load testing

### 6. "Multi-agent" should be described carefully

It is valid to call this a multi-agent or multi-specialist architecture, but do not imply autonomous agents negotiating freely.
Interviewers will notice the difference.

## Best way to present your contribution

Say this clearly:

> The main contribution was not inventing a new model. It was engineering an applied LLM system with controlled specialist reasoning, hybrid financial retrieval, deterministic grounding checks, and measurable evaluation gates.

Then add:

> I optimized for reliability over agent freedom. In finance, a constrained but inspectable workflow is often better than a more autonomous system that is harder to audit.

## Likely interview questions and strong answer angles

### "Walk me through the architecture."

Answer structure:

1. ingestion and normalization
2. chunking plus embedding index
3. profile-aware hybrid retrieval
4. specialist analysis nodes
5. contradiction handling
6. thesis synthesis
7. verification and repair
8. audit persistence and evaluation

### "Why use multiple agents or roles?"

Answer:

Because the evidence requirements differ by task.
Fundamentals, sentiment, and risk should not all reason over the same undifferentiated context.
The specialist split improves retrieval precision, reduces context pollution, and creates interpretable intermediate outputs.

### "How do you evaluate quality?"

Answer:

At three levels:

1. retrieval quality
   - precision@k
   - recall@k
   - off-ticker leakage
2. generation grounding
   - citation compliance
   - unsupported numeric claims
   - structured finding support
3. system resilience
   - contradiction surfacing
   - repair success rate
   - latency/token/cost telemetry

### "What would you improve next?"

Good answers:

- dynamic routing based on evidence sufficiency
- stronger provider layer and ingestion reliability
- model tiering and caching for cost control
- richer human eval rubric for final thesis quality
- online feedback loop from users / analysts
- confidence calibration and abstention policy

### "What happened when the model hallucinated?"

Answer:

I did not rely only on prompt instructions.
I added deterministic checks for citation syntax, uncited numeric claims, and unsupported structured findings.
If the report failed, the system triggered one repair pass and then revalidated.

### "How do you keep coding agents useful without lowering quality?"

Answer:

Use them to accelerate implementation, test generation, and refactoring, but keep strong contracts:

- typed schemas
- deterministic tests
- reviewable graph logic
- release gates
- observability

That answer maps directly to the job description mentioning Cursor/Codex.

## Honest feedback on your CV versus this repo

### What is already strong

- You already claim RAG, multi-agent orchestration, LangGraph, evaluation, observability, FastAPI, Docker, Postgres, and CI/CD.
- The repo supports those claims better than most interview projects do.
- Your background as a PhD researcher helps for "research to systems" positioning.

### What you should tighten in how you speak

- Avoid vague phrases like "autonomous workflows" unless you immediately define the control boundary.
- Say "controlled agentic workflow" or "role-specialized graph-based workflow" more often.
- Be precise that your evaluation is mostly offline and regression-oriented.
- Be precise that this is production-oriented engineering, not production at massive scale.

### What may be challenged

- depth of real LLM evaluation methodology
- whether the system is truly end to end
- whether the financial analysis quality is good enough for real users
- whether the multi-agent design adds measurable value over a simpler pipeline

You should answer those with evidence from the repo:

- structured findings and evidence ids
- release gates
- contradiction review
- hybrid retrieval policies
- verifier + repair loop

## Suggested 2-minute interview pitch

I built a production-oriented agentic RAG system for equity research. The architecture separates source ingestion, normalization, chunking, embedding/indexing, profile-aware hybrid retrieval, specialist analyst nodes, synthesis, and verification. Instead of one generic prompt over retrieved chunks, I used three specialist roles for fundamentals, sentiment, and risk, each with different retrieval policies over filings, transcripts, and news.

The part I focused on most was reliability. Analyst outputs are structured and carry evidence ids. The final thesis is generated from a constrained grounding packet, then checked by a verifier with deterministic rules for citation syntax, unsupported findings, and uncited numeric claims. If the report fails, the system runs a repair pass and revalidates. On top of that I added offline evaluation with golden-set release gates, retrieval metrics like precision and recall, contradiction surfacing, and cost/latency telemetry. So the key contribution is not just using LLMs, but turning them into a measurable and auditable applied system.

## Final blunt assessment

This project is good enough to discuss seriously in a Senior Research Engineer interview.

What makes it credible:

- real decomposition
- retrieval depth
- typed state and intermediate artifacts
- explicit grounding discipline
- evaluation beyond intuition

What would weaken you if you oversell it:

- calling it fully production-ready
- calling it a highly autonomous multi-agent system
- implying institutional-grade financial data quality

If you present it accurately, it is a strong example of applied generative AI systems engineering.
