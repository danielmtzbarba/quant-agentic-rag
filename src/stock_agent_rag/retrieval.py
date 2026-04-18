from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Protocol

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy import Select, cast, func, literal, or_, select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .db import ChunkEmbeddingORM, ChunkORM, VectorType, get_session_factory
from .logging import get_logger
from .schemas import EvidenceRecord

logger = get_logger(__name__)

PROFILE_DOC_TYPES: dict[str, set[str]] = {
    "fundamentals": {"filing"},
    "sentiment": {"transcript", "news"},
    "risk": {"filing", "news", "transcript"},
}

PROFILE_FACETS: dict[str, tuple[str, ...]] = {
    "fundamentals": (
        "revenue growth",
        "margins and profitability",
        "cash flow and balance sheet",
    ),
    "sentiment": (
        "management guidance",
        "earnings call tone",
        "recent news sentiment",
    ),
    "risk": (
        "risk factors",
        "regulation and legal exposure",
        "execution and balance sheet risk",
    ),
}


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    document_types: set[str]
    form_types: set[str] | None = None
    section_tokens: tuple[str, ...] = ()
    news_max_age_days: int | None = None
    transcript_max_age_days: int | None = None
    filing_max_age_days: int | None = None
    publisher_tokens: tuple[str, ...] = ()
    speaker_role_tokens: tuple[str, ...] = ()
    preferred_doc_order: tuple[str, ...] = ()
    diversity_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class FreshnessContext:
    latest_news_at: datetime | None
    latest_transcript_at: datetime | None
    latest_filing_by_form_type: dict[str, datetime]


@dataclass
class RetrievalCandidate:
    record: EvidenceRecord
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    freshness_score: float = 0.0
    metadata_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    matched_queries: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PlannedQuery:
    label: str
    text: str


@dataclass(frozen=True)
class RetrievalPlan:
    rewritten_query: str
    subqueries: list[PlannedQuery]
    rationale: str


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._client = OpenAIEmbeddings(
            model=resolved.embedding_model_name,
            api_key=resolved.openai_api_key,
        )

    def embed_query(self, text: str) -> list[float]:

        return list(self._client.embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(item) for item in self._client.embed_documents(texts)]


class QueryPlanOutput(BaseModel):
    rewritten_query: str
    rationale: str
    subqueries: list[str] = Field(default_factory=list)


class QueryPlanner(Protocol):
    def plan(self, *, query: str, ticker: str, profile: RetrievalProfile) -> RetrievalPlan:
        ...


class HeuristicQueryPlanner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def plan(self, *, query: str, ticker: str, profile: RetrievalProfile) -> RetrievalPlan:
        normalized = " ".join(query.strip().split())
        subqueries = [PlannedQuery(label="primary", text=f"{ticker} {normalized}".strip())]
        facets = PROFILE_FACETS.get(profile.name, ())
        limit = max(int(self.settings.retrieval_query_plan_limit), 1)
        for idx, facet in enumerate(facets, start=1):
            expanded = f"{ticker} {normalized} {facet}".strip()
            subqueries.append(PlannedQuery(label=f"facet_{idx}", text=expanded))
            if len(subqueries) >= limit:
                break
        deduped = _dedupe_planned_queries(subqueries)
        return RetrievalPlan(
            rewritten_query=subqueries[0].text,
            subqueries=deduped,
            rationale="heuristic financial query decomposition",
        )


class OpenAIQueryPlanner:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self.settings = resolved
        self.model = ChatOpenAI(
            model=resolved.reranker_model_name,
            temperature=0,
            api_key=resolved.openai_api_key,
        ).with_structured_output(QueryPlanOutput)

    def plan(self, *, query: str, ticker: str, profile: RetrievalProfile) -> RetrievalPlan:
        response = self.model.invoke(
            [
                (
                    "system",
                    "You are a financial retrieval planner. Rewrite the user query for "
                    "retrieval, then produce concise subqueries for parallel search. "
                    "Keep filing sections, transcript turns, and news evidence in mind.",
                ),
                (
                    "human",
                    f"Ticker: {ticker}\n"
                    f"Profile: {profile.name}\n"
                    f"Allowed document types: {sorted(profile.document_types)}\n"
                    f"User query: {query}\n"
                    f"Return at most {self.settings.retrieval_query_plan_limit} subqueries.",
                ),
            ]
        )
        subqueries = [
            PlannedQuery(label=f"facet_{idx}", text=f"{ticker} {text}".strip())
            for idx, text in enumerate(response.subqueries, start=1)
            if text.strip()
        ]
        if not subqueries:
            subqueries = [PlannedQuery(label="primary", text=f"{ticker} {query}".strip())]
        deduped = _dedupe_planned_queries(subqueries)
        return RetrievalPlan(
            rewritten_query=response.rewritten_query.strip() or f"{ticker} {query}".strip(),
            subqueries=deduped[: self.settings.retrieval_query_plan_limit],
            rationale=response.rationale.strip() or "openai retrieval planning",
        )


class RerankResult(BaseModel):
    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class RerankResults(BaseModel):
    results: list[RerankResult] = Field(default_factory=list)


class CandidateReranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> list[RetrievalCandidate]:
        ...


class HeuristicReranker:
    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> list[RetrievalCandidate]:
        query_terms = _normalize_terms(query)
        for candidate in candidates:
            overlap = _term_overlap_score(query_terms, candidate.record)
            diversity_bonus = min(len(candidate.matched_queries), 3) * 0.05
            candidate.rerank_score = (
                0.35 * _normalized_score(candidate.fused_score)
                + 0.15 * _normalized_score(candidate.lexical_score)
                + 0.15 * _normalized_score(candidate.semantic_score)
                + 0.15 * candidate.freshness_score
                + 0.1 * candidate.metadata_score
                + 0.05 * overlap
                + diversity_bonus
                + 0.05 * _document_priority_score(candidate.record, profile)
            )
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)


class OpenAIReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self.model = ChatOpenAI(
            model=resolved.reranker_model_name,
            temperature=0,
            api_key=resolved.openai_api_key,
        ).with_structured_output(RerankResults)

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        payload = "\n\n".join(
            (
                f"chunk_id={candidate.record.source_id}\n"
                f"title={candidate.record.title}\n"
                f"document_type={candidate.record.document_type}\n"
                f"form_type={candidate.record.form_type}\n"
                f"section={candidate.record.section}\n"
                f"speaker_role={candidate.record.speaker_role}\n"
                f"publisher={candidate.record.publisher}\n"
                f"published_at={candidate.record.published_at}\n"
                f"content={candidate.record.content[:1200]}"
            )
            for candidate in candidates
        )
        response = self.model.invoke(
            [
                (
                    "system",
                    "You are a financial retrieval reranker. Prefer document-aware evidence, "
                    "fresh relevant sources, and diversified support across filings, "
                    "transcripts, and news when appropriate for the profile.",
                ),
                (
                    "human",
                    f"Query: {query}\n"
                    f"Profile: {profile.name}\n"
                    f"Return scores between 0 and 1.\n\nCandidates:\n{payload}",
                ),
            ]
        )
        score_by_id = {item.chunk_id: item.score for item in response.results}
        for candidate in candidates:
            candidate.rerank_score = float(score_by_id.get(candidate.record.source_id, 0.0))
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        session_factory: sessionmaker[Session] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: CandidateReranker | None = None,
        query_planner: QueryPlanner | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory or get_session_factory()
        self.embedding_provider = embedding_provider
        self.reranker = reranker or self._build_reranker()
        self.query_planner = query_planner or self._build_query_planner()

    def search(
        self,
        *,
        query: str,
        ticker: str,
        top_k: int | None = None,
        profile: str | None = None,
    ) -> list[EvidenceRecord]:
        top_k = top_k or self.settings.default_top_k
        retrieval_profile = _build_profile(profile, self.settings)
        plan = self.query_planner.plan(
            query=query,
            ticker=ticker.upper(),
            profile=retrieval_profile,
        )
        logger.info(
            "retrieval plan created",
            extra={
                "ticker": ticker.upper(),
                "profile": retrieval_profile.name,
                "rewritten_query": plan.rewritten_query,
                "subqueries": [item.text for item in plan.subqueries],
                "rationale": plan.rationale,
            },
        )

        with self.session_factory() as session:
            freshness_context = _build_freshness_context(
                session=session,
                ticker=ticker,
                profile=retrieval_profile,
            )
            logger.info(
                "retrieval freshness policy applied",
                extra={
                    "ticker": ticker.upper(),
                    "profile": retrieval_profile.name,
                    "news_max_age_days": retrieval_profile.news_max_age_days,
                    "transcript_max_age_days": retrieval_profile.transcript_max_age_days,
                    "filing_max_age_days": retrieval_profile.filing_max_age_days,
                    "latest_news_at": freshness_context.latest_news_at,
                    "latest_transcript_at": freshness_context.latest_transcript_at,
                    "latest_filing_by_form_type": freshness_context.latest_filing_by_form_type,
                },
            )
            base_stmt = _build_filtered_stmt(
                ticker=ticker,
                profile=retrieval_profile,
            )
            ranked_lists: list[list[RetrievalCandidate]] = []
            lexical_total = 0
            semantic_total = 0
            candidate_limit = max(self.settings.retrieval_candidate_pool, top_k * 8)
            for planned_query in plan.subqueries:
                lexical = self._lexical_search(
                    session=session,
                    query=planned_query.text,
                    stmt=base_stmt,
                    limit=candidate_limit,
                    profile=retrieval_profile,
                    freshness_context=freshness_context,
                )
                semantic = self._semantic_search(
                    session=session,
                    query=planned_query.text,
                    stmt=base_stmt,
                    limit=candidate_limit,
                    profile=retrieval_profile,
                    freshness_context=freshness_context,
                )
                lexical_total += len(lexical)
                semantic_total += len(semantic)
                if lexical:
                    ranked_lists.append(lexical)
                if semantic:
                    ranked_lists.append(semantic)
                logger.info(
                    "retrieval subquery completed",
                    extra={
                        "ticker": ticker.upper(),
                        "profile": retrieval_profile.name,
                        "subquery_label": planned_query.label,
                        "subquery": planned_query.text,
                        "lexical_candidates": len(lexical),
                        "semantic_candidates": len(semantic),
                    },
                )

            fused = _fuse_ranked_lists(ranked_lists, rrf_k=self.settings.retrieval_rrf_k)
            logger.info(
                "retrieval fusion completed",
                extra={
                    "ticker": ticker.upper(),
                    "profile": retrieval_profile.name,
                    "subquery_count": len(plan.subqueries),
                    "lexical_candidates_total": lexical_total,
                    "semantic_candidates_total": semantic_total,
                    "fused_candidates": len(fused),
                },
            )

            rerank_input = fused[: self.settings.retrieval_rerank_top_n]
            reranked = self.reranker.rerank(
                query=plan.rewritten_query,
                candidates=rerank_input,
                profile=retrieval_profile,
            )
            logger.info(
                "retrieval rerank completed",
                extra={
                    "ticker": ticker.upper(),
                    "profile": retrieval_profile.name,
                    "rerank_input_count": len(rerank_input),
                    "reranked_count": len(reranked),
                },
            )

            diversified = self._select_diverse_candidates(
                candidates=reranked,
                top_k=top_k,
                profile=retrieval_profile,
            )
            with_neighbors = self._attach_neighbors(
                session=session,
                selected=diversified,
                top_k=top_k,
                freshness_context=freshness_context,
            )
            selected = with_neighbors[: top_k + self.settings.retrieval_neighbor_limit]
            for candidate in selected:
                candidate.record.score = max(
                    candidate.rerank_score,
                    candidate.fused_score,
                    candidate.lexical_score,
                    candidate.semantic_score,
                    candidate.freshness_score,
                )
            logger.info(
                "retrieval selection completed",
                extra={
                    "ticker": ticker.upper(),
                    "profile": retrieval_profile.name,
                    "selected_sources": [item.record.source_id for item in selected],
                    "selected_document_types": [item.record.document_type for item in selected],
                    "selected_documents": len(
                        {item.record.document_id for item in selected if item.record.document_id}
                    ),
                },
            )
            return [candidate.record for candidate in selected]

    def _lexical_search(
        self,
        *,
        session: Session,
        query: str,
        stmt: Select[tuple[ChunkORM]],
        limit: int,
        profile: RetrievalProfile,
        freshness_context: FreshnessContext,
    ) -> list[RetrievalCandidate]:
        dialect = session.bind.dialect.name if session.bind is not None else "unknown"
        if dialect == "postgresql":
            ts_document = func.to_tsvector(
                "english",
                func.concat(ChunkORM.title, literal(" "), ChunkORM.content),
            )
            ts_query = func.plainto_tsquery("english", query)
            rows = session.execute(
                stmt.add_columns(func.ts_rank_cd(ts_document, ts_query).label("lexical_score"))
                .where(ts_document.op("@@")(ts_query))
                .order_by(func.ts_rank_cd(ts_document, ts_query).desc())
                .limit(limit)
            ).all()
            candidates: list[RetrievalCandidate] = []
            for row in rows:
                record = _row_to_record(row[0])
                candidates.append(
                    self._candidate_from_record(
                        record=record,
                        query=query,
                        profile=profile,
                        freshness_context=freshness_context,
                        lexical_score=float(row[1] or 0.0),
                        matched_query=query,
                    )
                )
            return candidates

        query_terms = _normalize_terms(query)
        rows = list(session.scalars(stmt.limit(limit * 6)))
        candidates: list[RetrievalCandidate] = []
        for row in rows:
            record = _row_to_record(row)
            overlap = _term_overlap_score(query_terms, record)
            if overlap <= 0:
                continue
            candidates.append(
                self._candidate_from_record(
                    record=record,
                    query=query,
                    profile=profile,
                    freshness_context=freshness_context,
                    lexical_score=overlap,
                    matched_query=query,
                )
            )
        candidates.sort(key=lambda item: item.lexical_score, reverse=True)
        return candidates[:limit]

    def _semantic_search(
        self,
        *,
        session: Session,
        query: str,
        stmt: Select[tuple[ChunkORM]],
        limit: int,
        profile: RetrievalProfile,
        freshness_context: FreshnessContext,
    ) -> list[RetrievalCandidate]:
        if self.embedding_provider is None:
            return []
        query_embedding = self.embedding_provider.embed_query(query)
        dialect = session.bind.dialect.name if session.bind is not None else "unknown"
        if dialect == "postgresql":
            vector_value = _vector_literal(query_embedding)
            distance_expr = ChunkEmbeddingORM.embedding_vector.op("<=>")(
                cast(literal(vector_value), VectorType(self.settings.embedding_dimensions))
            )
            embedding_stmt = (
                select(ChunkORM, ChunkEmbeddingORM, distance_expr.label("semantic_distance"))
                .join(ChunkEmbeddingORM, ChunkEmbeddingORM.chunk_id == ChunkORM.chunk_id)
                .where(ChunkEmbeddingORM.embedding_model == self.settings.embedding_model_name)
                .where(ChunkEmbeddingORM.embedding_vector.is_not(None))
            )
            if stmt._where_criteria:
                for criterion in stmt._where_criteria:
                    embedding_stmt = embedding_stmt.where(criterion)
            embedding_stmt = embedding_stmt.order_by(
                distance_expr.asc(),
                ChunkORM.published_at.desc().nullslast(),
            ).limit(limit * 8)
            rows = list(session.execute(embedding_stmt).all())
            candidates: list[RetrievalCandidate] = []
            for chunk_row, _embedding_row, semantic_distance in rows:
                similarity = max(1.0 - float(semantic_distance or 0.0), 0.0)
                record = _row_to_record(chunk_row)
                candidates.append(
                    self._candidate_from_record(
                        record=record,
                        query=query,
                        profile=profile,
                        freshness_context=freshness_context,
                        semantic_score=similarity,
                        matched_query=query,
                    )
                )
            logger.info(
                "native pgvector semantic search completed",
                extra={
                    "query": query,
                    "candidate_count": len(candidates),
                    "embedding_model": self.settings.embedding_model_name,
                },
            )
            return candidates[:limit]

        embedding_stmt = (
            select(ChunkORM, ChunkEmbeddingORM)
            .join(ChunkEmbeddingORM, ChunkEmbeddingORM.chunk_id == ChunkORM.chunk_id)
            .where(ChunkEmbeddingORM.embedding_model == self.settings.embedding_model_name)
        )
        if stmt._where_criteria:
            for criterion in stmt._where_criteria:
                embedding_stmt = embedding_stmt.where(criterion)
        embedding_stmt = embedding_stmt.order_by(
            ChunkORM.published_at.desc().nullslast(),
            ChunkORM.chunk_id.asc(),
        ).limit(limit * 8)
        rows = list(session.execute(embedding_stmt).all())
        candidates: list[RetrievalCandidate] = []
        for chunk_row, embedding_row in rows:
            embedding = embedding_row.embedding_json or []
            if not embedding:
                continue
            similarity = _cosine_similarity(query_embedding, embedding)
            if similarity <= 0:
                continue
            record = _row_to_record(chunk_row)
            candidates.append(
                self._candidate_from_record(
                    record=record,
                    query=query,
                    profile=profile,
                    freshness_context=freshness_context,
                    semantic_score=similarity,
                    matched_query=query,
                )
            )
        candidates.sort(key=lambda item: item.semantic_score, reverse=True)
        logger.info(
            "python semantic fallback completed",
            extra={
                "query": query,
                "candidate_count": len(candidates),
                "embedding_model": self.settings.embedding_model_name,
            },
        )
        return candidates[:limit]

    def _candidate_from_record(
        self,
        *,
        record: EvidenceRecord,
        query: str,
        profile: RetrievalProfile,
        freshness_context: FreshnessContext,
        lexical_score: float = 0.0,
        semantic_score: float = 0.0,
        matched_query: str,
    ) -> RetrievalCandidate:
        metadata_score = _metadata_match_score(record, profile)
        freshness_score = _freshness_score(record, profile, freshness_context)
        candidate = RetrievalCandidate(
            record=record,
            lexical_score=lexical_score + metadata_score + 0.25 * freshness_score,
            semantic_score=semantic_score + 0.1 * metadata_score + 0.15 * freshness_score,
            freshness_score=freshness_score,
            metadata_score=metadata_score,
        )
        candidate.matched_queries.add(matched_query)
        return candidate

    def _select_diverse_candidates(
        self,
        *,
        candidates: list[RetrievalCandidate],
        top_k: int,
        profile: RetrievalProfile,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        doc_limits = {
            "news": self.settings.retrieval_max_news_chunks,
            "transcript": self.settings.retrieval_max_transcript_chunks,
            "filing": self.settings.retrieval_max_filing_chunks,
        }
        selected: list[RetrievalCandidate] = []
        seen_source_ids: set[str] = set()
        doc_type_counts: dict[str, int] = {}
        document_counts: dict[str, int] = {}

        for target_doc_type in profile.diversity_targets:
            for candidate in candidates:
                if len(selected) >= top_k:
                    break
                if candidate.record.document_type != target_doc_type:
                    continue
                if not self._can_select_candidate(
                    candidate=candidate,
                    seen_source_ids=seen_source_ids,
                    doc_type_counts=doc_type_counts,
                    document_counts=document_counts,
                    doc_limits=doc_limits,
                ):
                    continue
                _commit_selection(
                    candidate=candidate,
                    selected=selected,
                    seen_source_ids=seen_source_ids,
                    doc_type_counts=doc_type_counts,
                    document_counts=document_counts,
                )
                break

        for candidate in candidates:
            if len(selected) >= top_k:
                break
            if not self._can_select_candidate(
                candidate=candidate,
                seen_source_ids=seen_source_ids,
                doc_type_counts=doc_type_counts,
                document_counts=document_counts,
                doc_limits=doc_limits,
            ):
                continue
            _commit_selection(
                candidate=candidate,
                selected=selected,
                seen_source_ids=seen_source_ids,
                doc_type_counts=doc_type_counts,
                document_counts=document_counts,
            )

        logger.info(
            "retrieval diversity applied",
            extra={
                "profile": profile.name,
                "selected_count": len(selected),
                "doc_type_counts": doc_type_counts,
                "document_counts": document_counts,
            },
        )
        return selected

    def _can_select_candidate(
        self,
        *,
        candidate: RetrievalCandidate,
        seen_source_ids: set[str],
        doc_type_counts: dict[str, int],
        document_counts: dict[str, int],
        doc_limits: dict[str, int],
    ) -> bool:
        record = candidate.record
        if record.source_id in seen_source_ids:
            return False
        limit = doc_limits.get(record.document_type)
        if limit is not None and doc_type_counts.get(record.document_type, 0) >= limit:
            return False
        document_id = record.document_id or record.source_id
        if document_counts.get(document_id, 0) >= self.settings.retrieval_max_per_document:
            return False
        return True

    def _attach_neighbors(
        self,
        *,
        session: Session,
        selected: list[RetrievalCandidate],
        top_k: int,
        freshness_context: FreshnessContext,
    ) -> list[RetrievalCandidate]:
        if not selected:
            return []
        ordered: list[RetrievalCandidate] = []
        seen_source_ids: set[str] = set()
        neighbor_budget = self.settings.retrieval_neighbor_limit

        for candidate in selected:
            if candidate.record.source_id not in seen_source_ids:
                ordered.append(candidate)
                seen_source_ids.add(candidate.record.source_id)

            if neighbor_budget <= 0:
                continue
            base_index = candidate.record.chunk_index or 0
            lower_bound = max(base_index - self.settings.retrieval_neighbor_window, 0)
            upper_bound = base_index + self.settings.retrieval_neighbor_window
            stmt = (
                select(ChunkORM)
                .where(ChunkORM.document_id == candidate.record.document_id)
                .where(ChunkORM.chunk_index >= lower_bound)
                .where(ChunkORM.chunk_index <= upper_bound)
                .order_by(ChunkORM.chunk_index.asc())
            )
            for row in session.scalars(stmt):
                neighbor = _row_to_record(row)
                if (
                    neighbor.source_id in seen_source_ids
                    or neighbor.chunk_index == candidate.record.chunk_index
                ):
                    continue
                ordered.append(
                    RetrievalCandidate(
                        record=neighbor,
                        freshness_score=_freshness_score(
                            neighbor,
                            _build_profile(None, self.settings),
                            freshness_context,
                        ),
                        fused_score=max(candidate.fused_score - 0.01, 0.0),
                        rerank_score=max(candidate.rerank_score - 0.01, 0.0),
                    )
                )
                seen_source_ids.add(neighbor.source_id)
                neighbor_budget -= 1
                if (
                    neighbor_budget <= 0
                    or len(ordered) >= top_k + self.settings.retrieval_neighbor_limit
                ):
                    break
        return ordered

    def _build_reranker(self) -> CandidateReranker:
        if self._can_call_openai():
            try:
                return OpenAIReranker(self.settings)
            except Exception:
                logger.warning("falling back to heuristic reranker", exc_info=True)
        return HeuristicReranker()

    def _build_query_planner(self) -> QueryPlanner:
        if self._can_call_openai():
            try:
                return OpenAIQueryPlanner(self.settings)
            except Exception:
                logger.warning("falling back to heuristic query planner", exc_info=True)
        return HeuristicQueryPlanner(self.settings)

    def _can_call_openai(self) -> bool:
        key = (self.settings.openai_api_key or "").strip()
        return bool(key and not key.startswith("sk-test"))


def _build_profile(profile: str | None, settings: Settings) -> RetrievalProfile:
    if profile == "fundamentals":
        return RetrievalProfile(
            name="fundamentals",
            document_types={"filing"},
            form_types={"10-K", "10-Q"},
            section_tokens=(
                "item_1_business",
                "item_7_mda",
                "item_8_financial_statements",
                "part_i_item_2_mda",
            ),
            filing_max_age_days=550,
            preferred_doc_order=("filing",),
            diversity_targets=("filing",),
        )
    if profile == "sentiment":
        return RetrievalProfile(
            name="sentiment",
            document_types={"transcript", "news"},
            news_max_age_days=14,
            transcript_max_age_days=200,
            publisher_tokens=("reuters", "bloomberg", "wsj", "financial times", "cnbc"),
            speaker_role_tokens=("chief", "ceo", "cfo", "president", "investor relations"),
            preferred_doc_order=("transcript", "news"),
            diversity_targets=("transcript", "news"),
        )
    if profile == "risk":
        return RetrievalProfile(
            name="risk",
            document_types={"filing", "news", "transcript"},
            form_types={"10-K", "10-Q"},
            section_tokens=(
                "item_1a_risk_factors",
                "item_7a_market_risk",
                "part_ii_item_1a_risk_factors",
            ),
            news_max_age_days=30,
            transcript_max_age_days=200,
            filing_max_age_days=550,
            publisher_tokens=("reuters", "bloomberg", "wsj", "financial times", "cnbc"),
            speaker_role_tokens=("chief", "ceo", "cfo", "president", "investor relations"),
            preferred_doc_order=("filing", "news", "transcript"),
            diversity_targets=("filing", "news", "transcript"),
        )
    return RetrievalProfile(
        name=profile or "default",
        document_types=PROFILE_DOC_TYPES.get(
            profile or "",
            {"filing", "transcript", "news", "note"},
        ),
        preferred_doc_order=("filing", "transcript", "news", "note"),
        diversity_targets=("filing", "transcript", "news"),
    )


def _build_filtered_stmt(*, ticker: str, profile: RetrievalProfile) -> Select[tuple[ChunkORM]]:
    stmt = select(ChunkORM).where(ChunkORM.ticker == ticker.upper())
    if profile.document_types:
        stmt = stmt.where(ChunkORM.document_type.in_(sorted(profile.document_types)))
    if profile.form_types:
        stmt = stmt.where(
            or_(
                ChunkORM.form_type.is_(None),
                ChunkORM.form_type.in_(sorted(profile.form_types)),
            )
        )
    if profile.section_tokens:
        stmt = stmt.where(
            or_(
                ChunkORM.section.is_(None),
                ChunkORM.section.in_(profile.section_tokens),
            )
        )
    time_filters = []
    if profile.news_max_age_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=profile.news_max_age_days)
        time_filters.append(
            (ChunkORM.document_type != "news")
            | (ChunkORM.published_at.is_(None))
            | (ChunkORM.published_at >= cutoff)
        )
    if profile.transcript_max_age_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=profile.transcript_max_age_days)
        time_filters.append(
            (ChunkORM.document_type != "transcript")
            | (ChunkORM.published_at.is_(None))
            | (ChunkORM.published_at >= cutoff)
        )
    if profile.filing_max_age_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=profile.filing_max_age_days)
        time_filters.append(
            (ChunkORM.document_type != "filing")
            | (ChunkORM.published_at.is_(None))
            | (ChunkORM.published_at >= cutoff)
        )
    for condition in time_filters:
        stmt = stmt.where(condition)
    if profile.publisher_tokens:
        stmt = stmt.where(
            or_(
                ChunkORM.publisher.is_(None),
                *[
                    func.lower(ChunkORM.publisher).contains(token.lower())
                    for token in profile.publisher_tokens
                ],
            )
        )
    if profile.speaker_role_tokens:
        stmt = stmt.where(
            or_(
                ChunkORM.speaker_role.is_(None),
                *[
                    func.lower(ChunkORM.speaker_role).contains(token.lower())
                    for token in profile.speaker_role_tokens
                ],
            )
        )
    return stmt.order_by(ChunkORM.published_at.desc().nullslast(), ChunkORM.chunk_id.asc())


def _row_to_record(row: ChunkORM) -> EvidenceRecord:
    return EvidenceRecord(
        source_id=row.source_id,
        ticker=row.ticker,
        title=row.title,
        content=row.content,
        document_type=row.document_type,
        source_url=row.source_url,
        published_at=row.published_at,
        score=float(row.score or 0.0),
        provider=row.provider,
        section=row.section,
        form_type=row.form_type,
        document_id=row.document_id,
        accession_number=row.accession_number,
        chunk_index=row.chunk_index,
        metadata_version=row.metadata_version,
        speaker=row.speaker,
        speaker_role=row.speaker_role,
        publisher=row.publisher,
        sentiment_label=row.sentiment_label,
        sentiment_score=row.sentiment_score,
    )


def _normalize_terms(text: str) -> set[str]:
    return {token.strip(".,:;!?()[]{}").lower() for token in text.split() if token.strip()}


def _term_overlap_score(query_terms: set[str], record: EvidenceRecord) -> float:
    haystack = (
        f"{record.title}\n"
        f"{record.content}\n"
        f"{record.section}\n"
        f"{record.speaker_role}\n"
        f"{record.publisher}"
    ).lower()
    return float(sum(term in haystack for term in query_terms))


def _metadata_match_score(record: EvidenceRecord, profile: RetrievalProfile) -> float:
    score = 0.0
    if record.document_type in profile.document_types:
        score += 1.0
    if profile.form_types and record.form_type in profile.form_types:
        score += 0.6
    if profile.section_tokens and record.section in profile.section_tokens:
        score += 1.0
    if record.publisher and any(
        token in record.publisher.lower() for token in profile.publisher_tokens
    ):
        score += 0.25
    if record.speaker_role and any(
        token in record.speaker_role.lower() for token in profile.speaker_role_tokens
    ):
        score += 0.25
    return score


def _freshness_score(
    record: EvidenceRecord,
    profile: RetrievalProfile,
    context: FreshnessContext,
) -> float:
    if record.published_at is None:
        return 0.0
    published_at = record.published_at
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    age_days = max((datetime.now(UTC) - published_at).total_seconds() / 86400, 0.0)

    if record.document_type == "news":
        max_age = float(profile.news_max_age_days or 14)
        latest_news_at = context.latest_news_at
        recency_from_latest = 0.0
        if latest_news_at is not None:
            latest_news_at = _ensure_utc(latest_news_at)
            days_from_latest = max((latest_news_at - published_at).total_seconds() / 86400, 0.0)
            recency_from_latest = max(0.0, 1.0 - min(days_from_latest / 7.0, 1.0))
        age_component = max(0.0, 1.0 - min(age_days / max_age, 1.0))
        return min(1.0, 0.7 * age_component + 0.3 * recency_from_latest)
    if record.document_type == "transcript":
        latest_transcript_at = context.latest_transcript_at
        if latest_transcript_at is None:
            return max(
                0.0,
                1.0 - min(age_days / float(profile.transcript_max_age_days or 180), 1.0),
            )
        latest_transcript_at = _ensure_utc(latest_transcript_at)
        quarter_gap_days = max((latest_transcript_at - published_at).total_seconds() / 86400, 0.0)
        latest_quarter_score = max(0.0, 1.0 - min(quarter_gap_days / 120.0, 1.0))
        age_component = max(
            0.0,
            1.0 - min(age_days / float(profile.transcript_max_age_days or 200), 1.0),
        )
        return min(1.0, 0.75 * latest_quarter_score + 0.25 * age_component)
    if record.document_type == "filing":
        latest_for_form = context.latest_filing_by_form_type.get((record.form_type or "").upper())
        latest_form_score = 0.0
        if latest_for_form is not None:
            latest_for_form = _ensure_utc(latest_for_form)
            lag_days = max((latest_for_form - published_at).total_seconds() / 86400, 0.0)
            horizon = 200.0 if (record.form_type or "").upper() == "10-Q" else 450.0
            latest_form_score = max(0.0, 1.0 - min(lag_days / horizon, 1.0))
        age_horizon = 220.0 if (record.form_type or "").upper() == "10-Q" else 550.0
        age_component = max(0.0, 1.0 - min(age_days / age_horizon, 1.0))
        form_bonus = 0.15 if record.form_type in {"10-K", "10-Q"} else 0.0
        return min(1.0, 0.55 * latest_form_score + 0.3 * age_component + form_bonus)
    return 0.0


def _build_freshness_context(
    *,
    session: Session,
    ticker: str,
    profile: RetrievalProfile,
) -> FreshnessContext:
    latest_news_at = None
    latest_transcript_at = None
    latest_filing_by_form_type: dict[str, datetime] = {}

    if "news" in profile.document_types:
        latest_news_at = session.scalar(
            select(func.max(ChunkORM.published_at))
            .where(ChunkORM.ticker == ticker.upper())
            .where(ChunkORM.document_type == "news")
        )
    if "transcript" in profile.document_types:
        latest_transcript_at = session.scalar(
            select(func.max(ChunkORM.published_at))
            .where(ChunkORM.ticker == ticker.upper())
            .where(ChunkORM.document_type == "transcript")
        )
    if "filing" in profile.document_types:
        rows = session.execute(
            select(ChunkORM.form_type, func.max(ChunkORM.published_at))
            .where(ChunkORM.ticker == ticker.upper())
            .where(ChunkORM.document_type == "filing")
            .where(ChunkORM.form_type.in_(("10-K", "10-Q")))
            .group_by(ChunkORM.form_type)
        ).all()
        latest_filing_by_form_type = {
            str(form_type).upper(): published_at
            for form_type, published_at in rows
            if form_type and published_at is not None
        }

    return FreshnessContext(
        latest_news_at=latest_news_at,
        latest_transcript_at=latest_transcript_at,
        latest_filing_by_form_type=latest_filing_by_form_type,
    )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _document_priority_score(record: EvidenceRecord, profile: RetrievalProfile) -> float:
    for idx, doc_type in enumerate(profile.preferred_doc_order):
        if record.document_type == doc_type:
            return max(0.0, 1.0 - idx * 0.2)
    return 0.0


def _fuse_ranked_lists(
    ranked_lists: list[list[RetrievalCandidate]],
    *,
    rrf_k: int,
) -> list[RetrievalCandidate]:
    fused: dict[str, RetrievalCandidate] = {}
    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            source_id = candidate.record.source_id
            entry = fused.get(source_id)
            if entry is None:
                entry = RetrievalCandidate(
                    record=candidate.record,
                    lexical_score=candidate.lexical_score,
                    semantic_score=candidate.semantic_score,
                    freshness_score=candidate.freshness_score,
                    metadata_score=candidate.metadata_score,
                    fused_score=0.0,
                    rerank_score=candidate.rerank_score,
                    matched_queries=set(candidate.matched_queries),
                )
                fused[source_id] = entry
            else:
                entry.lexical_score = max(entry.lexical_score, candidate.lexical_score)
                entry.semantic_score = max(entry.semantic_score, candidate.semantic_score)
                entry.freshness_score = max(entry.freshness_score, candidate.freshness_score)
                entry.metadata_score = max(entry.metadata_score, candidate.metadata_score)
                entry.matched_queries.update(candidate.matched_queries)
            entry.fused_score += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)


def _commit_selection(
    *,
    candidate: RetrievalCandidate,
    selected: list[RetrievalCandidate],
    seen_source_ids: set[str],
    doc_type_counts: dict[str, int],
    document_counts: dict[str, int],
) -> None:
    selected.append(candidate)
    seen_source_ids.add(candidate.record.source_id)
    doc_type = candidate.record.document_type
    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    document_id = candidate.record.document_id or candidate.record.source_id
    document_counts[document_id] = document_counts.get(document_id, 0) + 1


def _dedupe_planned_queries(subqueries: list[PlannedQuery]) -> list[PlannedQuery]:
    seen: set[str] = set()
    deduped: list[PlannedQuery] = []
    for item in subqueries:
        normalized = " ".join(item.text.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalized_score(value: float) -> float:
    if value <= 0:
        return 0.0
    return value / (1.0 + value)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    embedding_provider: EmbeddingProvider | None = None
    key = (settings.openai_api_key or "").strip()
    if key and not key.startswith("sk-test"):
        embedding_provider = OpenAIEmbeddingProvider(settings)
    return HybridRetriever(
        settings=settings,
        embedding_provider=embedding_provider,
    )
