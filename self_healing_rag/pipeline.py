"""Self-healing RAG orchestrator and stage classes.

Control flow
------------
plan -> retrieve -> rerank -> CRAG -> (fallback?) -> generate -> ground

On rejection, the failure mode picks the next move: rewrite the query, widen
retrieval, escalate to web search, or bail out with a "cannot answer" response
after `max_attempts`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import RAGConfig
from .prompts import (
    ANSWER_PROMPT,
    DECOMPOSE_PROMPT,
    DOC_GRADE_PROMPT,
    GROUNDEDNESS_PROMPT,
    HYDE_PROMPT,
    REWRITE_PROMPT,
)
from .types import (
    DocGrade,
    Document,
    Embedder,
    FailureMode,
    LLM,
    PipelineResult,
    Reranker,
    StageTrace,
    VectorStore,
    WebSearch,
)
from .utils import cosine, dedupe_by_id, extract_json

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Query analysis and HyDE
# ----------------------------------------------------------------------------


class QueryPlanner:
    def __init__(self, llm: LLM, config: RAGConfig):
        self.llm = llm
        self.config = config

    def decompose(self, question: str) -> list[str]:
        """Return atomic sub-questions or [question] if it is already atomic."""
        tokens = question.split()
        looks_compound = (
            len(tokens) >= self.config.decompose_min_tokens
            or any(w in question.lower() for w in self.config.decompose_signal_words)
        )
        if not looks_compound:
            return [question]

        raw = self.llm.complete(DECOMPOSE_PROMPT.format(question=question))
        try:
            data = json.loads(extract_json(raw))
            subs = [s.strip() for s in data.get("subquestions", []) if s.strip()]
            return subs or [question]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("decompose parse failed, falling back to original question")
            return [question]

    def hyde(self, question: str) -> str:
        return self.llm.complete(HYDE_PROMPT.format(question=question))


# ----------------------------------------------------------------------------
# Retrieval and reranking
# ----------------------------------------------------------------------------


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        config: RAGConfig,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.config = config

    def retrieve(self, question: str, hyde_passage: str) -> list[Document]:
        vectors = self.embedder.embed([question, hyde_passage])
        query_hits = self.store.search(vectors[0], k=self.config.top_k)
        hyde_hits = self.store.search(vectors[1], k=self.config.top_k)
        merged = dedupe_by_id(query_hits + hyde_hits)
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[: self.config.top_k]

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        scores = self.reranker.score(question, docs)
        for doc, s in zip(docs, scores):
            doc.rerank_score = float(s)
        docs.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
        return docs[: self.config.rerank_k]


# ----------------------------------------------------------------------------
# CRAG evaluator
# ----------------------------------------------------------------------------


@dataclass
class CRAGDecision:
    action: str  # "use", "augment", "refuse"
    kept: list[Document]
    reason: str


class CRAGEvaluator:
    def __init__(self, llm: LLM, config: RAGConfig):
        self.llm = llm
        self.config = config

    def evaluate(self, question: str, docs: list[Document]) -> CRAGDecision:
        if not docs:
            return CRAGDecision("refuse", [], "no documents retrieved")

        for doc in docs:
            doc.grade = self._grade(question, doc)

        correct = [d for d in docs if d.grade == DocGrade.CORRECT]
        ambiguous = [d for d in docs if d.grade == DocGrade.AMBIGUOUS]

        if len(correct) >= self.config.min_relevant_docs:
            return CRAGDecision("use", correct, "sufficient correct docs")
        if correct or ambiguous:
            return CRAGDecision(
                "augment",
                correct + ambiguous,
                "partial coverage, augment with fallback",
            )
        return CRAGDecision("refuse", [], "no relevant docs, rewrite query")

    def _grade(self, question: str, doc: Document) -> DocGrade:
        raw = self.llm.complete(
            DOC_GRADE_PROMPT.format(question=question, passage=doc.text[:1500])
        )
        try:
            data = json.loads(extract_json(raw))
            return DocGrade(data.get("grade", "ambiguous"))
        except (json.JSONDecodeError, KeyError, ValueError):
            return DocGrade.AMBIGUOUS


# ----------------------------------------------------------------------------
# Answer generation and groundedness check
# ----------------------------------------------------------------------------


class AnswerGenerator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, question: str, docs: list[Document]) -> str:
        context = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))
        return self.llm.complete(
            ANSWER_PROMPT.format(question=question, context=context),
            temperature=0.2,
        )


@dataclass
class GroundingReport:
    grounded_fraction: float
    unsupported: list[str]
    confidence: float


class Grounder:
    def __init__(self, llm: LLM):
        self.llm = llm

    def check(
        self, question: str, answer: str, docs: list[Document]
    ) -> GroundingReport:
        context = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))
        raw = self.llm.complete(
            GROUNDEDNESS_PROMPT.format(
                question=question, context=context, answer=answer
            )
        )
        try:
            data = json.loads(extract_json(raw))
            return GroundingReport(
                grounded_fraction=float(data.get("grounded", 0.0)),
                unsupported=list(data.get("unsupported", [])),
                confidence=float(data.get("confidence", 0.0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return GroundingReport(0.0, ["parse failure"], 0.0)


# ----------------------------------------------------------------------------
# Dynamic learning cache
# ----------------------------------------------------------------------------


@dataclass
class CachedAnswer:
    query: str
    answer: str
    docs: list[Document]
    confidence: float
    vector: np.ndarray
    created_at: float


class LearningCache:
    """Stores past groundings and short-circuits repeat queries.

    Cosine-similar past queries whose confidence exceeded the current threshold
    are returned as a fast path. Capped by `config.cache_max_entries` with FIFO
    eviction.
    """

    def __init__(self, embedder: Embedder, config: RAGConfig):
        self.embedder = embedder
        self.config = config
        self._entries: list[CachedAnswer] = []

    def lookup(self, query: str) -> CachedAnswer | None:
        if not self._entries:
            return None
        qv = self.embedder.embed([query])[0]
        matrix = np.stack([e.vector for e in self._entries])
        sims = cosine(qv, matrix)
        idx = int(np.argmax(sims))
        if sims[idx] >= self.config.cache_similarity_threshold:
            return self._entries[idx]
        return None

    def store(
        self,
        query: str,
        answer: str,
        docs: list[Document],
        confidence: float,
    ) -> None:
        vector = self.embedder.embed([query])[0]
        entry = CachedAnswer(
            query=query,
            answer=answer,
            docs=docs,
            confidence=confidence,
            vector=vector,
            created_at=time.time(),
        )
        self._entries.append(entry)
        if len(self._entries) > self.config.cache_max_entries:
            self._entries = self._entries[-self.config.cache_max_entries:]


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


class SelfHealingRAG:
    def __init__(
        self,
        *,
        llm: LLM,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker,
        web_search: WebSearch | None = None,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.llm = llm
        self.planner = QueryPlanner(llm, self.config)
        self.retriever = Retriever(embedder, vector_store, reranker, self.config)
        self.crag = CRAGEvaluator(llm, self.config)
        self.generator = AnswerGenerator(llm)
        self.grounder = Grounder(llm)
        self.web_search = web_search
        self.cache = LearningCache(embedder, self.config)

    # -- public entry point --------------------------------------------------

    def answer(self, question: str) -> PipelineResult:
        trace: list[StageTrace] = []

        cached = self.cache.lookup(question)
        if cached and cached.confidence >= self.config.answer_confidence_threshold:
            trace.append(
                StageTrace(
                    "cache_hit",
                    0.0,
                    {"query": cached.query, "confidence": cached.confidence},
                )
            )
            return PipelineResult(
                query=question,
                answer=cached.answer,
                documents=cached.docs,
                confidence=cached.confidence,
                grounded=True,
                failure_mode=FailureMode.NONE,
                attempts=0,
                trace=trace,
                cached=True,
            )

        subs = self._time("decompose", trace, lambda: self.planner.decompose(question))
        if len(subs) > 1:
            return self._answer_compound(question, subs, trace)
        return self._answer_atomic(question, trace)

    # -- atomic loop ---------------------------------------------------------

    def _answer_atomic(
        self, question: str, trace: list[StageTrace]
    ) -> PipelineResult:
        current_query = question
        last_failure = FailureMode.NONE
        docs: list[Document] = []
        answer = ""
        report = GroundingReport(0.0, [], 0.0)

        for attempt in range(1, self.config.max_attempts + 1):
            hyde = self._time(
                f"hyde/{attempt}", trace, lambda: self.planner.hyde(current_query)
            )
            candidates = self._time(
                f"retrieve/{attempt}",
                trace,
                lambda: self.retriever.retrieve(current_query, hyde),
            )
            if not candidates:
                last_failure = FailureMode.RETRIEVAL_EMPTY
                current_query = self._rewrite(current_query, "no documents found")
                continue

            reranked = self._time(
                f"rerank/{attempt}",
                trace,
                lambda: self.retriever.rerank(current_query, candidates),
            )
            top_score = reranked[0].rerank_score or 0.0
            trace[-1].detail["top_rerank_score"] = top_score
            if top_score < self.config.retrieval_relevance_threshold:
                last_failure = FailureMode.RETRIEVAL_IRRELEVANT
                current_query = self._rewrite(current_query, "top result scored low")
                continue

            decision = self._time(
                f"crag/{attempt}",
                trace,
                lambda: self.crag.evaluate(current_query, reranked),
            )
            trace[-1].detail["action"] = decision.action

            if decision.action == "refuse":
                last_failure = FailureMode.CONTEXT_INSUFFICIENT
                current_query = self._rewrite(current_query, decision.reason)
                continue

            docs = decision.kept
            if decision.action == "augment" and self.web_search is not None:
                web_docs = self._time(
                    f"web/{attempt}",
                    trace,
                    lambda: self.web_search.search(current_query, k=4),
                )
                docs = dedupe_by_id(docs + web_docs)[: self.config.rerank_k]

            answer = self._time(
                f"generate/{attempt}",
                trace,
                lambda: self.generator.generate(current_query, docs),
            )
            report = self._time(
                f"ground/{attempt}",
                trace,
                lambda: self.grounder.check(current_query, answer, docs),
            )
            trace[-1].detail.update(
                {"grounded": report.grounded_fraction, "confidence": report.confidence}
            )

            if (
                report.grounded_fraction >= self.config.grounding_threshold
                and report.confidence >= self.config.answer_confidence_threshold
            ):
                self.cache.store(question, answer, docs, report.confidence)
                return PipelineResult(
                    query=question,
                    answer=answer,
                    documents=docs,
                    confidence=report.confidence,
                    grounded=True,
                    failure_mode=FailureMode.NONE,
                    attempts=attempt,
                    trace=trace,
                )

            last_failure = (
                FailureMode.GENERATION_UNGROUNDED
                if report.grounded_fraction < self.config.grounding_threshold
                else FailureMode.GENERATION_LOW_CONFIDENCE
            )
            current_query = self._rewrite(
                current_query,
                f"ungrounded claims: {', '.join(report.unsupported[:2])}",
            )

        return PipelineResult(
            query=question,
            answer=self._refusal(question, last_failure),
            documents=docs,
            confidence=report.confidence,
            grounded=False,
            failure_mode=last_failure,
            attempts=self.config.max_attempts,
            trace=trace,
        )

    # -- compound loop -------------------------------------------------------

    def _answer_compound(
        self, question: str, subs: list[str], trace: list[StageTrace]
    ) -> PipelineResult:
        sub_results = [self._answer_atomic(s, trace) for s in subs]
        merged_docs = dedupe_by_id(
            [d for r in sub_results for d in r.documents]
        )[: self.config.rerank_k]

        synthesis_prompt = (
            "Combine the following sub-answers into one coherent response to the "
            "original question. Preserve citations.\n\n"
            f"Original question: {question}\n\n"
            + "\n\n".join(
                f"Sub-question: {s}\nSub-answer: {r.answer}"
                for s, r in zip(subs, sub_results)
            )
            + "\n\nCombined answer:"
        )
        combined = self.llm.complete(synthesis_prompt, temperature=0.2)

        # A single weak leg pulls the whole answer down, which is the right default.
        confidence = min(r.confidence for r in sub_results) if sub_results else 0.0
        grounded = all(r.grounded for r in sub_results)
        failure = FailureMode.NONE if grounded else FailureMode.CONTEXT_INSUFFICIENT

        if grounded:
            self.cache.store(question, combined, merged_docs, confidence)

        return PipelineResult(
            query=question,
            answer=combined,
            documents=merged_docs,
            confidence=confidence,
            grounded=grounded,
            failure_mode=failure,
            attempts=max(r.attempts for r in sub_results),
            trace=trace,
        )

    # -- helpers -------------------------------------------------------------

    def _rewrite(self, question: str, reason: str) -> str:
        rewritten = self.llm.complete(
            REWRITE_PROMPT.format(question=question, reason=reason),
            temperature=0.3,
            max_tokens=120,
        ).strip().splitlines()[0]
        return rewritten or question

    def _refusal(self, question: str, mode: FailureMode) -> str:
        return (
            f"I cannot answer this reliably. Failure mode: {mode.value}. "
            f"Original question: {question}. Consider expanding the corpus or "
            f"loosening the confidence threshold."
        )

    def _time(
        self,
        stage: str,
        trace: list[StageTrace],
        fn: Callable[[], Any],
    ) -> Any:
        start = time.perf_counter()
        try:
            result = fn()
        finally:
            trace.append(StageTrace(stage, (time.perf_counter() - start) * 1000))
        return result
