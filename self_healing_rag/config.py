"""Tuning configuration for the self-healing loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    # Retrieval breadth. top_k feeds the reranker, rerank_k is what CRAG sees.
    top_k: int = 20
    rerank_k: int = 6
    min_relevant_docs: int = 2

    # Confidence thresholds gate the healing loop.
    retrieval_relevance_threshold: float = 0.55
    grounding_threshold: float = 0.70
    answer_confidence_threshold: float = 0.60

    # Bounded retries. Every retry costs an LLM call, so keep this small.
    max_attempts: int = 3

    # Decomposition triggers when a query looks compound.
    decompose_min_tokens: int = 12
    decompose_signal_words: tuple[str, ...] = (
        "and", "also", "compare", "versus", "vs", "difference", "both",
    )

    # Learning cache. Survives across queries and rehydrates on lookup.
    cache_similarity_threshold: float = 0.92
    cache_max_entries: int = 2_000
