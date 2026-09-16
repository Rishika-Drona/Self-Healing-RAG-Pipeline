"""Deterministic tests for the self-healing loop.

Uses fake LLM and embedder implementations so control flow can be verified
without API keys or network access. Three scenarios cover the main branches:

1. Happy path: retrieval hits, grounding passes, single attempt.
2. Recovery: first grounding fails, rewrite fixes the second pass.
3. Refusal: nothing in the corpus matches, pipeline bails cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from self_healing_rag import (
    Document,
    InMemoryVectorStore,
    RAGConfig,
    SelfHealingRAG,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Cheap deterministic embedder. Hashes tokens into a fixed vector."""

    dim = 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = hash(tok) % self.dim
                out[i, h] += 1.0
            n = np.linalg.norm(out[i]) + 1e-12
            out[i] /= n
        return out


class KeywordReranker:
    """Scores by shared token overlap."""

    def score(self, query, docs):
        q = set(query.lower().split())
        return [len(q & set(d.text.lower().split())) / max(len(q), 1) for d in docs]


class ScriptedLLM:
    """Returns pre-scripted responses based on prompt substring match."""

    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=800):
        self.calls.append(prompt[:80])
        for needle, response in self.rules:
            if needle in prompt:
                return response
        return "{}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CORPUS = [
    Document(
        "d1",
        "Apache Airflow is an orchestrator for data pipelines. DAGs describe "
        "task dependencies.",
    ),
    Document(
        "d2",
        "Snowflake is a cloud data warehouse with separated storage and compute "
        "layers.",
    ),
    Document(
        "d3",
        "dbt transforms data inside a warehouse using SQL models and Jinja "
        "templating.",
    ),
]


def make_pipeline(llm):
    embedder = HashEmbedder()
    store = InMemoryVectorStore(embedder)
    store.add(CORPUS)
    return SelfHealingRAG(
        llm=llm,
        embedder=embedder,
        vector_store=store,
        reranker=KeywordReranker(),
        config=RAGConfig(
            top_k=3,
            rerank_k=3,
            min_relevant_docs=1,
            retrieval_relevance_threshold=0.1,
            grounding_threshold=0.7,
            answer_confidence_threshold=0.6,
            max_attempts=3,
        ),
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_happy_path():
    llm = ScriptedLLM([
        ("Write a short passage", "Airflow orchestrates DAGs for pipelines."),
        ("grade", '{"grade": "correct", "reason": "matches"}'),
        ("Answer the question", "Airflow orchestrates pipelines via DAGs [1]."),
        (
            "Check whether the answer",
            '{"grounded": 0.95, "unsupported": [], "confidence": 0.9}',
        ),
    ])
    pipe = make_pipeline(llm)
    result = pipe.answer("What is Airflow?")
    assert result.grounded
    assert result.attempts == 1
    assert result.confidence >= 0.6


def test_recovery_after_ungrounded_first_pass():
    grounding_responses = iter([
        '{"grounded": 0.3, "unsupported": ["fake claim"], "confidence": 0.4}',
        '{"grounded": 0.9, "unsupported": [], "confidence": 0.85}',
    ])

    class DynamicLLM(ScriptedLLM):
        def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=800):
            self.calls.append(prompt[:80])
            if "Check whether the answer" in prompt:
                return next(grounding_responses)
            for needle, response in self.rules:
                if needle in prompt:
                    return response
            return "{}"

    llm = DynamicLLM([
        ("Write a short passage", "dbt transforms data with SQL models."),
        ("grade", '{"grade": "correct", "reason": "matches"}'),
        ("Answer the question", "dbt runs SQL models in a warehouse [1]."),
        ("Rewrite the search query", "dbt SQL transformation warehouse"),
    ])
    pipe = make_pipeline(llm)
    result = pipe.answer("How does dbt work?")
    assert result.grounded
    assert result.attempts == 2


def test_refusal_when_corpus_has_no_match():
    llm = ScriptedLLM([
        ("Write a short passage", "Something about unicorns and rainbows."),
        ("grade", '{"grade": "incorrect", "reason": "off-topic"}'),
        ("Rewrite the search query", "unicorn rainbow color spectrum"),
    ])
    pipe = make_pipeline(llm)
    result = pipe.answer("What color is a unicorn?")
    assert not result.grounded
    assert result.failure_mode.value in {
        "context_insufficient",
        "retrieval_irrelevant",
        "retrieval_empty",
    }


def test_cache_short_circuits_repeat_query():
    llm = ScriptedLLM([
        ("Write a short passage", "Airflow orchestrates DAGs."),
        ("grade", '{"grade": "correct", "reason": "matches"}'),
        ("Answer the question", "Airflow orchestrates DAGs [1]."),
        (
            "Check whether the answer",
            '{"grounded": 0.95, "unsupported": [], "confidence": 0.9}',
        ),
    ])
    pipe = make_pipeline(llm)
    first = pipe.answer("What is Airflow?")
    assert not first.cached
    calls_after_first = len(llm.calls)

    second = pipe.answer("What is Airflow?")
    assert second.cached
    assert len(llm.calls) == calls_after_first  # cache hit, no new LLM calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
