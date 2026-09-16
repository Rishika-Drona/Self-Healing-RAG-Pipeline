"""Core types and provider protocols.

The pipeline is written against small `Protocol` interfaces so any embedder,
vector store, reranker or LLM can be plugged in without touching the control
flow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np


class FailureMode(str, Enum):
    """Why a pass through the pipeline was rejected.

    Each mode maps to a distinct recovery strategy so retries are targeted
    rather than blindly repeating the same call.
    """

    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_IRRELEVANT = "retrieval_irrelevant"
    CONTEXT_INSUFFICIENT = "context_insufficient"
    CONTEXT_STALE = "context_stale"
    GENERATION_UNGROUNDED = "generation_ungrounded"
    GENERATION_LOW_CONFIDENCE = "generation_low_confidence"
    NONE = "none"


class DocGrade(str, Enum):
    """CRAG-style triage of a single retrieved document."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rerank_score: float | None = None
    grade: DocGrade | None = None

    def fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass
class StageTrace:
    stage: str
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: str
    answer: str
    documents: list[Document]
    confidence: float
    grounded: bool
    failure_mode: FailureMode
    attempts: int
    trace: list[StageTrace] = field(default_factory=list)
    cached: bool = False


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class VectorStore(Protocol):
    def search(self, vector: np.ndarray, k: int) -> list[Document]: ...


class Reranker(Protocol):
    def score(self, query: str, docs: list[Document]) -> list[float]: ...


class LLM(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str: ...


class WebSearch(Protocol):
    def search(self, query: str, k: int = 5) -> list[Document]: ...
