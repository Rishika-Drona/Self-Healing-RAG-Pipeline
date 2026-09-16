"""Self-Healing RAG.

A retrieval-augmented generation pipeline that treats RAG as a feedback system
rather than a single retrieve-then-generate call. Each stage is independently
evaluable so failures are attributable and recovery is targeted.
"""

from .adapters import (
    AnthropicLLM,
    CrossEncoderReranker,
    InMemoryVectorStore,
    OpenAIEmbedder,
    OpenAILLM,
)
from .config import RAGConfig
from .pipeline import (
    AnswerGenerator,
    CRAGDecision,
    CRAGEvaluator,
    CachedAnswer,
    Grounder,
    GroundingReport,
    LearningCache,
    QueryPlanner,
    Retriever,
    SelfHealingRAG,
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

__version__ = "0.1.0"

__all__ = [
    # orchestrator
    "SelfHealingRAG",
    "RAGConfig",
    # stages
    "QueryPlanner",
    "Retriever",
    "CRAGEvaluator",
    "CRAGDecision",
    "AnswerGenerator",
    "Grounder",
    "GroundingReport",
    "LearningCache",
    "CachedAnswer",
    # types
    "Document",
    "PipelineResult",
    "StageTrace",
    "FailureMode",
    "DocGrade",
    # protocols
    "Embedder",
    "VectorStore",
    "Reranker",
    "LLM",
    "WebSearch",
    # adapters
    "OpenAIEmbedder",
    "OpenAILLM",
    "AnthropicLLM",
    "CrossEncoderReranker",
    "InMemoryVectorStore",
]
