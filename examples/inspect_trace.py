"""Inspecting the healing trace on a compound query.

Shows how decomposition splits a compound question into atomic sub-questions
and how each sub-question runs its own healing loop. Useful when tuning
thresholds because you can see which stage rejected which pass.
"""

import os

import anthropic
import openai

from self_healing_rag import (
    AnthropicLLM,
    CrossEncoderReranker,
    Document,
    InMemoryVectorStore,
    OpenAIEmbedder,
    RAGConfig,
    SelfHealingRAG,
)


CORPUS = [
    Document(
        "spark-basics",
        "Apache Spark is a distributed compute framework. Jobs are broken into "
        "stages of tasks that run in parallel across executors.",
    ),
    Document(
        "spark-shuffle",
        "Shuffles happen when Spark needs to redistribute data across "
        "partitions, typically for wide transformations like joins and "
        "groupBy. Shuffles write to disk and are the main source of skew.",
    ),
    Document(
        "flink-basics",
        "Apache Flink is a stream-processing engine that treats batch as a "
        "special case of streaming. State is checkpointed to durable storage "
        "for fault tolerance.",
    ),
    Document(
        "flink-vs-spark",
        "Flink offers true event-time streaming with low latency. Spark "
        "Structured Streaming uses micro-batches, trading latency for a "
        "simpler execution model.",
    ),
]


def main():
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    embedder = OpenAIEmbedder(openai_client)
    store = InMemoryVectorStore(embedder)
    store.add(CORPUS)

    pipeline = SelfHealingRAG(
        llm=AnthropicLLM(anthropic_client),
        embedder=embedder,
        vector_store=store,
        reranker=CrossEncoderReranker(),
        config=RAGConfig(
            top_k=6,
            rerank_k=3,
            grounding_threshold=0.75,
        ),
    )

    question = (
        "How does Spark handle shuffles and how does Flink compare on "
        "streaming latency?"
    )
    result = pipeline.answer(question)

    print("=" * 60)
    print("Question:", question)
    print("=" * 60)
    print("\nAnswer:\n", result.answer)
    print(
        f"\nGrounded: {result.grounded}  Confidence: {result.confidence:.2f}  "
        f"Attempts: {result.attempts}"
    )
    print(f"Failure mode: {result.failure_mode.value}")

    print("\nStage trace:")
    for step in result.trace:
        detail = " ".join(f"{k}={v}" for k, v in step.detail.items())
        print(f"  {step.stage:20s} {step.duration_ms:6.1f}ms  {detail}")


if __name__ == "__main__":
    main()
