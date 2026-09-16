"""Quickstart example.

Requires:
    pip install anthropic openai sentence-transformers

Set environment variables:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...    # for embeddings
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
    SelfHealingRAG,
)


def main():
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    embedder = OpenAIEmbedder(openai_client)
    store = InMemoryVectorStore(embedder)

    corpus = [
        Document(
            "airflow",
            "Apache Airflow is an open-source workflow orchestrator originally "
            "developed at Airbnb in 2014. Workflows are defined as directed "
            "acyclic graphs (DAGs) of tasks written in Python.",
        ),
        Document(
            "snowflake",
            "Snowflake is a cloud data warehouse founded in 2012. It separates "
            "storage from compute so warehouses can be scaled independently of "
            "the data volume.",
        ),
        Document(
            "dbt",
            "dbt (data build tool) is a transformation framework that runs SQL "
            "models inside a data warehouse. Models can reference each other "
            "using the ref() Jinja macro.",
        ),
    ]
    store.add(corpus)

    pipeline = SelfHealingRAG(
        llm=AnthropicLLM(anthropic_client),
        embedder=embedder,
        vector_store=store,
        reranker=CrossEncoderReranker(),
    )

    result = pipeline.answer("What is Apache Airflow used for?")

    print("Answer:", result.answer)
    print("Grounded:", result.grounded)
    print("Confidence:", result.confidence)
    print("Attempts:", result.attempts)
    print("Failure mode:", result.failure_mode.value)
    print("\nTrace:")
    for step in result.trace:
        print(f"  {step.stage}: {step.duration_ms:.1f}ms {step.detail}")


if __name__ == "__main__":
    main()
