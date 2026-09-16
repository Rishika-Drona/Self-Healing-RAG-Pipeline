# Self-Healing RAG

A retrieval-augmented generation pipeline that recognizes when its own answer
is poorly supported and tries to fix the problem before returning it.

Most RAG systems are a single retrieve-then-generate call. If the retrieval is
weak, the generation is confidently wrong. This pipeline treats RAG as a
feedback system: every stage emits a signal, a controller decides whether the
signal is strong enough to proceed, and on rejection a targeted recovery
strategy kicks in rather than a blind retry.

## Features

- **HyDE query expansion** for better retrieval on short queries
- **Query decomposition** for compound questions
- **CRAG document grading** with correct / ambiguous / incorrect triage
- **Cross-encoder reranking** for precision on the shortlist
- **Groundedness checking** before the answer is returned
- **Bounded self-healing loop** with failure categories and targeted rewrites
- **Dynamic learning cache** that short-circuits repeat queries
- **Stage-level tracing** so latency and retry hotspots are visible

## Install

```bash
pip install -r requirements.txt
```

For the reference adapters you will also want:

```bash
pip install anthropic openai sentence-transformers
```

## Quickstart

```python
from self_healing_rag import (
    SelfHealingRAG, Document, InMemoryVectorStore,
    OpenAIEmbedder, AnthropicLLM, CrossEncoderReranker,
)
import anthropic, openai

embedder = OpenAIEmbedder(openai.OpenAI())
store = InMemoryVectorStore(embedder)
store.add([
    Document("d1", "Apache Airflow orchestrates data pipelines using DAGs."),
    Document("d2", "Snowflake is a cloud data warehouse."),
])

pipeline = SelfHealingRAG(
    llm=AnthropicLLM(anthropic.Anthropic()),
    embedder=embedder,
    vector_store=store,
    reranker=CrossEncoderReranker(),
)

result = pipeline.answer("What is Airflow?")
print(result.answer)
print(f"grounded={result.grounded} confidence={result.confidence:.2f}")
```

Full runnable examples live in [`examples/`](examples/).

## Architecture

```
question
  |
  v
[cache lookup] -- hit --> return
  |
  v
[decompose] -- compound? --> parallel atomic loops -> synthesize
  |
  v (atomic)
[HyDE] -> [retrieve k=20] -> [rerank k=6] -> [CRAG grade]
                                                |
                            +-------------------+-------------------+
                            |                   |                   |
                          use                augment              refuse
                            |                   |                   |
                            v                   v                   v
                        generate           add web hits         rewrite query
                            |               generate             (retry)
                            v                   |
                        [grounding check]       v
                            |               [grounding check]
                    ok / retry / refuse
```

## Recovery matrix

Every rejection maps to a specific action. Retries are never a blind repeat.

| Failure                   | Action                                    |
| ------------------------- | ----------------------------------------- |
| `RETRIEVAL_EMPTY`         | Rewrite query, widen embedding            |
| `RETRIEVAL_IRRELEVANT`    | Rewrite query using rerank feedback       |
| `CONTEXT_INSUFFICIENT`    | CRAG refuse, escalate to fallback search  |
| `CONTEXT_STALE`           | Add web hits, prefer recent metadata      |
| `GENERATION_UNGROUNDED`   | Rewrite with named unsupported claims     |
| `GENERATION_LOW_CONFIDENCE` | Retry with lower temperature and wider k |

## Swap points

The pipeline talks to five `Protocol` interfaces defined in
[`self_healing_rag/types.py`](self_healing_rag/types.py). Anything satisfying
them works.

- `Embedder` for query and doc vectors
- `VectorStore` for approximate nearest-neighbor retrieval
- `Reranker` for cross-encoder precision
- `LLM` for HyDE, decomposition, grading, generation, grounding
- `WebSearch` for CRAG fallback (optional)

Reference adapters in [`adapters.py`](self_healing_rag/adapters.py) cover
OpenAI embeddings, in-memory numpy search, sentence-transformers
cross-encoders, and both the Anthropic and OpenAI client. Replace
`InMemoryVectorStore` with Chroma, Qdrant, Milvus or pgvector in production.

## Configuration

Every threshold is on `RAGConfig`:

```python
from self_healing_rag import RAGConfig, SelfHealingRAG

config = RAGConfig(
    top_k=20,                          # retrieval breadth
    rerank_k=6,                        # what CRAG and the LLM see
    retrieval_relevance_threshold=0.55,
    grounding_threshold=0.7,           # tighten for factual domains
    answer_confidence_threshold=0.6,
    max_attempts=3,                    # bounded healing
)

pipeline = SelfHealingRAG(..., config=config)
```

## Cost controls

Every retry costs LLM calls, so the loop is bounded and instrumented:

- `max_attempts` caps healing at three passes by default
- `min_relevant_docs` gates augmentation to avoid a wasted web search
- Learning cache short-circuits repeat queries that already grounded well
- Cross-encoder reranking narrows the LLM-graded shortlist to `rerank_k` docs
- Every stage emits a `StageTrace` with latency and detail for observability

A single query in the worst case makes roughly 4-8 LLM calls (HyDE, optional
decomposition, one grade per doc, generation, grounding, plus a rewrite on
retry). Two practical ways to keep the bill in check:

1. Use a cheaper model for grading and grounding (Haiku, GPT-4o-mini) and
   reserve the expensive one for generation.
2. Add per-user rate limiting at the API gateway before anything hits the
   pipeline.

## Tuning notes

- Grounding threshold at 0.7 is a starting point. Domains with dense factual
  answers (docs, policy, code) tolerate 0.85. Open-ended domains do better
  around 0.6.
- HyDE helps most on short queries. If your queries are already keyword-rich,
  skip it and save an LLM call.
- Cross-encoder reranking is the biggest quality lever for a fixed budget.
  Prefer a larger reranker over more retrieved candidates.
- Decomposition is a tradeoff. It fixes compound queries but adds attempts.
  Tighten `decompose_signal_words` if you see over-decomposition.
- The grounding grader shares an LLM with generation by default. If they are
  the same model, a confidently wrong answer can talk itself into being
  grounded. Using a smaller or different family model as the grader is a
  cheap correctness win.

## Development

Run the tests without API keys:

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses deterministic fakes for LLM and embedder so it validates
control flow (happy path, recovery, refusal, cache short-circuit) offline.

## Project layout

```
self_healing_rag/
  __init__.py       # public exports
  types.py          # dataclasses, enums, Protocols
  config.py         # RAGConfig
  prompts.py        # prompt templates
  pipeline.py       # stages + orchestrator
  adapters.py       # OpenAI, Anthropic, in-memory store, cross-encoder
  utils.py          # helpers
tests/
  test_pipeline.py  # deterministic offline tests
examples/
  quickstart.py     # minimal end-to-end example
  inspect_trace.py  # compound query with stage trace
```

## License

MIT. See [LICENSE](LICENSE).
