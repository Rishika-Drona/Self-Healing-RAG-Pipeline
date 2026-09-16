"""Reference adapters for common providers.

The pipeline never touches these directly. It only knows the Protocol
interfaces from `types.py`, so any of these can be swapped for a Qdrant,
Chroma, Pinecone, Cohere Rerank or OpenAI-completion adapter.
"""

from __future__ import annotations

import numpy as np

from .types import Document, Embedder
from .utils import cosine


class OpenAIEmbedder:
    """Wraps the OpenAI Python client's embeddings endpoint."""

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class InMemoryVectorStore:
    """Numpy-backed store for small corpora and tests.

    Swap for Chroma, Qdrant, Pinecone or pgvector once your corpus outgrows RAM.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._docs: list[Document] = []
        self._matrix: np.ndarray | None = None

    def add(self, docs: list[Document]) -> None:
        vectors = self.embedder.embed([d.text for d in docs])
        self._docs.extend(docs)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )

    def search(self, vector: np.ndarray, k: int) -> list[Document]:
        if self._matrix is None or len(self._docs) == 0:
            return []
        sims = cosine(vector, self._matrix)
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            doc = self._docs[int(i)]
            hit = Document(
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=dict(doc.metadata),
                score=float(sims[int(i)]),
            )
            out.append(hit)
        return out


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder.

    Install with: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, docs: list[Document]) -> list[float]:
        pairs = [(query, d.text) for d in docs]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


class AnthropicLLM:
    """Wraps the Anthropic Python client's messages endpoint."""

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a careful, concise assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", "") == "text"
        )


class OpenAILLM:
    """Wraps the OpenAI Python client's chat completions endpoint."""

    def __init__(self, client, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
