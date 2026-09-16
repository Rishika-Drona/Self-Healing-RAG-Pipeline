"""Internal helpers."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from .types import Document


def extract_json(text: str) -> str:
    """Grab the first JSON object from an LLM response, tolerating prose around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no braces", text, 0)
    return text[start:end + 1]


def dedupe_by_id(docs: Iterable[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        key = d.doc_id or d.fingerprint()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def cosine(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v) + 1e-12
    m_norm = np.linalg.norm(m, axis=1) + 1e-12
    return (m @ v) / (m_norm * v_norm)
