"""Prompt templates.

Kept in one place so they can be tuned without hunting through the control
flow. All grader prompts return JSON so the LLM's answer is a parse away from
a decision.
"""

HYDE_PROMPT = """Write a short passage (3-5 sentences) that would be a strong,
factual answer to this question. Do not hedge. Write in the voice of a
reference document. The passage is used as a query representation, not shown
to a user.

Question: {question}

Passage:"""


DECOMPOSE_PROMPT = """Break this question into 1-4 atomic sub-questions that
can each be answered from a single passage. If it is already atomic, return
the original as the only element.

Return JSON: {{"subquestions": ["...", "..."]}}

Question: {question}"""


DOC_GRADE_PROMPT = """You are grading whether a passage is relevant to a
question. Read the passage and answer with one of correct, ambiguous or
incorrect, plus a one-sentence reason.

Return JSON: {{"grade": "correct|ambiguous|incorrect", "reason": "..."}}

Question: {question}
Passage: {passage}"""


ANSWER_PROMPT = """Answer the question using only the numbered passages below.
For every factual claim, cite the passage number in square brackets, for
example [2]. If the passages do not support an answer, say so plainly and
list what is missing.

Question: {question}

Passages:
{context}

Answer:"""


GROUNDEDNESS_PROMPT = """Check whether the answer is fully supported by the
passages. A claim is supported when a passage directly states it or clearly
implies it. Return:

- grounded: fraction of claims supported, from 0 to 1
- unsupported: list of unsupported claims, up to 3
- confidence: your confidence in the answer overall, from 0 to 1

Return JSON: {{"grounded": 0.0, "unsupported": ["..."], "confidence": 0.0}}

Question: {question}
Passages:
{context}
Answer: {answer}"""


REWRITE_PROMPT = """Rewrite the search query to improve retrieval. The current
retrieval was weak. Produce a more specific query that names concrete entities
or terms likely to appear in a source document. One line, no preamble.

Original query: {question}
Weak retrieval reason: {reason}

Rewritten query:"""
