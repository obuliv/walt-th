"""Ollama-backed BaseLLM: generates SQL candidates via a locally running Ollama server.

Ollama's HTTP API has no beam-search or multi-return ("n"/"best_of") parameter, so
generate_candidates makes `n` separate chat() calls with varied temperature/seed instead
of a single decode pass. BaseLLM's signature stays backend-agnostic so a future
transformers-based backend using real num_beams beam search can be swapped in later to
compare the two approaches without touching any caller (SqlAgent, evaluate.py, the CLI).

Requires a local Ollama server running with the target model pulled, e.g.:
    ollama pull llama3.2
"""
from __future__ import annotations

import ollama

from walt.agent.llm.base import BaseLLM

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a SQLite schema and a question, output "
    "ONLY one valid SQL query that answers the question. No explanation, no markdown "
    "fences, no commentary - just the SQL statement."
)


def _strip_sql(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower().startswith("sql"):
            text = text[3:]
    return text.strip().rstrip(";").strip()


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3.2", temperature: float = 0.8, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_candidates(self, question: str, schema_context: str, n: int = 5) -> list[str]:
        candidates = []
        for i in range(n):
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Schema:\n{schema_context}\n\nQuestion: {question}"},
                ],
                options={
                    "temperature": self.temperature,
                    "seed": i,
                    "num_predict": self.max_tokens,
                },
            )
            sql = _strip_sql(response["message"]["content"])
            if sql:
                candidates.append(sql)
        return candidates
