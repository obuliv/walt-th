"""Ollama-backed BaseLLM: generates SQL candidates via a locally running Ollama server.

Ollama's HTTP API has no beam-search or multi-return ("n"/"best_of") parameter, so
generate_candidates makes `n` separate chat() calls with varied temperature/seed instead
of a single decode pass. BaseLLM's signature stays backend-agnostic so a future
transformers-based backend using real num_beams beam search can be swapped in later to
compare the two approaches without touching any caller (SqlAgent, evaluate.py, the CLI).

Requires a local Ollama server running with the target model pulled, e.g.:
    ollama pull llama3.2

max_concurrency (default 1, i.e. today's sequential behavior) issues up to that many of
the n chat() calls at once via a thread pool -- ollama.chat() is a blocking HTTP call, so
plain threads (not multiprocessing) are enough; there's no CPU-bound work here to fight
the GIL over. This is a pure client-side optimization and does NOTHING on its own: the
actual Ollama/llama-server process has its own request-parallelism limit (llama.cpp's
`-np` / Ollama's OLLAMA_NUM_PARALLEL env var, commonly defaulting to 1 unless raised),
and concurrent client requests just queue at the server past that limit. Only raise this
above 1 after confirming the server itself accepts concurrent requests (see this repo's
CLAUDE.md / build_severity_dataset.py docstring for the exact steps) -- otherwise it's
pure overhead for zero throughput gain. Results are always returned in the same seed
order (0..n-1) regardless of which request finishes first or how many run concurrently,
so this is a drop-in speedup with no behavior change to callers.
"""
from __future__ import annotations

import concurrent.futures

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
    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.8,
        max_tokens: int = 512,
        max_concurrency: int = 1,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency

    def _generate_one(self, question: str, schema_context: str, seed: int) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema_context}\n\nQuestion: {question}"},
            ],
            options={
                "temperature": self.temperature,
                "seed": seed,
                "num_predict": self.max_tokens,
            },
        )
        return _strip_sql(response["message"]["content"])

    def generate_candidates(self, question: str, schema_context: str, n: int = 5) -> list[str]:
        if self.max_concurrency <= 1:
            results = [self._generate_one(question, schema_context, i) for i in range(n)]
        else:
            results = [""] * n
            workers = min(self.max_concurrency, n)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._generate_one, question, schema_context, i): i for i in range(n)}
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
        return [sql for sql in results if sql]
