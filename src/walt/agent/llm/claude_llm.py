"""Claude-backed BaseLLM: generates SQL candidates via the Anthropic API (default:
Haiku, a cheap/fast model — useful for comparing against OllamaLLM's llama3.2 in agent
eval, since the RM was trained on Claude-synthesized sql_good/sql_bad and this tests
whether it transfers better to candidates from the same model family it learned from).

The Messages API has no beam-search/multi-return parameter either, so this mirrors
OllamaLLM's approach: n separate calls at a nonzero temperature instead of one decode
pass. Requires ANTHROPIC_API_KEY (see .env.example) — same convention as
gen_training_data.py/fix_sql_bad.py.
"""
from __future__ import annotations

import anthropic
from dotenv import load_dotenv

from walt.agent.llm.base import BaseLLM

load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

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


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.8, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def generate_candidates(self, question: str, schema_context: str, n: int = 5) -> list[str]:
        candidates = []
        for _ in range(n):
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Schema:\n{schema_context}\n\nQuestion: {question}"}],
            )
            text = "".join(block.text for block in message.content if block.type == "text")
            sql = _strip_sql(text)
            if sql:
                candidates.append(sql)
        return candidates
