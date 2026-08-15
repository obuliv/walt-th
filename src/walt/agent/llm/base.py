"""Abstract interface for a SQL-candidate-generating LLM backend."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate_candidates(self, question: str, schema_context: str, n: int = 5) -> list[str]:
        """Return up to n candidate SQL strings answering `question`, given
        `schema_context` (CREATE TABLE / sample-data statements the SQL should target)."""
        raise NotImplementedError
