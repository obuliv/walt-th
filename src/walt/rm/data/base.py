"""Base class for text-to-SQL data adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Example:
    question: str
    sql_good: str
    source: str

    def to_dict(self) -> dict:
        return {"question": self.question, "sql_good": self.sql_good, "source": self.source}


class BaseAdapter(ABC):
    """Reads a source-specific file and yields standardized examples."""

    source: str

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    @abstractmethod
    def load(self) -> Iterator[Example]:
        """Parse self.file_path and yield Example records."""
        raise NotImplementedError
