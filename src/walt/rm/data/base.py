"""Base class for text-to-SQL data adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SQLBadCandidate:
    sql: str
    reason: str

    @staticmethod
    def from_dict(d: dict) -> "SQLBadCandidate":
        return SQLBadCandidate(sql=d["sql"], reason=d["reason"])

    def to_dict(self) -> dict:
        return {"sql": self.sql, "reason": self.reason}


@dataclass(frozen=True)
class Example:
    question: str
    sql_good: str
    source: str
    sql_bad: tuple[SQLBadCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.sql_good in {b.sql for b in self.sql_bad}:
            raise ValueError(f"sql_good duplicates a sql_bad entry for question: {self.question!r}")

    def to_dict(self) -> dict:
        d = {"question": self.question, "sql_good": self.sql_good, "source": self.source}
        if self.sql_bad:
            d["sql_bad"] = [b.to_dict() for b in self.sql_bad]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Example":
        sql_bad = tuple(SQLBadCandidate.from_dict(b) for b in d.get("sql_bad", []))
        return Example(question=d["question"], sql_good=d["sql_good"], source=d["source"], sql_bad=sql_bad)


class BaseAdapter(ABC):
    """Reads a source-specific file and yields standardized examples."""

    source: str

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    @abstractmethod
    def load(self) -> Iterator[Example]:
        """Parse self.file_path and yield Example records."""
        raise NotImplementedError
