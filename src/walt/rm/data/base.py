"""Base class for text-to-SQL data adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from walt.utils.sql_exec import clean_context


@dataclass(frozen=True)
class SQLBadCandidate:
    sql: str
    reason: str
    # 0-5, only populated by walt.rm.data.synth.enhance_severity_dataset: 0 = executes
    # to the same result as sql_good (non-distinguishing), 1 = low severity, 5 = high
    # severity. None on every other source/pipeline — see LRRewardModel.fit()'s
    # effective-rank rule for how each case is trained on.
    severity: int | None = None

    @staticmethod
    def from_dict(d: dict) -> "SQLBadCandidate":
        return SQLBadCandidate(sql=d["sql"], reason=d["reason"], severity=d.get("severity"))

    def to_dict(self) -> dict:
        d = {"sql": self.sql, "reason": self.reason}
        if self.severity is not None:
            d["severity"] = self.severity
        return d


@dataclass(frozen=True)
class Example:
    question: str
    sql_good: str
    source: str
    sql_bad: tuple[SQLBadCandidate, ...] = ()
    sql_context: tuple[str, ...] = ()
    sql_context_clean: tuple[str, ...] = ()  # sql_context with INSERT statements stripped — see clean_context()
    sql_context_valid: bool | None = None
    # Path to a real .sqlite file, relative to $DATA_PATH, that sql executes against — an
    # alternative to embedding sql_context inline for sources where the real DB already
    # exists on disk (e.g. walt.rm.data.synth) and duplicating its CREATE TABLE/INSERT INTO
    # dump into every row would be wasteful. sql_context is left empty when this is set; see
    # walt.utils.sql_exec.execute_with_context, which resolves either form transparently.
    sql_context_path: str | None = None
    split: str = "trainval"  # "trainval" | "val" — RM training/CV never sees "val" rows

    def __post_init__(self) -> None:
        if self.sql_good in {b.sql for b in self.sql_bad}:
            raise ValueError(f"sql_good duplicates a sql_bad entry for question: {self.question!r}")

    def to_dict(self) -> dict:
        d = {
            "question": self.question,
            "sql_good": self.sql_good,
            "source": self.source,
            "split": self.split,
        }
        if self.sql_bad:
            d["sql_bad"] = [b.to_dict() for b in self.sql_bad]
        if self.sql_context:
            d["sql_context"] = list(self.sql_context)
        if self.sql_context_clean:
            d["sql_context_clean"] = list(self.sql_context_clean)
        if self.sql_context_valid is not None:
            d["sql_context_valid"] = self.sql_context_valid
        if self.sql_context_path is not None:
            d["sql_context_path"] = self.sql_context_path
        return d

    @staticmethod
    def from_dict(d: dict) -> "Example":
        sql_bad = tuple(SQLBadCandidate.from_dict(b) for b in d.get("sql_bad", []))
        sql_context = tuple(d.get("sql_context", []))
        # Fall back to computing it for rows written before sql_context_clean existed,
        # so older enhanced JSONL files keep working without regenerating them.
        sql_context_clean = (
            tuple(d["sql_context_clean"]) if "sql_context_clean" in d else clean_context(sql_context)
        )
        return Example(
            question=d["question"],
            sql_good=d["sql_good"],
            source=d["source"],
            sql_bad=sql_bad,
            sql_context=sql_context,
            sql_context_clean=sql_context_clean,
            sql_context_valid=d.get("sql_context_valid"),
            sql_context_path=d.get("sql_context_path"),
            split=d.get("split", "trainval"),
        )


class BaseAdapter(ABC):
    """Reads a source-specific file and yields standardized examples."""

    source: str

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    @abstractmethod
    def load(self) -> Iterator[Example]:
        """Parse self.file_path and yield Example records."""
        raise NotImplementedError
