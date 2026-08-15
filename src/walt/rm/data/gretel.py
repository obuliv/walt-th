"""Adapter for the gretelai/synthetic_text_to_sql HuggingFace dataset.

Unlike Spider/DBASQL, this source already ships a `sql_context` (SQLite CREATE
TABLE/INSERT INTO statements) alongside each (question, sql) pair, so the adapter
splits and locally verifies it here rather than leaving that to an LLM synthesis step
(see gen_training_data.py) — a later sql_bad-generation pass over this source only
needs to add negatives, reusing this context as-is.

https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import requests
import sqlglot

from walt.rm.data.base import BaseAdapter, Example
from walt.utils.sql_exec import run_sql

HF_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "test": "https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def download_split(split: str, dest_dir: Path) -> Path:
    """Downloads the HF-hosted parquet file for `split` into dest_dir, if not already
    present there, and returns its local path."""
    if split not in HF_PARQUET_URLS:
        raise ValueError(f"Unknown gretel split {split!r}, expected one of {list(HF_PARQUET_URLS)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{split}.parquet"
    if dest_path.exists():
        return dest_path

    response = requests.get(HF_PARQUET_URLS[split], timeout=120)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    return dest_path


def split_sql_context(sql_context: str) -> tuple[str, ...]:
    """Splits gretel's single semicolon-joined sql_context string into individual
    statements, using sqlglot (rather than a naive str.split(";")) so semicolons
    inside quoted string literals don't produce bogus statement boundaries. Falls back
    to the naive split on a sqlglot parse error (a handful of gretel rows use syntax
    sqlglot's sqlite dialect rejects) — run_sql()/sql_context_valid catches whatever
    that fallback still gets wrong, consistent with how this project already tracks
    context-validity as a percentage rather than requiring 100%."""
    try:
        statements = sqlglot.parse(sql_context, dialect="sqlite")
        return tuple(stmt.sql(dialect="sqlite") for stmt in statements if stmt is not None)
    except sqlglot.errors.ParseError:
        return tuple(s.strip() for s in sql_context.split(";") if s.strip())


class GretelAdapter(BaseAdapter):
    source = "gretel"

    def load(self) -> Iterator[Example]:
        table = pq.read_table(self.file_path, columns=["sql_prompt", "sql_context", "sql"])
        for row in table.to_pylist():
            question = row["sql_prompt"].strip()
            sql_good = row["sql"].strip()
            raw_context = row["sql_context"].strip()
            if not question or not sql_good:
                continue

            sql_context = split_sql_context(raw_context) if raw_context else ()
            sql_context_valid = run_sql(sql_context, sql_good).success if sql_context else None

            yield Example(
                question=question,
                sql_good=sql_good,
                source=self.source,
                sql_context=sql_context,
                sql_context_valid=sql_context_valid,
            )
