"""Adapter for the Spider text-to-SQL CSV dataset."""
from __future__ import annotations

import csv
from typing import Iterator

from walt.rm.data.base import BaseAdapter, Example


class SpiderAdapter(BaseAdapter):
    source = "spider"

    def load(self) -> Iterator[Example]:
        with self.file_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["text_query"].strip()
                sql = row["sql_command"].strip()
                if not question or not sql:
                    continue
                yield Example(question=question, sql_good=sql, source=self.source)
