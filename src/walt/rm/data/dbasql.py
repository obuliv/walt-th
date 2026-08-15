"""Adapter for the DBASQL JSON dataset."""
from __future__ import annotations

import json
import re
from typing import Iterator

from walt.rm.data.base import BaseAdapter, Example

# input_text entries are suffixed with a literal "[SQL]" tag, e.g. "Insert a new student [SQL]".
_SQL_SUFFIX_RE = re.compile(r"\s*\[SQL\]\s*$", re.IGNORECASE)


class DBASQLAdapter(BaseAdapter):
    source = "dbasql"

    def load(self) -> Iterator[Example]:
        with self.file_path.open(encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            question = _SQL_SUFFIX_RE.sub("", record["input_text"]).strip()
            sql = record["target_text"].strip()
            if not question or not sql:
                continue
            yield Example(question=question, sql_good=sql, source=self.source)
