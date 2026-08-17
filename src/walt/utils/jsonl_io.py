"""Transparent gzip support for JSONL I/O: a path ending in .gz is
gzip-compressed/decompressed automatically; every other path is plain text."""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO


def open_jsonl(path: str | Path, mode: str = "rt") -> IO[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8")
