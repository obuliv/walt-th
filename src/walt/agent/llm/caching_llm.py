"""Disk-backed cache wrapping any BaseLLM, so re-running evaluation with a different RM
(different --rm-model, hyperparameters, or reward model class entirely) doesn't require
regenerating candidates from the LLM every time — only the (question, schema_context)
pair matters for what the LLM produces; RM tuning changes nothing upstream of it.

Cache key deliberately excludes `n`: a request for fewer candidates than are cached is
served by slicing, and a request for more triggers a fresh generate_candidates() call
that overwrites the cache entry — so raising --n-candidates between runs still costs a
regeneration, but lowering or holding it steady (the common case while tuning the RM)
is always a cache hit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from walt.agent.llm.base import BaseLLM


class CachingLLM(BaseLLM):
    def __init__(self, inner: BaseLLM, cache_path: str | Path, cache_namespace: str | None = None):
        self.inner = inner
        self.cache_path = Path(cache_path)
        # Distinguishes cache files shared across LLM configs (e.g. different Ollama
        # models) — defaults to the inner LLM's own `model` attribute if it has one.
        self.cache_namespace = cache_namespace or getattr(inner, "model", type(inner).__name__)
        self._cache: dict[str, list[str]] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def _key(self, question: str, schema_context: str) -> str:
        raw = json.dumps(
            {"namespace": self.cache_namespace, "question": question, "schema_context": schema_context},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate_candidates(self, question: str, schema_context: str, n: int = 5) -> list[str]:
        key = self._key(question, schema_context)
        cached = self._cache.get(key, [])
        if len(cached) >= n:
            return cached[:n]
        candidates = self.inner.generate_candidates(question, schema_context, n)
        self._cache[key] = candidates
        self._save()
        return candidates

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2))
