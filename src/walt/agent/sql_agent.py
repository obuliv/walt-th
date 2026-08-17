"""Ties together an LLM candidate generator, a reward model, and SQL execution:
generate N candidate SQL queries, score them with the RM, execute the RM's top pick
against an in-memory SQLite database seeded from schema_context, and report the result.

Usage:
    python -m walt.agent.sql_agent --input data/output/rm_enhanced.jsonl --index 0
    python -m walt.agent.sql_agent --question "..." --schema-file schema.sql
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from walt.agent.llm.base import BaseLLM
from walt.agent.llm.caching_llm import CachingLLM
from walt.agent.llm.claude_llm import ClaudeLLM
from walt.agent.llm.ollama_llm import OllamaLLM
from walt.rm.model.base import BaseRewardModel, ScoredCandidate
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.lr.lr_model_v6 import LRRewardModelV6
from walt.utils.sql_exec import ExecutionResult, clean_context, execute_with_context


@dataclass(frozen=True)
class AgentResult:
    question: str
    raw_candidates: list[str]  # LLM output, in generation order, before RM reranking
    scored_candidates: list[ScoredCandidate]
    best_sql: str
    execution: ExecutionResult
    final_answer: str | None
    critique: str | None = None  # placeholder — not implemented yet

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "raw_candidates": self.raw_candidates,
            "scored_candidates": [
                {"sql": c.sql, "score": c.score, "rank": c.rank, "error_code": c.error_code}
                for c in self.scored_candidates
            ],
            "best_sql": self.best_sql,
            "execution": {
                "success": self.execution.success,
                "columns": list(self.execution.columns) if self.execution.columns else None,
                "rows": [list(row) for row in self.execution.rows] if self.execution.rows else None,
                "error": self.execution.error,
            },
            "final_answer": self.final_answer,
            "critique": self.critique,
        }


def _format_answer(execution: ExecutionResult) -> str | None:
    if not execution.success:
        return f"Query failed: {execution.error}"
    if execution.rows is None:
        return "Query executed successfully (no result rows)."
    if not execution.rows:
        return "No rows returned."
    header = ", ".join(execution.columns) if execution.columns else ""
    lines = [", ".join(str(v) for v in row) for row in execution.rows]
    return "\n".join(([header] if header else []) + lines)


class SqlAgent:
    def __init__(
        self,
        llm: BaseLLM,
        rm: BaseRewardModel,
        n_candidates: int = 5,
        strip_llm_context: bool = False,
    ):
        self.llm = llm
        self.rm = rm
        self.n_candidates = n_candidates
        # When True, the LLM generates blind (no schema) while execution/scoring still
        # use the real schema_context — isolates the effect of schema-grounded generation
        # from everything else in the pipeline (RM scoring, SQL execution).
        self.strip_llm_context = strip_llm_context

    def run(
        self,
        question: str,
        schema_context: list[str],
        sql_context_clean: list[str] | None = None,
        sql_context_path: str | None = None,
    ) -> AgentResult:
        # LLM/RM see only the schema (CREATE TABLE etc, no INSERT rows) — sample data is
        # noise for "does this candidate SQL fit the schema", and is only needed below
        # for actually executing the chosen candidate. sql_context_clean lets a caller
        # pass the already-computed DDL-only schema directly (needed when schema_context
        # itself is empty by design — see sql_context_path below — so there'd be nothing
        # for clean_context() to derive it from); defaults to deriving it the old way.
        clean_schema = sql_context_clean if sql_context_clean is not None else clean_context(schema_context)
        llm_context = "" if self.strip_llm_context else "\n".join(clean_schema)
        candidates = self.llm.generate_candidates(question, llm_context, self.n_candidates)
        if not candidates:
            raise RuntimeError(f"LLM produced no candidate SQL for question: {question!r}")
        scored = self.rm.rank(question, candidates, sql_context=clean_schema)
        best = scored[0]
        # sql_context_path lets execution fall back to a real .sqlite file (relative to
        # $DATA_PATH, cached across calls) when schema_context itself is empty — see
        # walt.utils.sql_exec.execute_with_context. When schema_context is non-empty this
        # is identical to the old run_sql(schema_context, best.sql) call.
        execution = execute_with_context(schema_context, sql_context_path, best.sql)
        return AgentResult(
            question=question,
            raw_candidates=candidates,
            scored_candidates=scored,
            best_sql=best.sql,
            execution=execution,
            final_answer=_format_answer(execution),
            critique=None,
        )


def build_llm(
    model: str, llm_cache_path: str | Path | None, backend: str = "ollama", ollama_concurrency: int = 1
) -> BaseLLM:
    """backend selects the candidate-generating LLM: "ollama" (default, local
    OllamaLLM — model is e.g. "llama3.2") or "claude" (ClaudeLLM via the Anthropic API
    — model is e.g. "claude-haiku-4-5-20251001", requires ANTHROPIC_API_KEY).
    ollama_concurrency is ignored for backend="claude" — see OllamaLLM's docstring for
    why raising it only helps once the Ollama server itself accepts concurrent
    requests."""
    llm: BaseLLM
    if backend == "claude":
        llm = ClaudeLLM(model=model)
    elif backend == "ollama":
        llm = OllamaLLM(model=model, max_concurrency=ollama_concurrency)
    else:
        raise ValueError(f"Unknown llm backend: {backend!r} (expected 'ollama' or 'claude')")
    if llm_cache_path is not None:
        llm = CachingLLM(llm, cache_path=llm_cache_path)
    return llm


def run_agent(
    question: str,
    schema_context: list[str],
    *,
    sql_context_clean: list[str] | None = None,
    sql_context_path: str | None = None,
    rm_model_path: str | Path = "data/output/rm_model.joblib",
    ollama_model: str = "llama3.2",
    llm_backend: str = "ollama",
    n_candidates: int = 5,
    llm_cache_path: str | Path | None = "data/output/llm_cache.json",
    strip_llm_context: bool = False,
    ollama_concurrency: int = 1,
) -> AgentResult:
    embedding_provider = SentenceTransformerEmbedding()
    rm = LRRewardModelV6.load(rm_model_path, embedding_provider=embedding_provider)
    llm = build_llm(ollama_model, llm_cache_path, backend=llm_backend, ollama_concurrency=ollama_concurrency)
    agent = SqlAgent(llm=llm, rm=rm, n_candidates=n_candidates, strip_llm_context=strip_llm_context)
    return agent.run(question, schema_context, sql_context_clean=sql_context_clean, sql_context_path=sql_context_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="JSONL with question/sql_context rows (e.g. rm_enhanced.jsonl)")
    parser.add_argument("--index", type=int, default=0, help="Row index to use from --input")
    parser.add_argument("--question", help="Ad hoc question (alternative to --input/--index)")
    parser.add_argument("--schema-file", type=Path, help="File with one SQL statement per line, used with --question")
    parser.add_argument("--rm-model", type=Path, default=Path("data/output/rm_model.joblib"))
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--ollama-concurrency", type=int, default=1, help="Concurrent Ollama requests per row's candidate generation. Only helps once the Ollama server itself accepts concurrent requests (see build_severity_dataset.py docstring) — otherwise pure overhead.")
    parser.add_argument("--llm-cache", type=Path, default=Path("data/output/llm_cache.json"), help="Cache LLM candidates here, keyed by (question, schema_context) — reused across RM changes")
    parser.add_argument("--no-llm-cache", action="store_true", help="Always call the LLM fresh, ignoring/skipping the cache")
    parser.add_argument("--strip-context", action="store_true", help="Don't show the LLM schema_context when generating candidates (still used for execution) — tests unconditioned generation")
    args = parser.parse_args()

    sql_context_clean = None
    sql_context_path = None
    if args.input:
        with args.input.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        record = rows[args.index]
        question = record["question"]
        schema_context = record.get("sql_context", [])
        sql_context_clean = record.get("sql_context_clean")
        sql_context_path = record.get("sql_context_path")
    elif args.question:
        question = args.question
        schema_context = args.schema_file.read_text().splitlines() if args.schema_file else []
    else:
        parser.error("Either --input/--index or --question is required")
        return

    result = run_agent(
        question,
        schema_context,
        sql_context_clean=sql_context_clean,
        sql_context_path=sql_context_path,
        rm_model_path=args.rm_model,
        ollama_model=args.ollama_model,
        n_candidates=args.n_candidates,
        llm_cache_path=None if args.no_llm_cache else args.llm_cache,
        strip_llm_context=args.strip_context,
        ollama_concurrency=args.ollama_concurrency,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
