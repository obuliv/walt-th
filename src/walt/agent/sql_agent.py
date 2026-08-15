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
from walt.agent.llm.ollama_llm import OllamaLLM
from walt.rm.model.base import BaseRewardModel, ScoredCandidate
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.lr_model_v3 import LRRewardModelV3
from walt.utils.sql_exec import ExecutionResult, run_sql


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

    def run(self, question: str, schema_context: list[str]) -> AgentResult:
        llm_context = "" if self.strip_llm_context else "\n".join(schema_context)
        candidates = self.llm.generate_candidates(question, llm_context, self.n_candidates)
        if not candidates:
            raise RuntimeError(f"LLM produced no candidate SQL for question: {question!r}")
        scored = self.rm.rank(question, candidates, sql_context=tuple(schema_context))
        best = scored[0]
        execution = run_sql(schema_context, best.sql)
        return AgentResult(
            question=question,
            raw_candidates=candidates,
            scored_candidates=scored,
            best_sql=best.sql,
            execution=execution,
            final_answer=_format_answer(execution),
            critique=None,
        )


def build_llm(ollama_model: str, llm_cache_path: str | Path | None) -> BaseLLM:
    llm: BaseLLM = OllamaLLM(model=ollama_model)
    if llm_cache_path is not None:
        llm = CachingLLM(llm, cache_path=llm_cache_path)
    return llm


def run_agent(
    question: str,
    schema_context: list[str],
    *,
    rm_model_path: str | Path = "data/output/rm_model.joblib",
    ollama_model: str = "llama3.2",
    n_candidates: int = 5,
    llm_cache_path: str | Path | None = "data/output/llm_cache.json",
    strip_llm_context: bool = False,
) -> AgentResult:
    embedding_provider = SentenceTransformerEmbedding()
    rm = LRRewardModelV3.load(rm_model_path, embedding_provider=embedding_provider)
    llm = build_llm(ollama_model, llm_cache_path)
    agent = SqlAgent(llm=llm, rm=rm, n_candidates=n_candidates, strip_llm_context=strip_llm_context)
    return agent.run(question, schema_context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="JSONL with question/sql_context rows (e.g. rm_enhanced.jsonl)")
    parser.add_argument("--index", type=int, default=0, help="Row index to use from --input")
    parser.add_argument("--question", help="Ad hoc question (alternative to --input/--index)")
    parser.add_argument("--schema-file", type=Path, help="File with one SQL statement per line, used with --question")
    parser.add_argument("--rm-model", type=Path, default=Path("data/output/rm_model.joblib"))
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--llm-cache", type=Path, default=Path("data/output/llm_cache.json"), help="Cache LLM candidates here, keyed by (question, schema_context) — reused across RM changes")
    parser.add_argument("--no-llm-cache", action="store_true", help="Always call the LLM fresh, ignoring/skipping the cache")
    parser.add_argument("--strip-context", action="store_true", help="Don't show the LLM schema_context when generating candidates (still used for execution) — tests unconditioned generation")
    args = parser.parse_args()

    if args.input:
        with args.input.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        record = rows[args.index]
        question = record["question"]
        schema_context = record.get("sql_context", [])
    elif args.question:
        question = args.question
        schema_context = args.schema_file.read_text().splitlines() if args.schema_file else []
    else:
        parser.error("Either --input/--index or --question is required")
        return

    result = run_agent(
        question,
        schema_context,
        rm_model_path=args.rm_model,
        ollama_model=args.ollama_model,
        n_candidates=args.n_candidates,
        llm_cache_path=None if args.no_llm_cache else args.llm_cache,
        strip_llm_context=args.strip_context,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
