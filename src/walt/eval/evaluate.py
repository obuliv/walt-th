"""Agent-level evaluation on the held-out val split (rows with split="val", produced by
pre_process.py and never seen by RM training/CV — see rm/data/pre_process.py).

Reports three things:
  1. RM accuracy selecting correct SQL among sql_good vs sql_bad (BaseRewardModel.evaluate()).
  2. SQL execution pass/fail: does the agent's chosen SQL actually run against sql_context?
  3. End-to-end QA accuracy: does the agent's executed result match sql_good's?

Usage:
    python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from walt.agent.llm.ollama_llm import OllamaLLM
from walt.agent.sql_agent import SqlAgent
from walt.rm.data.base import Example
from walt.rm.model.base import load_examples
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.lr_model_v3 import LRRewardModelV3
from walt.utils.sql_exec import ExecutionResult, run_sql


def _rows_match(a: ExecutionResult, b: ExecutionResult) -> bool:
    if a.columns != b.columns:
        return False
    return sorted(a.rows or ()) == sorted(b.rows or ())


def evaluate_agent(val_examples: list[Example], agent: SqlAgent) -> dict[str, Any]:
    executable = [ex for ex in val_examples if ex.sql_context]
    pass_count = 0
    qa_examples = [ex for ex in executable if ex.sql_context_valid]
    qa_correct = 0

    for ex in executable:
        try:
            result = agent.run(ex.question, list(ex.sql_context))
        except RuntimeError as exc:
            # The LLM produced zero usable candidates for this row (can happen with a
            # small n_candidates against a reasoning model that burns its token budget
            # on thinking) — count it as a failed row rather than aborting the whole
            # evaluation run.
            print(f"  WARNING: {exc}")
            continue
        if result.execution.success:
            pass_count += 1
        if ex.sql_context_valid:
            reference = run_sql(ex.sql_context, ex.sql_good)
            if _rows_match(result.execution, reference):
                qa_correct += 1

    return {
        "sql_pass_rate": {
            "passed": pass_count,
            "total": len(executable),
            "rate": pass_count / len(executable) if executable else None,
        },
        "qa_accuracy": {
            "correct": qa_correct,
            "total": len(qa_examples),
            "rate": qa_correct / len(qa_examples) if qa_examples else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/rm_enhanced.jsonl"))
    parser.add_argument("--rm-model", type=Path, default=Path("data/output/rm_model.joblib"))
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N val rows (smoke testing)")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON summary")
    args = parser.parse_args()

    examples = load_examples(args.input)
    val_examples = [ex for ex in examples if ex.split == "val"]
    if args.limit:
        val_examples = val_examples[: args.limit]
    print(f"Evaluating on {len(val_examples)} val examples")

    embedding_provider = SentenceTransformerEmbedding()
    rm = LRRewardModelV3.load(args.rm_model, embedding_provider=embedding_provider)
    rm.warm_cache(val_examples)

    rm_metrics = rm.evaluate(val_examples)
    print("\n1. RM accuracy (sql_good vs sql_bad):")
    rm.publish_metrics(rm_metrics)

    agent = SqlAgent(llm=OllamaLLM(model=args.ollama_model), rm=rm, n_candidates=args.n_candidates)
    agent_metrics = evaluate_agent(val_examples, agent)

    sp = agent_metrics["sql_pass_rate"]
    print("\n2. SQL execution pass/fail:")
    print(f"  {sp['passed']}/{sp['total']} passed" + (f" ({sp['rate']:.1%})" if sp["rate"] is not None else ""))

    qa = agent_metrics["qa_accuracy"]
    print("\n3. End-to-end QA accuracy:")
    print(f"  {qa['correct']}/{qa['total']} correct" + (f" ({qa['rate']:.1%})" if qa["rate"] is not None else ""))

    summary = {"rm_metrics": rm_metrics, **agent_metrics}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.output}")


if __name__ == "__main__":
    main()
