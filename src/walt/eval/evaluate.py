"""Agent-level evaluation on the held-out val split (rows with split="val", produced by
pre_process.py and never seen by RM training/CV — see rm/data/pre_process.py).

Reports three things:
  1. RM accuracy selecting correct SQL among sql_good vs sql_bad (BaseRewardModel.evaluate()).
  2. SQL execution pass/fail and 3. end-to-end QA accuracy, each reported *with* RM
     reranking (the agent's actual top-ranked pick) and *without* it (the first LLM
     candidate, i.e. what a single-shot call with no RM would have produced) — using the
     same generated candidates for both, so this isolates the RM's contribution without
     doubling the (expensive) LLM calls.

Usage:
    python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from walt.agent.sql_agent import SqlAgent, build_llm
from walt.rm.data.base import Example
from walt.rm.model.base import load_examples
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.lr_model_v3 import LRRewardModelV3
from walt.rm.model.tracking import log_run
from walt.utils.sql_exec import ExecutionResult, run_sql


def _rows_match(a: ExecutionResult, b: ExecutionResult) -> bool:
    if a.columns != b.columns:
        return False
    return sorted(a.rows or ()) == sorted(b.rows or ())


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "rate": count / total if total else None}


def evaluate_agent(val_examples: list[Example], agent: SqlAgent) -> dict[str, Any]:
    executable = [ex for ex in val_examples if ex.sql_context]
    qa_examples = [ex for ex in executable if ex.sql_context_valid]

    rm_pass = base_pass = 0
    rm_qa = base_qa = 0

    print(f"Running agent over {len(executable)} executable val rows...")
    for i, ex in enumerate(executable, 1):
        try:
            result = agent.run(ex.question, list(ex.sql_context))
        except RuntimeError as exc:
            # The LLM produced zero usable candidates for this row (can happen with a
            # small n_candidates against a reasoning model that burns its token budget
            # on thinking) — count it as a failed row rather than aborting the whole
            # evaluation run.
            print(f"  WARNING: {exc}")
            continue

        # With RM reranking: the agent's actual top-ranked pick (already executed).
        rm_execution = result.execution
        # Without RM: the first LLM candidate, i.e. what a single-shot call with no
        # reranking would have produced. Reuses the candidates already generated for
        # this row instead of calling the LLM again.
        baseline_sql = result.raw_candidates[0]
        base_execution = (
            rm_execution if baseline_sql == result.best_sql else run_sql(ex.sql_context, baseline_sql)
        )

        rm_pass += int(rm_execution.success)
        base_pass += int(base_execution.success)

        if ex.sql_context_valid:
            reference = run_sql(ex.sql_context, ex.sql_good)
            rm_qa += int(_rows_match(rm_execution, reference))
            base_qa += int(_rows_match(base_execution, reference))

        if i % 10 == 0 or i == len(executable):
            print(
                f"  [{i}/{len(executable)}] pass so far — with RM: {rm_pass}/{i} "
                f"({100 * rm_pass / i:.1f}%), without RM: {base_pass}/{i} ({100 * base_pass / i:.1f}%)",
                flush=True,
            )

    return {
        "sql_pass_rate": {
            "with_rm": _rate(rm_pass, len(executable)),
            "without_rm": _rate(base_pass, len(executable)),
        },
        "qa_accuracy": {
            "with_rm": _rate(rm_qa, len(qa_examples)),
            "without_rm": _rate(base_qa, len(qa_examples)),
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
    parser.add_argument("--llm-cache", type=Path, default=Path("data/output/llm_cache.json"), help="Cache LLM candidates here, keyed by (question, schema_context) — so re-running with a different --rm-model reuses candidates instead of re-calling the LLM")
    parser.add_argument("--no-llm-cache", action="store_true", help="Always call the LLM fresh, ignoring/skipping the cache")
    parser.add_argument("--run-name", default=None, help="Label for this run in the run log (default: derived from --rm-model and --ollama-model)")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/eval_runs"), help="Directory where agent-eval run records are logged for later comparison (separate from RM training's data/output/runs/ — different metric shape)")
    parser.add_argument("--no-log-run", action="store_true", help="Skip writing a run record (e.g. for throwaway/debug runs)")
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

    llm_cache_path = None if args.no_llm_cache else args.llm_cache
    llm = build_llm(args.ollama_model, llm_cache_path)
    agent = SqlAgent(llm=llm, rm=rm, n_candidates=args.n_candidates)
    agent_metrics = evaluate_agent(val_examples, agent)

    def _print_line(label: str, stats: dict[str, Any]) -> None:
        rate = f" ({stats['rate']:.1%})" if stats["rate"] is not None else ""
        print(f"  {label}: {stats['count']}/{stats['total']}{rate}")

    sp = agent_metrics["sql_pass_rate"]
    print("\n2. SQL execution pass/fail (RM reranking vs no reranking):")
    _print_line("with RM   ", sp["with_rm"])
    _print_line("without RM", sp["without_rm"])

    qa = agent_metrics["qa_accuracy"]
    print("\n3. End-to-end QA accuracy (RM reranking vs no reranking):")
    _print_line("with RM   ", qa["with_rm"])
    _print_line("without RM", qa["without_rm"])

    summary = {"rm_metrics": rm_metrics, **agent_metrics}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.output}")

    if not args.no_log_run:
        run_name = args.run_name or f"{args.rm_model.stem}_{args.ollama_model}"
        run_path = log_run(
            args.runs_dir,
            run_name=run_name,
            model_class=f"{type(rm).__name__}+{args.ollama_model}",
            config={
                "input": str(args.input),
                "rm_model": str(args.rm_model),
                "ollama_model": args.ollama_model,
                "n_candidates": args.n_candidates,
                "n_val_examples": len(val_examples),
            },
            # Flat so a future comparison table/chart can key straight into it, same as
            # rm/model/visualize.py does for top1_accuracy/pairwise_accuracy/mrr.
            metrics={
                "rm_top1_accuracy": rm_metrics["top1_accuracy"],
                "rm_pairwise_accuracy": rm_metrics["pairwise_accuracy"],
                "rm_mrr": rm_metrics["mrr"],
                "sql_pass_rate_with_rm": sp["with_rm"]["rate"],
                "sql_pass_rate_without_rm": sp["without_rm"]["rate"],
                "qa_accuracy_with_rm": qa["with_rm"]["rate"],
                "qa_accuracy_without_rm": qa["without_rm"]["rate"],
            },
            training={
                "n_executable": sp["with_rm"]["total"],
                "n_qa_examples": qa["with_rm"]["total"],
                "sql_pass_rate": agent_metrics["sql_pass_rate"],
                "qa_accuracy": agent_metrics["qa_accuracy"],
            },
        )
        print(f"Logged run to {run_path}")


if __name__ == "__main__":
    main()
