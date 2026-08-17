"""Agent-level evaluation on the held-out val split (rows with split="val", produced by
pre_process.py and never seen by RM training/CV — see rm/data/pre_process.py).

Reports four things:
  1. RM accuracy selecting correct SQL among sql_good vs sql_bad (BaseRewardModel.evaluate()).
  2. SQL execution pass/fail and 3. end-to-end QA accuracy, each reported *with* RM
     reranking (the agent's actual top-ranked pick) and *without* it (the first LLM
     candidate, i.e. what a single-shot call with no RM would have produced) — using the
     same generated candidates for both, so this isolates the RM's contribution without
     doubling the (expensive) LLM calls.
  4. Oracle ceiling: of the n candidates the LLM generated for each question (before any
     reranking), how many are actually correct? Bucketed into all_correct (any pick
     wins), zero_correct (unreachable by any reranker — an LLM generation-quality
     ceiling, not an RM problem), and mixed (selection actually matters) — with the
     mixed bucket's with-RM/without-RM achieved rates compared against the random-chance
     expectation, so a low "achieved" number can be told apart from "there was nothing
     to achieve." Answers "how much room does reranking have to grow" independent of
     whether the RM specifically is any good.

Usage:
    python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from walt.agent.sql_agent import SqlAgent, build_llm
from walt.rm.data.base import Example
from walt.rm.model.base import load_examples
from walt.rm.model.constant_model import ConstantRewardModel
from walt.rm.model.distilbert_model import DistilBertRewardModel
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.lr.lr_model_v3 import LRRewardModelV3
from walt.rm.model.lr.lr_model_v4 import LRRewardModelV4
from walt.rm.model.lr.lr_model_v6 import LRRewardModelV6
from walt.rm.model.lr.lr_model_v7 import LRRewardModelV7
from walt.rm.model.schema_filter import SchemaFilteredRewardModel
from walt.rm.model.tracking import log_run

# "constant" needs no --rm-model file (ConstantRewardModel has no trainable state) —
# combine with --schema-filter to test "first schema-valid LLM candidate, no learned
# reranking at all" as a baseline against the real RM classes.
RM_CLASS_CHOICES = ["lr_v3", "lr_v4", "lr_v6", "lr_v7", "distilbert", "constant"]
RM_CLASS_BY_NAME = {"lr_v3": LRRewardModelV3, "lr_v4": LRRewardModelV4, "lr_v6": LRRewardModelV6, "lr_v7": LRRewardModelV7}
from walt.utils.sql_exec import (
    ExecutionResult,
    capture_db_state,
    execute_with_context,
    resolve_context_statements,
)


def _row_sort_key(row: tuple) -> tuple:
    # sorted() on raw SQL rows can raise TypeError if a column mixes None with a
    # comparable type across rows (e.g. one row's value is None, another's is a float,
    # at the same position) — None doesn't order against non-None in Python 3. Encode
    # "is this None" as a leading bool per value so every row sorts against every other
    # without ever comparing None to a non-None value directly.
    return tuple((v is None, v) if v is not None else (True, "") for v in row)


def _rows_match(a: ExecutionResult, b: ExecutionResult) -> bool | None:
    # Deliberately ignores a.columns/b.columns: SQLite auto-labels a result column from
    # the expression text itself when there's no explicit alias (e.g. `AVG(Age)` vs
    # `AVG(T1.Age)` for a table-qualified rewrite of the same column) — column *names*
    # are cosmetic, not part of whether the agent got the right answer. Only the actual
    # row values (and their count/order within a row) are compared.
    #
    # A failed execution (rows=None, success=False) is never a match, regardless of what
    # the other side looks like — it must NOT be treated as "returned zero rows", or any
    # candidate that crashes gets silently credited as correct on every question whose
    # real answer happens to be empty. Returns None (rather than True/False) when both
    # sides succeeded but neither has a row set at all — non-SELECT statements
    # (UPDATE/DELETE/INSERT/DDL) always report rows=None on success, so "both None" isn't
    # evidence of a match either; the caller must fall back to a different comparison
    # (see _effect_match) for that case instead of defaulting to "equal".
    if not a.success or not b.success:
        return False
    if a.rows is None and b.rows is None:
        return None
    if a.rows is None or b.rows is None:
        return False
    return sorted(a.rows, key=_row_sort_key) == sorted(b.rows, key=_row_sort_key)


MAX_SAMPLE_ROWS = 20  # cap on rows embedded per execution in a --sample-log entry


def _execution_detail(execution: ExecutionResult) -> dict[str, Any]:
    if execution.rows is None:
        return {"success": execution.success, "rows": None, "n_rows": None, "error": execution.error}
    return {
        "success": execution.success,
        "rows": [list(r) for r in execution.rows[:MAX_SAMPLE_ROWS]],
        "n_rows": len(execution.rows),
        "error": execution.error,
    }


def _effect_match(context_statements: Sequence[str], sql_a: str, sql_b: str) -> bool:
    """Fallback for _rows_match's None case (both statements are non-SELECT, so neither
    has a row set to compare): re-runs context_statements + each statement from scratch
    via capture_db_state and compares the full resulting database state (every table's
    rows) instead. Each side gets its own fresh in-memory DB, so sql_a's effect can never
    leak into sql_b's run."""
    ok_a, state_a = capture_db_state(context_statements, sql_a)
    ok_b, state_b = capture_db_state(context_statements, sql_b)
    return ok_a and ok_b and state_a == state_b


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "rate": count / total if total else None}


def evaluate_agent(
    val_examples: list[Example], agent: SqlAgent, sample_n: int = 0, sample_seed: int = 42
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Returns (metrics, samples) -- samples is a list of per-row detail dicts (question,
    expected vs. actual SQL/results, both with and without RM reranking) for `sample_n`
    rows chosen uniformly at random (seeded by `sample_seed`, so a repeat run with the
    same inputs picks the same rows) from the executable set, for human review. Empty
    when sample_n is 0 (the default) -- capturing costs nothing extra since every value
    in a sample dict is already computed for the aggregate metrics either way."""
    executable = [ex for ex in val_examples if ex.sql_context or ex.sql_context_path]
    qa_examples = [ex for ex in executable if ex.sql_context_valid]
    sample_indices = set(random.Random(sample_seed).sample(range(len(executable)), min(sample_n, len(executable))))
    samples: list[dict[str, Any]] = []

    # Rows that carry sql_context_path (see walt.utils.sql_exec.execute_with_context)
    # reuse a cached in-memory connection across calls that share the same path — sorting
    # groups those consecutive so the cache stays warm instead of rebuilding per row. Rows
    # without a path (gretel/spider/dbasql — full sql_context embedded directly) sort
    # together too, but gain nothing from it since they were never cached to begin with.
    executable = sorted(executable, key=lambda ex: ex.sql_context_path or "")

    rm_pass = base_pass = 0
    rm_qa = base_qa = 0

    # Oracle ceiling: of the n candidates the LLM generated (before any reranking), how
    # many are actually correct? This bounds how much reranking — RM or otherwise —
    # could ever achieve. Bucketed per row: all_correct (any pick wins, nothing to
    # rerank), zero_correct (unreachable — a pure LLM generation-quality ceiling, not an
    # RM problem), mixed (0 < n_correct < n — the only bucket where selection quality
    # actually matters). Within the mixed bucket, "achieved" (with/without RM) is
    # compared against the random-chance expectation (weighted by n_correct/n per row)
    # to tell whether reranking is doing better than a coin flip.
    n_all_correct = n_zero_correct = n_mixed = 0
    mixed_rm_correct = mixed_base_correct = 0
    mixed_expected_random = 0.0

    print(f"Running agent over {len(executable)} executable val rows...")
    progress_interval = 10  # a status line every N rows -- each involves n_candidates
    # real LLM calls (Ollama or Claude) plus SQL execution, slow enough that silent
    # per-row progress can look stuck on a large val set.
    start = time.perf_counter()
    for i, ex in enumerate(executable, 1):
        if i % progress_interval == 0 or i == len(executable):
            elapsed = time.perf_counter() - start
            rate_per_min = i / elapsed * 60 if elapsed > 0 else 0.0
            eta_min = (len(executable) - i) / rate_per_min if rate_per_min > 0 else float("inf")
            print(
                f"  {i}/{len(executable)} rows ({100 * i / len(executable):.1f}%) | "
                f"{rate_per_min:.1f} rows/min | elapsed {elapsed / 60:.1f}min | ETA ~{eta_min:.0f}min"
            )
        try:
            result = agent.run(
                ex.question,
                list(ex.sql_context),
                sql_context_clean=list(ex.sql_context_clean),
                sql_context_path=ex.sql_context_path,
            )
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
            rm_execution
            if baseline_sql == result.best_sql
            else execute_with_context(ex.sql_context, ex.sql_context_path, baseline_sql)
        )

        rm_pass += int(rm_execution.success)
        base_pass += int(base_execution.success)

        if ex.sql_context_valid:
            reference = execute_with_context(ex.sql_context, ex.sql_context_path, ex.sql_good)

            # Lazily resolved: only needed when _rows_match can't decide from row sets
            # alone (both sides non-SELECT) — never touched for a purely-SELECT dataset
            # like Spider, so this costs nothing there.
            _resolved_context_cache: list[tuple[str, ...] | None] = [None]

            def _resolved_context() -> tuple[str, ...]:
                if _resolved_context_cache[0] is None:
                    _resolved_context_cache[0] = resolve_context_statements(ex.sql_context, ex.sql_context_path)
                return _resolved_context_cache[0]

            def _qa_match(execution: ExecutionResult, sql: str) -> bool:
                match = _rows_match(execution, reference)
                if match is not None:
                    return match
                return _effect_match(_resolved_context(), sql, ex.sql_good)

            rm_correct = _qa_match(rm_execution, result.best_sql)
            base_correct = _qa_match(base_execution, baseline_sql)
            rm_qa += int(rm_correct)
            base_qa += int(base_correct)

            # Reuse the two executions already computed above instead of re-running
            # identical SQL — result.best_sql/baseline_sql are always among raw_candidates.
            executions_by_sql = {result.best_sql: rm_execution, baseline_sql: base_execution}
            n_correct = 0
            for sql in result.raw_candidates:
                execution = executions_by_sql.setdefault(
                    sql, execute_with_context(ex.sql_context, ex.sql_context_path, sql)
                )
                n_correct += int(_qa_match(execution, sql))

            n_candidates = len(result.raw_candidates)
            if n_correct == n_candidates:
                n_all_correct += 1
            elif n_correct == 0:
                n_zero_correct += 1
            else:
                n_mixed += 1
                mixed_rm_correct += int(rm_correct)
                mixed_base_correct += int(base_correct)
                mixed_expected_random += n_correct / n_candidates

        if (i - 1) in sample_indices:
            if ex.sql_context_valid:
                oracle_bucket = "all_correct" if n_correct == n_candidates else "zero_correct" if n_correct == 0 else "mixed"
                reference_detail = _execution_detail(reference)
            else:
                rm_correct = base_correct = oracle_bucket = None
                reference_detail = None
            samples.append(
                {
                    "index": i,
                    "question": ex.question,
                    "schema": list(ex.sql_context_clean),
                    "sql_good": ex.sql_good,
                    "context_valid": ex.sql_context_valid,
                    "reference": reference_detail,
                    "with_rm": {"sql": result.best_sql, **_execution_detail(rm_execution), "correct": rm_correct},
                    "without_rm": {"sql": baseline_sql, **_execution_detail(base_execution), "correct": base_correct},
                    "oracle_bucket": oracle_bucket,
                }
            )

        if i % 10 == 0 or i == len(executable):
            print(
                f"  [{i}/{len(executable)}] pass so far — with RM: {rm_pass}/{i} "
                f"({100 * rm_pass / i:.1f}%), without RM: {base_pass}/{i} ({100 * base_pass / i:.1f}%)",
                flush=True,
            )

    metrics = {
        "sql_pass_rate": {
            "with_rm": _rate(rm_pass, len(executable)),
            "without_rm": _rate(base_pass, len(executable)),
        },
        "qa_accuracy": {
            "with_rm": _rate(rm_qa, len(qa_examples)),
            "without_rm": _rate(base_qa, len(qa_examples)),
        },
        "oracle": {
            # "ceiling" = fraction of rows where at least one candidate is correct —
            # the best any reranker (RM or otherwise) could ever achieve on this data.
            "ceiling": _rate(n_all_correct + n_mixed, len(qa_examples)),
            "all_candidates_correct": _rate(n_all_correct, len(qa_examples)),
            "zero_candidates_correct": _rate(n_zero_correct, len(qa_examples)),
            "mixed": _rate(n_mixed, len(qa_examples)),
            "mixed_achieved_with_rm": _rate(mixed_rm_correct, n_mixed),
            "mixed_achieved_without_rm": _rate(mixed_base_correct, n_mixed),
            "mixed_expected_random": {
                "expected_correct": round(mixed_expected_random, 2),
                "total": n_mixed,
                "rate": mixed_expected_random / n_mixed if n_mixed else None,
            },
        },
    }
    return metrics, samples


def _append_rows_block(lines: list[str], detail: dict[str, Any], label: str = "Rows") -> None:
    if not detail["success"]:
        lines.append(f"❌ execution error: `{detail['error']}`")
        return
    if detail["rows"] is None:
        lines.append("_(non-SELECT statement — no row set)_")
        return
    if not detail["rows"]:
        lines.append("_(0 rows)_")
        return
    lines.append(f"{label}:")
    lines.append("```")
    lines.extend(repr(tuple(r)) for r in detail["rows"])
    if detail["n_rows"] > len(detail["rows"]):
        lines.append(f"... and {detail['n_rows'] - len(detail['rows'])} more row(s)")
    lines.append("```")


def _append_execution_section(lines: list[str], title: str, sql: str, detail: dict[str, Any], correct: bool | None) -> None:
    lines.append(f"**{title}:**")
    lines.append("```sql")
    lines.append(sql)
    lines.append("```")
    _append_rows_block(lines, detail)
    if correct is not None:
        lines.append(f"QA correct: {'✅ yes' if correct else '❌ no'}")
    lines.append("")


def write_samples_md(samples: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    """Human-readable Markdown log of --sample-log's randomly sampled rows: question,
    schema, expected (sql_good) vs. actual SQL/results with and without RM reranking,
    and the oracle bucket that row falls into. Row content (execution success/rows/
    error) is exactly what evaluate_agent() already computed for the aggregate
    metrics — this just renders a subset of it for human review instead of collapsing
    it into counts."""
    lines = [
        "# Eval sample log",
        "",
        f"- input: `{args.input}`",
        f"- rm-model: `{args.rm_model}` (`{args.rm_class}`, schema-filter={args.schema_filter})",
        f"- {len(samples)} row(s) sampled (seed={args.sample_seed})",
        "",
    ]
    for s in samples:
        lines.append(f"## Row {s['index']}: {s['question']}")
        lines.append("")
        lines.append("**Schema:**")
        lines.append("```sql")
        lines.append("\n".join(s["schema"]))
        lines.append("```")
        lines.append("")
        lines.append("**Expected (`sql_good`):**")
        lines.append("```sql")
        lines.append(s["sql_good"])
        lines.append("```")
        if s["reference"] is None:
            lines.append("_expected rows not verified (`sql_context_valid=False` for this row)_")
        else:
            _append_rows_block(lines, s["reference"], label="Expected rows")
        lines.append("")

        _append_execution_section(lines, "With RM reranking", s["with_rm"]["sql"], s["with_rm"], s["with_rm"]["correct"])
        _append_execution_section(lines, "Without RM (first candidate)", s["without_rm"]["sql"], s["without_rm"], s["without_rm"]["correct"])

        lines.append(f"Oracle bucket: `{s['oracle_bucket'] or 'n/a'}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    # Stdout is fully buffered by default when redirected to a file (e.g. a
    # backgrounded run), so evaluate_agent's progress prints wouldn't be visible until
    # the process exits without this — see build_severity_dataset.py's main().
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/rm_enhanced.jsonl"))
    parser.add_argument("--rm-model", type=Path, default=Path("data/output/rm_model.joblib"))
    parser.add_argument("--rm-class", choices=RM_CLASS_CHOICES, default="lr_v6", help="Which BaseRewardModel subclass --rm-model was saved from")
    parser.add_argument("--llm-backend", choices=["ollama", "claude"], default="ollama", help="Which LLM generates candidate SQL: local Ollama (default) or the Anthropic API (Claude)")
    parser.add_argument("--ollama-model", default="llama3.2", help="Model name when --llm-backend ollama")
    parser.add_argument("--claude-model", default="claude-haiku-4-5-20251001", help="Model name when --llm-backend claude (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--ollama-concurrency", type=int, default=1, help="[--llm-backend ollama only] concurrent Ollama requests per row's candidate generation. Only helps once the Ollama server itself accepts concurrent requests (see build_severity_dataset.py docstring) — otherwise pure overhead.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N val rows (smoke testing)")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON summary")
    parser.add_argument("--llm-cache", type=Path, default=None, help="Cache LLM candidates here, keyed by (question, schema_context) — so re-running with a different --rm-model reuses candidates instead of re-calling the LLM. Defaults to data/output/llm_cache.json for --llm-backend ollama, data/output/llm_cache_claude.json for --llm-backend claude — separate files so switching backends never overwrites the other's cached candidates")
    parser.add_argument("--no-llm-cache", action="store_true", help="Always call the LLM fresh, ignoring/skipping the cache")
    parser.add_argument("--strip-context", action="store_true", help="Don't show the LLM schema_context when generating candidates (still used for execution/reference) — tests unconditioned generation")
    parser.add_argument("--run-name", default=None, help="Label for this run in the run log (default: derived from --rm-model and the LLM model)")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/eval_runs"), help="Directory where agent-eval run records are logged for later comparison (separate from RM training's data/output/runs/ — different metric shape)")
    parser.add_argument("--no-log-run", action="store_true", help="Skip writing a run record (e.g. for throwaway/debug runs)")
    parser.add_argument("--schema-filter", action="store_true", help="Wrap --rm-model in SchemaFilteredRewardModel — a hard is_schema_valid pre-filter applied before the RM's own scoring (see schema_filter.py). No-op for lr_v6 (already learns this internally); meant for models with no equivalent feature, e.g. distilbert")
    parser.add_argument("--sample-log", type=int, default=0, help="Write a human-readable Markdown log of N randomly sampled rows (question, expected vs. actual SQL/results, with and without RM reranking) — see --sample-log-output. 0 (default) disables this.")
    parser.add_argument("--sample-log-output", type=Path, default=None, help="Where to write the --sample-log Markdown file. Defaults to --output with its suffix replaced by _samples.md, or data/output/eval_samples.md if --output isn't given.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for --sample-log's row selection — same seed + same input file (same executable row order) always picks the same rows, even across separate --rm-model runs -- lets a caller merge two runs' samples by row index (see scripts/evaluate_best.sh's 3-way with-RM/plain-filter/no-rerank merge)")
    parser.add_argument("--samples-json", type=Path, default=None, help="[--sample-log only] also dump the raw sampled-row list as JSON here (machine-readable; --sample-log-output's Markdown is for humans) -- e.g. so a wrapper script can merge two separate runs' samples (matched by row index) into one comparison")
    args = parser.parse_args()

    examples = load_examples(args.input)
    val_examples = [ex for ex in examples if ex.split == "val"]
    if args.limit:
        val_examples = val_examples[: args.limit]
    print(f"Evaluating on {len(val_examples)} val examples")

    if args.rm_class == "constant":
        rm = ConstantRewardModel()
    elif args.rm_class == "distilbert":
        rm = DistilBertRewardModel.load(args.rm_model)
    else:
        embedding_provider = SentenceTransformerEmbedding()
        rm = RM_CLASS_BY_NAME[args.rm_class].load(args.rm_model, embedding_provider=embedding_provider)
    if args.schema_filter:
        rm = SchemaFilteredRewardModel(rm)
    rm.warm_cache(val_examples)

    rm_metrics = rm.evaluate(val_examples)
    print("\n1. RM accuracy (sql_good vs sql_bad):")
    rm.publish_metrics(rm_metrics)

    llm_model = args.claude_model if args.llm_backend == "claude" else args.ollama_model
    if args.no_llm_cache:
        llm_cache_path = None
    elif args.llm_cache is not None:
        llm_cache_path = args.llm_cache
    else:
        default_cache_name = "llm_cache_claude.json" if args.llm_backend == "claude" else "llm_cache.json"
        llm_cache_path = Path("data/output") / default_cache_name
    llm = build_llm(llm_model, llm_cache_path, backend=args.llm_backend, ollama_concurrency=args.ollama_concurrency)
    agent = SqlAgent(llm=llm, rm=rm, n_candidates=args.n_candidates, strip_llm_context=args.strip_context)
    agent_metrics, samples = evaluate_agent(val_examples, agent, sample_n=args.sample_log, sample_seed=args.sample_seed)

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

    oracle = agent_metrics["oracle"]
    print("\n4. Oracle ceiling (how many of the n generated candidates are even correct — how much room reranking has to grow):")
    _print_line("any candidate correct (ceiling)", oracle["ceiling"])
    _print_line("all candidates correct         ", oracle["all_candidates_correct"])
    _print_line("zero candidates correct        ", oracle["zero_candidates_correct"])
    _print_line("mixed (selection matters)      ", oracle["mixed"])
    if oracle["mixed"]["count"]:
        exp = oracle["mixed_expected_random"]
        print("  within the mixed bucket:")
        _print_line("    achieved with RM   ", oracle["mixed_achieved_with_rm"])
        _print_line("    achieved without RM", oracle["mixed_achieved_without_rm"])
        rate = f" ({exp['rate']:.1%})" if exp["rate"] is not None else ""
        print(f"    expected by random pick: {exp['expected_correct']:.1f}/{exp['total']}{rate}")

    summary = {"rm_metrics": rm_metrics, **agent_metrics}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.output}")

    if args.sample_log:
        sample_log_output = args.sample_log_output or (
            args.output.with_name(f"{args.output.stem}_samples.md") if args.output else Path("data/output/eval_samples.md")
        )
        write_samples_md(samples, sample_log_output, args)
        print(f"Wrote {len(samples)} sample row(s) to {sample_log_output}")
        if args.samples_json:
            args.samples_json.parent.mkdir(parents=True, exist_ok=True)
            args.samples_json.write_text(json.dumps(samples, indent=2))
            print(f"Wrote raw sample data to {args.samples_json}")

    if not args.no_log_run:
        run_name = args.run_name or f"{args.rm_model.stem}_{args.llm_backend}_{llm_model}"
        run_path = log_run(
            args.runs_dir,
            run_name=run_name,
            model_class=f"{type(rm).__name__}+{args.llm_backend}:{llm_model}",
            config={
                "input": str(args.input),
                "rm_model": str(args.rm_model),
                "rm_class": args.rm_class,
                "schema_filter": args.schema_filter,
                "llm_backend": args.llm_backend,
                "llm_model": llm_model,
                "n_candidates": args.n_candidates,
                "n_val_examples": len(val_examples),
                "strip_context": args.strip_context,
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
                "oracle_ceiling": oracle["ceiling"]["rate"],
                "oracle_all_correct_rate": oracle["all_candidates_correct"]["rate"],
                "oracle_zero_correct_rate": oracle["zero_candidates_correct"]["rate"],
                "oracle_mixed_rate": oracle["mixed"]["rate"],
                "oracle_mixed_achieved_with_rm": oracle["mixed_achieved_with_rm"]["rate"],
                "oracle_mixed_achieved_without_rm": oracle["mixed_achieved_without_rm"]["rate"],
                "oracle_mixed_expected_random_rate": oracle["mixed_expected_random"]["rate"],
            },
            training={
                "n_executable": sp["with_rm"]["total"],
                "n_qa_examples": qa["with_rm"]["total"],
                "sql_pass_rate": agent_metrics["sql_pass_rate"],
                "qa_accuracy": agent_metrics["qa_accuracy"],
                "oracle": oracle,
            },
        )
        print(f"Logged run to {run_path}")


if __name__ == "__main__":
    main()
