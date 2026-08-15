# RM-Powered SQL Agent

## Context

`walt` currently ends at a trained reward model (`LRRewardModelV3`, saved to
`data/output/rm_model.joblib`) that can score/rank candidate SQL strings against a
question. `src/walt/agent/` and `src/walt/eval/` are empty placeholder packages with no
implementation. There is no SQL-generation LLM, no toy-database execution path, and the
RM training data (`rm_enhanced.jsonl`) has no schema/data context to actually run SQL
against.

This plan builds the missing piece: a `SqlAgent` that generates N candidate SQL queries
with a local Ollama LLM, scores them with the existing RM, executes the best one against
an in-memory SQLite database, and returns a structured result. It also extends
`gen_training_data.py` so each training row carries the `CREATE TABLE`/`INSERT INTO`
context needed to actually execute `sql_good` — reusing the same SQLite executor the
agent uses, so the data-gen script can locally verify the context it synthesizes and
report how much of the generated data is actually trustworthy before a training run is
sunk into it.

Confirmed with you across this planning session:
- LLM backend is **Ollama running `llama3.2` locally** (not ONNX — initial mention
  corrected). Ollama's API has no beam-search/multi-return parameter, so candidate
  generation is 5 separate calls with varied temperature/seed (see §2) — the `BaseLLM`
  abstraction is kept backend-agnostic so a future HF-transformers backend with true
  `num_beams` beam search is a drop-in swap for comparison later.
- The dataset gets a held-out **val** split the RM never trains or cross-validates on;
  a new `walt.eval.evaluate` script scores the *agent* (not just the RM) end-to-end on
  that val split.
- The shared SQLite executor lives in a new `walt.utils` package, not at the top level.
- `gen_training_data.py` prints an aggregate pass-rate summary — how many `sql_good` rows
  actually executed successfully against their generated `sql_context` — so data quality
  can be checked before committing to a full training run.
- `BAD_SQL_REASONS` is replaced with four new categories: missing filters, wrong
  aggregation, unsafe patterns (`SELECT *`, cross joins), misjoined tables.

## New/modified files

```
src/walt/
├── utils/
│   ├── __init__.py                 NEW  (new package)
│   └── sql_exec.py                 NEW  shared in-memory SQLite executor
├── agent/
│   ├── __init__.py                 (exists, empty)
│   ├── llm/
│   │   ├── __init__.py             NEW
│   │   ├── base.py                 NEW  BaseLLM ABC
│   │   └── ollama_llm.py           NEW  OllamaLLM(BaseLLM)
│   └── sql_agent.py                NEW  SqlAgent orchestrator + AgentResult + CLI
├── eval/
│   ├── __init__.py                 (exists, empty)
│   └── evaluate.py                 NEW  agent-level eval over the val split
└── rm/
    ├── data/
    │   ├── base.py                 MODIFY  Example: + sql_context, sql_context_valid, split
    │   ├── pre_process.py          MODIFY  stamp each row with split: "trainval" | "val"
    │   └── gen_training_data.py    MODIFY  sql_context gen + verify + summary + new reasons
    └── model/base.py               (unchanged — group_split/k_fold_split keep working exactly
                                      as today, just always called on the "trainval" pool)
```

`pyproject.toml`: add `ollama` (official Python client for the local Ollama HTTP API).
No ONNX/optimum/onnxruntime deps needed. Operational prerequisite (outside repo scope):
Ollama installed and running locally with `ollama pull llama3.2`.

## 1. Shared SQL executor — `src/walt/utils/sql_exec.py`

```python
@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    columns: tuple[str, ...] | None
    rows: tuple[tuple, ...] | None
    error: str | None

def run_sql(context_statements: Sequence[str], sql: str, timeout: float = 5.0) -> ExecutionResult:
    ...
```

Opens a fresh `sqlite3.connect(":memory:", timeout=timeout)`, executes each context
statement then `sql`, captures `cursor.description`/`fetchall()` when present (SELECT-like
statements) else `columns=rows=None` (DDL/DML), catches `sqlite3.Error` into
`ExecutionResult(success=False, error=str(e))`, always closes the connection. Lives in the
new `walt.utils` package (rather than top-level `walt.sql_exec` or inside `rm/`) since it's
shared, general-purpose infra used by both `gen_training_data.py` (context verification)
and `sql_agent.py` (executing the RM's chosen candidate) — one execution path, no
duplicated SQLite handling, and a natural home for other cross-cutting helpers later.

## 2. LLM abstraction — `src/walt/agent/llm/`

`base.py`:
```python
class BaseLLM(ABC):
    @abstractmethod
    def generate_candidates(self, question: str, schema_context: str, n: int = 5) -> list[str]:
        """Return up to n candidate SQL strings for question, given CREATE TABLE schema_context."""
```

`ollama_llm.py` — `OllamaLLM(BaseLLM)`:
- `__init__(self, model: str = "llama3.2", temperature: float = 0.8, max_tokens: int = 256)`
- Uses the `ollama` package's `chat()` with a system message ("You are a text-to-SQL
  assistant. Given a SQLite schema and a question, output ONLY one valid SQL query — no
  explanation, no markdown fences.") + user message containing `schema_context` and
  `question`.
- `generate_candidates` loops `n` times, one `ollama.chat()` call per candidate with
  `options={"temperature": temperature, "seed": i}` for diversity, strips markdown
  fences/prose from each response, keeps non-empty results.
- Kept behind `BaseLLM` so a future `AnthropicLLM` or a different Ollama model is a
  drop-in swap — nothing else in the agent depends on the concrete class.

**On beam search**: confirmed Ollama's `options` schema (temperature/top_p/top_k/seed/
mirostat/num_predict/...) has no `num_beams`/`n`/`best_of` parameter — that's a
llama.cpp-server-API limitation Ollama inherits, not something `walt` can configure
around. `OllamaLLM.generate_candidates` therefore makes 5 separate calls with varied
temperature/seed rather than one beam-search call. `BaseLLM`'s signature
(`generate_candidates(question, schema_context, n) -> list[str]`) is deliberately
backend-agnostic (no Ollama-specific params leak into it), so a future
`HFTransformersLLM(BaseLLM)` using `transformers.generate(num_beams=n,
num_return_sequences=n)` for true single-call beam search is a drop-in swap for
comparing the two approaches later — no changes needed to `SqlAgent`, `evaluate.py`, or
the CLI.

## 3. Reward model — reuse as-is

No new RM code. `SqlAgent` takes any `walt.rm.model.base.BaseRewardModel`; construct it
by loading the existing baseline:
```python
model = LRRewardModelV3.load("data/output/rm_model.joblib")
```
`rank(question, candidates) -> list[ScoredCandidate]` already exists on the base class and
is exactly "score a list of candidate SQL strings against a question" — the agent calls it
directly rather than looping `score()` itself.

## 4. Agent orchestrator — `src/walt/agent/sql_agent.py`

```python
@dataclass(frozen=True)
class AgentResult:
    question: str
    scored_candidates: list[ScoredCandidate]   # from rm.rank() — sql, score, rank, error_code
    best_sql: str
    execution: ExecutionResult
    final_answer: str | None                    # simple formatted rows, or "Query failed: ..."
    critique: str | None = None                  # placeholder — always None for now

class SqlAgent:
    def __init__(self, llm: BaseLLM, rm: BaseRewardModel, n_candidates: int = 5): ...

    def run(self, question: str, schema_context: list[str]) -> AgentResult:
        candidates = self.llm.generate_candidates(question, "\n".join(schema_context), self.n_candidates)
        scored = self.rm.rank(question, candidates)
        best = scored[0]
        execution = run_sql(schema_context, best.sql)
        final_answer = _format_answer(execution)
        return AgentResult(question, scored, best.sql, execution, final_answer, critique=None)
```

`run_sql` here is `walt.utils.sql_exec.run_sql`.

CLI (`python -m walt.agent.sql_agent`): `--input <jsonl with sql_context> --index N`
(pulls `question`/`sql_context` from a generated row) or `--question`/`--schema-file` for
ad hoc use, plus `--rm-model` (default `data/output/rm_model.joblib`), `--ollama-model`
(default `llama3.2`), `--n-candidates` (default 5). Prints `AgentResult` as JSON. The core
`run_agent(...)` logic stays importable separately from the CLI's `main()` so
`walt.eval.evaluate` can reuse it without shelling out.

## 5. `gen_training_data.py` — `sql_context` generation, verification, new reason categories

**New reason taxonomy.** Replace the current 6-entry `BAD_SQL_REASONS`
(`wrong_columns_or_tables`, `wrong_join_or_aggregation`, `wrong_filter_or_sort`,
`type_or_null_handling`, `syntax_error`, `inefficient_query`) with 4:

| name | description |
|---|---|
| `missing_filters` | Omits a `WHERE`/`HAVING` condition the question implies, returning too many rows. |
| `wrong_aggregation` | Wrong aggregate function, `GROUP BY`, or aggregates over the wrong column. |
| `unsafe_patterns` | Risky patterns like `SELECT *` or an unqualified/cross join instead of the specific columns/join the question calls for. |
| `misjoined_tables` | Joins on the wrong column(s) or the wrong pair of tables, producing an incorrect result set. |

This is a straight replacement, not an addition — update `BAD_SQL_REASONS`, the
`TOOL_SCHEMA`/`RESULT_SCHEMA` `reason` enum, `SYSTEM_PROMPT`'s reasons block, and
re-tag every `FEW_SHOT_EXAMPLES` entry's `sql_bad` items to one of the 4 new categories
(rewriting the examples' SQL where needed so each one is a genuine instance of its
category, not just a relabeled old one). `sql_bad` keeps its existing `minItems: 3,
maxItems: 5` — categories can repeat across an example's variants. Note this changes the
categories `pairwise_accuracy_by_reason` breaks down by going forward; existing
`rm_metrics.json`/run logs using the old 6 categories (e.g. CLAUDE.md's `syntax_error`
narrative for V3) reflect data generated under the old taxonomy and won't map onto new
runs — that's an expected consequence of retraining on regenerated data, not something
this plan needs to reconcile.

**`sql_context` generation** (unchanged from earlier in this plan): `TOOL_SCHEMA`/
`RESULT_SCHEMA` gain a required `sql_context` property — `{"type": "array", "items":
{"type": "string"}}`, described as SQLite-compatible `CREATE TABLE`/`INSERT INTO`
statements, in execution order, minimal enough that `sql_good` runs directly against them
and returns a meaningful result; explicitly instruct: if `sql_good` is itself DDL
(`CREATE`/`ALTER`/`DROP TABLE`) rather than a query over existing data, return an empty
array. Same instruction added to `SYSTEM_PROMPT`; `FEW_SHOT_EXAMPLES` gain a matching
`"sql_context"` array per example.

**Verification + summary.** `enhance_record(record, result)` merges `sql_context` and
`sql_bad` as before, then calls `walt.utils.sql_exec.run_sql(merged["sql_context"],
merged["sql_good"])` and sets `merged["sql_context_valid"] = execution.success`. Both
`cmd_test` and `cmd_collect` call `enhance_record`, so this runs in both modes for free
(`cmd_submit` only submits batch requests, unaffected). After processing all rows, both
`cmd_test` and `cmd_collect` print an aggregate summary before exiting:

```
sql_good execution check: 142/150 passed (94.7%)
```

i.e. count of `sql_context_valid is True` over total rows processed — a fast, explicit
signal ("do I trust this generated data enough to train on it, or does the prompt/context
generation need work?") separate from the per-row `sql_bad` synthesis output that already
prints/writes today.

## 6. `Example` dataclass — `src/walt/rm/data/base.py`

Mirror the existing `sql_bad` pattern:
```python
sql_context: tuple[str, ...] = ()
sql_context_valid: bool | None = None
split: str = "trainval"        # "trainval" | "val"
```
Included conditionally in `to_dict()` / parsed in `from_dict()` the same way `sql_bad` is.
`sql_context`/`sql_context_valid` aren't consumed by RM training/scoring today — carried
through so `evaluate.py` and the agent CLI's `--input`/`--index` path can load a row's
generated context via the existing `load_examples()` helper instead of re-parsing JSONL.

## 7. Train/test/val split — `src/walt/rm/data/pre_process.py`

**Design decision, flagging the tradeoff explicitly:** the RM pipeline's existing
train/test boundary is not fixed today — `group_split(test_size=0.2, seed=...)` in
`rm/model/base.py` re-derives it per run, and `cross_validate.py`'s `k_fold_split`
re-folds the whole loaded set into k folds itself. Neither respects a persisted
row-level train/test label — that's deliberate (CLAUDE.md documents re-splitting/CV as
the load-bearing methodology here, since a single fixed split was shown to be noisy). So
rather than persisting three fixed labels (`train`/`test`/`val`) where "train"/"test"
would silently go unused by `cross_validate.py`, `pre_process.py` stamps each row with
just two: `split = "val"` (held out) or `"trainval"` (everything RM training/CV is
allowed to see — it keeps doing its own internal train/test division exactly as today,
unchanged). This satisfies "RM can see only train and test" — the val rows are never in
the pool `train.py`/`cross_validate.py` load — while leaving 100% of the existing
`group_split`/`k_fold_split`/CV infrastructure untouched. Flagging this so you can
redirect if you actually want a fixed, non-CV train/test boundary persisted too.

Implementation: after the existing seeded shuffle of the combined, downsampled example
list, slice off a `--val-fraction` (default `0.15`) tail as `split="val"`, label the rest
`split="trainval"`. One `--val-fraction` CLI flag, one seeded slice — no change to the
existing per-source proportional downsampling logic. `gen_training_data.py` enhances
every row regardless of split (val rows need `sql_bad`/`sql_context` too, for
`evaluate.py`'s metrics below); `enhance_record` already merges onto a copy of the
original record, so the `split` field rides through untouched.

`train.py`/`cross_validate.py` gain one filter when loading: only rows with
`split == "trainval"` enter `group_split`/`k_fold_split` — val rows are invisible to
both.

## 8. Agent-level evaluation — `src/walt/eval/evaluate.py`

Loads `rm_enhanced.jsonl` via `load_examples`, filters to `split == "val"`. Reuses the RM
(`LRRewardModelV3.load(...)`), `SqlAgent`, and `walt.utils.sql_exec.run_sql` built above —
no duplicated scoring/execution logic. Computes exactly the three metrics requested:

1. **RM accuracy selecting correct SQL among good vs bad**: `rm.evaluate(val_examples)` —
   already computes `top1_accuracy`/`pairwise_accuracy`/`mrr` (+ per-reason breakdown,
   now under the 4 new categories) treating `sql_good` vs `sql_bad` as the candidate set;
   called directly, no new logic.
2. **SQL pass/fail execution stats**: for each val example, run
   `SqlAgent.run(question, sql_context)` and record `execution.success`; report the pass
   rate (denominator: val rows with non-empty `sql_context`, since DDL-only rows have
   nothing to execute against).
3. **End-to-end QA accuracy**: for each val example, also compute the reference result via
   `run_sql(sql_context, sql_good)` and compare its `(columns, rows)` against the agent's
   `execution` result — order-insensitive row-set equality (SQL result order isn't
   guaranteed without explicit `ORDER BY`). Accuracy = fraction matching. Restricted to
   rows where `sql_context_valid` is `True` (can't judge correctness against a reference
   that doesn't itself execute) — reported denominator makes this explicit rather than
   silently dropping rows.

CLI: `python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl --rm-model
data/output/rm_model.joblib --ollama-model llama3.2 --n-candidates 5 --output
data/output/eval_results.json`. Prints/writes a JSON summary with all three metric
groups plus the row counts/denominators used for each, so results are auditable.

## Verification

1. `uv add ollama`; confirm `ollama pull llama3.2` + `ollama serve` running locally.
2. `python -c "from walt.utils.sql_exec import run_sql; print(run_sql(['CREATE TABLE t(id INT)','INSERT INTO t VALUES (1)'], 'SELECT * FROM t'))"` — confirms the executor in isolation.
3. `uv run python -m walt.rm.data.pre_process --target-count 5000 --val-fraction 0.15 --output data/output/rm_data.jsonl` — confirm output rows carry `split` and roughly 15% land in `"val"`.
4. `uv run python -m walt.rm.data.gen_training_data test --input data/output/rm_data.jsonl --limit 10` — inspect output rows for `sql_context`, `sql_context_valid`, new 4-category `sql_bad` reasons, `split` passed through unchanged, and confirm the printed pass-rate summary line appears.
5. `uv run python -m walt.rm.model.train ...` — confirm it still runs and its metrics are computed only over `split == "trainval"` rows (e.g. log/print the row count going into `group_split` and sanity-check it excludes val).
6. Run the new agent CLI against one generated row (`--input <test output> --index 0`) and confirm it prints ranked candidates with scores, `best_sql`, execution `rows`/`columns`, a non-null `final_answer`, and `critique: null`.
7. Confirm `LRRewardModelV3.load("data/output/rm_model.joblib")` still loads and `.rank()` runs unmodified against agent-generated candidates (no RM code changes, so this is a regression check).
8. `uv run python -m walt.eval.evaluate --input data/output/rm_enhanced.jsonl --limit 5` (small `--limit`/subset flag for a fast smoke test) — confirm it prints all three metric groups without error.

## Post-implementation notes (added after execution)

The plan above was executed largely as written, plus several follow-on changes made
during the same working session that this document was not updated to reflect inline
(see `CLAUDE.md` for the authoritative, current state):

- `sql_agent.py`'s CLI/`run_agent()` and `evaluate.py` both gained `--llm-cache`
  (default `data/output/llm_cache.json`) via a new `agent/llm/caching_llm.py`
  (`CachingLLM(BaseLLM)`), keyed by `(model, question, schema_context)` — not part of
  this plan, added after the fact so RM re-tuning doesn't require re-calling Ollama.
- `evaluate.py`'s with-RM/without-RM comparison (agent's actual top pick vs. the first
  generated candidate) wasn't in this plan either — added afterward once the first
  eval run raised the question of how much the RM was actually contributing.
- `evaluate.py` and a new `eval/visualize.py` now log/compare run history via the
  existing `rm/model/tracking.py` (`log_run`/`load_runs`, reused as-is), pointed at a
  separate `data/output/eval_runs/` directory — mirroring `train.py`'s existing
  history-tracking convention, at the user's request after the first full eval run.
- Data regeneration (§5) was actually run at `--target-count 1000` (matching the
  pre-existing baseline's scale, not a fresh choice) via the real Anthropic Message
  Batch API — see `CLAUDE.md`'s "Data regeneration" section for the actual result
  (980/1000 rows, 91.7% `sql_context_valid`, 20 schema-rejects from a minor
  prompt-robustness gap where the model sometimes omits `sql_context` instead of `[]`).
- First full agent-eval finding: the RM does **not** transfer to `llama3.2`'s candidate
  distribution — with-RM and without-RM tie on end-to-end QA accuracy (56.9% either
  way), and among rows where RM's pick differs from the naive first-candidate baseline
  it's an exact 15-15 split on correctness (a coin flip, not an improvement). Oracle
  ceiling analysis: 77.2% of val rows have at least one correct candidate among 5, vs
  56.9% currently achieved — most of that gap sits in a 54-row "mixed" bucket where
  current selection (29/54) is barely above chance (~26.8/54 expected at random). Full
  writeup and the suggested next step (retrain with `llama3.2`'s own wrong candidates
  folded in as `sql_bad`-style negatives, targeting the real distribution mismatch
  instead of assuming synthetic negatives generalize) is in `CLAUDE.md`.
