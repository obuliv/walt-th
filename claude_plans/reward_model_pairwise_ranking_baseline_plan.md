# Pairwise-Ranking SQL Reward Model

## Context

The repo has a complete data-generation pipeline (`src/walt/rm/data/`) that produced
`data/output/rm_enhanced.jsonl` — 987 records of `{question, source, sql_good, sql_bad:
[{sql, reason}, ...]}` (5 negatives per question, 1 row has 4). There is currently zero
model/training code anywhere in the repo, and zero ML dependencies installed
(`pyproject.toml` only has `python-dotenv`, `anthropic`, `jsonschema`).

Goal: build a reusable, algorithm-agnostic `BaseRewardModel` (parsing, question-level
train/test split, ranking, evaluation, metrics publishing) and a first concrete
implementation, `LRRewardModel`, that scores SQL candidates via a pluggable local
embedding model + scikit-learn logistic regression trained pairwise. A `train.py` CLI
chains load → split → fit → evaluate → publish → save.

Confirmed with user: embeddings must be swappable/testable (pluggable
`EmbeddingProvider`), default local model is `jinaai/jina-embeddings-v2-base-code`
(code-aware, via `sentence-transformers`). Error-code prediction is a base-class hook
only — the LR implementation does not implement it this iteration.

## Design

### File layout (new subpackage, sibling to `src/walt/rm/data/`)

```
src/walt/rm/data/base.py   # EXTEND existing Example, add SQLBadCandidate
src/walt/rm/model/
    __init__.py
    embeddings.py  # EmbeddingProvider ABC + SentenceTransformerEmbedding
    base.py        # ScoredCandidate, load_examples, group_split, BaseRewardModel ABC
    lr_model.py    # LRRewardModel(BaseRewardModel)
    train.py       # CLI chaining everything together
```

Import direction (no cycles): `train.py` → `{base, embeddings, lr_model}`;
`lr_model.py` → `{base, embeddings, walt.rm.data.base.Example}`; `model/base.py` →
`walt.rm.data.base.Example`; `embeddings.py` → nothing internal.

**No separate `RMExample` type.** Extend the existing `walt.rm.data.base.Example`
(currently `{question, sql_good, source}`) with an optional `sql_bad` field instead of
introducing a parallel type — one class describes the record shape across the whole
pipeline (adapters → pre_process → gen_training_data → RM training), not two.

### Changes to `src/walt/rm/data/base.py`

```python
@dataclass(frozen=True)
class SQLBadCandidate:
    sql: str
    reason: str

    @staticmethod
    def from_dict(d: dict) -> "SQLBadCandidate": ...
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class Example:
    question: str
    sql_good: str
    source: str
    sql_bad: tuple[SQLBadCandidate, ...] = ()   # new, defaulted — existing adapters unaffected

    def __post_init__(self):
        # fail fast on malformed rows rather than silently mis-scoring in evaluate()
        if self.sql_good in {b.sql for b in self.sql_bad}:
            raise ValueError(f"sql_good duplicates a sql_bad entry for question: {self.question!r}")

    def to_dict(self) -> dict:
        d = {"question": self.question, "sql_good": self.sql_good, "source": self.source}
        if self.sql_bad:
            d["sql_bad"] = [b.to_dict() for b in self.sql_bad]
        return d   # omits sql_bad when empty -> rm_data.jsonl output from pre_process.py is unchanged

    @staticmethod
    def from_dict(d: dict) -> "Example":
        sql_bad = tuple(SQLBadCandidate.from_dict(b) for b in d.get("sql_bad", []))
        return Example(question=d["question"], sql_good=d["sql_good"], source=d["source"], sql_bad=sql_bad)
```

`Example` is still constructed positionally as `Example(question, sql_good, source)` by
`spider.py`/`dbasql.py` — adding `sql_bad` as a 4th, defaulted field is backward
compatible with every existing call site.

### `model/base.py`'s `ScoredCandidate`

```python
@dataclass(frozen=True)
class ScoredCandidate:
    sql: str
    score: float
    rank: int                    # 1-based, ties broken by stable sort (input order)
    error_code: str | None = None
```

A free helper (not a method on `Example`, to keep `data/base.py` free of RM-specific
concerns) lives in `model/base.py`:

```python
def all_candidates(example: Example) -> list[str]:
    return [example.sql_good] + [b.sql for b in example.sql_bad]
```

### `embeddings.py`

```python
class EmbeddingProvider(ABC):
    dim: int
    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...
    @property
    @abstractmethod
    def config(self) -> dict: ...   # JSON-serializable identity, for model persistence

class SentenceTransformerEmbedding(EmbeddingProvider):
    DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

    def __init__(self, model_name=DEFAULT_MODEL_NAME, max_seq_length=1024,
                 trust_remote_code=True, device=None):
        from sentence_transformers import SentenceTransformer  # deferred: heavy import
        ...

    def embed(self, texts, batch_size=32) -> np.ndarray:
        return self._model.encode(texts, batch_size=batch_size,
                                   normalize_embeddings=True,  # cosine sim -> dot product
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    @property
    def config(self) -> dict: ...  # {"type": "sentence_transformer", "model_name", "max_seq_length", "trust_remote_code", "dim"}
```

Confirmed loading requirements for `jinaai/jina-embeddings-v2-base-code`: 768-dim
(from `config.json`), requires `trust_remote_code=True` (custom JinaBERT/ALiBi modeling
code, no vanilla `AutoModel` path), and requires the `einops` package installed
separately (used internally by the custom attention code, not a declared
`sentence-transformers` dependency — omitting it fails on first `.encode()` call, not
at construction). Model card default is `max_seq_length = 1024` (extrapolates to 8192
via ALiBi, but 1024 is ample headroom for question/SQL text and much cheaper).

### `base.py`

```python
def load_examples(path) -> list[Example]: ...   # parse rm_enhanced.jsonl via Example.from_dict

def group_split(examples, test_size=0.2, seed=42) -> tuple[list[Example], list[Example]]:
    # question-level split (random.Random(seed), matching pre_process.py's convention) —
    # every pair from one question stays entirely in train or entirely in test

class BaseRewardModel(ABC):
    @abstractmethod
    def fit(self, train_examples: list[Example]) -> None: ...

    @abstractmethod
    def score(self, question: str, sql: str) -> float: ...

    def predict_error_code(self, question: str, sql: str) -> str | None:
        return None   # hook only; unimplemented by LRRewardModel this iteration

    def rank(self, question: str, candidates: list[str]) -> list[ScoredCandidate]:
        ...  # stable sort descending by score(), 1-based rank

    def evaluate(self, test_examples: list[Example]) -> dict:
        # generic over any subclass, built only on rank()/score() plus the
        # all_candidates(example) helper defined above:
        #   top1_accuracy   — is sql_good ranked #1 among [sql_good] + sql_bad
        #   pairwise_accuracy — fraction of (good,bad) pairs correctly ordered (strict >, ties = incorrect)
        #   mrr             — mean reciprocal rank of sql_good
        #   n_examples, n_pairs
        ...

    def publish_metrics(self, metrics: dict, output_path: Path | None = None) -> None:
        ...  # print formatted summary; optionally write metrics JSON
```

### `lr_model.py`

Feature function: `phi(question, sql) = concat(embed(sql), [cosine_sim(embed(question), embed(sql))])`
— embed question and SQL **separately** through the same model (not one joint string).
Rationale: jina-code is a retrieval-style model trained to align matching query/code
pairs, so cosine-sim is a meaningful relevance signal; the raw SQL embedding alone
should also carry a lot of "is this well-formed SQL" signal. Since embeddings are
L2-normalized, cosine similarity is a plain dot product.

```python
class LRRewardModel(BaseRewardModel):
    def __init__(self, embedding_provider: EmbeddingProvider, seed: int = 42):
        self.embedding_provider = embedding_provider
        self.seed = seed
        self.coef_: np.ndarray | None = None   # shape (dim+1,), no intercept
        self._question_cache: dict[str, np.ndarray] = {}
        self._sql_cache: dict[str, np.ndarray] = {}

    def warm_cache(self, examples: list[Example]) -> None:
        # public: embeds the unique set of questions and unique set of SQL strings
        # (good + bad, via all_candidates()) in `examples` not already cached, one
        # batched call each. train.py calls this for both train and test examples so
        # evaluate()/score() never falls back to slow per-string embedding calls.

    def _phi(self, question: str, sql: str) -> np.ndarray: ...  # concat(sql_vec, [dot(q_vec, sql_vec)])

    def fit(self, train_examples: list[Example]) -> None:
        self.warm_cache(train_examples)
        # for each (sql_good, bad) pair: randomly assign which is "A"/"B" (label=1 if A
        # is good) so the LR intercept doesn't pick up positional bias; feature = phi(A) - phi(B)
        # fit sklearn LogisticRegression(max_iter=1000, random_state=self.seed); keep only coef_
        # (drop intercept — only relative order across a candidate list matters for ranking)

    def score(self, question: str, sql: str) -> float:
        # embeds on demand if not cached (defensive fallback), then coef_ . phi(question, sql)

    def save(self, path) -> None:
        # joblib.dump({"coef": self.coef_, "seed": self.seed,
        #              "embedding_config": self.embedding_provider.config}, path)
        # never re-serializes the embedding model itself, only its config for reconstruction

    @classmethod
    def load(cls, path, embedding_provider=None) -> "LRRewardModel": ...
```

### `train.py`

argparse CLI matching the style of `pre_process.py`/`gen_training_data.py` (module
docstring + `Usage:`, plain `main()`).

Flags: `--input` (default `data/output/rm_enhanced.jsonl`), `--test-size` (0.2),
`--seed` (42), `--embedding-model` (default = jina-code), `--device` (None = let
sentence-transformers auto-pick; allows `mps` on this Apple Silicon machine),
`--model-output` (default `data/output/rm_model.joblib`), `--metrics-output` (optional
JSON path).

Chain: `load_examples` → `group_split` → build `SentenceTransformerEmbedding` → build
`LRRewardModel` → `fit(train)` → `warm_cache(test)` → `evaluate(test)` →
`publish_metrics` → `save(model_output)`.

## Dependencies

```bash
uv add numpy scikit-learn sentence-transformers einops
```

`torch` and `joblib` come in transitively (via `sentence-transformers` and
`scikit-learn` respectively) — no need to add them directly. `einops` must be added
explicitly since it's a runtime-only dependency of Jina's custom modeling code, not a
declared package dependency anywhere.

Known risk: the repo's `.venv` runs Python 3.14, while `pyproject.toml` only requires
`>=3.11`. If `uv add` can't resolve prebuilt `torch`/`sentence-transformers` wheels for
cp314, fall back to pinning the venv to a supported version (`uv python pin 3.12`)
rather than fighting 3.14 compatibility — nothing in this design depends on 3.14.

## Not doing this iteration

- No pytest/test-runner setup — none exists in the repo today (confirmed in
  `CLAUDE.md`), and it wasn't asked for. Verification is the end-to-end `train.py` run
  against real data (below), not a unit-test suite.
- No error-code prediction implementation (hook only, per requirements).
- No Voyage/OpenAI/TF-IDF embedding providers — just the ABC + one working
  implementation, structured so adding another provider later doesn't touch `base.py`
  or `lr_model.py`.

## Verification

1. **Loading smoke test** (validates `trust_remote_code`/`einops` chain in isolation):
   ```bash
   uv add numpy scikit-learn sentence-transformers einops
   uv run python -c "
   from sentence_transformers import SentenceTransformer
   m = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
   m.max_seq_length = 1024
   v = m.encode(['SELECT 1'], normalize_embeddings=True)
   print(v.shape, v.dtype)
   "
   ```
   Expect `(1, 768) float32`.

2. **End-to-end run** on the real 987-row dataset:
   ```bash
   uv run python -m walt.rm.model.train \
     --input data/output/rm_enhanced.jsonl \
     --model-output data/output/rm_model.joblib \
     --metrics-output data/output/rm_metrics.json
   ```

3. **Sanity-check the metrics.** Chance-level top-1 accuracy is ~1/6 ≈ 0.167 (1 good +
   5 bad candidates per question), chance MRR ≈ 0.373. A working model should land
   clearly above that — rough expectation: top-1 accuracy ~0.45–0.75, pairwise accuracy
   ~0.65–0.85 (capped somewhat since the negatives were generated specifically to be
   close misses). Red flags:
   - **Near-chance numbers** → check the random A/B pair-label assignment in `fit()`
     and that `sql_good`/`sql_bad` aren't swapped in `Example.from_dict`.
   - **Suspiciously perfect (~1.0) accuracy** → check `group_split` is actually
     splitting at the `Example`/question level, not a flattened pair list (assert
     `{ex.question for ex in train} & {ex.question for ex in test} == set()`).

4. Inspect `data/output/rm_metrics.json` and confirm `n_pairs` == sum of `len(sql_bad)`
   across the test split (catches the one 4-negative row being miscounted).

## Critical files

- `src/walt/rm/data/base.py` (extend: add `SQLBadCandidate`, add `sql_bad` field to `Example`)
- `src/walt/rm/model/embeddings.py` (new)
- `src/walt/rm/model/base.py` (new)
- `src/walt/rm/model/lr_model.py` (new)
- `src/walt/rm/model/train.py` (new)
- `pyproject.toml` (dependency additions via `uv add`)

---

## Post-plan evolution (not part of the original approved plan, recorded for context)

Everything above is the plan as originally approved, before implementation started.
Since then, the following was added in response to follow-up requests (see git history
and `CLAUDE.md`'s Architecture section for full detail):

- `LRRewardModelV2` (`lr_model_v2.py`) — raw-dot-product feature ablations (didn't beat V1).
- `LRRewardModelV3` (`lr_model_v3.py`) — added `is_sql_valid` (via `sqlglot`); real win.
- `GBMRewardModel` (`gbm_model.py`) — pointwise gradient boosting; lost decisively to LR.
- `cross_validate.py` / `base.py`'s `cross_validate()` — k-fold CV, needed because a
  single 80/20 split turned out too noisy to trust small deltas between configs.
- `C` (LR regularization) tuning via CV — sklearn's default `C=1.0` was substantially
  under-fit for this data size; `C=30` is the tuned default.
- `tracking.py` / `visualize.py` — per-run JSON logging under `data/output/runs/` plus
  a comparison table/chart, so different approaches can be tracked over time.

Current best baseline: `lr_v3` with `C=30` (top1_accuracy 0.546, pairwise_accuracy
0.864, mrr 0.738 on the standard 80/20 split), now `train.py`'s default.
