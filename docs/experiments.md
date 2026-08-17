# Experiment log

Archived from `CLAUDE.md` on 2026-08-16, verbatim. This is the full chronological
record of RM/agent experiments — baselines, ablations, negative results, and the
reasoning behind current defaults. `CLAUDE.md` keeps only the structural summary and
links here.

## Architecture

**Data pipeline (`src/walt/rm/data/`)** runs in two sequential stages:

1. **Adapters → standardized examples** (`base.py`, `spider.py`, `dbasql.py`):
   Each source has a `BaseAdapter` subclass that parses its own raw file
   format (Spider's CSV, DBASQL's JSON) and yields `Example(question,
   sql_good, source)` records. To add a new dataset, subclass `BaseAdapter`,
   implement `load()`, and register it in `pre_process.py`'s `SOURCES` dict.

2. **`pre_process.py`** (module name is `walt.rm.data.pre_process`, entry
   point historically called "launcher" — check the module docstring, it
   lags a prior rename) loads every registered source, proportionally
   downsamples each to hit `--target-count` total (never upsamples), shuffles
   with a fixed seed, and writes standardized JSONL.

3. **`gen_training_data.py`** takes that JSONL and, per row, calls Claude
   with a tool-forced schema (`emit_sql_review`) to: (a) correct `sql_good`
   if it has a bug, (b) synthesize 3-5 `sql_bad` variants, each tagged with a
   mistake category from `BAD_SQL_REASONS` (`missing_filters`,
   `wrong_aggregation`, `unsafe_patterns`, `misjoined_tables`), and (c)
   synthesize `sql_context` — SQLite `CREATE TABLE`/`INSERT INTO` statements
   `sql_good` can actually execute against, verified locally afterward (no
   extra API call, see "Data regeneration" below). The system prompt + few-shot
   examples are defined in this file — edit `FEW_SHOT_EXAMPLES` and
   `BAD_SQL_REASONS` together when tuning quality. Supports `test` (sync,
   cheap iteration), `submit` (async Message Batch, cheaper at scale), and
   `collect` (poll + merge) modes; batch state is cached locally in
   `src/walt/rm/data/.batch_state/<batch_id>.json` so `collect` can be
   re-run independently of `submit`.

Output convention: JSONL files under `data/output/`, one JSON object per
line (`rm_data.jsonl` = stage 2 output, `rm_enhanced.jsonl` = stage 3
output).

**gretel (`gretel.py`/`build_gretel_dataset.py`, 2026-08-15)** is a second, parallel
pipeline outside the `pre_process.py`/`SOURCES` flow above, for
`gretelai/synthetic_text_to_sql` — added because that dataset already ships its own
`sql_context` per row (no LLM synthesis needed) and its own train/test split (no
re-splitting needed). `GretelAdapter` downloads the HF-hosted parquet directly
(`pyarrow`, no `datasets` dependency needed), splits the dataset's single
semicolon-joined `sql_context` string into individual statements via `sqlglot` (falls
back to a naive `;`-split on a sqlglot parse error — a handful of rows use
non-SQLite-dialect syntax sqlglot's sqlite mode rejects), and verifies each locally via
`run_sql()` at extraction time rather than deferring to `gen_training_data.py`.
`build_gretel_dataset.py` samples exactly `--train-count`/`--test-count` rows from
gretel's own `train`/`test` splits (mapped to our `split="trainval"`/`"val"` — never
re-splitting their train data ourselves) and writes to the separate
`data/output/gretel/gretel_data.jsonl` (not merged into `rm_data.jsonl`). Raw parquet is
cached (gitignored, ~32MB) under `data/gretel/`.

`gen_training_data.py` now branches per-row on whether `sql_context` is already
populated: rows without one go through the original full flow (`emit_sql_review`) as
described above; rows that already have one (gretel) go through a second
`emit_sql_bad`-only tool call, told not to touch `sql_good`/`sql_context` and just
generate `sql_bad` against them (`has_context()`/`BAD_ONLY_*` in the module).
`is_llm_ready()` additionally skips, before either `test` or `submit` spends a call,
rows whose `sql_context_valid` was already computed `False` at extraction time — bad-only
mode can't fix those (~532/2200 for gretel, mostly non-SQLite-dialect source schemas).

**Reward model (`src/walt/rm/model/`)** scores/ranks SQL candidates for a question.
All `lr_model*.py` files below (the LR variants and `sql_features`-adjacent
`lr_model_context.py`) live under `rm/model/lr/`, not directly in `rm/model/` — moved
there (2026-08-15) once the flat directory grew to 7 LR variants; import via
`walt.rm.model.lr.lr_model_v3` or the re-exporting `walt.rm.model.lr` package `__init__`.
`BaseRewardModel` (`base.py`) is algorithm-agnostic: question-level train/test split
(`group_split` — splits by `Example`, never by pair, to avoid leaking a question's
other candidates across the split), `rank()`/`evaluate()` (top-1 accuracy, pairwise
accuracy, MRR) built generically on a subclass's `score(question, sql)`, and an
unused-so-far `predict_error_code()` hook for future subclasses. `LRRewardModel`
(`lr_model.py`) is the first implementation: scores via `w · phi(question, sql)` where
`phi = concat(embed(sql), [cosine_sim(embed(question), embed(sql))])`, fit by training
sklearn `LogisticRegression` on `phi(q,A) - phi(q,B)` differences for
(`sql_good`, `sql_bad`) pairs (Bradley-Terry style — a per-candidate scorer, not a
pairwise classifier, so it generalizes to ranking any-size candidate lists). Embeddings
come from a swappable `EmbeddingProvider` (`embeddings.py`); the default,
`SentenceTransformerEmbedding`, uses `jinaai/jina-embeddings-v2-base-code` and requires
`trust_remote_code=True` + the `einops` package. `Example` (`walt/rm/data/base.py`) now
carries an optional `sql_bad: tuple[SQLBadCandidate, ...]` field for this — defaulted to
`()` so the data-adapter pipeline is unaffected. `load_examples()` skips and warns on
rows where `sql_good` duplicates a `sql_bad` entry (~9/987 rows in the current
`rm_enhanced.jsonl`, a labeling artifact from the LLM data-generation step) rather than
aborting the load.

`LRRewardModelV2` (`lr_model_v2.py`) subclasses `LRRewardModel`, overriding only
`_embed_unique`/`_phi`: V1's `embed()` always L2-normalizes, so its "cosine
similarity" feature already *is* a dot product (of unit vectors) — V2 fetches raw
(unnormalized) embeddings instead and adds the raw dot product as a second,
magnitude-sensitive interaction feature alongside the (now locally-computed) cosine
similarity (770-dim phi vs V1's 769-dim), with `use_cosine_sim`/`use_dot_product`/
`standardize_dot_product` constructor flags to ablate each piece. On the real dataset
none of these combinations beat V1 (confirmed via CV, not just a single split) — the
raw dot product just isn't informative here, standardized or not.

**Establishing a baseline (CV-driven)**: a naive single 80/20 split is too noisy to
trust small deltas between configs — `cross_validate.py`'s k-fold CV surfaced two real
improvements a single split had been masking:
- **Regularization**: sklearn's default `C=1.0` was substantially under-fitting for
  this data size (~4700 training pairs, 769 features). Sweeping `C` via CV
  (`cross_validate.py --model lr_v1 --C ...`) showed top1_accuracy climbing
  monotonically from 0.246 (`C=0.01`) to ~0.47 (`C=10`-`C=30`, tight ±0.005-0.013 std)
  before diminishing returns and rising variance set in past `C=100` (early overfitting
  signal). `C=30` is the sweet spot — `LRRewardModel`/`V2`/`V3` all take `C` as a
  constructor arg (persisted via `save()`/`load()`), and `train.py`/`cross_validate.py`
  both default to `--C 30`.
- **`LRRewardModelV3`** (`lr_model_v3.py`) subclasses `LRRewardModel`, appending one
  handcrafted feature to V1's phi: `is_sql_valid(sql)` (`sql_features.py`, via
  `sqlglot` — pure syntax parsing, no schema/database needed). Targeted directly at the
  `syntax_error` mistake category, which was consistently the weakest or
  second-weakest in every embedding-only variant's `pairwise_accuracy_by_reason`
  breakdown (~0.70-0.73). Under CV at `C=30`, V3 beats V1 (top1 0.494 vs 0.469); on the
  full train.py run this shows up concretely as `syntax_error` accuracy jumping from
  0.73 to 0.89. Note: sqlglot's default dialect treats a lot of unmodeled DDL/DCL
  (MySQL's `MODIFY COLUMN`, `GRANT`/`REVOKE`, `RENAME TABLE`, ...) as a lenient
  `Command` fallback instead of raising — `is_sql_valid` treats that fallback as
  "invalid too" (catches real mistakes like `GRANT ... FROM` that should be `TO`, at
  the cost of ~0.7% false positives on genuinely-valid `sql_good` DDL); see the
  docstring in `sql_features.py` for the recall/false-positive numbers this trades off.
- **`GBMRewardModel`** (`gbm_model.py`) tries a nonlinear model class instead of more
  features: `sklearn.ensemble.HistGradientBoostingClassifier`, trained *pointwise*
  (label = is this candidate `sql_good`, features = `phi(question, sql)` directly, no
  pairwise differencing) rather than pairwise, because a nonlinear model trained on
  feature *differences* loses the linear-scorer's clean decompose-into-a-per-candidate-
  score property (see the module docstring for why that matters for ranking an
  any-size candidate list). It's a strict feature superset of V3 (same phi) — but loses
  decisively under CV regardless of hyperparameters tried (default and a shallower/
  fewer-rounds config both land well below plain `lr_v1`, e.g. top1 ~0.36-0.37 vs
  0.469). Expected, not a bug: gradient-boosted trees split on individual features one
  at a time, which doesn't suit dense embedding dimensions where the signal is a
  weighted combination across all ~768 of them — exactly what a linear model is good at
  and trees are bad at.

**Data regeneration: 4-category `sql_bad` taxonomy + held-out `val` split (2026-08-15).**
`gen_training_data.py`'s `BAD_SQL_REASONS` was replaced wholesale — the old 6 categories
(`wrong_columns_or_tables`, `wrong_join_or_aggregation`, `wrong_filter_or_sort`,
`type_or_null_handling`, `syntax_error`, `inefficient_query`) are gone, replaced by 4:
`missing_filters`, `wrong_aggregation`, `unsafe_patterns`, `misjoined_tables`. Rows also
now carry `sql_context` (LLM-synthesized SQLite `CREATE TABLE`/`INSERT INTO` statements
so `sql_good` is actually executable — empty when `sql_good` is itself DDL) and
`sql_context_valid` (verified locally via `walt/utils/sql_exec.py`'s `run_sql()`, no
extra API call). `pre_process.py` also now stamps every row with `split`
(`"trainval"`/`"val"`, via `--val-fraction`) — `val` is a held-out set for agent-level
evaluation (see below) that `train.py`/`cross_validate.py` filter out before
`group_split`/`k_fold_split` ever run, rather than changing how those existing
train/test-within-`trainval` mechanisms work (a single fixed train/test boundary was
already shown too noisy to trust — see CV discussion above — so only the val/trainval
boundary is persisted). Regenerating at the same scale as before (`--target-count 1000`,
same seed → same underlying question/SQL sample) via Anthropic Message Batch: 980/1000
rows succeeded (20 rejected — the model occasionally omits the now-required
`sql_context` field instead of returning an empty array; a minor, unfixed
prompt-robustness gap, ~2% of rows). Of the 980: 836 `trainval` / 144 `val`, and
`sql_context_valid` passed for 899/980 (91.7%) — the 4 reason categories are reasonably
balanced (830-1078 `sql_bad` instances each).

**Superseded baseline (spider/dbasql only): `lr_v3` with `C=30`, retrained on the
regenerated data above** — `top1_accuracy` 0.6587, `pairwise_accuracy` 0.8982, `mrr`
0.8179 on the standard 80/20 `trainval` split (5-fold CV: 0.6448 ± 0.0303 / 0.8955 ±
0.0097 / 0.8117 ± 0.0165), with a moderate overfitting gap (+0.078/+0.028/+0.045 — wider
than the ~+0.02-0.03 seen pre-regeneration, plausibly just a smaller effective
`trainval` pool: 836 rows vs the old 978). Not directly comparable to the
pre-regeneration numbers (`top1_accuracy` 0.546, `pairwise_accuracy` 0.864, `mrr` 0.738,
vs the original untuned `lr_v1` baseline's 0.424/0.811/0.658) — both the `sql_bad`
taxonomy and the row set changed, not just `C`. See below for the current baseline —
kept here only as a before/after reference point.

**Current best baseline: `lr_v3` with `C=1000`, spider/dbasql + gretel combined
(2026-08-15).** `data/output/rm_enhanced_with_gretel.jsonl` = the 980-row
`rm_enhanced.jsonl` above plus the 1666-row `gretel_enhanced.jsonl` (gretel data run
through `gen_training_data.py`'s bad-only mode — see gretel pipeline above), 2646 rows
total / 2642 usable (4 more hit the same `sql_good`-duplicates-`sql_bad` skip as
before) — 2355 `trainval` / 291 `val`. On this combined set, `C=30` (tuned on the old,
smaller `trainval` pool) noticeably *underperforms*: CV-sweeping `C` from 3 to 3000
(`cross_validate.py --model lr_v3 --C ...`) showed top1_accuracy still climbing well
past the old sweet spot — 0.542 (`C=3`) → 0.580 (`C=30`) → 0.602 (`C=300`) — before
plateauing at ~0.60 through `C=3000`, with `C=1000` landing on that plateau at the
tightest variance by far (top1 0.6006 ± 0.0056, vs ±0.015-0.024 for every other `C`
tried) — more data supporting much less regularization, as expected. Standard 80/20
split at `C=1000`: `top1_accuracy` 0.6128, `pairwise_accuracy` 0.8794, `mrr` 0.7877,
overfitting gap +0.016/+0.013/+0.014 (tighter than the superseded baseline's, despite
~3x the `trainval` rows — plausibly gretel's added diversity, not just row count).
**These numbers are lower than the superseded spider/dbasql-only baseline above**
(0.6587/0.8982/0.8179) despite the re-tuned `C` — confirmed by both CV and the single
split, so it isn't split noise. Likely cause: gretel spans far more diverse
domains/schemas per question than spider/dbasql, so `phi`'s embedding-similarity signal
has more surface area to get confused by; per-category breakdown on the combined set is
`unsafe_patterns` 0.95, `wrong_aggregation` 0.87, `missing_filters` 0.85,
`misjoined_tables` 0.84 (weakest, consistent with the superseded baseline). Read this as
a harder, more realistic baseline rather than a regression to fix — the smaller,
narrower old dataset was measuring an easier task.

**Making the RM consume `sql_context` (the schema `sql` executes against) — two things
tried, neither beats plain `lr_v3` (2026-08-15).** `BaseRewardModel.score()`/`rank()`
and `LRRewardModel._phi()`/`fit()` now thread an optional `sql_context: tuple[str, ...]`
through end to end (`evaluate()` passes `ex.sql_context`; `sql_agent.py` passes
`schema_context`) — plumbing any future context-aware variant needs, kept even though
neither variant below won:
- **`LRRewardModelV4`** (`lr_model_v4.py`): appends one scalar,
  `cosine_sim(embed(sql_context), embed(sql))`, to V3's phi (context embedded once per
  example, keyed by the joined statement text — see `lr_model_context.py`'s
  `ContextAwareLRRewardModel`, shared by V4/V5). The feature is real — a fitted
  coefficient of 1.15 at `C=1000`, not zeroed out — but CV-tied with plain `lr_v3` at
  every `C` tried (300/1000/3000; e.g. `C=1000` top1 0.5993 ± 0.0196 vs V3's
  0.6006 ± 0.0056), with consistently *higher* variance than V3 alone. No measurable
  benefit.
- **`LRRewardModelV5`** (`lr_model_v5.py`): concatenates the full `embed(sql_context)`
  vector (not just a scalar) alongside `embed(sql)`. This one isn't a "no signal"
  result — it's structurally inert. `LRRewardModel.fit()` trains on the *pairwise
  difference* `phi(q, sql_good) - phi(q, sql_bad)`; `sql_context` is identical across
  every candidate for a given question, so `embed(sql_context)` cancels to *exactly*
  zero in every training row regardless of what it encodes. Confirmed directly: the
  fitted coefficient block for those 768 dims is `0.0` to the last bit, and V5's
  predictions are byte-identical to V3's on the standard 80/20 split. Concatenating a
  per-question-constant feature can never contribute a gradient under this
  pairwise-difference objective — fixing it for real would need either a per-candidate
  interaction (e.g. `embed(sql_context) * embed(sql)` elementwise, generalizing V4's
  scalar to a full vector instead of a raw concat) or switching to pointwise training
  like `GBMRewardModel` (which already underperformed pairwise `lr_v1` here — see
  below). Left undone; this is documented as a dead end, not attempted further.

**Embedding model choice matters more than any single feature/hyperparameter change
tried so far.** CV-swept `lr_v3`/`C=30` against two general-purpose alternatives —
`BAAI/bge-base-en-v1.5` (top1 0.372) and `sentence-transformers/all-mpnet-base-v2`
(top1 0.377) — both landed well below `jina-embeddings-v2-base-code`'s 0.494, and
close to each other despite being different model families. This is decent evidence
that `jina-embeddings-v2-base-code`'s code-specific training (query/code retrieval
alignment) is doing real work here, not just a plausible-sounding default — a generic
strong text-embedding model isn't a substitute for one that's actually seen code.

**Scaling the `embed(sql)` block of phi doesn't help either — same null-result pattern
as the dot-product scaling test.** `LRRewardModelV3Scaled` (`lr_model_v3_scaled.py`,
`--scaling` on `cross_validate.py`) ablates how the ~768-dim `embed(sql)` portion of
phi is scaled before concatenation with `cosine_sim`/`is_sql_valid`, everything else
identical to the `lr_v3`/`C=30` baseline. `embed(sql)` is already L2-normalized by the
embedding provider's default (`normalize=True`), so `scaling="l2_normalize"` is a
no-op by construction and `scaling="l2_normalize_standardize"` collapses to plain
`scaling="standardize"` — both still implemented and CV-run rather than assumed, and
the numbers confirm the equivalence empirically (l2_normalize: top1 0.492±0.013 vs
baseline 0.494±0.014; l2_normalize_standardize: 0.473±0.016 vs standardize-alone:
0.470±0.014). The one real result: per-dimension standardizing (sklearn
`StandardScaler`, fit per CV fold on that fold's training data only) *costs* ~2.3pp
top1 (0.470 vs 0.494) — outside CV noise, not a fluke. Likely explanation: `C=30` was
CV-tuned against the raw (unnormalized-per-dimension) embedding scale, and
standardizing changes which dimensions the L2 penalty effectively favors; if a scaling
scheme were ever adopted, `C` would need re-tuning, but since nothing here beats the
unscaled baseline that's moot for now. Net: no scaling variant beats the current
unscaled baseline — keep `embed(sql)` as-is.

**L1 regularization (vs the default L2) is a promising but unconfirmed lead, not yet a
win.** `LRRewardModel` (and V2/V3/V3Scaled via inheritance) now take `penalty`/
`l1_ratio` (`SOLVER_BY_PENALTY = {l2: lbfgs, l1: liblinear, elasticnet: saga}` in
`lr_model.py`; sklearn requires a non-`lbfgs` solver for L1/elasticnet), exposed as
`--penalty`/`--l1-ratio` on `cross_validate.py`. Confirmed L1 does genuine feature
selection first (sklearn 1.8+ emits a `penalty`-is-being-deprecated-in-favor-of-
`l1_ratio` warning here, but still applies it correctly): a direct fit at `C=30` zeroed
395/770 coefficients under `penalty=l1` vs 0/770 under `l2`. CV-swept `lr_v3`/`C=30`
against the `l2` baseline at two seeds — L1 comes out numerically ahead on all 6
metric comparisons (top1/pairwise/mrr x 2 seeds), a consistent direction, but the
margin isn't clean: seed=42 shows top1 0.515±0.027 (l1) vs 0.494±0.014 (l2), a gap
inside roughly one *combined* std since L1's own variance is ~2x L2's there; seed=7
shows top1 0.509±0.046 (l1) vs 0.498±0.060 (l2), an even smaller gap with both
penalties noisier (matches the pre-existing seed-7-vs-42 fold-heterogeneity note
above). Net: not yet a confirmed win — a real, consistently-directional lead worth
revisiting, not a new baseline. If pursued further, `C` should be re-swept
specifically for `penalty=l1` (L1's optimal regularization strength isn't guaranteed
to be the same 30 tuned for L2) — not attempted here to keep this run isolated to
penalty type alone. The ~50% coefficient sparsity itself could also be a reason to
prefer L1 independent of the accuracy question (e.g. simpler/faster inference), if
that becomes a project goal.

`tracking.py` (`log_run`/`load_runs`) is the shared run-logging mechanism any
`BaseRewardModel` subclass's training script can reuse — not tied to `LRRewardModel`.
`visualize.py` reads everything under a runs directory and renders a comparison
table + line chart (top1_accuracy/pairwise_accuracy/mrr per run, chronological).
`base.py`'s `cross_validate()`/`k_fold_split()` are similarly algorithm-agnostic (take
a `model_factory` closure) — shares one pre-warmed embedding cache across all folds
(embedding a string doesn't depend on which fold it's in) rather than re-embedding
per fold, so k-fold CV costs about the same wall-clock time as a single split.

**Gretel-only scope (2026-08-15).** All work from this point on targets
`data/output/gretel/gretel_enhanced.jsonl` exclusively (1666 rows: 1519 `trainval` /
147 `val`, 4 more skipped for the usual `sql_good`-duplicates-`sql_bad` reason — 1662
usable), not the combined spider/dbasql+gretel set used above. There was no lr_v3
baseline trained purely on gretel data before this — every earlier "gretel" run was
actually the combined set — so one was established fresh: 5-fold CV sweeping `C` from
1 to 30000 found the same plateau pattern as the combined-set sweep but at a similar
`C`, peaking at **`C=1000`** (top1 0.5624 ± 0.0104, pairwise 0.8624 ± 0.0064, mrr
0.7571 ± 0.0072 — tightest variance of any `C` tried, consistent with the "more data
supports less regularization" pattern already seen). Matched single 80/20 split
(`seed=42`, same split later used for `DistilBertRewardModel` below): top1 0.5050,
pairwise 0.8479, mrr 0.7286, overfitting gap +0.174/+0.062/+0.100 (wider than the
combined-set gap — expected, gretel-only `trainval` is smaller: 1212 train rows).
Per-category on this split: `unsafe_patterns` 0.937 (strongest, as everywhere else in
this file), `misjoined_tables` 0.733 (weakest).

**`DistilBertRewardModel` (`distilbert_model.py`) — first fine-tuned-transformer RM,
does not beat `lr_v3` (2026-08-15).** Tries letting a model jointly attend over
`(sql_context, question, sql)` in one forward pass instead of embedding each piece
separately and combining via dot product, hypothesizing this would help most on
schema-reasoning mistakes (`misjoined_tables`/`missing_filters`). Architecture:
`distilbert-base-cased` (AutoModel, fully unfrozen) + a linear head on `[CLS]`,
manually encoding `[CLS] sql_context [SEP] question [SEP] sql [SEP]` (DistilBERT has
no `token_type_ids` — no segment embeddings — so segment structure comes purely from
`[SEP]` positions + `attention_mask`; DistilBertTokenizer's native API only supports
2-segment input, hence the manual construction). Trained genuinely pairwise, mirroring
`LRRewardModel.fit`'s exact anti-positional-bias RNG pattern: `score_A`/`score_B` are
two *independent* forward passes (each a single `(context, question, sql)` input), and
only `BCEWithLogitsLoss(score_A - score_B, label)` combines them — unlike
`GBMRewardModel`'s pointwise design (which exists because pairwise-differencing
*features* into one joint nonlinear classifier loses per-candidate score
decomposability), this stays a well-defined, always-transitive per-candidate scorer
despite being nonlinear, since the model never sees A and B jointly.

Two pre-flight checks (run before committing to a full fit(), see
`distilbert_preflight.py`) came back clean: token length using the real tokenizer and
format, 99.9% (8074/8079) of candidate sequences fit within 512 tokens untruncated
(median 162, p95 321, max 514) — truncation is a non-issue here, so the simple
truncate-`sql_context`-from-the-end fallback (no sqlglot-based schema filtering) was
used as-is; and an MPS sanity check (real model/tokenizer, tiny 8-pair batch, 15
forward+backward steps) showed no CPU-fallback ops, no errors, and a clean loss trend,
confirming native MPS support on this hardware with no
`PYTORCH_ENABLE_MPS_FALLBACK` needed.

Full run (defaults: `lr=2e-5`, `batch_size=8`, `grad_accum_steps=2` [effective 16],
`max_length=512`, early-stopping on validation `pairwise_accuracy` with
`patience=3`, question-level internal 90/10 split via the existing `group_split`):
stopped at epoch 7/15 (best weights from epoch 4 restored), ~30.3 min wall-clock on
an Apple Silicon Mac Mini via MPS (peak RSS 1.2GB; the MPS allocator separately
reported ~25.6GB, almost certainly inflated by caching-allocator pool growth across
~5700 variably-shaped micro-batches from dynamic padding rather than real working
set — RSS is the trustworthy figure here). On the same matched split as the gretel-
only `lr_v3` baseline above: top1 0.4983, pairwise 0.8387, mrr 0.7213 — **all three
slightly below `lr_v3`** (−0.007/−0.009/−0.007). Per-category: `unsafe_patterns` 0.967
(+3.0pp vs `lr_v3`), `missing_filters` 0.826 (+0.2pp, flat), `wrong_aggregation` 0.801
(−4.2pp), **`misjoined_tables` 0.698 (−3.5pp, the category the joint-attention
hypothesis predicted would improve most — instead it's the one that got worse)**.
Overfitting gap +0.322/+0.112/+0.184 — roughly 2x `lr_v3`'s on the identical split,
unsurprising for a fully-unfrozen 66M-param backbone fine-tuned on only 1091 internal
training examples (4724 pairs). Net: for this dataset size, `lr_v3` remains the
better reward model — far cheaper (38s vs ~30min) and no worse on any category that
matters. Not wired into `cross_validate.py` (no shared-cache path exists for it there
— each fold would fully re-fit from scratch — and a single held-out split matching
`lr_v3`'s shape was the goal here). The untried next lever, if this is revisited:
freezing most of the backbone and fine-tuning only the top layers + head, which would
cut both the overfitting gap and training cost — a different experiment from this one.

**SQL agent (`src/walt/agent/`)** wires an LLM candidate generator to the RM and a toy
SQLite executor: `agent/llm/base.py`'s `BaseLLM.generate_candidates(question,
schema_context, n) -> list[str]` is the swappable interface; `agent/llm/ollama_llm.py`'s
`OllamaLLM` is the only implementation so far, backed by a local Ollama server. Ollama's
HTTP API has no beam-search/multi-return (`num_beams`/`n`/`best_of`) parameter, so
`generate_candidates` makes `n` separate `chat()` calls with varied temperature/seed
rather than one decode pass — `BaseLLM`'s signature is deliberately backend-agnostic so
a future `transformers`-based backend with true `num_beams` beam search is a drop-in
swap for comparison later. `agent/sql_agent.py`'s `SqlAgent.run(question,
schema_context)` generates candidates, calls `rm.rank()` (the existing
`BaseRewardModel` method — no new RM code needed) to pick the top-scored one, executes
it via `walt/utils/sql_exec.py`'s `run_sql()` against a fresh in-memory SQLite DB seeded
from `schema_context`, and returns an `AgentResult` (`raw_candidates`,
`scored_candidates`, `best_sql`, `execution`, `final_answer`, and a `critique` field
that's always `None` — not implemented). `agent/llm/caching_llm.py`'s `CachingLLM`
wraps any `BaseLLM` with a disk cache (default `data/output/llm_cache.json`) keyed by
`(model, question, schema_context)` — deliberately *not* keyed by `n`, so a request for
fewer candidates than cached is served by slicing and only a request for *more*
triggers a real call — so re-running evaluation against a different RM (retrained,
different hyperparameters, even a different `BaseRewardModel` subclass) costs zero new
LLM calls. Enabled by default on both `sql_agent.py`'s CLI and `evaluate.py`
(`--no-llm-cache` to disable).

**Agent-level evaluation (`src/walt/eval/evaluate.py`)** runs the agent over the
held-out `split == "val"` rows (never seen by RM training/CV — see above) and reports
four things: (1) RM accuracy discriminating `sql_good` vs `sql_bad`, via the existing
`BaseRewardModel.evaluate()` — no new logic; (2) SQL execution pass/fail, and (3)
end-to-end QA accuracy (row-set match between the agent's executed result and
`run_sql(sql_context, sql_good)`'s reference result) — both (2) and (3) reported *with*
RM reranking (the agent's actual top pick) *and* without it (the first generated
candidate, i.e. what a single-shot call with no RM would produce), reusing the same
generated candidates for both so the comparison costs no extra LLM calls; and (4) an
**oracle ceiling** (added 2026-08-15, previously a one-off ad hoc analysis — now a
standard part of every run): of the `n_candidates` the LLM generated per question, how
many actually execute to the correct answer, bucketed per row into `all_correct` (any
pick wins — nothing to rerank), `zero_correct` (unreachable by *any* reranker — an LLM
generation-quality ceiling, not an RM problem), and `mixed` (0 < n_correct < n —
the only bucket where selection quality actually matters). `ceiling` =
`(all_correct + mixed) / n_qa_examples` is the best any reranker could ever score; within
`mixed`, the with-RM/without-RM achieved rates are compared against the random-chance
expectation (`Σ n_correct/n_candidates` over that bucket) so a low achieved number can
be told apart from "there was nothing to achieve." Reuses `rm_execution`/`base_execution`
already computed for (2)/(3) instead of re-running identical SQL, so this costs no extra
LLM calls and only a handful of extra local `run_sql()` calls (up to `n_candidates - 2`
per row). Each run logs a record to `data/output/eval_runs/` via `rm/model/tracking.py`'s
`log_run`/`load_runs` (reused as-is — already algorithm-agnostic, not tied to RM
training) with a flat `metrics` dict (`rm_top1_accuracy`, `sql_pass_rate_with_rm`/
`_without_rm`, `qa_accuracy_with_rm`/`_without_rm`, `oracle_ceiling` and friends) so
`eval/visualize.py` — the same table+chart pattern as `rm/model/visualize.py`, one color
per metric group and solid/dashed linestyle for with/without RM, plus a third dotted
line for `oracle_ceiling` (no with/without split — it's a property of the LLM's
candidates, not the reranker) — can track the RM-vs-no-RM gap *and* the ceiling it's
bounded by across runs over time.

**Finding: the RM does not transfer to `llama3.2`'s candidate distribution
(2026-08-15).** On the 144-row val set (123 executable rows, `n_candidates=5`),
RM-reranked and no-rerank-baseline are *tied* on end-to-end QA accuracy — 70/123
(56.9%) either way — and the RM's pick actually executes successfully slightly *less*
often than the naive first-candidate baseline (103/123 vs 107/123). This isn't the RM
quietly doing nothing: it disagrees with the baseline pick on 69/123 (56%) rows, but
among those disagreements it's an exact 15-15 split on which pick is actually correct
(39 ties) — when RM changes the answer, it's a coin flip, not an improvement. Likely
cause: the RM was trained to discriminate `sql_good` from Claude-*synthesized*
`sql_bad` negatives (four deliberate, clean mistake categories — see above), a
different error distribution than `llama3.2`'s actual generation mistakes at
`temperature=0.8`; the discriminative signal doesn't transfer out-of-distribution. Not
yet fixed — the natural next step is retraining with `llama3.2`'s own wrong candidates
folded in as `sql_bad`-style negatives, targeting the mismatch directly instead of
assuming synthetic negatives generalize.

**Oracle ceiling: 77.2% (95/123), vs 56.9% achieved — most of the gap isn't RM's to
close.** Among the 5 cached `llama3.2` candidates per val question: 41/123 (33%) rows
have *all 5* candidates correct (any pick wins, nothing to rerank), 28/123 (23%) have
*zero* correct candidates (unreachable by any reranker — a `llama3.2`
generation-quality ceiling, not an RM problem), leaving 54/123 (44%) as the "mixed"
bucket where selection quality actually matters. In that bucket, current achieved
(29/54) is barely above the ~26.8/54 a purely random pick would be expected to get by
chance (weighted by how many of the 5 are correct per row) — confirming the RM has
~no discriminative signal on `llama3.2`'s candidates specifically, consistent with the
disagreement finding above. Originally a one-off ad hoc analysis; re-run against the
now-integrated `evaluate.py` oracle-ceiling logic (see above) and reproduced exactly
(77.2%/41/28/54/29/54), confirming the two methodologies agree.

**Agent baseline on gretel's val split (2026-08-15): the transfer gap is worse here, not
just present.** Same setup (`n_candidates=5`, `llama3.2`), run against
`data/output/gretel/gretel_enhanced.jsonl`'s 147 executable val rows with the current
best RM (`lr_v3`/`C=1000` on the combined dataset — see baseline above). RM accuracy
(`sql_good` vs `sql_bad`) on this split: top1 0.544 — noticeably below the 0.613 the
same RM gets on the combined 80/20 split, consistent with gretel being the harder
subset throughout this file. At the agent level, RM reranking doesn't just fail to
help as on spider/dbasql (tied) — it actively *hurts* both metrics: SQL execution pass
131/147 (89.1%) with RM vs 135/147 (91.8%) without, QA accuracy 55/147 (37.4%) with RM
vs 59/147 (40.1%) without (a ~2.7pp regression each, not noise-sized given the n=147
sample). Separately, raw QA accuracy here (37-40%) is far below spider/dbasql's ~57%
regardless of reranking — `llama3.2` itself generates correct SQL far less often for
gretel's more domain-diverse questions, so this isn't purely an RM transfer problem;
generation quality is also lower on this harder distribution. Logged to
`data/output/eval_runs/20260815T235055Z_lr_v3_C1000_with_gretel_on_gretel_val.json`.

Oracle ceiling on this split confirms it's a generation-quality problem more than a
reranking problem: **51.0% (75/147)**, vs spider/dbasql's 77.2% — a much lower ceiling.
`zero_correct` is 49.0% (72/147, more than double spider/dbasql's 22.8%) — nearly half
the val set is unreachable by *any* reranker, `all_correct` is 24.5% (36/147), leaving
`mixed` at 26.5% (39/147). Within that mixed bucket the pattern from spider/dbasql
repeats and sharpens: achieved-without-RM (59.0%, 23/39) clearly beats both
achieved-with-RM (48.7%, 19/39) *and* the random-chance expectation (45.6%, 17.8/39) —
RM reranking is worse than doing nothing, and only barely better than chance, on
gretel's candidates specifically.

**`sql_bad` label quality: ~48% of candidates weren't distinguishing negatives, mostly
fixed by a 2nd-pass LLM step (2026-08-15).** An ad hoc audit of
`gretel_enhanced.jsonl` — executing every `sql_good`/`sql_bad` pair against its own
row's `sql_context` and diffing result sets — found 3,459/7,214 `sql_bad` candidates
(47.9%, across 85.4% of rows) actually returned the *same* result as `sql_good`, for
two distinct reasons: (1) sparse/homogeneous sample data (e.g. a `missing_filters`
candidate drops `WHERE quarter = 1`, but every seeded row already has `quarter = 1`, so
dropping it is a no-op on that data even though it's a real mistake in general), and (2)
genuinely non-distinguishing SQL — a cosmetic rewrite (renamed alias, reordered clauses)
that computes the same thing regardless of data. `fix_sql_bad.py` (new module,
`test`/`submit`/`collect` shape matching `gen_training_data.py`) detects every flagged
candidate locally via `run_sql` (no LLM needed for detection) and, per affected row,
asks Claude to either append `sql_context` sample rows (case 1) or replace the
candidate with a genuine different-result mistake in the same reason category (case 2),
locally re-verifying the result before accepting it — never touching `sql_good`, never
modifying/removing an existing `sql_context` statement. Run once against the full
1,666-row dataset (1,423 rows sent to the LLM; 47 batch results failed validation and
were left as their original, still-flagged content — no row is ever silently dropped,
output line count always matches input): row-level flagged rate fell from 85.4% to
39.3%, candidate-level from 47.9% to 18.1%.

Caveat that surfaced during verification: the result-set comparison this relies on is
blind to non-`SELECT` statements (`UPDATE`/`DELETE`/`INSERT`/`CREATE VIEW`, ~185/1,666
rows here) — `run_sql` returns `rows=None` for those (no `cursor.description`), so two
*different* UPDATE statements both look like a trivial "match" (`None == None`) even
though they'd have completely different effects on real data. Restricting to the
1,481 rows where `sql_good` is actually a `SELECT` (where the comparison is valid), the
real improvement is: row-level flagged rate 83.7% → 31.7%, candidate-level 43.7% →
10.5% — the non-`SELECT` rows are ~unchanged (184/185 before and after) because the fix
pass's own detection has the same blind spot, so the LLM correctly had nothing
meaningful to fix there. Properly verifying non-`SELECT` statements would need
comparing database state before/after execution rather than a result set — not
attempted here, out of scope for this pass.

**`gen_training_data.py` strengthened to reduce how often rows need the fix pass above
(2026-08-15).** Two changes: (1) `SYSTEM_PROMPT`/`BAD_ONLY_SYSTEM_PROMPT` now explicitly
require every WHERE/HAVING/JOIN/aggregation condition appearing anywhere in `sql_good`
or any `sql_bad` to actually matter for the sample data provided, and require mentally
confirming each `sql_bad` candidate would execute to a genuinely different result before
finalizing it — not just look different. (2) A new active mechanism,
`needs_enrichment()`/`enrich_context()`: after a row is generated, if `sql_good` is a
`SELECT`-shaped statement that executes successfully but returns zero rows, one
follow-up `emit_context_enrichment` tool call asks for more `INSERT INTO` rows (append
-only, existing tables only), locally re-verified before being accepted — wired into
both `cmd_test` (immediately, synchronously) and `cmd_collect` (a pass over the merged
batch results at the end); falls back to the original `sql_context` unchanged if the
addition breaks `sql_good` or still returns zero rows, no retry loop. `needs_enrichment`
deliberately checks `rows is not None and len(rows) == 0`, not just falsy `rows` — an
earlier version used `not execution.rows`, which also (incorrectly) matched
`rows=None` and fired on every DDL/`INSERT`/etc. row (~11% of gretel), asking the LLM to
"enrich" a schema that was never empty in the first place; caught via `cmd_test`
inspection before any real submit/collect run used it. Note this enrichment step runs
*after* `sql_bad` is already generated in the same call, so it doesn't guarantee
already-generated candidates stay meaningfully different post-enrichment (confirmed
directly: a `>` vs `>=` `missing_filters` candidate stayed a false negative because the
enrichment call, unaware of that candidate, added data that didn't happen to fall in the
boundary) — closing that fully would mean re-verifying `sql_bad` after enrichment too,
which `fix_sql_bad.py` above already does generically and can be re-run on any future
output if needed, so it wasn't duplicated here.

**`LRRewardModelV6` — a schema-validity feature closes most of the agent's RM-transfer
gap, now the default everywhere (2026-08-16).** The RM-doesn't-transfer finding above
traced to a distribution mismatch: `is_sql_valid` (pure syntax, `sql_features.py`)
essentially never fires on `llama3.2`'s actual mistakes, because its candidates almost
always parse fine — the real failure mode is referencing a table/column that doesn't
exist in the schema, a class of error pure-syntax checking can't see and the four
synthetic `sql_bad` mistake categories never model either.
`is_schema_valid(sql, sql_context)` (`sql_features.py`) targets this directly: executes
`sql` against `sql_context`'s schema via `run_sql`, cached on `(sql, sql_context)`.
`sql_context` here is expected to be the clean, data-free `CREATE TABLE`-only context
(`clean_context()`, see below), not the full context with sample rows — otherwise a
legitimate query could spuriously "fail" on a data-only constraint (e.g. a UNIQUE
conflict on inserted rows) that has nothing to do with whether the query itself is
well-formed for the schema. No context at all (e.g. `sql_good` is itself DDL) is
treated as valid/neutral. `LRRewardModelV6` (`lr_model_v6.py`) is `lr_v3`'s phi plus
this one feature — cheap, no embedding call, same pattern as `is_sql_valid` itself.

(Aside: `Example.sql_context_clean` — `sql_context` with `INSERT` statements stripped
via `clean_context()`, sqlite-`INSERT` statements identified via `sqlglot` — was added
earlier the same day so the LLM candidate generator and the RM both see only the
schema, not sample-data noise; `fit()`/`evaluate()` and `sql_agent.py`'s `run()` all
use it in place of the full `sql_context` for everything except actual execution, which
still needs the real data.)

On synthetic `sql_bad` pairwise accuracy (5-fold CV, `C=1000`, gretel-only) the feature
barely moves the needle — plain gretel top1 0.5731 vs `lr_v3`'s 0.5725, gretel_opus
(see below) 0.5918 vs 0.5829 — expected, since the synthetic negatives were written to
reference real schema elements, just with wrong logic. The win is entirely at the
agent level, where `llama3.2`'s actual candidates do hallucinate schema: on
gretel_opus's val split, SQL execution pass rate with RM reranking jumped from 76.6%
(`lr_v3`, *worse* than the 85.8% no-rerank baseline) to 96.5% (*better* than baseline);
QA accuracy's with/without-RM gap shrank from -5.6pp to -1.4pp; and the mixed-bucket
(selection-matters) achieved rate flipped from below random-chance expectation (37.3%
vs 45.1%) to above it (49.0%). Reproduced on plain gretel too (SQL pass 80.3%→97.3% vs
an 86.4% baseline, mixed-bucket 39.6%→50.0% vs 46.2% chance) — not a
gretel_opus-specific artifact. `sql_agent.py`'s `run_agent()`, `evaluate.py`'s
`--rm-class`, and `train.py`/`cross_validate.py`'s `--model` now all default to
`lr_v6`.

**`lr_v6` `C` re-swept on gretel-only data (2026-08-16): `C=300`, not the inherited
`lr_v3` defaults of `30`/`1000`.** Same sweep methodology as the original `lr_v3`
sweeps (5-fold CV, `C` from 1 to 30000), run separately on `gretel_opus` and plain
gretel. Both plateau over roughly the same `C=100`-`30000` range as `lr_v3` did, but
the peak sits lower this time: `gretel_opus` peaks at `C=300` (top1 0.5993±0.0266,
pairwise 0.8791±0.0077, mrr 0.7766±0.0141), plain gretel peaks at `C=100` (top1
0.5863±0.0322) with `C=300` a close second (0.5791±0.0335) — close enough between the
two datasets' peaks (within ~1pp, inside fold noise) that a single shared value beats
splitting by dataset, and `C=300` is the better shared pick since it's gretel_opus's
actual peak and only marginally off plain gretel's. Both `train.py` and
`cross_validate.py` now default to `--C 300`. This beats the old inherited default
(`C=30`, tuned for `lr_v3` on spider/dbasql) by a consistent +1.7-1.8pp top1 on both
datasets — a real gain, not noise. Not yet swept on the combined spider/dbasql+gretel
pool (`lr_v3`'s other tuned point, `C=1000`, was for that larger dataset) — if `lr_v6`
is ever trained on the combined set, `C` should be re-swept there too rather than
assuming `300` transfers.

QA accuracy with RM reranking still trails the no-rerank baseline by a small margin on
both datasets (39.7% vs 41.1% gretel_opus, 40.1% vs 41.5% plain gretel) —
`is_schema_valid` fixes "does the picked SQL even execute" almost completely but says
nothing about whether its logic is actually correct, so a schema-valid-but-wrong-answer
candidate can still outrank a correct one. Not addressed here.

**`gretel_opus` (2026-08-16)** is the same gretel pipeline above, run a second time with
`ANTHROPIC_MODEL=claude-opus-5` instead of the default `claude-sonnet-5`, output kept
separate under `data/output/gretel_opus/` (never overwrites the Sonnet-generated
`data/output/gretel/`) so the two are directly comparable. `gen_training_data.py`
bad-only mode against the same 1,668 llm-ready rows: 1,601 succeeded (67 failed tool
validation, same "occasionally omits a required field" gap seen with Sonnet, just a
different nested-shape failure mode). Pre-`fix_sql_bad.py`, Opus's negatives were
already more distinguishing than Sonnet's on the same data (68.3% rows / 26.5%
candidates flagged vs Sonnet's 80.8% / 42.7%); post-fix, 21.1% / 5.7% vs Sonnet's
27.0% / 8.7%. `lr_v3`/`C=1000` CV on `gretel_opus` beats plain gretel on all three
headline metrics (top1 0.5829±0.0219 vs 0.5725±0.0330, pairwise 0.8778±0.0082 vs
0.8625±0.0140, mrr 0.7703±0.0128 vs 0.7642±0.0205) — a real but modest gain, smaller
than `lr_v6`'s agent-level effect above.

**`walt.rm.data.synth` — Spider-based deterministic RM dataset, no LLM involved
(2026-08-16).** A third, independent pipeline (`src/walt/rm/data/synth/`, not registered
in `pre_process.py`'s `SOURCES`) built on the official Spider release (real per-database
SQLite files + gold SQL, manually downloaded — see Setup) rather than a flattened
question/SQL CSV or an LLM-synthesized schema. Two things distinguish it from every
other source in this file:

1. **`sql_bad` is generated by a rule-based sqlglot-AST corruption engine
   (`corrupt.py`), not an LLM.** Five categories — `missing_filters` (drop a WHERE/HAVING
   conjunct), `wrong_aggregation` (swap/drop an agg function), `unsafe_patterns`
   (`SELECT *` or an ON-join → `CROSS JOIN`), `misjoined_tables` (swap a join column/table
   for a wrong-but-plausible one, using a lightweight `schema.py` model parsed from real
   DDL), and `compound` (two of the above applied in sequence, or the same one twice when
   a query structurally supports only one — e.g. no JOIN/aggregation present). Reuses the
   exact 4 base reason strings from `gen_training_data.py`'s `BAD_SQL_REASONS` so this
   source's rows merge cleanly into any `pairwise_accuracy_by_reason` breakdown. Every
   candidate is verified against the real DB (never assumed): candidates that fail to
   execute are kept as strong negatives; candidates that execute but return the *same*
   result as `sql_good` are still kept (never silently dropped) but flagged in
   `synth_bad_candidate_qc.json` — 6.9% of 6,812 generated candidates on the first real
   run, a healthy rate (cf. the ~44-48% pre-`fix_sql_bad.py` rate on LLM-generated
   negatives elsewhere in this file).

2. **Rows don't embed `sql_context` — they carry `sql_context_path` instead.** `Example`
   (`rm/data/base.py`) gained a new optional field, `sql_context_path: str | None`
   (relative to `$DATA_PATH`, e.g. `"spider/database/restaurants/restaurants.sqlite"`),
   left empty for every other source. Traced every `sql_context` consumer before adding
   this: RM training/CV/`evaluate()` already only read `sql_context_clean` (DDL-only,
   still always populated, annotated with `-- one of: ...` categorical-value comments for
   low-cardinality TEXT columns via `annotate_ddl`) — untouched by this change, including
   `lr_v6`'s `is_schema_valid` feature. Only real SQL execution (the agent, and
   `eval/evaluate.py`'s baseline/reference/oracle-ceiling checks) needed updating:
   `sql_exec.py`'s new `execute_with_context(sql_context, sql_context_path, sql)` is a
   drop-in replacement for `run_sql(ex.sql_context, sql)` everywhere it was called, and
   `SqlAgent.run()` gained matching `sql_context_clean`/`sql_context_path` parameters
   (both default to the old derive-from-`schema_context` behavior, so every pre-existing
   caller is unaffected). When `sql_context_path` resolves a real `.sqlite` file (not the
   embedded-`sql_context` case), `execute_with_context` reuses one in-memory connection
   across every call sharing that path — built once via `sql_exec.load_context_from_sqlite`
   (reads `sqlite_master` + one `INSERT` per row directly, **not**
   `sqlite3.Connection.iterdump()`: Python 3.12+'s `iterdump()` runs `PRAGMA
   foreign_key_check` internally and raised on real FK/encoding issues found in Spider's
   own `yelp.sqlite`/an HR-domain DB's `last_name` column — data problems this module has
   no reason to fail on, since it only needs the CREATE TABLE/INSERT text) — and is
   invalidated (closed, evicted, rebuilt fresh next call) the instant a mutating statement
   (INSERT/UPDATE/DELETE/CREATE/DROP/ALTER, via a conservative sqlglot check) runs against
   it, so no candidate's side effect can leak into another's result. `eval/evaluate.py`
   additionally sorts its val rows by `sql_context_path` before the main loop so rows
   sharing a database run consecutively and the cache actually stays warm.

Dataset selection had to become multi-database: Spider's `train_spider.json`/
`train_others.json` (146 databases) and `dev.json` (20 databases) are **fully disjoint
database sets** (confirmed by direct computation, zero overlap) — Spider's own
cross-database-generalization design — so `trainval` rows are pooled from the first two
files' databases and `val` rows from `dev.json`'s, a stronger held-out guarantee than any
other source here (different schemas entirely, not just different rows). No single
3-6-table database reaches the target row counts (best single candidate: `restaurants`,
125 pairs) — `spider_source.shortlist_candidates` instead ranks all `min_tables<=N<=
max_tables` (default 3-8) databases by FK-density-then-pair-count, and
`select_dbs_for_target` greedily accumulates databases (verifying each against its real
data) until the target is reached, only processing as many as actually needed. One
empirically-discovered guard: `MAX_SQLITE_SIZE_BYTES` (5MB) excludes `soccer_1`
(317MB/196,823 rows), `wta_1` (105MB), and `baseball_1` (30MB) from candidacy — three
huge outliers among otherwise sub-4MB databases that turned a 2-second run into 7+
minutes (and, for `wta_1`, would have hit the val pool) for a handful of pairs each, with
no relevance to schema/join complexity. Default run (`--train-count 2000 --val-count
300`): 2,300 examples across 38 databases (33 `trainval`, 5 `val`, zero `db_id` overlap
between them) in ~2 seconds end-to-end, verified round-tripping through
`Example.from_dict`/`load_examples()` and a live `evaluate.py`/`sql_agent.py` run against
Ollama + a real `lr_v6` model.

**Synth baseline: `lr_v6`/`C=300` (2026-08-16) — RM discrimination saturates, but this is
the first dataset where reranking transfers cleanly to `llama3.2` (2026-08-16).** 5-fold
CV and the standard 80/20 split both land at **top1/pairwise/mrr ≈ 1.00**, every mistake
category included — confirmed not an `is_schema_valid`-specific artifact (`lr_v3` without
it still hits 99.6% top1 via CV). Read this as the deterministic corruptions being far
more surface-distinguishable than the subtle LLM-crafted negatives used everywhere else
in this file, not as the RM task being meaningfully "solved" — a `C` sweep was skipped as
pointless here (no headroom to tune against). The real signal is agent-level, on the
300-row val set (`llama3.2`, 5 candidates/question, held-out on entirely disjoint
databases from `trainval` — see above): **SQL execution pass 286/300 (95.3%) with RM vs
216/300 (72.0%) without; QA accuracy 166/300 (55.3%) with RM vs 139/300 (46.3%)
without.** Oracle ceiling 68.0% (204/300) — 16.0% all-correct, 32.0% zero-correct
(unreachable by any reranker), 52.0% mixed (selection matters); within that mixed bucket,
RM-achieved (75.6%) clearly beats both no-rerank (58.3%) and random-chance expectation
(51.8%). This is the **first source in this file where `lr_v6` reranking transfers
cleanly** to `llama3.2`'s actual candidates on both metrics — spider/dbasql was tied,
gretel/gretel_opus saw reranking actively hurt both metrics (see above). Plausible cause,
not confirmed: Spider's gold SQL and this module's corruptions both come from/target a
narrower, more template-like query distribution than gretel's, closer to what `llama3.2`
itself tends to generate, so the RM's learned signal (schema-validity + embedding
similarity) may simply generalize better here — not re-examined further. Artifacts:
`data/output/rm_model_synth_lr_v6_C300.joblib`,
`data/output/rm_metrics_synth_lr_v6_C300.json`, `data/output/eval_synth_lr_v6_C300.json`,
runs logged to `data/output/runs/`/`data/output/eval_runs/` as `synth_lr_v6_C300*`.

**Synth ablations: the learned RM adds ~nothing on top of a hard schema-validity filter
— confirmed across `lr_v6`, distilbert, and a cross-domain model (2026-08-16).** Four
follow-up experiments, all against the same 300-row synth val set / cached `llama3.2`
candidates as the baseline above (so directly comparable), each written to its own new
`data/output/synth_*/` folder — nothing under `data/output/synth/` or the shared
`data/output/runs/`/`eval_runs/` was touched:

- **`add_llama_negatives.py` on synth's `trainval` split** (`data/output/synth_llama/`)
  — this script predates `sql_context_path` and read `record.get("sql_context") or []`
  directly, which is always empty for synth rows; fixed by adding
  `sql_exec.resolve_context_statements(sql_context, sql_context_path)` (returns
  `sql_context` as-is if non-empty, else loads the full context from the path — the same
  resolution `execute_with_context` does, minus the caching, for callers like this one
  that need the raw statement list rather than to execute one query) and using it in
  `llama_negatives_for_row`. 2,000 `trainval` rows in ≈13s (helped by the LLM cache);
  1,423 rows (71.2%) got at least one new `reason="llama"` negative, 2,875 total. `val`
  rows confirmed byte-identical before/after (verified directly, not assumed). Training
  `lr_v6`/`C=300` on this: CV top1 dropped 0.9985→0.9840, val top1 1.0000→0.9733 (two of
  the four *original* categories got measurably worse: `missing_filters` 100%→95.89%,
  `wrong_aggregation` 100%→98.82%) — and at the agent level, QA accuracy dropped
  166/300→159/300 and mixed-bucket achieved 118/156→111/156. **The opposite of what this
  technique fixed for gretel** (see the RM-transfer finding above) — plausible cause: the
  synth RM was already transferring cleanly, so there was no gap for llama negatives to
  close, and `C=300` was never re-tuned for the harder, more diverse signal this adds.

- **Cross-domain check: evaluating synth's val set with `rm_model_lr_v6_C300_gretel_llama.joblib`**
  (a model trained on an entirely different dataset — gretel+llama, no Spider schemas
  ever seen) (`data/output/synth_cross_eval/`) — SQL pass rate is *identical* to the
  synth-trained model (286/300, 95.3%) and QA accuracy is close (163/300, 54.3% vs the
  synth-trained model's 166/300, 55.3% — a 3-row difference on n=300). The RM's own
  pairwise discrimination accuracy on synth's `sql_good`/`sql_bad` pairs is where the real
  gap is (87.0% cross-domain vs 100% synth-trained) — that gap just doesn't propagate
  through to agent-level behavior. **Of the ~9pp total QA-accuracy lift from reranking,
  ~8pp (≈89%) is already captured by a model that never saw this dataset's training
  data** — most of what looks like "the RM helping" here isn't specific to synth's own
  training data at all.

- **The actual driver, isolated directly**: `evaluate.py` gained `--rm-class constant`
  (`rm/model/constant_model.py` — scores every candidate `0.0`, so `rank()`'s stable sort
  just preserves the LLM's original generation order) and `--schema-filter`
  (`rm/model/schema_filter.py` — wraps any model, adding a large penalty to any candidate
  failing `is_schema_valid` before delegating to the inner model's own score, so a
  schema-valid candidate always outranks an invalid one regardless of what the inner
  model prefers among survivors). `constant` with no filter is byte-for-byte identical to
  no reranking at all (216/300 SQL pass either way — confirms the wiring: a tied score
  really does degrade to "first candidate, LLM's own order"). `constant` **+**
  `--schema-filter` — zero learned signal, zero training data, just "reject
  schema-invalid candidates, keep the LLM's order among the rest" — hits 286/300 (95.3%)
  SQL pass and **168/300 (56.0%) QA accuracy, the best QA accuracy of every configuration
  tested in this file, beating the synth-trained `lr_v6` model itself** (166/300, 55.3%).
  SQL pass rate is 286/300 across *every* schema-filtered configuration regardless of
  which model sits underneath — 100% of that lift is the filter, 0% is anything learned.

- **Distilbert, synth+llama, with and without the filter** (`data/output/synth_llama/`,
  `rm_model_distilbert_synth_llama.pt`) — training itself lands in the same "synth is
  easy" pattern as `lr_v6`: top1/pairwise/mrr 0.9300/0.9812/0.9627, vs the reference
  gretel+llama distilbert run's 0.5526/0.8830/0.7420 (same architecture, same recipe,
  wildly different difficulty — confirms this isn't `lr_v6`-specific). At the agent
  level, **without** `--schema-filter` it's actively harmful — 211/300 (70.3%) SQL pass
  and 128/300 (42.7%) QA accuracy, *both below* the no-rerank baseline (216/300 / 46.3%),
  and its mixed-bucket achieved rate (51.3%) sits at/below random chance (51.8%) —
  reproducing on synth exactly the gretel-side finding that motivated building
  `schema_filter.py` in the first place (a model that sometimes actively prefers a
  schema-invalid candidate). **With** `--schema-filter` it converges to numbers
  essentially identical to every other schema-filtered configuration — 286/300 (95.3%) /
  166/300 (55.3%) / 118/156 (75.6%), matching `constant`+filter and the synth-trained
  `lr_v6` almost exactly.

**Net takeaway for this dataset**: the real baseline to beat isn't "no reranking"
(72.0%/46.3%) — it's "schema-filter, no learned model at all" (95.3%/56.0%), and nothing
trained here (three `lr_v6` variants, one distilbert variant) actually clears that bar.
`is_schema_valid` is carrying essentially the entire agent-level result on this val set.

**Severity-scored synth pipeline: llama3.2-sourced sql_bad + Claude-assigned 0-5
severity, no rule-based corruption (2026-08-16).** A second Spider-based pipeline,
`src/walt/rm/data/synth/build_severity_dataset.py` (stage 1) + `enhance_severity_dataset.py`
(stage 2), reusing `spider_source.py`'s DB/pair selection completely unchanged but
replacing `corrupt.py`'s deterministic AST corruptor entirely — this pipeline's
`sql_bad` comes solely from `llama3.2`'s own real generation mistakes (via a local
Ollama server) plus Claude backfill, on the theory that real generation mistakes are a
better training signal than synthetic ones for a reward model meant to rerank a real
LLM's candidates (see the RM-transfer finding elsewhere in this file). Runs the
identical flow for both `trainval` and `val` pools — no special-casing — so
`evaluate.py`'s RM-discrimination metric has `sql_bad` on val rows exactly like
training does. Output lives under `data/output/synth_severity/`, entirely separate
from `data/output/synth/`; `build_synth_dataset.py` is untouched.

Stage 1 generates up to `--n-ollama-candidates` (default 5) candidates per row via
`OllamaLLM`/`CachingLLM` (sharing the project's normal `llm_cache.json`), verifies each
via `corrupt.verify_candidate()` (reused as-is — a generic execution/result-match
check, not tied to the corruptor) against the row's own DB connection. Unlike
`add_llama_negatives.py`, same-result candidates are *kept*, not dropped — a
`matches_gold: bool` hint travels with each raw candidate into an intermediate,
pipeline-private JSONL (`sql_bad` entries are `{"sql", "matches_gold"}`, no `reason`
yet — nothing upstream assigns one). A non-SELECT-shaped candidate (Spider's
`sql_good` is always a SELECT) is recorded as a real, un-executed mistake rather than
run against the shared per-DB connection at all, since a hallucinated DML statement
executing there would corrupt every later pair's verification sharing that connection
— corrupt.py's own candidates never had this risk, being SELECT-shaped AST mutations
by construction, but an LLM's raw output carries no such guarantee.

Stage 2 is one combined Claude tool call (`emit_severity_review`) per row,
`test`/`submit`/`collect`-shaped exactly like `gen_training_data.py`/`fix_sql_bad.py`
(batch state cached the same way, shared helpers reused directly). Reason taxonomy:
the same 5 base categories (`missing_filters`/`wrong_aggregation`/`unsafe_patterns`/
`misjoined_tables`/`compound`) plus `llama` as an explicit catch-all — but `llama` is
valid *only* for categorizing an already-generated (llama3.2-sourced) candidate whose
mistake doesn't cleanly fit the 5; a *new* candidate Claude proposes to fill thin
coverage must always target one of the 5 (enforced by a stricter, separate schema on
that array — the model tried to slip `llama` into a proposed-new-candidate's reason
often enough during `test`-mode iteration that this needed its own JSON-schema-level
enum, not just prompt wording). Coverage rule: backfill up to 3 new candidates when
the pool has fewer than 3 non-zero-severity candidates or doesn't span both a "low"
(1-2) and "high" (4-5) severity band.

**Severity=0 handling — a real design tension, resolved as a hint, not a hard rule
(2026-08-16).** Initial version force-set `severity=0` whenever the local
`matches_gold` hint was true, overriding whatever the LLM said — but `matches_gold`
only proves the candidate matched *on this row's specific sample data*, not general
equivalence, and a live example surfaced the gap immediately: `LIKE '%Paper%'` vs
`= 'Paper'` matched only because no other value in that table's small category list
happened to contain the substring "Paper" — a real, general `missing_filters` mistake
the sample just didn't expose, exactly the same class of problem `fix_sql_bad.py` was
built to catch for the original LLM-negatives pipeline (see its own section above).
Fixed: the LLM is now trusted when it keeps a nonzero severity despite the hint (an
"override" — it's making a judgment call about generalization the hint can't make),
but a severity=0 *claim* for a candidate the hint proves genuinely differs is still
clamped to 1 — that's a plain factual error (proven by real execution), not a judgment
call, so it stays hard-enforced. `new_candidates`' non-executing entries are still
hard-rejected regardless (no provenance guarantee the way an existing llama3.2-sourced
or corrupt.py-sourced candidate has). At full scale (2300 rows), this asymmetry
mattered: 1,122 candidates kept a nonzero severity despite a same-result hint, vs
2,710 confirmed severity=0 and 202 false-zero claims clamped — the override case is
common enough that forcing it to 0 unconditionally would have discarded real signal on
roughly a third of the hinted candidates.

**Data model: `SQLBadCandidate.severity` (2026-08-16).** New optional field,
`severity: int | None = None` — 0-5, only ever populated by
`enhance_severity_dataset.py`; every other source's rows keep `severity=None` forever
(the field is omitted from `to_dict()` output when `None`, so pre-existing JSONL files
round-trip byte-identical). `LRRewardModel.fit()` (`lr_model.py`, shared by every LR
variant via inheritance) now builds pairwise training data via an effective-rank rule
instead of pure good-vs-bad: `sql_good` always beats every bad; `severity is None`
bads are only ever paired against `sql_good` (never each other) — this is what
guarantees an all-`None` dataset produces bit-for-bit identical training pairs, in the
identical RNG-consumption order, as before this change; `severity == 0` bads are
excluded from every pair entirely (no signal in a tied comparison); `severity in 1..5`
bads are additionally paired against every *other* `1..5` bad on the same example with
a *different* severity, teaching relative ranking, not just a binary boundary.
`BaseRewardModel.evaluate()` mirrors this — severity=0 candidates are excluded from
both pairwise accuracy and the `top1_accuracy`/`mrr` ranking, since a model ranking one
above `sql_good` isn't making a real mistake. Confirmed via direct pair-count
comparison on real data: the training set produces 78 pairs without severity signal vs
112 with it engaged (severities `[0,2,2,5]`-style rows), and a full `train.py`
regression run against an existing no-severity dataset (`gretel_enhanced.jsonl`)
produced metrics byte-identical to before this change.

**`--severity-zero-as-positive`: an opt-in, not-yet-adopted ablation (2026-08-16).**
`LRRewardModel` also takes `severity_zero_as_positive: bool = False` (wired through
`train.py`/`cross_validate.py` for every LR variant) — when on, `fit()` additionally
treats every severity=0 candidate as a second positive anchor, paired against every
real bad exactly like `sql_good` is (never against `sql_good` or another severity=0
candidate — those are tied, no signal). Default off deliberately: severity=0 is the
least-trustworthy label in the pipeline (see above), so promoting it to a trusted
*positive* example multiplies the damage of any mislabel rather than just wasting one
skipped pair — this needs a real A/B comparison before being adopted, not a default
flip. `evaluate()` is untouched by this flag on purpose, so both configs are scored by
an identical criterion. Caught one real bug enabling it for the first time:
`warm_cache()` was building its embed-cache set via `all_candidates()`, which excludes
severity=0 SQL (correct for `evaluate()`'s ranking) — `fit()` with the flag on then
hit a `KeyError` looking up an embedding that was never cached. Fixed by having
`warm_cache()` embed every `sql_bad` candidate directly, regardless of severity (a
cached-but-unused embedding is free; a missing one is a hard crash).

**A/B result: rejected — the ablation is a net negative (2026-08-16).** 5-fold CV,
`lr_v6`/`C=300`, full 2,000-row trainval severity-scored synth set, flag off vs on:
top1 0.9265±0.0119 vs 0.9170±0.0201, pairwise 0.9754±0.0041 vs 0.9724±0.0081, mrr
0.9587±0.0067 vs 0.9537±0.0120 — worse on all three headline metrics *and* roughly
1.8-2x the fold-to-fold variance with the flag on. Consistent with the caution above:
severity=0 remains the least-trustworthy label in the pipeline even after the Q1 hint-
not-rule fix, and promoting it to a trusted positive anchor adds noise rather than
useful signal. Verdict: keep the default off — this is a tested, rejected ablation,
not a live option. Runs logged as `synth_severity_lr_v6_C300_zero_off`/`_zero_on` in
`data/output/runs/`.

**Client-side Ollama concurrency + a real server-side bottleneck (2026-08-16).**
`OllamaLLM` takes `max_concurrency: int = 1` (default off — today's exact sequential
behavior) — when raised, the `n` candidate-generation calls for one row fire
concurrently via a thread pool, always returned in seed order regardless of completion
order. Deliberately scoped to *within* one row's `generate_candidates()` call, not
across rows: only one such call is ever in flight at a time from any caller in this
codebase, so `CachingLLM`'s cache write (documented as not concurrency-safe) never
sees concurrent writers even with this enabled. Wired through everywhere `OllamaLLM`
gets constructed (`build_severity_dataset.py`, `sql_agent.py`/`evaluate.py`,
`add_llama_negatives.py`) as `--ollama-concurrency`. Critically, this alone does
nothing: inspecting the actual `llama-server` process Ollama spawns showed it running
with `-np 1` (llama.cpp's parallel-request-slot flag) by default on this machine —
concurrent client requests just queue at the server with no throughput gain until the
*server's* slot count is raised. Fix (system-level, not code):
`launchctl setenv OLLAMA_NUM_PARALLEL 4` (a plain shell `export` doesn't reach it —
Ollama.app isn't launched from the shell), then quit and reopen Ollama.app; verified
via `ps aux | grep llama-server` showing `-np 4` after the restart. Restarting Ollama
kills any in-flight request from a currently-running script with no retry logic, so
this needs to happen between runs, not during one. Separately: `CachingLLM`'s cache
key (`sha256(namespace, question, schema_context)`) does **not** include the prompt
text (`OllamaLLM.SYSTEM_PROMPT`/message template) — editing the prompt silently does
not invalidate existing cache entries for the same `(question, schema_context)`, a
known, not-yet-fixed gap worth remembering before ever tuning that prompt.

**Progress visibility for long-running Ollama/Claude-loop scripts (2026-08-16).**
Python fully buffers stdout by default whenever it isn't a TTY — a real problem for a
backgrounded multi-hour run (`build_severity_dataset.py`'s full-scale run looked
completely silent for 40+ minutes despite processing normally; progress was only
inferable indirectly, from `llm_cache.json`'s growing entry count). Fixed by calling
`sys.stdout.reconfigure(line_buffering=True)` at the top of `main()` in every
long-running Ollama/Claude-loop script (`build_severity_dataset.py`,
`enhance_severity_dataset.py`, `add_llama_negatives.py`, `evaluate.py`). Separately,
`build_severity_dataset.py`'s only progress signal used to be one line per *database*
fully finished (`ProgressTracker`, new) — too coarse for a run spanning few, large
DBs; now prints every 10 rows (and always on the final row) with a rate/ETA estimate,
denominated against the *actual* pairs-to-attempt count (`select_dbs_for_target`
commonly overshoots `--train-count`/`--val-count` before the final downsample, so this
is not the same number as the CLI target). `add_llama_negatives.py` and
`evaluate.py`'s agent loop got the same periodic-print treatment (`evaluate.py`'s
additionally reports running with/without-RM pass rates alongside the ETA, since
that's the metric that actually matters mid-run).

**Full-scale run results: `lr_v6`/`C=300` on the severity-scored synth dataset
(2026-08-16).** `--train-count 2000 --val-count 300` (2,300 rows total, matching the
deterministic-corruptor synth baseline's scale for direct comparison) — stage 1
generated 9,000 raw candidates across 2,290/2,300 rows (10 got none from `llama3.2`,
left for stage 2 to backfill), 3,849 (43%) flagged same-result by the local hint.
Stage 2's Message Batch (2,300 requests) completed cleanly: 2,282 succeeded, 18
(0.78%) fell back to local-only categorization after invalid tool-call output (a few
residual `llama`-in-`new_candidates` slips despite the stricter schema, and the model
occasionally wrapping its whole response as a string instead of the expected object —
both handled gracefully via `apply_severity_review`'s fallback path, never dropping a
row). Training (`train.py --model lr_v6 --C 300`, no re-sweep — inherited from the
gretel-only tuning): top1 0.9225, pairwise 0.9693, mrr 0.9548 on the standard 80/20
split, overfitting gap a modest +0.026/+0.016/+0.018 — well above the gretel baselines
(~0.55-0.60 top1) and below the old deterministic-corruptor synth baseline (~1.00,
"too easy" by design), landing in between as expected for real-but-noisier negatives.

Agent-level eval on the 300-row val split (fully held-out on disjoint Spider
databases, `llama3.2`, 5 candidates/question) confirmed RM discrimination transfers
cleanly to the held-out DBs (top1 0.9133/pairwise 0.9704/mrr 0.9503, close to the
trainval-split numbers) and, more importantly, reranking clearly helps at the agent
level: SQL execution pass 279/300 (93.0%) with RM vs 216/300 (72.0%) without; QA
accuracy 162/300 (54.0%) with RM vs 139/300 (46.3%) without. Oracle ceiling 68.0%
(204/300) — 16.0% all-correct, 32.0% zero-correct, 52.0% mixed; within the mixed
bucket, achieved-with-RM (73.1%) clearly beats both achieved-without-RM (58.3%) and
random-chance expectation (51.8%). This is the **second** source in this file (after
the deterministic-corruptor synth pipeline) where reranking transfers cleanly and
helps — unlike gretel/gretel_opus, where it was tied or actively harmful — landing
within ~2pp of the old synth baseline's with-RM numbers (95.3%/55.3%) despite
replacing rule-based corruption entirely with real-mistake-derived negatives.
Artifacts: `data/output/rm_model_synth_severity_lr_v6.joblib`,
`data/output/rm_metrics_synth_severity_lr_v6.json`,
`data/output/eval_synth_severity_lr_v6_C300.json`, runs logged to
`data/output/runs/`/`data/output/eval_runs/` as `synth_severity_lr_v6_C300*`.

**The "Synth ablations" finding reproduces on the severity-scored dataset too — and
here schema-filtering doesn't just tie the trained model, it beats it (2026-08-16).**
Same isolation methodology as the deterministic-corruptor synth ablations above
(`--rm-class constant --schema-filter`: zero learned signal, zero training data, just
reject schema-invalid candidates and keep the LLM's own order among survivors), run
against this dataset's identical 300-row val set: SQL execution pass 286/300 (95.3%,
vs trained `lr_v6`'s 279/300, 93.0%), QA accuracy 168/300 (56.0%, vs 162/300, 54.0%),
mixed-bucket achieved 76.9% (vs 73.1%) — `constant`+schema-filter wins on every
metric. Confirms the embeddings + severity-aware training aren't adding agent-level
value beyond `is_schema_valid` on this dataset either; right now they're costing a
couple points, not gaining any. Telling detail in the RM-accuracy numbers:
`constant`+schema-filter's `pairwise_accuracy` is 0.2915 — *below* random chance —
because `SchemaFilteredRewardModel` only distinguishes schema-valid from
schema-invalid candidates; when `sql_good` and a `sql_bad` are both schema-valid (the
common case — most negatives here are syntactically fine, just semantically wrong),
they tie at the same constant score, and `evaluate()`'s strict `>` comparison counts a
tie as a miss. Schema validity is strong at filtering genuinely broken/hallucinated
LLM candidates but has ~no power over the semantic mistakes this pipeline's severity
scoring was actually built to teach — that gap is exactly where a learned model
*should* be adding value and currently isn't. Artifacts:
`data/output/eval_synth_severity_constant_schemafilter.json`, logged as
`synth_severity_constant_schemafilter` in `data/output/eval_runs/`.

**Full ablation ladder: without `is_schema_valid`, reranking is actively harmful, not
just unhelpful (2026-08-16).** Two more points complete the picture from the previous
paragraph — `lr_v3` (embeddings + `is_sql_valid` only, `is_schema_valid` absent from
both training and inference — no `--schema-filter` either) trained and evaluated on
the identical setup: pairwise RM-discrimination is essentially unchanged from `lr_v6`
(top1 0.9225/pairwise 0.9710/mrr 0.9553 vs `lr_v6`'s 0.9225/0.9693/0.9548 — confirms
`is_schema_valid` was never doing much for synthetic-pair discrimination, matching its
original finding elsewhere in this file), but at the **agent** level it's a different
story entirely: SQL pass 211/300 (70.3%) and QA 135/300 (45.0%), both *below* the
no-rerank baseline (216/300 72.0% / 139/300 46.3%), and the mixed-bucket achieved rate
(55.8%) falls below random-chance expectation (51.8%) — reranking with this feature
missing is actively worse than doing nothing, reproducing the same RM-transfer failure
already documented for gretel/gretel_opus. Full ladder on the identical 300-row val
set: no-rerank 72.0%/46.3% → `lr_v3` (no schema signal anywhere) 70.3%/45.0% (worse
than no-rerank) → `lr_v6` (schema signal learned in) 93.0%/54.0% → `constant`+
schema-filter (schema signal only, no embeddings) 95.3%/56.0% (best). Conclusion:
`is_schema_valid` isn't just the dominant contributor to `lr_v6`'s agent-level result
— on this dataset it is the *entire* reason reranking helps at all, and the embeddings
signal the rest of the model relies on is actively counterproductive once real LLM
candidates (rather than curated synthetic pairs) are being ranked. Artifacts:
`data/output/rm_model_synth_severity_lr_v3.joblib`,
`data/output/eval_synth_severity_lr_v3_C300.json`, logged as
`synth_severity_lr_v3_C300` in `data/output/runs/`/`data/output/eval_runs/`.

**Training exclusively on schema-valid pairs + a hard filter at inference — a
plausible hypothesis that didn't pay off (2026-08-16).** Since `is_schema_valid`
already handles schema-invalid candidates for free via a hard filter (no training
needed — see above), and the same filter has ~no power over schema-valid-vs-valid
discrimination (`pairwise_accuracy` 0.29, below random chance, when both sides of a
pair are schema-valid), the natural next question: does removing schema-invalid
`sql_bad` from training let a linear model's limited capacity focus entirely on the
actually-hard problem? `walt.rm.data.filter_schema_valid` (new, reuses
`sql_features.is_schema_valid` against each row's own `sql_context_clean`) drops
schema-invalid candidates from every row uniformly (both splits, no special-casing):
2,649/13,634 (19.4%) dropped, 14/2,300 rows left with zero `sql_bad`. Training `lr_v6`/
`C=300` (unchanged, not re-swept for the smaller/harder pair distribution) on the
result: pairwise discrimination *within* schema-valid pairs is 0.9647 (test set) —
genuinely harder than the unfiltered model's 0.9693-0.9710 on the full pair set,
confirming the filter did remove the easy cases, not just noise. But combined with
`--schema-filter` at inference, agent-level results came in *worse* than both
`constant`+schema-filter and the original full-data `lr_v6`: SQL pass matches exactly
(95.3% — expected, since with the hard filter applied, pass rate is governed almost
entirely by "does ≥1 schema-valid candidate exist," independent of which scorer breaks
ties among survivors), but QA accuracy (53.7% vs 56.0%/54.0%) and mixed-bucket
achieved (72.4% vs 76.9%/73.1%) both dropped. Two honest caveats before calling this a
clean negative — `C` was never re-tuned for the new, harder pair distribution, and
14/2,300 rows lost all their negatives entirely — but directionally this doesn't
support the capacity-focusing theory: the dropped schema-invalid negatives evidently
carried real embedding-space signal beyond "this is schema-invalid," and removing them
didn't sharpen semantic discrimination. Artifacts:
`data/output/synth_severity/synth_severity_enhanced_schemavalid.jsonl`,
`data/output/rm_model_synth_severity_schemavalid_lr_v6.joblib`,
`data/output/eval_synth_severity_schemavalid_lr_v6_C300_schemafilter.json`, logged as
`synth_severity_schemavalid_lr_v6_C300*` in `data/output/runs/`/`data/output/eval_runs/`.

**`lr_v4` (soft schema-cosine-similarity, no `is_schema_valid`) reproduces its gretel
null result here too — and sharper (2026-08-16).** V4/V5 (see the original gretel
ablation above) were never tried on the severity-scored synth dataset; V4 adds one
scalar, `cosine_sim(embed(sql_context_clean), embed(sql))`, on top of V3's phi — no
hard execution check anywhere, the exact "soft schema awareness, no `is_schema_valid`"
constraint. `evaluate.py`'s `RM_CLASS_CHOICES`/`RM_CLASS_BY_NAME` gained `lr_v4`
(wasn't wired in before; `lr_v3`/`lr_v6`/`distilbert`/`constant` only). Pairwise
discrimination is again tied with plain V3 (top1 0.9125/pairwise 0.9670/mrr 0.9495 vs
V3's 0.9225/0.9710/0.9553 — slightly lower, noise-level). At the agent level it's a
clean null-to-negative result: SQL pass exactly ties the no-rerank baseline (216/300,
72.0% either way), and QA accuracy comes in *below* no-rerank (131/300, 43.7% vs
139/300, 46.3%) — essentially indistinguishable from V3 (no schema signal at all,
70.3%/45.0%), nowhere close to `lr_v6`'s 93.0%/54.0%. `lr_v5` (full context-vector
concat) was not re-run — its null result is a proven mathematical property of the
pairwise-difference training objective (`sql_context` is per-example-constant, so
`embed(sql_context)` cancels to exactly `0.0` in every training row, confirmed to the
last coefficient bit previously), not a data-dependent empirical question, so
re-running it on a new dataset can't produce a different answer. Conclusion: schema
awareness only pays off as a *hard, symbolic, execution-verified* fact — a
hallucinated column can still sit close to the real schema in embedding space (so
cosine similarity barely drops) while definitively failing to execute; no amount of
soft embedding proximity substitutes for actually trying to run the query. Artifacts:
`data/output/rm_model_synth_severity_lr_v4.joblib`,
`data/output/eval_synth_severity_lr_v4_C300.json`, logged as
`synth_severity_lr_v4_C300` in `data/output/runs/`/`data/output/eval_runs/`.

**`lr_v4` under the same schema-valid-only-training + hard-filter treatment as
`lr_v6` above: the cosine-sim feature finally shows real signal, but still trails
both `lr_v6`'s equivalent and the zero-training filter baseline (2026-08-16).** The
`lr_v4` run above trained on the full dataset with no `--schema-filter` at inference —
not an apples-to-apples comparison against the schema-valid-only `lr_v6` variant.
Retrained on `synth_severity_enhanced_schemavalid.jsonl`, evaluated with
`--schema-filter`: SQL pass 286/300 (95.3%, identical to every other filtered
config — expected, the filter alone governs this), QA accuracy 156/300 (52.0%),
mixed-bucket achieved 108/156 (69.2%) — clearly above random-chance expectation
(51.8%) for the first time across every `lr_v4` configuration tried, and a real jump
from the unfiltered version's 53.2%. So the hard filter *does* let the cosine-sim
feature contribute genuine value on top of raw embeddings — it just still isn't
enough: `lr_v6`'s schema-valid-only equivalent beats it (53.7%/72.4%), and
`constant`+filter (zero training at all) beats it by a wider margin (56.0%/76.9%).
Net: under the "no `is_schema_valid`" constraint, this is the best configuration
found so far, but it still costs ~4pp QA / ~8pp mixed-bucket relative to doing
nothing beyond the hard filter. Artifacts:
`data/output/rm_model_synth_severity_schemavalid_lr_v4.joblib`,
`data/output/eval_synth_severity_schemavalid_lr_v4_C300_schemafilter.json`, logged as
`synth_severity_schemavalid_lr_v4_C300*` in `data/output/runs/`/`data/output/eval_runs/`.

**`sql_good` and every `sql_bad` are now sqlglot-normalized before embedding — closes
a real, previously untested formatting leak, and moves `lr_v6`'s tuned `C` from 300 to
0.1 (2026-08-17).** Hypothesis: `sql_good` (Spider's hand-written gold, or gretel's,
etc.) and `sql_bad` never went through the same text-rendering pipeline. For the
deterministic corruptor (`corrupt.py`), every negative is emitted via
`mutated.sql(dialect="sqlite")` — a canonical sqlglot render — while `sql_good` kept
its original hand-formatted casing/spacing verbatim. For the severity pipeline,
negatives are raw `llama3.2` completions with their own generation-style formatting,
again never reconciled with Spider's gold formatting. Either way, a model could
partly be learning "which formatting style does this look like" instead of "is this
query correct" — a shortcut with zero relevance once real LLM candidates (never
hand-written gold, never a bare AST re-render) are what actually gets scored at
inference.

First attempt normalized `sql_good` at *data-generation* time
(`spider_source.normalize_sql()`, a `sqlglot.parse_one(sql, dialect="sqlite").sql(...)`
round-trip, falling back to the original text on a parse error) — folded into
`load_pairs()`, so it's shared by both `build_synth_dataset.py` and
`build_severity_dataset.py`. Regenerating the deterministic-corruptor dataset under
this alone (`data/output/synth_normalized/`) dropped CV top1/pairwise/mrr from ≈1.00
to 0.9875±0.0032/0.9958±0.0011/0.9938±0.0016 — real but small, since that dataset's
corruptions are already too structurally easy to need a formatting shortcut (see the
original synth-saturation finding above). A second pass hand-normalizing the
llama-only severity dataset post hoc (`data/output/synth_severity_normalized/`, via a
one-off script, since `sql_bad` there is free-form LLM text with no pipeline corruption
step to intercept) showed a much bigger effect: CV top1 0.9780±0.0080 → 0.9275±0.0217,
pairwise 0.9897±0.0040 → 0.9611±0.0101 — llama's own generation style is a much
stronger formatting tell than sqlglot's AST re-render is. Agent-level QA accuracy
dipped slightly in both cases (deterministic: 55.3%→53.3%; llama-only: 54.3%→52.0%) —
closing the leak didn't unlock better real-world reranking on its own, just made the
RM's self-reported numbers honest.

That data-generation-time fix has a real gap, though: it only touches text stored on
disk, so real `llama3.2` candidates scored at inference (`sql_agent.py`/
`evaluate.py`) never get normalized — train/inference formatting distributions would
stay mismatched. Fixed properly by moving normalization into the embedding boundary
itself: `LRRewardModel.score()`/`fit()`/`warm_cache()` (`rm/model/lr/lr_model.py`, the
shared base for every `lr_v1`..`lr_v7` variant, plus `gbm_model.py`) now call
`normalize_sql()` (moved to `walt.utils.sql_exec`, shared by both the data and model
layers) on every SQL string immediately before it's cached/embedded/compared against
`is_sql_valid`/`is_schema_valid`. This is automatic for every dataset and every
inference-time candidate, no dataset regeneration needed, and sidesteps the
data-generation approach's "drop the candidate that now duplicates `sql_good`"
bookkeeping — a post-normalization collision just embeds identically and contributes
a zero-row pair to `fit()` instead of needing to be found and removed. `lr_v7`
(overrides `score()`/`warm_cache()` without calling `super()`) and `gbm_model.py`
(overrides `fit()`/`score()` directly against the inherited, now-normalized
`warm_cache()`) needed matching fixes to avoid a cache-key mismatch between what gets
embedded and what gets looked up — `LRRewardModelV2`/`V3`/`V4`/`V5`/`V6` and
`ContextAwareLRRewardModel`/`LRRewardModelV3Scaled` inherit `score()`/`warm_cache()`
unchanged and needed no changes.

Re-swept `C` for `lr_v6` on the production dataset
(`data/output/synth_severity/synth_severity_enhanced.jsonl`) under the new
always-on normalization: the plateau **inverted** — CV top1 falls monotonically as
`C` grows (C=1: 0.8035 → C=300: 0.7625 → C=10000: 0.7460), peaking instead at
C≈0.1-0.3 (top1 0.8130/0.8125). Makes sense: with the formatting shortcut gone, there
is less real signal to fit, so the old low-regularization default now overfits.
Retrained the default at **C=0.1**: 80/20-split top1/pairwise/mrr dropped from
0.9225/0.9693/0.9548 to **0.805/0.9322/0.8888** — a large, honest-signal drop, not a
regression, per the normalization-leak finding above. Agent-level, though, held up or
improved: SQL pass 279/300 (93.0%) → **286/300 (95.3%)**, QA accuracy 162/300 (54.0%)
→ 160/300 (53.3%, noise-level). Net: closing the leak cost nothing at the agent level
and arguably helped SQL pass rate, while making the RM's own reported accuracy numbers
trustworthy again — **this is now the default everywhere** (`CLAUDE.md`'s "Current
state" section still says `C=300` as of this writing and needs updating to `C=0.1`).
Artifacts: `data/output/rm_model_synth_severity_normalized_lr_v6_C0.1.joblib`,
`data/output/rm_metrics_synth_severity_normalized_lr_v6_C0.1.json`,
`data/output/eval_synth_severity_normalized_lr_v6_C0.1.json`, logged as
`synth_severity_normalized_lr_v6_C0.1_{train,eval}` in `data/output/runs/`/
`data/output/eval_runs/`.

**`ignore_sql_good`: training only on llama-vs-llama correctness (never Spider's
literal gold text) scores far worse on the RM's own pairwise test but reranks real
`llama3.2` candidates better than any trained model has so far — the first config to
beat `constant`+schema-filter on QA accuracy (2026-08-17).** New `LRRewardModel`
constructor flag (threaded through `lr_v1`/`v3`/`v4`/`v5`/`v6`/`v7`, persisted in
`save()`/`load()`, exposed as `--ignore-sql-good` on `train.py`/`cross_validate.py`):
when set, `fit()` drops `sql_good` from `positive_anchors` entirely and uses *only*
`severity==0` `sql_bad` candidates ("executes to the same result as `sql_good`",
`enhance_severity_dataset.py`'s label) as the positive anchor — forces
`severity_zero_as_positive`'s effect on regardless of its own setting, since otherwise
a row with no `severity==0` candidate would lose every positive-vs-negative pair with
no anchor left at all. Combined with the ollama-only dataset
(`synth_severity_enhanced_ollamaonly.jsonl`, `filter_ollama_only.py` — every
`sql_bad`, at every severity, already restricted to `llama3.2`-origin text), this
means Spider's hand-written gold SQL never appears anywhere in training: both the
positive and negative class are llama's own generations, differing only by
correctness. `evaluate()` is deliberately left untouched — it still ranks against the
real `sql_good` — so headline CV/pairwise metrics stay comparable to every other run
in this file; only what `fit()` treats as ground truth changes.

C-swept (0.01-300, 6 points) on the ollama-only dataset: peaks at **C=10**
(top1 0.7465±0.0317/pairwise 0.8374±0.0187/mrr 0.8495±0.0187) — every value tried
lands far below the `sql_good`-anchored baseline at its own best `C`
(top1 0.9255±0.0253/pairwise 0.9592±0.0123/mrr 0.9581±0.0140 at C=300, same dataset,
same normalized-embedding code), so the gap is real, not a tuning artifact. Trained
both configs on the identical raw file and ran the agent-level eval on the same
300-row val split (no-rerank baseline byte-identical across both: 216/300 SQL /
139/300 QA, confirming apples-to-apples):

| | `sql_good`-anchored (C=300) | `ignore_sql_good` (C=10) |
|---|---|---|
| CV pairwise | 0.9592 | 0.8374 |
| SQL pass w/RM | 272/300 (90.7%) | 284/300 (94.7%) |
| QA accuracy w/RM | 160/300 (53.3%) | **175/300 (58.3%)** |
| mixed-bucket achieved | 112/156 (71.8%) | 127/156 (81.4%) |

`ignore_sql_good` scores dramatically worse on its own pairwise-discrimination test
but reranks real candidates dramatically better. Working hypothesis: `evaluate()`'s
pairwise metric always tests the model against the literal `sql_good` string as one of
the candidates, so a `sql_good`-anchored model can partly learn "does this look like
hand-written Spider SQL" — a source/style tell that scores well on that test but is
irrelevant at deployment, where the agent only ever reranks `llama3.2`'s own
candidates, never hand-written gold. `ignore_sql_good` can't lean on that shortcut, so
it's forced to learn the llama-correct-vs-llama-incorrect distinction that actually
transfers. **58.3% QA accuracy is the best number anywhere in this file** — it beats
`constant`+schema-filter's severity-dataset result (168/300, 56.0% QA, 286/300, 95.3%
SQL pass — see above) on QA accuracy and mixed-bucket achieved (81.4% vs 76.9%), while
narrowly losing SQL pass rate (94.7% vs 95.3%). Every other trained config in this
file has lost to `constant`+schema-filter on QA accuracy; this is the first one that
wins — directly on the open problem flagged in `CLAUDE.md`/this file ("schema
filtering has ~no power over semantic mistakes... that is exactly where a learned
model should add value and currently doesn't"). Stacking `--schema-filter` on top of
the `ignore_sql_good` model closes even that gap: SQL pass 286/300 (95.3%, now tied
with the pure filter) with QA accuracy unchanged at 175/300 (58.3%) — the hard gate
catches the couple of schema-invalid candidates `ignore_sql_good` was still
occasionally out-ranking, at zero cost to QA accuracy. **`ignore_sql_good` +
`--schema-filter` now strictly dominates every other configuration in this file on
both metrics simultaneously.** Not yet tried: re-validating on the full
(non-ollama-only) severity dataset. Artifacts:
`data/output/rm_model_ollamaonly_sqlgood_lr_v6_C300.joblib`,
`data/output/rm_model_ollamaonly_ignoresqlgood_lr_v6_C10.joblib`,
`data/output/eval_ollamaonly_sqlgood_lr_v6_C300.json`,
`data/output/eval_ollamaonly_ignoresqlgood_lr_v6_C10.json`, logged as
`ollamaonly_sqlgood_lr_v6_C300*`/`ollamaonly_ignoresqlgood_lr_v6_C10*` in
`data/output/runs/`/`data/output/eval_runs/`; the filtered combination is
`data/output/eval_ollamaonly_ignoresqlgood_lr_v6_C10_schemafilter.json`, logged as
`ollamaonly_ignoresqlgood_lr_v6_C10_schemafilter_eval`.

**`lr_v7` on llama-only + schema-valid-only training reproduces the "no hard schema
check = actively harmful" pattern in its sharpest form yet — but the filter fully
rescues it (2026-08-17).** `filter_schema_valid.py` applied on top of the ollama-only
dataset (`filter_ollama_only.py`'s output) drops schema-invalid `sql_bad` candidates
entirely (8,999→6,350, 29.4%; 81/2,300 rows lost every negative) →
`synth_severity_enhanced_ollamaonly_schemavalid.jsonl`. RM discrimination alone looks
fine — CV top1 0.9495±0.0080/pairwise 0.9557±0.0067/mrr 0.9721±0.0044, 80/20 split
0.9475/0.9580/0.9717 (n=400) — but `v7` deliberately has no `is_schema_valid` feature
(see its docstring), and this training set now contains *zero* schema-invalid
examples for it to learn from either way. Agent-level, unfiltered: **actively
harmful** — SQL pass 215/300 (71.7%, *below* the 216/300 no-rerank baseline), QA
accuracy 132/300 (44.0%, below the 139/300 no-rerank baseline), mixed-bucket achieved
84/156 (53.8%, below random-chance expectation 51.8%) — reranking is choosing worse
candidates than doing nothing. Stacking `--schema-filter` on top fully rescues it: SQL
pass 286/300 (95.3%, the standard filtered ceiling), QA accuracy 163/300 (54.3%,
clearly above no-rerank), mixed-bucket achieved 115/156 (73.7%) — but still below
`constant`+filter (56.0%) and well below `ignore_sql_good`+filter (58.3%). Confirms
`is_schema_valid` (or an equivalent hard gate) isn't optional for this candidate
distribution regardless of which model sits underneath. Artifacts:
`data/output/rm_model_ollamaonly_schemavalid_lr_v7_C300.joblib`,
`data/output/eval_ollamaonly_schemavalid_lr_v7_C300.json` (unfiltered),
`data/output/eval_ollamaonly_schemavalid_lr_v7_C300_schemafilter.json` (filtered),
logged as `ollamaonly_schemavalid_lr_v7_cosine_C300_cv`/
`ollamaonly_schemavalid_lr_v7_C300_{train,eval}`/
`ollamaonly_schemavalid_lr_v7_C300_schemafilter_eval`.

**`lr_v7` gains an `include_schema_valid` flag (V6's own feature, appended to phi_v7)
— trained on the full llama-only dataset (schema-invalid negatives included this
time, so the feature has something to learn from), it just reproduces `lr_v6`, no
better (2026-08-17).** New `LRRewardModel`-family constructor flag
(`include_schema_valid: bool = False`, default off for backward compat with existing
saved `v7` models), exposed as `--v7-schema-valid`, appends
`is_schema_valid(sql, sql_context)` as one more phi dimension (781 total vs. `v7`'s
780) — same feature `lr_v6` already has, just added on top of `v7`'s extra
symbolic/lexical block instead of replacing it. `sql_good` still the training anchor
(not combined with `ignore_sql_good` in this entry — see the next one). CV on the
*full* ollama-only dataset (not schema-valid-filtered) at `C=300`: top1
0.9285±0.0262/pairwise 0.9620±0.0112/mrr 0.9600±0.0140 — statistically tied with plain
`lr_v6`'s own baseline on the identical dataset/`C` (0.9255/0.9592/0.9581). A coarse
`C`-sweep (0.01/0.1/1/10, log-decade steps) plateaus by `C=1`
(top1 0.9275/pairwise 0.9617/mrr 0.9594, `C=10` statistically identical at
0.9260/0.9619/0.9589) — picked `C=1`. Agent-level: unfiltered at `C=300` gives SQL
pass 274/300 (91.3%), QA accuracy 161/300 (53.7%), mixed-bucket achieved 113/156
(72.4%); the tuned `C=1` model with `--schema-filter` gives SQL pass 286/300 (95.3%,
the standard filtered ceiling) and **the identical 161/300 (53.7%) QA accuracy** — the
filter only closes the SQL-pass gap, doesn't move QA at all. 53.7% is within noise of
plain `lr_v6` default's 53.3% and clearly behind `ignore_sql_good`+filter's 58.3%:
**`v7`'s extra symbolic/lexical features (length ratio, structural counts,
question-arg overlap, commands/args embedding split) add nothing on top of
`is_schema_valid`, whether soft (this entry) or hard-filtered.** Artifacts:
`data/output/rm_model_ollamaonly_lr_v7_schemavalid_feature_C300.joblib` (unfiltered
test point), `data/output/rm_model_ollamaonly_lr_v7_schemavalid_feature_C1.joblib`
(final, filtered), `data/output/eval_ollamaonly_lr_v7_schemavalid_feature_C300.json`,
`data/output/eval_ollamaonly_lr_v7_schemavalid_feature_C1_schemafilter.json`, logged
as `ollamaonly_lr_v7_schemavalid_feature_C300_{cv,train,eval}`/
`ollamaonly_lr_v7_schemavalid_feature_C1_schemafilter_eval`.

**Stacking `ignore_sql_good` on top of `v7`+`is_schema_valid` closes most of the gap
to the best config, but still doesn't beat plain `lr_v6`+`ignore_sql_good`+filter
(2026-08-17).** Same `v7` + `include_schema_valid` model as above, but trained with
`--ignore-sql-good` too (`C=10`, reused from the closest comparable sweeps — not
independently tuned for this exact combination), `--schema-filter` at inference.
Agent-level: SQL pass 286/300 (95.3%, filtered ceiling), QA accuracy **171/300
(57.0%)**, mixed-bucket achieved 123/156 (78.8%) — a large jump over the same model
without `ignore_sql_good` (53.7%→57.0%), confirming `ignore_sql_good` is doing almost
all of the work yet again, `v7`'s extra features contributing a little (57.0% vs.
`lr_v6`+`ignore_sql_good`+filter's 58.3% on the identical setup) but not enough to
overtake it. 80/20-split RM discrimination for this combination is the lowest of any
`v7` variant tried (top1 0.74/pairwise 0.8462/mrr 0.8483, n=400) — expected, same
"scores worse on its own pairwise test, reranks better in practice" pattern
`ignore_sql_good` already showed on `lr_v6`. Current full ranking on the ollama-only
val set (all filtered where noted):

| config | SQL pass | QA accuracy |
|---|---|---|
| `lr_v6` + `ignore_sql_good` + filter | 95.3% | **58.3%** |
| `lr_v7` + `is_schema_valid` + `ignore_sql_good` + filter | 95.3% | 57.0% |
| `constant` + filter (no training at all) | 95.3% | 56.0% |
| `lr_v7`, schema-valid-only training, + filter | 95.3% | 54.3% |
| `lr_v7` + `is_schema_valid` (no `ignore_sql_good`) + filter | 95.3% | 53.7% |
| `lr_v6` default (`sql_good`-anchored) | 90.7% | 53.3% |
| `lr_v7`, schema-valid-only training, no filter | 71.7% | 44.0% (below no-rerank) |

`ignore_sql_good` remains the single biggest lever found in this file — bigger than
which `lr_*` variant, bigger than which extra features, bigger than the hard filter
alone. Artifacts: `data/output/rm_model_ollamaonly_lr_v7_schemavalid_ignoresqlgood_C10.joblib`,
`data/output/eval_ollamaonly_lr_v7_schemavalid_ignoresqlgood_C10_schemafilter.json`,
logged as `ollamaonly_lr_v7_schemavalid_ignoresqlgood_C10_{train,schemafilter_eval}`.

**Dropping the graded-severity bad-vs-bad pairs (severity 3 vs. 2, 4 vs. 2, etc.) on
top of `ignore_sql_good` edges out the previous best config — a new top result, though
the margin is small enough to be noise (2026-08-17).** New `LRRewardModel` constructor
flag `drop_bad_vs_bad_pairs` (threaded through `lr_v1`/`v3`/`v4`/`v5`/`v6`/`v7`,
persisted in `save()`/`load()`, exposed as `--drop-bad-vs-bad-pairs` on
`train.py`/`cross_validate.py`): when set, `fit()` skips the pairing loop that pits
two `severity in 1..5` bads against each other whenever their severities differ,
leaving only pairs where one side is a `positive_anchor` (`sql_good` or, under
`ignore_sql_good`, a `severity==0` bad). The question this tests: do those
graded-severity pairs ("this is a *worse* mistake than that one") teach the model
anything a real deployment can use, or is "correct vs. incorrect" the only
distinction that actually transfers? Combined with `ignore_sql_good` on the
ollama-only dataset (same setup as the entry above, just with this flag added), a
`C`-sweep (0.1/1/10/30) plateaus almost immediately — pairwise 0.8140 (`C=0.1`) →
0.8258 (`C=1`) → 0.8310 (`C=10`) → 0.8314 (`C=30`, picked, but statistically
indistinguishable from `C=10`) — and lands far below `ignore_sql_good`-with-pairs' own
best (`pairwise` 0.8374 at `C=10` on the identical dataset, see above): removing those
pairs shrinks the training set enough (roughly half the pairs, since every row's
severity-1..5 bads no longer pair against each other) that the model has strictly less
to learn from by its own metric. Agent-level, though, it's at least as good, arguably
a hair better: SQL pass 286/300 (95.3%, tied with every other filtered config), QA
accuracy **177/300 (59.0%)**, mixed-bucket achieved 129/156 (82.7%) — vs.
`ignore_sql_good`+filter's own 175/300 (58.3%) and 127/156 (81.4%). Same "scores worse
on its own pairwise test, reranks better in practice" pattern `ignore_sql_good` itself
showed relative to the `sql_good`-anchored default: whatever the graded-severity pairs
teach the model about relative badness doesn't seem to be information a real
`llama3.2` candidate distribution rewards at inference time. Caveat: +2/300 rows is a
small enough delta that it could plausibly be noise rather than a real effect — worth
re-running with a different seed or on the full (non-ollama-only) severity dataset
before treating this as the new default. Current full ranking on the ollama-only val
set (all filtered where noted):

| config | SQL pass | QA accuracy |
|---|---|---|
| `lr_v6` + `ignore_sql_good` + `drop_bad_vs_bad_pairs` + filter | 95.3% | **59.0%** |
| `lr_v6` + `ignore_sql_good` + filter | 95.3% | 58.3% |
| `lr_v7` + `is_schema_valid` + `ignore_sql_good` + filter | 95.3% | 57.0% |
| `constant` + filter (no training at all) | 95.3% | 56.0% |
| `lr_v7`, schema-valid-only training, + filter | 95.3% | 54.3% |
| `lr_v7` + `is_schema_valid` (no `ignore_sql_good`) + filter | 95.3% | 53.7% |
| `lr_v6` default (`sql_good`-anchored) | 90.7% | 53.3% |
| `lr_v7`, schema-valid-only training, no filter | 71.7% | 44.0% (below no-rerank) |

Artifacts (a new top-level `runs/` directory, not `data/output/` — see
`scripts/build_best_rm.sh`/`scripts/evaluate_best.sh`): `runs/ablation_drop_bad_vs_bad/rm_model.joblib`,
`runs/ablation_drop_bad_vs_bad/rm_metrics.json`, `runs/ablation_drop_bad_vs_bad/eval_results.json`
(trained model), `runs/ablation_drop_bad_vs_bad/eval_baseline_constant_schemafilter.json`
(filter-only baseline, byte-identical to the `ignore_sql_good` entry's own baseline —
same dataset, same LLM cache), C-sweep + train logs under
`runs/ablation_drop_bad_vs_bad/runs/`, eval logs under
`runs/ablation_drop_bad_vs_bad/eval_runs/`.

**DistilBERT under the identical `ignore_sql_good`+`drop_bad_vs_bad_pairs`+ollama-only
setup: clearly weaker on its own held-out discrimination, but — after `--schema-filter`
— lands within 0.7pp QA of `lr_v6` on the same 300-row val set (2026-08-17).**
`build_pairs()` (`distilbert_model.py`, previously always sql_good-vs-every-bad,
ignoring severity entirely) now takes the same `ignore_sql_good`/
`severity_zero_as_positive`/`drop_bad_vs_bad_pairs` flags as `LRRewardModel.fit()`,
with the identical pairing rule (verified pair-for-pair against `lr_model.py`'s output
on a synthetic example) and the same backward-compatibility guarantee (all-`None`
severity + every flag off reproduces the exact old pairs/rng draws). Threaded through
`train.py`'s `--ignore-sql-good`/`--severity-zero-as-positive`/`--drop-bad-vs-bad-pairs`
(now `[..., distilbert only]` too) and persisted in `save()`/`load()`'s
`hyperparams` dict. This shrinks the training set sharply — `distilbert_preflight.py`'s
time estimate (which uses default, unflagged pairing) assumed 8,725 train pairs; the
actual flagged run had only 3,554 — so a full run finishes far faster than the
preflight's naive estimate suggests. Early-stopped after 6 epochs (best epoch 3,
patience 3): held-out (val split) `top1_accuracy` **0.6167**, `pairwise_accuracy`
**0.7756** — clearly behind `lr_v6`'s own numbers on the identical val split/config
(top1 0.7367, pairwise 0.8678) and close to chance on `top1` specifically. Agent-level,
with `--schema-filter` (SQL pass tied at 286/300, 95.3%, same as every filtered
config): QA accuracy **175/300 (58.3%)**, mixed-bucket achieved **127/156 (81.4%)** —
vs. `lr_v6`'s 177/300 (59.0%) / 129/156 (82.7%) on the exact same val rows, same
candidates (shared LLM cache). Despite scoring meaningfully worse on its own
discrimination test, DistilBERT nearly matches `lr_v6` at the agent level and still
clearly beats `constant`+filter (168/300, 56.0%) — the same "self-reported pairwise
accuracy doesn't predict agent-level transfer" pattern seen throughout this file,
now shown to hold even across model *families*, not just training-anchor choices.
Doesn't overturn CLAUDE.md's "never beats `lr_v6`" verdict (it still loses, on both
metrics, at ~50x the training cost) but the margin is much narrower here than in
earlier DistilBERT entries — plausibly because `--schema-filter` does most of the
heavy lifting regardless of which model sits underneath it, leaving less room for the
better-discriminating model to pull ahead. Artifacts:
`runs/ablation_drop_bad_vs_bad/rm_model_distilbert.pt`,
`runs/ablation_drop_bad_vs_bad/rm_metrics_distilbert.json`,
`runs/ablation_drop_bad_vs_bad/eval_results_distilbert.json`, logged as
`ablation_drop_bad_vs_bad_distilbert_{train,eval}` in
`runs/ablation_drop_bad_vs_bad/runs/`/`runs/ablation_drop_bad_vs_bad/eval_runs/`.
