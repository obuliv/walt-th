"""Trains a pairwise-ranking reward model on rm_enhanced.jsonl, evaluates it on a
held-out question-level split, and checks the train/test gap for overfitting.

Defaults (--model lr_v6, --C 300) reflect the best config found via 5-fold CV
(see cross_validate.py and CLAUDE.md) — cosine-sim + is_sql_valid + is_schema_valid
features, and much less L2 regularization than sklearn's C=1.0 default, which was
clearly under-using the model's capacity for this data size (~4700 training pairs,
769-770 features). --C 300 is CV-tuned for lr_v6 on gretel-only data specifically
(peaks around C=100-300 there, unlike lr_v3's C=30/C=1000 tuned on other datasets)
— see CLAUDE.md.

Usage:
    python -m walt.rm.model.train --input data/output/rm_enhanced.jsonl --model-output data/output/rm_model.joblib
    python -m walt.rm.model.train --model gbm  # nonlinear pointwise alternative; see gbm_model.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from walt.rm.model.base import group_split, load_examples, overfitting_gap
from walt.rm.model.distilbert_model import DEFAULT_MODEL_NAME as DISTILBERT_DEFAULT_MODEL_NAME
from walt.rm.model.distilbert_model import DistilBertRewardModel
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.gbm_model import GBMRewardModel
from walt.rm.model.lr import (
    LRRewardModel,
    LRRewardModelV2,
    LRRewardModelV3,
    LRRewardModelV4,
    LRRewardModelV5,
    LRRewardModelV6,
    LRRewardModelV7,
)
from walt.rm.model.lr.lr_model_v7 import EMBEDDING_DIFF_MODES
from walt.rm.model.tracking import log_run

MODEL_CHOICES = ["lr_v1", "lr_v2", "lr_v3", "lr_v4", "lr_v5", "lr_v6", "lr_v7", "gbm", "distilbert"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/rm_enhanced.jsonl"), help="Input JSONL (question, sql_good, sql_bad, source)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of questions held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and pair-label randomization")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="lr_v6", help="Which reward model implementation to train")
    parser.add_argument("--embedding-model", default=SentenceTransformerEmbedding.DEFAULT_MODEL_NAME, help="sentence-transformers model name/id")
    parser.add_argument("--device", default=None, help="Torch device for the embedding model (e.g. 'mps', 'cpu'); default lets sentence-transformers auto-select")
    parser.add_argument("--model-output", type=Path, default=Path("data/output/rm_model.joblib"), help="Where to save the fitted model")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional path to write test-set evaluation metrics as JSON")
    parser.add_argument("--run-name", default=None, help="Label for this run in the run log (default: derived from --model and the embedding model name)")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/runs"), help="Directory where per-run metric records are logged for later comparison")
    parser.add_argument("--no-log-run", action="store_true", help="Skip writing a run record (e.g. for throwaway/debug runs)")
    parser.add_argument("--C", type=float, default=300.0, help="[lr_v1/lr_v2/.../lr_v6 only] LogisticRegression inverse regularization strength — 300 is CV-tuned for lr_v6 on gretel-only data (sklearn's own default is 1.0); other models/datasets were tuned separately, see CLAUDE.md")
    parser.add_argument("--severity-zero-as-positive", action="store_true", help="[lr_v1/lr_v3/lr_v4/lr_v5/lr_v6/lr_v7/distilbert only] treat severity==0 sql_bad candidates (from enhance_severity_dataset.py) as extra positive anchors paired against every real bad, instead of excluding them from every pair. No-op on datasets without severity.")
    parser.add_argument("--ignore-sql-good", action="store_true", help="[lr_v1/lr_v3/lr_v4/lr_v5/lr_v6/lr_v7/distilbert only] drop sql_good from positive_anchors entirely and use ONLY severity==0 sql_bad candidates as the positive anchor (forces severity_zero_as_positive's effect on regardless of its own flag). A row with no severity==0 candidate contributes zero good-vs-bad pairs (its ranked-bad-vs-ranked-bad pairs, if any, are unaffected). evaluate() is unchanged — still ranks against the real sql_good.")
    parser.add_argument("--drop-bad-vs-bad-pairs", action="store_true", help="[lr_v1/lr_v3/lr_v4/lr_v5/lr_v6/lr_v7/distilbert only] drop the ranked-bad-vs-ranked-bad pairs (e.g. severity 3 vs. severity 2) entirely, keeping only pairs with a positive_anchor (sql_good or a severity==0 bad) on one side. Tests whether graded-severity pairs add signal beyond correct-vs-incorrect.")
    parser.add_argument("--embedding-diff-mode", choices=EMBEDDING_DIFF_MODES, default="cosine", help="[lr_v7 only] how the commands/args-vs-question embedding comparison is folded into phi: 'cosine' (2 scalar cosine similarities) or 'raw' (2x768 raw vector differences)")
    parser.add_argument("--v7-schema-valid", dest="v7_schema_valid", action="store_true", help="[lr_v7 only] append is_schema_valid(sql, sql_context) to phi_v7 (V7 excludes it by default, unlike lr_v6)")
    parser.add_argument("--v2-cosine-sim", dest="v2_cosine_sim", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only] include the cosine-similarity interaction feature")
    parser.add_argument("--v2-dot-product", dest="v2_dot_product", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only] include the raw dot-product interaction feature")
    parser.add_argument("--v2-standardize-dot", dest="v2_standardize_dot", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only] standardize the raw dot-product feature using training-set mean/std")
    parser.add_argument("--gbm-max-iter", type=int, default=200, help="[gbm only] number of boosting rounds")
    parser.add_argument("--gbm-max-depth", type=int, default=None, help="[gbm only] max tree depth (None = sklearn default)")
    parser.add_argument("--gbm-learning-rate", type=float, default=0.1, help="[gbm only]")
    parser.add_argument("--distilbert-model-name", default=DISTILBERT_DEFAULT_MODEL_NAME, help="[distilbert only] HF model id for the backbone")
    parser.add_argument("--distilbert-epochs", type=int, default=15, help="[distilbert only] max epochs (early stopping usually stops sooner)")
    parser.add_argument("--distilbert-lr", type=float, default=2e-5, help="[distilbert only] AdamW learning rate")
    parser.add_argument("--distilbert-batch-size", type=int, default=8, help="[distilbert only] pairs per micro-batch (2x forward passes)")
    parser.add_argument("--distilbert-grad-accum-steps", type=int, default=2, help="[distilbert only] gradient accumulation steps")
    parser.add_argument("--distilbert-max-length", type=int, default=512, help="[distilbert only] max token sequence length")
    parser.add_argument("--distilbert-warmup-ratio", type=float, default=0.1, help="[distilbert only] fraction of total steps spent warming up")
    parser.add_argument("--distilbert-weight-decay", type=float, default=0.01, help="[distilbert only] AdamW weight decay")
    parser.add_argument("--distilbert-val-fraction", type=float, default=0.1, help="[distilbert only] internal question-level split for early stopping")
    parser.add_argument("--distilbert-early-stop-patience", type=int, default=3, help="[distilbert only] epochs without improvement before stopping")
    parser.add_argument("--distilbert-early-stop-metric", choices=["pairwise_accuracy", "loss"], default="pairwise_accuracy", help="[distilbert only] early-stopping criterion")
    args = parser.parse_args()
    run_start = time.perf_counter()

    load_stats: dict = {}
    examples = load_examples(args.input, stats=load_stats)
    n_loaded = len(examples)
    examples = [ex for ex in examples if ex.split != "val"]
    train_examples, test_examples = group_split(examples, test_size=args.test_size, seed=args.seed)
    print(
        f"Loaded {n_loaded} examples ({n_loaded - len(examples)} val rows excluded): "
        f"{len(train_examples)} train / {len(test_examples)} test"
    )

    embedding_provider = (
        None if args.model == "distilbert" else SentenceTransformerEmbedding(model_name=args.embedding_model, device=args.device)
    )
    if args.model == "lr_v1":
        model = LRRewardModel(embedding_provider=embedding_provider, seed=args.seed, C=args.C, severity_zero_as_positive=args.severity_zero_as_positive, ignore_sql_good=args.ignore_sql_good, drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs)
    elif args.model == "lr_v2":
        model = LRRewardModelV2(
            embedding_provider=embedding_provider,
            seed=args.seed,
            C=args.C,
            use_cosine_sim=args.v2_cosine_sim,
            use_dot_product=args.v2_dot_product,
            standardize_dot_product=args.v2_standardize_dot,
        )
    elif args.model == "lr_v3":
        model = LRRewardModelV3(embedding_provider=embedding_provider, seed=args.seed, C=args.C, severity_zero_as_positive=args.severity_zero_as_positive, ignore_sql_good=args.ignore_sql_good, drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs)
    elif args.model == "lr_v4":
        model = LRRewardModelV4(embedding_provider=embedding_provider, seed=args.seed, C=args.C, severity_zero_as_positive=args.severity_zero_as_positive, ignore_sql_good=args.ignore_sql_good, drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs)
    elif args.model == "lr_v5":
        model = LRRewardModelV5(embedding_provider=embedding_provider, seed=args.seed, C=args.C, severity_zero_as_positive=args.severity_zero_as_positive, ignore_sql_good=args.ignore_sql_good, drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs)
    elif args.model == "lr_v6":
        model = LRRewardModelV6(embedding_provider=embedding_provider, seed=args.seed, C=args.C, severity_zero_as_positive=args.severity_zero_as_positive, ignore_sql_good=args.ignore_sql_good, drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs)
    elif args.model == "lr_v7":
        model = LRRewardModelV7(
            embedding_provider=embedding_provider,
            seed=args.seed,
            C=args.C,
            severity_zero_as_positive=args.severity_zero_as_positive,
            ignore_sql_good=args.ignore_sql_good,
            drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs,
            embedding_diff_mode=args.embedding_diff_mode,
            include_schema_valid=args.v7_schema_valid,
        )
    elif args.model == "distilbert":
        model = DistilBertRewardModel(
            model_name=args.distilbert_model_name,
            seed=args.seed,
            device=args.device,
            max_length=args.distilbert_max_length,
            learning_rate=args.distilbert_lr,
            num_epochs=args.distilbert_epochs,
            batch_size=args.distilbert_batch_size,
            grad_accum_steps=args.distilbert_grad_accum_steps,
            warmup_ratio=args.distilbert_warmup_ratio,
            weight_decay=args.distilbert_weight_decay,
            val_fraction=args.distilbert_val_fraction,
            early_stop_patience=args.distilbert_early_stop_patience,
            early_stop_metric=args.distilbert_early_stop_metric,
            ignore_sql_good=args.ignore_sql_good,
            severity_zero_as_positive=args.severity_zero_as_positive,
            drop_bad_vs_bad_pairs=args.drop_bad_vs_bad_pairs,
        )
    else:
        model = GBMRewardModel(
            embedding_provider=embedding_provider,
            seed=args.seed,
            max_iter=args.gbm_max_iter,
            max_depth=args.gbm_max_depth,
            learning_rate=args.gbm_learning_rate,
        )

    model.fit(train_examples)
    model.warm_cache(test_examples)

    eval_start = time.perf_counter()
    test_metrics = model.evaluate(test_examples)
    eval_seconds = time.perf_counter() - eval_start
    print("Test set:")
    model.publish_metrics(test_metrics, output_path=args.metrics_output)

    # cache is already warm for train_examples from fit() — this is just more score()
    # calls, no extra embedding calls, so it's cheap to always check.
    train_eval_start = time.perf_counter()
    train_metrics = model.evaluate(train_examples)
    train_eval_seconds = time.perf_counter() - train_eval_start
    print("\nTrain set (for overfitting check):")
    model.publish_metrics(train_metrics)

    gap = overfitting_gap(train_metrics, test_metrics)
    print("\nOverfitting check (train - test; large positive = overfitting):")
    for key, value in gap.items():
        print(f"  {key}: {value:+.4f}")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)
    print(f"\nSaved model to {args.model_output}")

    if not args.no_log_run:
        run_name = args.run_name or (
            f"distilbert_{args.distilbert_model_name.split('/')[-1]}"
            if args.model == "distilbert"
            else f"{args.model}_{args.embedding_model.split('/')[-1]}"
        )
        run_path = log_run(
            args.runs_dir,
            run_name=run_name,
            model_class=type(model).__name__,
            config={
                "input": str(args.input),
                "model": args.model,
                "embedding_model": args.embedding_model,
                "device": args.device,
                "test_size": args.test_size,
                "seed": args.seed,
                "n_train": len(train_examples),
                "n_test": len(test_examples),
                **load_stats,  # n_rows_total, n_rows_skipped
                **({"C": args.C} if args.model in ("lr_v1", "lr_v2", "lr_v3", "lr_v4", "lr_v5", "lr_v6", "lr_v7") else {}),
                **(
                    {
                        "severity_zero_as_positive": args.severity_zero_as_positive,
                        "ignore_sql_good": args.ignore_sql_good,
                        "drop_bad_vs_bad_pairs": args.drop_bad_vs_bad_pairs,
                    }
                    if args.model in ("lr_v1", "lr_v3", "lr_v4", "lr_v5", "lr_v6", "lr_v7", "distilbert")
                    else {}
                ),
                **(
                    {"embedding_diff_mode": args.embedding_diff_mode, "v7_schema_valid": args.v7_schema_valid}
                    if args.model == "lr_v7"
                    else {}
                ),
                **(
                    {
                        "v2_cosine_sim": args.v2_cosine_sim,
                        "v2_dot_product": args.v2_dot_product,
                        "v2_standardize_dot": args.v2_standardize_dot,
                    }
                    if args.model == "lr_v2"
                    else {}
                ),
                **(
                    {"gbm_max_iter": args.gbm_max_iter, "gbm_max_depth": args.gbm_max_depth, "gbm_learning_rate": args.gbm_learning_rate}
                    if args.model == "gbm"
                    else {}
                ),
                **(
                    {
                        "distilbert_model_name": args.distilbert_model_name,
                        "distilbert_epochs": args.distilbert_epochs,
                        "distilbert_lr": args.distilbert_lr,
                        "distilbert_batch_size": args.distilbert_batch_size,
                        "distilbert_grad_accum_steps": args.distilbert_grad_accum_steps,
                        "distilbert_max_length": args.distilbert_max_length,
                        "distilbert_warmup_ratio": args.distilbert_warmup_ratio,
                        "distilbert_weight_decay": args.distilbert_weight_decay,
                        "distilbert_val_fraction": args.distilbert_val_fraction,
                        "distilbert_early_stop_patience": args.distilbert_early_stop_patience,
                        "distilbert_early_stop_metric": args.distilbert_early_stop_metric,
                    }
                    if args.model == "distilbert"
                    else {}
                ),
            },
            metrics=test_metrics,
            training={
                **model.fit_info,  # n_train_pairs, embedding_dim, feature_dim, label_balance,
                # lr_max_iter/lr_n_iter/lr_converged, embed_seconds, fit_seconds
                "eval_seconds": round(eval_seconds, 3),
                "train_eval_seconds": round(train_eval_seconds, 3),
                "total_seconds": round(time.perf_counter() - run_start, 3),
            },
            train_metrics=train_metrics,
            overfitting_gap=gap,
        )
        print(f"Logged run to {run_path}")


if __name__ == "__main__":
    main()
