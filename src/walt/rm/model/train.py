"""Trains a pairwise-ranking reward model on rm_enhanced.jsonl, evaluates it on a
held-out question-level split, and checks the train/test gap for overfitting.

Defaults (--model lr_v3, --C 30) reflect the best config found via 5-fold CV
(see cross_validate.py and CLAUDE.md) — cosine-sim + is_sql_valid features, and much
less L2 regularization than sklearn's C=1.0 default, which was clearly under-using the
model's capacity for this data size (~4700 training pairs, 769-770 features).

Usage:
    python -m walt.rm.model.train --input data/output/rm_enhanced.jsonl --model-output data/output/rm_model.joblib
    python -m walt.rm.model.train --model gbm  # nonlinear pointwise alternative; see gbm_model.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from walt.rm.model.base import group_split, load_examples, overfitting_gap
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.gbm_model import GBMRewardModel
from walt.rm.model.lr import LRRewardModel, LRRewardModelV2, LRRewardModelV3, LRRewardModelV4, LRRewardModelV5
from walt.rm.model.tracking import log_run

MODEL_CHOICES = ["lr_v1", "lr_v2", "lr_v3", "lr_v4", "lr_v5", "gbm"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/rm_enhanced.jsonl"), help="Input JSONL (question, sql_good, sql_bad, source)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of questions held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and pair-label randomization")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="lr_v3", help="Which reward model implementation to train")
    parser.add_argument("--embedding-model", default=SentenceTransformerEmbedding.DEFAULT_MODEL_NAME, help="sentence-transformers model name/id")
    parser.add_argument("--device", default=None, help="Torch device for the embedding model (e.g. 'mps', 'cpu'); default lets sentence-transformers auto-select")
    parser.add_argument("--model-output", type=Path, default=Path("data/output/rm_model.joblib"), help="Where to save the fitted model")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional path to write test-set evaluation metrics as JSON")
    parser.add_argument("--run-name", default=None, help="Label for this run in the run log (default: derived from --model and the embedding model name)")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/runs"), help="Directory where per-run metric records are logged for later comparison")
    parser.add_argument("--no-log-run", action="store_true", help="Skip writing a run record (e.g. for throwaway/debug runs)")
    parser.add_argument("--C", type=float, default=30.0, help="[lr_v1/lr_v2/lr_v3 only] LogisticRegression inverse regularization strength (CV-tuned; sklearn's own default is 1.0)")
    parser.add_argument("--v2-cosine-sim", dest="v2_cosine_sim", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only] include the cosine-similarity interaction feature")
    parser.add_argument("--v2-dot-product", dest="v2_dot_product", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only] include the raw dot-product interaction feature")
    parser.add_argument("--v2-standardize-dot", dest="v2_standardize_dot", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only] standardize the raw dot-product feature using training-set mean/std")
    parser.add_argument("--gbm-max-iter", type=int, default=200, help="[gbm only] number of boosting rounds")
    parser.add_argument("--gbm-max-depth", type=int, default=None, help="[gbm only] max tree depth (None = sklearn default)")
    parser.add_argument("--gbm-learning-rate", type=float, default=0.1, help="[gbm only]")
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

    embedding_provider = SentenceTransformerEmbedding(model_name=args.embedding_model, device=args.device)
    if args.model == "lr_v1":
        model = LRRewardModel(embedding_provider=embedding_provider, seed=args.seed, C=args.C)
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
        model = LRRewardModelV3(embedding_provider=embedding_provider, seed=args.seed, C=args.C)
    elif args.model == "lr_v4":
        model = LRRewardModelV4(embedding_provider=embedding_provider, seed=args.seed, C=args.C)
    elif args.model == "lr_v5":
        model = LRRewardModelV5(embedding_provider=embedding_provider, seed=args.seed, C=args.C)
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
        run_name = args.run_name or f"{args.model}_{args.embedding_model.split('/')[-1]}"
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
                **({"C": args.C} if args.model in ("lr_v1", "lr_v2", "lr_v3", "lr_v4", "lr_v5") else {}),
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
