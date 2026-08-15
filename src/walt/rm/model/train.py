"""Trains the LR pairwise-ranking reward model on rm_enhanced.jsonl and evaluates it
on a held-out, question-level split.

Usage:
    python -m walt.rm.model.train --input data/output/rm_enhanced.jsonl --model-output data/output/rm_model.joblib
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from walt.rm.model.base import group_split, load_examples
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.lr_model import LRRewardModel
from walt.rm.model.tracking import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/rm_enhanced.jsonl"), help="Input JSONL (question, sql_good, sql_bad, source)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of questions held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and pair-label randomization")
    parser.add_argument("--embedding-model", default=SentenceTransformerEmbedding.DEFAULT_MODEL_NAME, help="sentence-transformers model name/id")
    parser.add_argument("--device", default=None, help="Torch device for the embedding model (e.g. 'mps', 'cpu'); default lets sentence-transformers auto-select")
    parser.add_argument("--model-output", type=Path, default=Path("data/output/rm_model.joblib"), help="Where to save the fitted model")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional path to write evaluation metrics as JSON")
    parser.add_argument("--run-name", default=None, help="Label for this run in the run log (default: derived from the embedding model name)")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/runs"), help="Directory where per-run metric records are logged for later comparison")
    parser.add_argument("--no-log-run", action="store_true", help="Skip writing a run record (e.g. for throwaway/debug runs)")
    args = parser.parse_args()
    run_start = time.perf_counter()

    load_stats: dict = {}
    examples = load_examples(args.input, stats=load_stats)
    train_examples, test_examples = group_split(examples, test_size=args.test_size, seed=args.seed)
    print(f"Loaded {len(examples)} examples: {len(train_examples)} train / {len(test_examples)} test")

    embedding_provider = SentenceTransformerEmbedding(model_name=args.embedding_model, device=args.device)
    model = LRRewardModel(embedding_provider=embedding_provider, seed=args.seed)

    model.fit(train_examples)
    model.warm_cache(test_examples)

    eval_start = time.perf_counter()
    metrics = model.evaluate(test_examples)
    eval_seconds = time.perf_counter() - eval_start
    model.publish_metrics(metrics, output_path=args.metrics_output)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)
    print(f"Saved model to {args.model_output}")

    if not args.no_log_run:
        run_name = args.run_name or args.embedding_model.split("/")[-1]
        run_path = log_run(
            args.runs_dir,
            run_name=run_name,
            model_class=type(model).__name__,
            config={
                "input": str(args.input),
                "embedding_model": args.embedding_model,
                "device": args.device,
                "test_size": args.test_size,
                "seed": args.seed,
                "n_train": len(train_examples),
                "n_test": len(test_examples),
                **load_stats,  # n_rows_total, n_rows_skipped
            },
            metrics=metrics,
            training={
                **model.fit_info,  # n_train_pairs, embedding_dim, feature_dim, label_balance,
                # lr_max_iter/lr_n_iter/lr_converged, embed_seconds, fit_seconds
                "eval_seconds": round(eval_seconds, 3),
                "total_seconds": round(time.perf_counter() - run_start, 3),
            },
        )
        print(f"Logged run to {run_path}")


if __name__ == "__main__":
    main()
