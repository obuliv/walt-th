"""k-fold cross-validation for reward model configs, so comparisons between approaches
are trustworthy — a single 80/20 split has real variance (e.g. --seed 7 vs --seed 42 on
lr_v1 swung top1_accuracy by ~1.5pp on this dataset, comparable in size to some of the
differences between approaches we've been trying to compare on single splits).

Usage:
    python -m walt.rm.model.cross_validate --model lr_v1
    python -m walt.rm.model.cross_validate --model lr_v1 --C 0.1
    python -m walt.rm.model.cross_validate --model lr_v3
    python -m walt.rm.model.cross_validate --model gbm --gbm-max-depth 3
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from walt.rm.model.base import cross_validate, load_examples, publish_cv_summary
from walt.rm.model.embeddings import SentenceTransformerEmbedding
from walt.rm.model.gbm_model import GBMRewardModel
from walt.rm.model.lr import (
    SCALING_CHOICES,
    SOLVER_BY_PENALTY,
    LRRewardModel,
    LRRewardModelV2,
    LRRewardModelV3,
    LRRewardModelV3Scaled,
    LRRewardModelV4,
    LRRewardModelV5,
    LRRewardModelV6,
)
from walt.rm.model.tracking import log_run

MODEL_CHOICES = ["lr_v1", "lr_v2", "lr_v3", "lr_v3_scaled", "lr_v4", "lr_v5", "lr_v6", "gbm"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/rm_enhanced.jsonl"), help="Input JSONL (question, sql_good, sql_bad, source)")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of trainval examples to keep (question-level, seeded shuffle) before CV — for data-scaling experiments")
    parser.add_argument("--k", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold assignment and pair-label randomization")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="lr_v6", help="Which reward model implementation to evaluate")
    parser.add_argument("--embedding-model", default=SentenceTransformerEmbedding.DEFAULT_MODEL_NAME, help="sentence-transformers model name/id")
    parser.add_argument("--device", default=None, help="Torch device for the embedding model")
    parser.add_argument("--C", type=float, default=300.0, help="[lr_v1/lr_v2/.../lr_v6 only] LogisticRegression inverse regularization strength — 300 is CV-tuned for lr_v6 on gretel-only data (sklearn's own default is 1.0); other models/datasets were tuned separately, see CLAUDE.md")
    parser.add_argument("--penalty", choices=sorted(SOLVER_BY_PENALTY), default="l2", help="[lr_v1/lr_v2/lr_v3/lr_v3_scaled only] LogisticRegression penalty type")
    parser.add_argument("--l1-ratio", type=float, default=None, help="[penalty=elasticnet only] elastic-net mixing parameter in [0,1] (0=l2, 1=l1)")
    parser.add_argument("--v2-cosine-sim", dest="v2_cosine_sim", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only]")
    parser.add_argument("--v2-dot-product", dest="v2_dot_product", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only]")
    parser.add_argument("--v2-standardize-dot", dest="v2_standardize_dot", action=argparse.BooleanOptionalAction, default=True, help="[lr_v2 only]")
    parser.add_argument("--scaling", choices=SCALING_CHOICES, default="none", help="[lr_v3_scaled only] how to scale the embed(sql) block of phi before concatenation")
    parser.add_argument("--gbm-max-iter", type=int, default=200, help="[gbm only] number of boosting rounds")
    parser.add_argument("--gbm-max-depth", type=int, default=None, help="[gbm only] max tree depth (None = sklearn default)")
    parser.add_argument("--gbm-learning-rate", type=float, default=0.1, help="[gbm only]")
    parser.add_argument("--run-name", default=None, help="Label for this run in the run log (default: derived from --model)")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/runs"), help="Directory where run records are logged")
    parser.add_argument("--no-log-run", action="store_true", help="Skip writing a run record")
    args = parser.parse_args()

    examples = load_examples(args.input)
    n_loaded = len(examples)
    examples = [ex for ex in examples if ex.split != "val"]
    n_trainval = len(examples)
    if args.train_fraction < 1.0:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        n_keep = round(len(shuffled) * args.train_fraction)
        examples = shuffled[:n_keep]
    print(
        f"Loaded {n_loaded} examples ({n_loaded - n_trainval} val rows excluded), "
        f"subsampled to {len(examples)}/{n_trainval} ({args.train_fraction:.0%}) trainval rows, "
        f"running {args.k}-fold CV with model={args.model}"
    )

    embedding_provider = SentenceTransformerEmbedding(model_name=args.embedding_model, device=args.device)

    def model_factory():
        if args.model == "lr_v1":
            return LRRewardModel(embedding_provider=embedding_provider, seed=args.seed, C=args.C, penalty=args.penalty, l1_ratio=args.l1_ratio)
        if args.model == "lr_v2":
            return LRRewardModelV2(
                embedding_provider=embedding_provider,
                seed=args.seed,
                C=args.C,
                penalty=args.penalty,
                l1_ratio=args.l1_ratio,
                use_cosine_sim=args.v2_cosine_sim,
                use_dot_product=args.v2_dot_product,
                standardize_dot_product=args.v2_standardize_dot,
            )
        if args.model == "lr_v3":
            return LRRewardModelV3(embedding_provider=embedding_provider, seed=args.seed, C=args.C, penalty=args.penalty, l1_ratio=args.l1_ratio)
        if args.model == "lr_v4":
            return LRRewardModelV4(embedding_provider=embedding_provider, seed=args.seed, C=args.C, penalty=args.penalty, l1_ratio=args.l1_ratio)
        if args.model == "lr_v5":
            return LRRewardModelV5(embedding_provider=embedding_provider, seed=args.seed, C=args.C, penalty=args.penalty, l1_ratio=args.l1_ratio)
        if args.model == "lr_v6":
            return LRRewardModelV6(embedding_provider=embedding_provider, seed=args.seed, C=args.C, penalty=args.penalty, l1_ratio=args.l1_ratio)
        if args.model == "lr_v3_scaled":
            return LRRewardModelV3Scaled(
                embedding_provider=embedding_provider,
                seed=args.seed,
                C=args.C,
                penalty=args.penalty,
                l1_ratio=args.l1_ratio,
                scaling=args.scaling,
            )
        if args.model == "gbm":
            return GBMRewardModel(
                embedding_provider=embedding_provider,
                seed=args.seed,
                max_iter=args.gbm_max_iter,
                max_depth=args.gbm_max_depth,
                learning_rate=args.gbm_learning_rate,
            )
        raise ValueError(args.model)

    cv_result = cross_validate(model_factory, examples, k=args.k, seed=args.seed)
    publish_cv_summary(cv_result)

    if not args.no_log_run:
        run_name = args.run_name or f"cv_{args.model}"
        mean_metrics = {key: stats["mean"] for key, stats in cv_result["summary"].items()}
        mean_metrics["n_examples"] = len(examples)
        run_path = log_run(
            args.runs_dir,
            run_name=run_name,
            model_class=model_factory().__class__.__name__,
            config={
                "input": str(args.input),
                "train_fraction": args.train_fraction,
                "model": args.model,
                "embedding_model": args.embedding_model,
                "k": args.k,
                "seed": args.seed,
                "C": args.C,
                "penalty": args.penalty,
                "l1_ratio": args.l1_ratio,
                **(
                    {"v2_cosine_sim": args.v2_cosine_sim, "v2_dot_product": args.v2_dot_product, "v2_standardize_dot": args.v2_standardize_dot}
                    if args.model == "lr_v2"
                    else {}
                ),
                **({"scaling": args.scaling} if args.model == "lr_v3_scaled" else {}),
                **(
                    {"gbm_max_iter": args.gbm_max_iter, "gbm_max_depth": args.gbm_max_depth, "gbm_learning_rate": args.gbm_learning_rate}
                    if args.model == "gbm"
                    else {}
                ),
            },
            metrics=mean_metrics,
            training={"cv": cv_result},
        )
        print(f"Logged run to {run_path}")


if __name__ == "__main__":
    main()
