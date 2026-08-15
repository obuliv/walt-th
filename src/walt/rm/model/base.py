"""Base class for pairwise-ranking SQL reward models.

Concrete subclasses implement `fit()` and `score()`; this class supplies the generic
ranking, evaluation, and metrics-reporting machinery on top.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from walt.rm.data.base import Example


def load_examples(path: str | Path, stats: dict | None = None) -> list[Example]:
    """Load Examples from an rm_enhanced.jsonl-shaped file (question, sql_good, source, sql_bad).
    Rows that fail Example's validation (e.g. sql_good duplicating a sql_bad entry, a
    labeling mistake seen in the LLM-generated negatives) are skipped with a warning
    rather than aborting the whole load. If `stats` is given, it's filled in with
    {n_rows_total, n_rows_skipped} for callers that want to record it (e.g. in a run log)."""
    examples = []
    skipped = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(Example.from_dict(json.loads(line)))
            except ValueError as exc:
                skipped += 1
                print(f"WARNING: skipping malformed row: {exc}")
    if skipped:
        print(f"Skipped {skipped} malformed row(s) out of {skipped + len(examples)}")
    if stats is not None:
        stats["n_rows_total"] = skipped + len(examples)
        stats["n_rows_skipped"] = skipped
    return examples


def group_split(
    examples: list[Example], test_size: float = 0.2, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Split at the question (Example) level so every pair from one question stays
    entirely in train or entirely in test — pair-level splitting would leak schema/
    phrasing signal for that question across the split."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_test = round(len(shuffled) * test_size)
    test, train = shuffled[:n_test], shuffled[n_test:]
    return train, test


def all_candidates(example: Example) -> list[str]:
    return [example.sql_good] + [b.sql for b in example.sql_bad]


@dataclass(frozen=True)
class ScoredCandidate:
    sql: str
    score: float
    rank: int  # 1-based; ties broken by stable sort (input order)
    error_code: str | None = None


class BaseRewardModel(ABC):
    @abstractmethod
    def fit(self, train_examples: list[Example]) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, question: str, sql: str) -> float:
        """Higher = more likely correct. Not assumed to be calibrated/probabilistic —
        only relative order across candidates for the same question is meaningful."""
        raise NotImplementedError

    def predict_error_code(self, question: str, sql: str) -> str | None:
        """Optional hook for subclasses that classify *why* a candidate is bad. Default:
        unsupported."""
        return None

    def rank(self, question: str, candidates: list[str]) -> list[ScoredCandidate]:
        scored = [(sql, self.score(question, sql)) for sql in candidates]
        ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [
            ScoredCandidate(
                sql=sql,
                score=score,
                rank=i + 1,
                error_code=self.predict_error_code(question, sql),
            )
            for i, (sql, score) in enumerate(ordered)
        ]

    def evaluate(self, test_examples: list[Example]) -> dict:
        n_examples = len(test_examples)
        top1_hits = 0
        rr_sum = 0.0
        pair_correct = 0
        pair_total = 0
        reason_correct: dict[str, int] = {}
        reason_total: dict[str, int] = {}

        for ex in test_examples:
            ranked = self.rank(ex.question, all_candidates(ex))
            good_rank = next(r.rank for r in ranked if r.sql == ex.sql_good)
            top1_hits += int(good_rank == 1)
            rr_sum += 1.0 / good_rank

            good_score = self.score(ex.question, ex.sql_good)
            for bad in ex.sql_bad:
                bad_score = self.score(ex.question, bad.sql)
                pair_total += 1
                correct = int(good_score > bad_score)
                pair_correct += correct
                reason_total[bad.reason] = reason_total.get(bad.reason, 0) + 1
                reason_correct[bad.reason] = reason_correct.get(bad.reason, 0) + correct

        # per-mistake-category breakdown — too granular for the headline trend chart, but
        # kept in the metrics record for future debugging (e.g. "does the new approach
        # actually fix syntax_error cases, or just get lucky on wrong_filter_or_sort").
        pairwise_accuracy_by_reason = {
            reason: reason_correct[reason] / reason_total[reason] for reason in sorted(reason_total)
        }

        return {
            "n_examples": n_examples,
            "n_pairs": pair_total,
            "top1_accuracy": top1_hits / n_examples if n_examples else float("nan"),
            "pairwise_accuracy": pair_correct / pair_total if pair_total else float("nan"),
            "mrr": rr_sum / n_examples if n_examples else float("nan"),
            "pairwise_accuracy_by_reason": pairwise_accuracy_by_reason,
            "n_pairs_by_reason": dict(sorted(reason_total.items())),
        }

    def publish_metrics(self, metrics: dict, output_path: str | Path | None = None) -> None:
        print("Reward model evaluation:")
        for key in ("n_examples", "n_pairs", "top1_accuracy", "pairwise_accuracy", "mrr"):
            value = metrics.get(key)
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        by_reason = metrics.get("pairwise_accuracy_by_reason")
        if by_reason:
            print("  pairwise_accuracy_by_reason:")
            for reason, acc in by_reason.items():
                n = metrics["n_pairs_by_reason"][reason]
                print(f"    {reason}: {acc:.4f} (n={n})")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2))
            print(f"Wrote metrics to {output_path}")
