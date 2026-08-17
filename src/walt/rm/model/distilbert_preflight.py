"""Two cheap checks to run BEFORE committing to a full DistilBertRewardModel training
run — either could change the truncation strategy, batch size, or device fallback
plan, so both are reported here and this script deliberately stops without launching
any real fit().

Part A: token-length distribution over the target dataset with the real tokenizer and
the real [CLS] context [SEP] question [SEP] sql [SEP] format, so max_length/truncation
is chosen from actual data rather than assumed.

Part B: a real forward+backward MPS sanity check (actual model/head/tokenizer, tiny
batch) — watching for silent CPU-fallback ops, hard MPS errors, and basic loss-goes-
down sanity over a few steps on a fixed tiny batch.

Usage:
    uv run python -m walt.rm.model.distilbert_preflight --input data/output/gretel/gretel_enhanced.jsonl
"""
from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from walt.rm.model.base import all_candidates, group_split, load_examples
from walt.rm.model.distilbert_model import DEFAULT_MODEL_NAME, build_pairs, resolve_device


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, math.ceil(p * len(sorted_values)) - 1))
    return sorted_values[idx]


def token_length_report(examples, tokenizer, max_length: int) -> None:
    total_lengths: list[int] = []
    context_lengths: list[int] = []
    for ex in examples:
        context_text = "\n".join(ex.sql_context_clean)
        context_ids = tokenizer.encode(context_text, add_special_tokens=False)
        question_ids = tokenizer.encode(ex.question, add_special_tokens=False)
        context_lengths.append(len(context_ids))
        for sql in all_candidates(ex):
            sql_ids = tokenizer.encode(sql, add_special_tokens=False)
            total_lengths.append(4 + len(context_ids) + len(question_ids) + len(sql_ids))  # 4 = [CLS] + 3x[SEP]

    total_lengths.sort()
    context_lengths.sort()
    n = len(total_lengths)
    n_fits = sum(1 for t in total_lengths if t <= max_length)
    print(f"\n=== Token-length distribution (n={n} candidate sequences, {len(context_lengths)} examples) ===")
    print(
        f"total sequence length: min={total_lengths[0]} median={_percentile(total_lengths, 0.5)} "
        f"p95={_percentile(total_lengths, 0.95)} max={total_lengths[-1]}"
    )
    print(f"fraction fitting within max_length={max_length} untruncated: {n_fits}/{n} ({n_fits/n:.1%})")
    print(
        f"context-only length (the only segment truncation touches): "
        f"min={context_lengths[0]} median={_percentile(context_lengths, 0.5)} "
        f"p95={_percentile(context_lengths, 0.95)} max={context_lengths[-1]}"
    )


def mps_sanity_check(examples, seed: int, n_pairs: int, n_steps: int, max_length: int) -> dict:
    device = resolve_device(None)
    print(f"\n=== MPS sanity check ===")
    print(f"torch={torch.__version__} transformers={transformers.__version__} selected device={device}")

    def run(fallback_enabled: bool):
        if fallback_enabled:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        from walt.rm.model.distilbert_model import DistilBertRewardModel  # local import: env var must be set first

        model = DistilBertRewardModel(device=str(device), max_length=max_length, seed=seed)
        pairs = build_pairs(examples, seed=seed)[:n_pairs]
        pretok = model._pretokenize(pairs)
        optimizer = torch.optim.AdamW(model.net.parameters(), lr=2e-5)
        model.net.train()
        losses, step_seconds = [], []
        fallback_ops: set[str] = set()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(n_steps):
                t0 = time.perf_counter()
                optimizer.zero_grad()
                logits_a, logits_b, labels = model._forward_pairs(pretok)
                loss = model._loss_fn(logits_a - logits_b, labels)
                loss.backward()
                optimizer.step()
                step_seconds.append(time.perf_counter() - t0)
                losses.append(loss.item())
            for w in caught:
                msg = str(w.message)
                if "not currently supported on the MPS backend" in msg:
                    fallback_ops.add(msg.split("'")[1] if "'" in msg else msg)
        return losses, step_seconds, fallback_ops

    hard_error = None
    fallback_used = False
    try:
        losses, step_seconds, fallback_ops = run(fallback_enabled=False)
    except Exception as e:
        hard_error = e
        losses, step_seconds, fallback_ops = [], [], set()
        if device.type == "mps":
            print(f"Native MPS run failed ({e!r}); retrying with PYTORCH_ENABLE_MPS_FALLBACK=1")
            fallback_used = True
            try:
                losses, step_seconds, fallback_ops = run(fallback_enabled=True)
                hard_error = None
            except Exception as e2:
                hard_error = e2

    print(f"fallback warnings fired: {sorted(fallback_ops) or 'none'}")
    print(f"hard error: {repr(hard_error) if hard_error else 'none'}")
    if losses:
        print(f"loss over {len(losses)} steps: first={losses[0]:.4f} last={losses[-1]:.4f} all={[round(l, 4) for l in losses]}")
        avg_step = sum(step_seconds) / len(step_seconds)
        print(f"avg step (forward+backward+optimizer.step) time: {avg_step*1000:.1f}ms")
    else:
        avg_step = None

    recommendation = "STOP AND ASK — MPS appears broadly broken"
    if hard_error is None and not fallback_used and fallback_ops == set():
        recommendation = "GO — proceed natively on MPS"
    elif hard_error is None and (fallback_used or fallback_ops):
        recommendation = f"GO WITH CAUTION — MPS works but with fallback for: {sorted(fallback_ops) or ['(recovered via env var)']}"
    elif hard_error is not None and device.type != "mps":
        recommendation = "N/A — no MPS device available, running on CPU by default"
    print(f"recommendation: {recommendation}")

    return {
        "device": str(device),
        "hard_error": repr(hard_error) if hard_error else None,
        "fallback_ops": sorted(fallback_ops),
        "losses": losses,
        "avg_step_seconds": avg_step,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/output/gretel/gretel_enhanced.jsonl"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pairs", type=int, default=8, help="Tiny batch size for the MPS sanity check")
    parser.add_argument("--n-steps", type=int, default=15, help="Number of forward+backward steps in the sanity check")
    parser.add_argument("--batch-size", type=int, default=8, help="Planned real training batch size, for the time estimate")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Planned real grad accumulation, for the time estimate")
    parser.add_argument("--num-epochs", type=int, default=15, help="Planned real epoch cap, for the time estimate")
    args = parser.parse_args()

    examples = load_examples(args.input)
    examples = [ex for ex in examples if ex.split != "val"]
    print(f"Loaded {len(examples)} trainval examples from {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    token_length_report(examples, tokenizer, args.max_length)

    sanity = mps_sanity_check(examples, args.seed, args.n_pairs, args.n_steps, args.max_length)

    if sanity["avg_step_seconds"] is not None:
        internal_train, _ = group_split(examples, test_size=0.1, seed=args.seed + 1)
        n_train_pairs = len(build_pairs(internal_train, seed=args.seed))
        micro_batches_per_epoch = math.ceil(n_train_pairs / args.batch_size)
        seconds_per_epoch = sanity["avg_step_seconds"] * micro_batches_per_epoch
        total_capped_seconds = seconds_per_epoch * args.num_epochs
        print(
            f"\n=== Rough full-run time estimate (extrapolated from the tiny sanity batch — "
            f"real per-example cost may differ) ==="
        )
        print(f"planned n_train_pairs (internal 90% split): {n_train_pairs}, {micro_batches_per_epoch} micro-batches/epoch")
        print(f"estimated seconds/epoch: {seconds_per_epoch:.1f}s (~{seconds_per_epoch/60:.1f} min)")
        print(
            f"estimated total (if all {args.num_epochs} capped epochs run, no early stop): "
            f"{total_capped_seconds:.0f}s (~{total_capped_seconds/60:.1f} min)"
        )
        print("Early stopping will very likely stop before the cap — this is an upper bound.")

    print("\nStopping here per plan — no full fit() run launched. Review the above before proceeding.")


if __name__ == "__main__":
    main()
