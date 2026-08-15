"""Builds a downsampled, standardized JSONL training set from all registered data sources.

Usage:
    python -m walt.rm.data.launcher --target-count 5000 --output rm_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from walt.rm.data.base import BaseAdapter, Example
from walt.rm.data.dbasql import DBASQLAdapter
from walt.rm.data.spider import SpiderAdapter

load_dotenv()

try:
    DATA_DIR = Path(os.environ["DATA_PATH"]).expanduser().resolve()
except KeyError as exc:
    raise RuntimeError(
        "DATA_PATH is not set. Add it to a .env file (see .env.example) "
        "or export it in your environment."
    ) from exc

# source name -> (file_path, adapter class)
SOURCES: dict[str, tuple[Path, type[BaseAdapter]]] = {
    "spider": (DATA_DIR / "spider_text_sql.csv", SpiderAdapter),
    "dbasql": (DATA_DIR / "DBASQL.json", DBASQLAdapter),
}


def compute_sample_sizes(sizes: dict[str, int], target_total: int) -> dict[str, int]:
    """Proportionally allocate target_total across sources, capped at each source's size.

    If target_total exceeds the number of examples available, every source is used in
    full (there's no downsampling to do, and this adapter never upsamples/duplicates).
    """
    total = sum(sizes.values())
    if target_total >= total:
        return dict(sizes)

    raw = {name: size / total * target_total for name, size in sizes.items()}
    allocated = {name: int(raw[name]) for name in sizes}
    remainder = target_total - sum(allocated.values())
    # Give the leftover slots to whichever sources lost the most to flooring.
    order = sorted(sizes, key=lambda name: raw[name] - allocated[name], reverse=True)
    for name in order[:remainder]:
        allocated[name] += 1
    return allocated


def build_dataset(target_count: int, seed: int = 42) -> list[Example]:
    rng = random.Random(seed)

    examples_by_source: dict[str, list[Example]] = {}
    for name, (file_path, adapter_cls) in SOURCES.items():
        examples_by_source[name] = list(adapter_cls(file_path).load())

    sizes = {name: len(examples) for name, examples in examples_by_source.items()}
    sample_sizes = compute_sample_sizes(sizes, target_count)

    combined: list[Example] = []
    for name, examples in examples_by_source.items():
        n = sample_sizes[name]
        combined.extend(examples if n >= len(examples) else rng.sample(examples, n))

    rng.shuffle(combined)
    return combined


def write_jsonl(examples: list[Example], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, help="Desired total number of examples across all sources", default=1000)
    parser.add_argument("--output", type=Path, default=Path(DATA_DIR, "output", "rm_data.jsonl"), help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling and shuffling")
    args = parser.parse_args()

    examples = build_dataset(args.target_count, seed=args.seed)

    write_jsonl(examples, args.output)

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.source] = counts.get(example.source, 0) + 1

    print(f"Wrote {len(examples)} examples to {args.output} (target was {args.target_count})")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
