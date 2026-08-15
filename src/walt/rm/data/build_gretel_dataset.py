"""Downloads and samples the gretelai/synthetic_text_to_sql dataset into a standardized
JSONL, separate from the other sources' pre_process.py pipeline since this source
already carries its own sql_context (see gretel.py) and its own train/test split.

gretel's "test" split is mapped to our `split="val"` (held out for agent-level eval,
invisible to RM training/CV — same meaning as pre_process.py's val_fraction split);
its "train" split maps to `split="trainval"`.

Usage:
    python -m walt.rm.data.build_gretel_dataset --train-count 2000 --test-count 200
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from walt.rm.data.base import Example
from walt.rm.data.gretel import GretelAdapter, download_split

load_dotenv()

try:
    DATA_DIR = Path(os.environ["DATA_PATH"]).expanduser().resolve()
except KeyError as exc:
    raise RuntimeError(
        "DATA_PATH is not set. Add it to a .env file (see .env.example) "
        "or export it in your environment."
    ) from exc

RAW_DIR = DATA_DIR / "gretel"
DEFAULT_OUTPUT = DATA_DIR / "output" / "gretel" / "gretel_data.jsonl"


def sample_split(hf_split: str, count: int, our_split: str, seed: int) -> list[Example]:
    parquet_path = download_split(hf_split, RAW_DIR)
    examples = list(GretelAdapter(parquet_path).load())
    if count < len(examples):
        examples = random.Random(seed).sample(examples, count)
    return [dataclasses.replace(ex, split=our_split) for ex in examples]


def write_jsonl(examples: list[Example], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-count", type=int, default=2000, help="Rows to sample from gretel's train split (-> split=trainval)")
    parser.add_argument("--test-count", type=int, default=200, help="Rows to sample from gretel's test split (-> split=val)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling and shuffling")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    args = parser.parse_args()

    trainval = sample_split("train", args.train_count, "trainval", args.seed)
    val = sample_split("test", args.test_count, "val", args.seed)

    combined = trainval + val
    random.Random(args.seed).shuffle(combined)

    write_jsonl(combined, args.output)

    valid = sum(1 for ex in combined if ex.sql_context_valid)
    print(f"Wrote {len(combined)} examples to {args.output}")
    print(f"  trainval: {len(trainval)}")
    print(f"  val: {len(val)}")
    print(f"sql_context execution check: {valid}/{len(combined)} passed ({100 * valid / len(combined):.1f}%)")


if __name__ == "__main__":
    main()
