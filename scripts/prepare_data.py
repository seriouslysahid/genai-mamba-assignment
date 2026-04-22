"""
prepare_data.py — Download and tokenize a small subset of The Pile
for the Mamba reproduction experiments.

Produces data/train.bin and data/val.bin as memory-mapped uint16 arrays.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --num_train 50000 --num_val 2000
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading tokenizer (EleutherAI/gpt-neox-20b)...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print("Loading The Pile (streaming)...")
    ds = load_dataset("EleutherAI/pile", split="train", streaming=True)

    for split_name, n_examples in [("train", args.num_train), ("val", args.num_val)]:
        print(f"\nTokenizing {split_name} split ({n_examples} examples)...")
        all_ids = []
        count = 0
        for example in tqdm(ds, total=n_examples, desc=split_name):
            if count >= n_examples:
                break
            text = example.get("text", "")
            if not text.strip():
                continue
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
            count += 1

        arr = np.array(all_ids, dtype=np.uint16)
        out_path = os.path.join(args.out_dir, f"{split_name}.bin")
        arr.tofile(out_path)
        n_tokens = len(arr)
        print(f"  → {n_tokens:,} tokens saved to {out_path} "
              f"({n_tokens * 2 / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=50_000,
                        help="Number of Pile examples for training split")
    parser.add_argument("--num_val", type=int, default=2_000,
                        help="Number of Pile examples for validation split")
    parser.add_argument("--out_dir", type=str, default="data")
    main(parser.parse_args())
