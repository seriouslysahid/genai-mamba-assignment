"""
prepare_data.py - Download and tokenize a small subset of The Pile
for the Mamba reproduction experiments.

Produces data/train.bin and data/val.bin as memory-mapped uint16 arrays.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --num_train 50000 --num_val 2000
    python scripts/prepare_data.py --dataset_name monology/pile-uncopyrighted
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer ({args.tokenizer_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Loading dataset ({args.dataset_name}, streaming train split)...")
    # Stream the dataset, batched processing is much faster
    ds = load_dataset(args.dataset_name, split="train", streaming=True)

    for split_name, n_examples in [("train", args.num_train), ("val", args.num_val)]:
        print(f"\nTokenizing {split_name} split ({n_examples} examples)...")
        all_ids = []
        
        # Take the required number of examples from the stream
        ds_subset = ds.take(n_examples)
        
        # We can map with batched=True to speed up tokenization
        def tokenize_batch(batch):
            # Tokenize and filter out empty sequences
            texts = [t for t in batch["text"] if t and t.strip()]
            if not texts:
                return {"ids": []}
            # encode produces a list of lists of ints
            ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
            # Flatten the batch
            flat_ids = [i for seq in ids for i in seq]
            return {"ids": flat_ids}

        # Apply map operation - IterableDatasets support batched=True mapping
        ds_tokenized = ds_subset.map(tokenize_batch, batched=True, batch_size=1000, remove_columns=list(next(iter(ds_subset)).keys()))

        count = 0
        for item in tqdm(ds_tokenized, total=n_examples, desc=f"tokenizing {split_name}", unit="batch"):
            all_ids.extend(item["ids"])
            count += 1
            if count >= n_examples:
                break

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
    parser.add_argument("--dataset_name", type=str,
                        default="monology/pile-uncopyrighted",
                        help="Hugging Face dataset to stream")
    parser.add_argument("--tokenizer_name", type=str,
                        default="EleutherAI/gpt-neox-20b",
                        help="Tokenizer used for uint16 GPT-NeoX token IDs")
    parser.add_argument("--out_dir", type=str, default="data")
    main(parser.parse_args())
