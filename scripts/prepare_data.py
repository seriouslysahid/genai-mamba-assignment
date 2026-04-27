# download and prepare the pile dataset
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

NUM_TRAIN = 300_000
NUM_VAL = 5_000
DATASET = "monology/pile-uncopyrighted"
TOKENIZER = "EleutherAI/gpt-neox-20b"
OUT_DIR = "data"

if __name__ == "__main__":
    print(f"loading dataset {DATASET}...")
    dataset = load_dataset(DATASET, split="train", streaming=True)
    
    print(f"loading tokenizer {TOKENIZER}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def get_tokens(num_tokens):
        tokens = []
        for row in dataset:
            toks = tokenizer.encode(row["text"])
            tokens.extend(toks)
            if len(tokens) >= num_tokens:
                break
        return tokens[:num_tokens]

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"processing {NUM_TRAIN} train tokens...")
    train_tokens = get_tokens(NUM_TRAIN)
    train_arr = np.memmap(os.path.join(OUT_DIR, "train.bin"), dtype=np.uint16, mode="w+", shape=(len(train_tokens),))
    train_arr[:] = train_tokens
    train_arr.flush()

    print(f"processing {NUM_VAL} val tokens...")
    val_tokens = get_tokens(NUM_VAL)
    val_arr = np.memmap(os.path.join(OUT_DIR, "val.bin"), dtype=np.uint16, mode="w+", shape=(len(val_tokens),))
    val_arr[:] = val_tokens
    val_arr.flush()

    print("done!")
