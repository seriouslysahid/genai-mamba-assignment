"""
Evaluate trained Mamba and Transformer checkpoints.
Computes held-out perplexity and generates sample text.

Usage:
    python scripts/evaluate_mamba.py --model mamba --ckpt out/mamba_final.pt
    python scripts/evaluate_mamba.py --model transformer --ckpt out/transformer_final.pt
"""

from __future__ import annotations

import argparse
from itertools import islice
import json
import logging
import math
import os
import textwrap
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging as hf_logging

from train_mamba import DEFAULTS, build_mamba, build_transformer


# Keep CLI output clean.
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


class CleanPileShardDataset(torch.utils.data.Dataset):
    """
    Memory-mapped dataset over pre-tokenized .bin files.

    The memmap is copied into a writable NumPy array so PyTorch does not emit
    the non-writable tensor warning when constructing tensors from it.
    """

    def __init__(self, data_dir: str, split: str, seq_len: int):
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run prepare_data.py first."
            )

        arr = np.array(np.memmap(path, dtype="uint16", mode="r"), copy=True)
        self.data = torch.from_numpy(arr).long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


@torch.inference_mode()
def evaluate(model, loader, device, max_batches=None):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    batches = islice(loader, max_batches) if max_batches else loader

    for x, y in tqdm(
        batches,
        total=total_batches,
        desc="Evaluating",
        unit="batch",
        leave=False,
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x).logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def _wrap_text(text: str, width: int = 92) -> str:
    paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
    if not paragraphs:
        return ""
    return "\n\n".join(textwrap.fill(p, width=width) for p in paragraphs)


@torch.inference_mode()
def generate_samples(
    model,
    tokenizer,
    device,
    prompts,
    max_new_tokens=128,
    is_mamba=False,
):
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    results = []

    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if is_mamba:
            # Mamba generate() uses max_length and does not accept attention_mask.
            out = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                temperature=0.8,
                top_k=40,
                top_p=0.95,
                cg=True,
            )
        else:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=40,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        results.append(
            {
                "prompt": prompt,
                "generation": text,
            }
        )

    return results


def pretty_print_results(args, dtype, n_params, loss, ppl, bpb):
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"Model           : {args.model}")
    print(f"Checkpoint      : {args.ckpt}")
    print(f"Parameters      : {n_params / 1e6:.1f}M")
    print(f"DType           : {str(dtype).replace('torch.', '')}")
    print(f"Sequence Length : {args.seq_len}")
    print(f"Validation Loss : {loss:.4f}")
    print(f"Perplexity      : {ppl:.2f}")
    print(f"Bits-per-byte   : {bpb:.4f}")
    print("=" * 100)


def pretty_print_samples(model_name, samples):
    print("\n" + "=" * 100)
    print(f"{model_name.upper()} GENERATION SAMPLES")
    print("=" * 100)

    for idx, sample in enumerate(samples, 1):
        print("\n" + "-" * 100)
        print(f"SAMPLE {idx}")
        print("-" * 100)

        print("\nPROMPT:")
        print(_wrap_text(sample["prompt"]))

        print("\nGENERATION:")
        print(_wrap_text(sample["generation"][:1200]))

    print("\n" + "=" * 100)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_mamba = args.model == "mamba"

    print(f"\nLoading {args.model} model from {args.ckpt}...")

    dtype = resolve_dtype(args.dtype)

    model = build_mamba() if is_mamba else build_transformer(args.seq_len)

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device=device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    val_ds = CleanPileShardDataset(args.data_dir, "val", args.seq_len)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    loss, ppl = evaluate(
        model,
        val_loader,
        device,
        max_batches=args.max_batches,
    )
    bpb = round(loss / math.log(2), 4)

    pretty_print_results(args, dtype, n_params, loss, ppl, bpb)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [
        "The future of artificial intelligence",
        "In a recent study, researchers found that",
        "State space models are different from transformers because",
    ]

    samples = generate_samples(
        model,
        tokenizer,
        device,
        prompts,
        is_mamba=is_mamba,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        "model": args.model,
        "checkpoint": args.ckpt,
        "dtype": str(dtype).replace("torch.", ""),
        "parameters_M": round(n_params / 1e6, 1),
        "val_loss": round(loss, 4),
        "perplexity": round(ppl, 2),
        "bpb": bpb,
        "samples": samples,
    }

    out_path = os.path.join(args.out_dir, f"{args.model}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out_path}")
    pretty_print_samples(args.model, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mamba", "transformer"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    main(parser.parse_args())