"""
Evaluate trained Mamba and Transformer checkpoints.
Computes held-out perplexity and generates sample text.

Usage:
    python scripts/evaluate_mamba.py --model mamba --ckpt out/mamba_final.pt
    python scripts/evaluate_mamba.py --model transformer --ckpt out/transformer_final.pt
"""

import argparse
from itertools import islice
import json
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from train_mamba import PileShardDataset, build_mamba, build_transformer, DEFAULTS


def resolve_dtype(dtype):
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[dtype]


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    batches = islice(loader, max_batches) if max_batches else loader
    for x, y in tqdm(batches, total=total_batches, desc="evaluating", unit="batch"):
        x, y = x.to(device), y.to(device)
        logits = model(x).logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


def generate_samples(model, tokenizer, device, prompts, max_new=128, is_mamba=False):
    model.eval()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if is_mamba:
            out = model.generate(
                input_ids=ids, max_length=ids.shape[1] + max_new,
                temperature=0.8, top_k=40, cg=True,
            )
            text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        else:
            out = model.generate(
                ids, max_new_tokens=max_new, temperature=0.8, top_k=40,
                do_sample=True,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generation": text})
    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_mamba = args.model == "mamba"

    print(f"Loading {args.model} model from {args.ckpt}...")
    dtype = resolve_dtype(args.dtype)
    model = build_mamba() if is_mamba else build_transformer(args.seq_len)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device=device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    seq_len = args.seq_len
    val_ds = PileShardDataset(args.data_dir, "val", seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)

    loss, ppl = evaluate(model, val_loader, device, max_batches=args.max_batches)
    print(f"Val loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    prompts = [
        "The future of artificial intelligence",
        "In a recent study, researchers found that",
        "State space models are different from transformers because",
    ]
    samples = generate_samples(model, tokenizer, device, prompts, is_mamba=is_mamba)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        "model": args.model,
        "checkpoint": args.ckpt,
        "dtype": str(dtype).replace("torch.", ""),
        "parameters_M": round(n_params / 1e6, 1),
        "val_loss": round(loss, 4),
        "perplexity": round(ppl, 2),
        "samples": samples,
    }
    out_path = os.path.join(args.out_dir, f"{args.model}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")

    for s in samples:
        print(f"\n--- Prompt: {s['prompt']}")
        print(s["generation"][:300])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mamba", "transformer"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"],
                        default="auto")
    main(parser.parse_args())
