"""
Measure inference throughput and peak GPU memory for Mamba vs Transformer
at various sequence lengths.

Usage:
    python scripts/benchmark_mamba.py
    python scripts/benchmark_mamba.py --seq_lens 512 1024 2048 4096
"""

import argparse
import json
import os
import time

import torch
from tqdm.auto import tqdm
from train_mamba import build_mamba, build_transformer

WARMUP_ITERS = 10
BENCH_ITERS = 50


def resolve_dtype(dtype):
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[dtype]


def measure_throughput(model, input_ids, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Returns (tokens/sec, peak_memory_mb)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # reset after warmup for clean measurement

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = input_ids.numel() * iters / elapsed
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return tok_per_sec, peak_mem_mb


def measure_training_throughput(model, input_ids, optimizer,
                                warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Returns (training tokens/sec, peak_memory_mb) for a full
    forward + backward + optimizer step. This is the relevant measure
    for the paper's efficiency claims, which concern training cost."""
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    tok_per_sec = input_ids.numel() * iters / elapsed
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    model.eval()
    return tok_per_sec, peak_mem_mb


def run(args):
    device = "cuda"
    dtype = resolve_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    results = []

    model_builders = [("mamba", build_mamba), ("transformer", build_transformer)]
    for name, builder in tqdm(model_builders, desc="benchmark models", unit="model"):
        print(f"\n{'='*50}")
        print(f"Benchmarking: {name}")
        print(f"{'='*50}")

        max_seq = max(args.seq_lens)
        model = builder() if name == "mamba" else builder(seq_len=max_seq)
        model = model.to(device=device, dtype=dtype).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params / 1e6:.1f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

        for seq_len in tqdm(args.seq_lens, desc=f"{name} sequence lengths", unit="seq_len"):
            inf_ids = torch.randint(0, 50277, (args.batch_size, seq_len), device=device)
            trn_ids = torch.randint(0, 50277, (args.train_batch_size, seq_len), device=device)
            
            try:
                inf_tps, inf_mem = measure_throughput(model, inf_ids)
                print(f"  seq_len={seq_len:>5d} | inf {inf_tps:>10.0f} tok/s | "
                      f"{inf_mem:>8.1f} MB")
                
                try:
                    trn_tps, trn_mem = measure_training_throughput(model, trn_ids, optimizer)
                    print(f"  seq_len={seq_len:>5d} | trn {trn_tps:>10.0f} tok/s | "
                          f"{trn_mem:>8.1f} MB")
                except torch.cuda.OutOfMemoryError:
                    trn_tps, trn_mem = None, None
                    print(f"  seq_len={seq_len:>5d} | trn OOM")
                    torch.cuda.empty_cache()
                
                results.append({
                    "model": name, "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "train_batch_size": args.train_batch_size,
                    "inference_tokens_per_sec": round(inf_tps, 1),
                    "training_tokens_per_sec":  round(trn_tps, 1) if trn_tps else None,
                    "inference_peak_memory_mb": round(inf_mem, 1),
                    "training_peak_memory_mb":  round(trn_mem, 1) if trn_mem else None,
                    "parameters_M": round(n_params / 1e6, 1),
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  seq_len={seq_len:>5d} | OOM (inference)")
                results.append({
                    "model": name, "seq_len": seq_len,
                    "inference_tokens_per_sec": None,
                    "training_tokens_per_sec":  None,
                    "inference_peak_memory_mb": None,
                    "training_peak_memory_mb":  None,
                    "parameters_M": round(n_params / 1e6, 1),
                    "error": "OOM",
                })
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "dtype": str(dtype).replace("torch.", ""),
    }
    out_path = os.path.join(args.out_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", type=int, nargs="+",
                        default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training throughput benchmark "
                             "(smaller than inference due to backward memory cost)")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"],
                        default="auto")
    parser.add_argument("--out_dir", type=str, default="out")
    run(parser.parse_args())
