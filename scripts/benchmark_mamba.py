"""
Beautiful benchmarking script for comparing Mamba vs Transformer.

Measures:
- Inference throughput
- Training throughput
- Peak GPU memory
- OOM behavior
- Scaling with sequence length

Usage:
    python scripts/benchmark_mamba.py

Custom sequence lengths:
    python scripts/benchmark_mamba.py --seq_lens 512 1024 2048 4096
"""

import argparse
import json
import os
import time
import warnings

import torch
from tqdm.auto import tqdm

from train_mamba import build_mamba, build_transformer

# Suppress noisy warnings
warnings.filterwarnings("ignore")

WARMUP_ITERS = 10
BENCH_ITERS = 50


# ======================================================================================
# UTILS
# ======================================================================================

def resolve_dtype(dtype):
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def format_num(x):
    if x is None:
        return "OOM"

    if x >= 1e6:
        return f"{x / 1e6:.2f}M"

    if x >= 1e3:
        return f"{x / 1e3:.2f}K"

    return f"{x:.2f}"


def print_header(title):
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)


def print_subheader(title):
    print("\n" + "-" * 100)
    print(title.center(100))
    print("-" * 100)


# ======================================================================================
# BENCHMARK FUNCTIONS
# ======================================================================================

@torch.no_grad()
def measure_inference(model, input_ids, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """
    Measures inference throughput and memory.
    Returns:
        (tokens_per_sec, peak_memory_mb)
    """

    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        model(input_ids)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    for _ in range(iters):
        model(input_ids)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = (input_ids.numel() * iters) / elapsed
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return tok_per_sec, peak_mem_mb


def measure_training(model, input_ids, optimizer,
                     warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """
    Measures training throughput and memory.
    Includes:
    - forward
    - backward
    - optimizer step
    """

    model.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
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

    tok_per_sec = (input_ids.numel() * iters) / elapsed
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model.eval()

    return tok_per_sec, peak_mem_mb


# ======================================================================================
# MAIN
# ======================================================================================

def run(args):
    device = "cuda"

    dtype = resolve_dtype(args.dtype)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)

    print_header("MAMBA vs TRANSFORMER BENCHMARK")

    print(f"GPU                : {gpu_name}")
    print(f"DType             : {str(dtype).replace('torch.', '')}")
    print(f"Inference Batch   : {args.batch_size}")
    print(f"Training Batch    : {args.train_batch_size}")
    print(f"Sequence Lengths  : {args.seq_lens}")
    print(f"Warmup Iterations : {WARMUP_ITERS}")
    print(f"Benchmark Iters   : {BENCH_ITERS}")

    results = []

    model_builders = [
        ("mamba", build_mamba),
        ("transformer", build_transformer),
    ]

    for model_name, builder in model_builders:

        print_header(f"BENCHMARKING: {model_name.upper()}")

        max_seq = max(args.seq_lens)

        if model_name == "mamba":
            model = builder()
        else:
            model = builder(seq_len=max_seq)

        model = model.to(device=device, dtype=dtype)

        n_params = sum(p.numel() for p in model.parameters())

        print(f"Parameters         : {n_params / 1e6:.1f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

        model_results = []

        for seq_len in tqdm(args.seq_lens, desc=f"{model_name} seq lengths"):

            print_subheader(f"{model_name.upper()} | SEQ LEN = {seq_len}")

            inf_ids = torch.randint(
                0,
                50277,
                (args.batch_size, seq_len),
                device=device,
            )

            trn_ids = torch.randint(
                0,
                50277,
                (args.train_batch_size, seq_len),
                device=device,
            )

            # ------------------------------------------------------------------
            # Inference
            # ------------------------------------------------------------------

            try:
                inf_tps, inf_mem = measure_inference(model, inf_ids)

                print("INFERENCE")
                print(f"  Throughput : {format_num(inf_tps)} tok/s")
                print(f"  Peak Memory: {inf_mem:.1f} MB")

            except torch.cuda.OutOfMemoryError:

                inf_tps, inf_mem = None, None

                print("INFERENCE")
                print("  STATUS     : OOM")

                torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Training
            # ------------------------------------------------------------------

            try:
                trn_tps, trn_mem = measure_training(
                    model,
                    trn_ids,
                    optimizer,
                )

                print("\nTRAINING")
                print(f"  Throughput : {format_num(trn_tps)} tok/s")
                print(f"  Peak Memory: {trn_mem:.1f} MB")

            except torch.cuda.OutOfMemoryError:

                trn_tps, trn_mem = None, None

                print("\nTRAINING")
                print("  STATUS     : OOM")

                torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Speedup
            # ------------------------------------------------------------------

            model_results.append({
                "model": model_name,
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "train_batch_size": args.train_batch_size,
                "inference_tokens_per_sec": round(inf_tps, 1) if inf_tps else None,
                "training_tokens_per_sec": round(trn_tps, 1) if trn_tps else None,
                "inference_peak_memory_mb": round(inf_mem, 1) if inf_mem else None,
                "training_peak_memory_mb": round(trn_mem, 1) if trn_mem else None,
                "parameters_M": round(n_params / 1e6, 1),
            })

        results.extend(model_results)

        del model
        torch.cuda.empty_cache()

    # ==================================================================================
    # SAVE RESULTS
    # ==================================================================================

    os.makedirs(args.out_dir, exist_ok=True)

    meta = {
        "gpu": gpu_name,
        "dtype": str(dtype).replace("torch.", ""),
    }

    out_path = os.path.join(args.out_dir, "benchmark_results.json")

    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": meta,
                "results": results,
            },
            f,
            indent=2,
        )

    print_header("BENCHMARK COMPLETE")

    print(f"Results saved to: {out_path}")

    # ==================================================================================
    # FINAL SUMMARY TABLE
    # ==================================================================================

    print_header("FINAL SUMMARY")

    print(
        f"{'MODEL':<15}"
        f"{'SEQ':<10}"
        f"{'INF TOK/S':<18}"
        f"{'TRN TOK/S':<18}"
        f"{'INF MEM (MB)':<18}"
        f"{'TRN MEM (MB)':<18}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['model']:<15}"
            f"{r['seq_len']:<10}"
            f"{str(r['inference_tokens_per_sec']):<18}"
            f"{str(r['training_tokens_per_sec']):<18}"
            f"{str(r['inference_peak_memory_mb']):<18}"
            f"{str(r['training_peak_memory_mb']):<18}"
        )

    print("\nDone.")


# ======================================================================================
# ENTRYPOINT
# ======================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="out",
    )

    run(parser.parse_args())