"""
Generate plots from training logs and benchmark results.
Outputs PNGs for the slide deck.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --out_dir out --log_dir out
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (7, 4.5),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {"mamba": "#4C72B0", "transformer": "#DD8452"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_training_loss(mamba_log, transformer_log, out_dir):
    fig, ax = plt.subplots()
    ax.plot(mamba_log["step"], mamba_log["train_loss"],
            label="Mamba-130M", color=COLORS["mamba"], linewidth=1.5)
    ax.plot(transformer_log["step"], transformer_log["train_loss"],
            label="Transformer-130M", color=COLORS["transformer"], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss — Mamba vs Transformer")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "training_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_val_perplexity(mamba_log, transformer_log, out_dir):
    fig, ax = plt.subplots()
    if mamba_log.get("val_loss"):
        steps = [v["step"] for v in mamba_log["val_loss"]]
        ppls = [v["ppl"] for v in mamba_log["val_loss"]]
        ax.plot(steps, ppls, "o-", label="Mamba-130M",
                color=COLORS["mamba"], linewidth=1.5, markersize=4)
    if transformer_log.get("val_loss"):
        steps = [v["step"] for v in transformer_log["val_loss"]]
        ppls = [v["ppl"] for v in transformer_log["val_loss"]]
        ax.plot(steps, ppls, "s-", label="Transformer-130M",
                color=COLORS["transformer"], linewidth=1.5, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity — Mamba vs Transformer")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "val_perplexity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_throughput(bench_results, out_dir):
    mamba = [r for r in bench_results if r["model"] == "mamba" and r.get("inference_tokens_per_sec")]
    trans = [r for r in bench_results if r["model"] == "transformer" and r.get("inference_tokens_per_sec")]

    fig, ax = plt.subplots()
    if mamba:
        ax.plot([r["seq_len"] for r in mamba],
                [r["inference_tokens_per_sec"] for r in mamba],
                "o-", label="Mamba-130M", color=COLORS["mamba"], linewidth=1.5)
    if trans:
        ax.plot([r["seq_len"] for r in trans],
                [r["inference_tokens_per_sec"] for r in trans],
                "s-", label="Transformer-130M", color=COLORS["transformer"], linewidth=1.5)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Inference Throughput vs Sequence Length")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    path = os.path.join(out_dir, "throughput.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_training_throughput(bench_results, out_dir):
    mamba = [r for r in bench_results
             if r["model"] == "mamba" and r.get("training_tokens_per_sec")]
    trans = [r for r in bench_results
             if r["model"] == "transformer" and r.get("training_tokens_per_sec")]
    
    fig, ax = plt.subplots()
    if mamba:
        ax.plot([r["seq_len"] for r in mamba],
                [r["training_tokens_per_sec"] for r in mamba],
                "o-", label="Mamba-130M", color=COLORS["mamba"], linewidth=1.5)
    if trans:
        ax.plot([r["seq_len"] for r in trans],
                [r["training_tokens_per_sec"] for r in trans],
                "s-", label="Transformer-130M",
                color=COLORS["transformer"], linewidth=1.5)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Training Tokens / sec")
    ax.set_title("Training Throughput vs Sequence Length\n"
                 "(forward + backward + optimizer step)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    path = os.path.join(out_dir, "training_throughput.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_memory(bench_results, out_dir):
    mamba = [r for r in bench_results if r["model"] == "mamba" and r.get("inference_peak_memory_mb")]
    trans = [r for r in bench_results if r["model"] == "transformer" and r.get("inference_peak_memory_mb")]

    fig, ax = plt.subplots()
    if mamba:
        ax.plot([r["seq_len"] for r in mamba],
                [r["inference_peak_memory_mb"] / 1024 for r in mamba],
                "o-", label="Mamba-130M", color=COLORS["mamba"], linewidth=1.5)
    if trans:
        ax.plot([r["seq_len"] for r in trans],
                [r["inference_peak_memory_mb"] / 1024 for r in trans],
                "s-", label="Transformer-130M", color=COLORS["transformer"], linewidth=1.5)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Memory Usage vs Sequence Length")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    path = os.path.join(out_dir, "memory.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_speedup(bench_results, out_dir):
    mamba = {r["seq_len"]: r["inference_tokens_per_sec"] for r in bench_results
             if r["model"] == "mamba" and r.get("inference_tokens_per_sec")}
    trans = {r["seq_len"]: r["inference_tokens_per_sec"] for r in bench_results
             if r["model"] == "transformer" and r.get("inference_tokens_per_sec")}
    common = sorted(set(mamba) & set(trans))
    if not common:
        return

    ratios = [mamba[s] / trans[s] for s in common]
    fig, ax = plt.subplots()
    ax.bar([str(s) for s in common], ratios, color=COLORS["mamba"], alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mamba / Transformer Throughput Ratio")
    ax.set_title("Speedup: Mamba vs Transformer")
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "speedup.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def print_slide_summary(log_dir):
    """Print a ready-to-copy table for slide 15."""
    import math
    
    summary = {}
    for model in ["mamba", "transformer"]:
        path = os.path.join(log_dir, f"{model}_eval.json")
        if os.path.exists(path):
            data = load_json(path)
            summary[model] = {
                "perplexity": data.get("perplexity", "—"),
                "bpb":        data.get("bpb", "—"),
            }
    
    bench_path = os.path.join(log_dir, "benchmark_results.json")
    if os.path.exists(bench_path):
        raw = load_json(bench_path)
        results = raw["results"] if isinstance(raw, dict) else raw
        
        for model in ["mamba", "transformer"]:
            if model not in summary:
                summary[model] = {}
            
            for r in results:
                if r["model"] != model:
                    continue
                sl = r["seq_len"]
                if sl == 1024:
                    summary[model]["inf_tps_1024"] = (
                        f"{r['inference_tokens_per_sec']:,.0f}"
                        if r.get("inference_tokens_per_sec") else "OOM")
                    summary[model]["trn_tps_1024"] = (
                        f"{r['training_tokens_per_sec']:,.0f}"
                        if r.get("training_tokens_per_sec") else "OOM")
                if sl == 4096:
                    summary[model]["inf_tps_4096"] = (
                        f"{r['inference_tokens_per_sec']:,.0f}"
                        if r.get("inference_tokens_per_sec") else "OOM")
                    summary[model]["trn_tps_4096"] = (
                        f"{r['training_tokens_per_sec']:,.0f}"
                        if r.get("training_tokens_per_sec") else "OOM")
                    summary[model]["trn_mem_4096"] = (
                        f"{r['training_peak_memory_mb']/1024:.2f} GB"
                        if r.get("training_peak_memory_mb") else "OOM")
    
    print("\n" + "=" * 62)
    print("SLIDE 15 — Benchmark Results (copy into slides)")
    print("=" * 62)
    print(f"{'Metric':<32} {'Mamba':>12} {'Transformer':>14}")
    print("-" * 62)
    
    rows = [
        ("Perplexity",                    "perplexity",  "perplexity"),
        ("BPB",                           "bpb",         "bpb"),
        ("Inf. throughput @ 1024 (tok/s)","inf_tps_1024","inf_tps_1024"),
        ("Trn. throughput @ 1024 (tok/s)","trn_tps_1024","trn_tps_1024"),
        ("Inf. throughput @ 4096 (tok/s)","inf_tps_4096","inf_tps_4096"),
        ("Trn. throughput @ 4096 (tok/s)","trn_tps_4096","trn_tps_4096"),
        ("Trn. memory @ 4096",            "trn_mem_4096","trn_mem_4096"),
    ]
    
    for label, mkey, tkey in rows:
        m = summary.get("mamba", {}).get(mkey, "—")
        t = summary.get("transformer", {}).get(tkey, "—")
        print(f"{label:<32} {str(m):>12} {str(t):>14}")
    
    print("=" * 62)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Training curves
    mamba_log_path = os.path.join(args.log_dir, "mamba_log.json")
    trans_log_path = os.path.join(args.log_dir, "transformer_log.json")
    if os.path.exists(mamba_log_path) and os.path.exists(trans_log_path):
        mamba_log = load_json(mamba_log_path)
        trans_log = load_json(trans_log_path)
        plot_training_loss(mamba_log, trans_log, args.out_dir)
        plot_val_perplexity(mamba_log, trans_log, args.out_dir)
    else:
        print(f"Training logs not found in {args.log_dir}. Skipping loss/ppl plots.")

    # Benchmark results
    bench_path = os.path.join(args.log_dir, "benchmark_results.json")
    if os.path.exists(bench_path):
        raw = load_json(bench_path)
        # handle both old (list) and new ({meta, results}) formats
        bench = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
        plot_throughput(bench, args.out_dir)
        plot_training_throughput(bench, args.out_dir)
        plot_memory(bench, args.out_dir)
        plot_speedup(bench, args.out_dir)
    else:
        print(f"Benchmark results not found at {bench_path}. Skipping throughput/memory plots.")
    
    print_slide_summary(args.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="out")
    parser.add_argument("--out_dir", type=str, default="out")
    main(parser.parse_args())
