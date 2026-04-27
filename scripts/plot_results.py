# plot mamba results
import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

LOG_DIR = "out"
OUT_DIR = "out"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_training_loss(log_dir, out_dir):
    print("plotting training loss...")
    mamba = load_json(os.path.join(log_dir, "mamba_log.json"))
    transformer = load_json(os.path.join(log_dir, "transformer_log.json"))

    plt.figure(figsize=(10, 6))
    plt.plot(mamba["step"], mamba["train_loss"], label="Mamba", linewidth=2)
    plt.plot(transformer["step"], transformer["train_loss"], label="Transformer", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=220, bbox_inches="tight")
    plt.close()

def plot_val_perplexity(log_dir, out_dir):
    print("plotting validation perplexity...")
    mamba = load_json(os.path.join(log_dir, "mamba_log.json"))
    transformer = load_json(os.path.join(log_dir, "transformer_log.json"))

    m_steps, m_ppl = [x["step"] for x in mamba["val_loss"]], [x["ppl"] for x in mamba["val_loss"]]
    t_steps, t_ppl = [x["step"] for x in transformer["val_loss"]], [x["ppl"] for x in transformer["val_loss"]]

    plt.figure(figsize=(10, 6))
    plt.plot(m_steps, m_ppl, marker="o", linewidth=2, label="Mamba")
    plt.plot(t_steps, t_ppl, marker="o", linewidth=2, label="Transformer")
    plt.xlabel("Training Step")
    plt.ylabel("Validation Perplexity")
    plt.title("Validation Perplexity Comparison")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "val_perplexity.png"), dpi=220, bbox_inches="tight")
    plt.close()

def plot_benchmarks(log_dir, out_dir):
    bench_path = os.path.join(log_dir, "benchmark_results.json")
    if not os.path.exists(bench_path):
        print("benchmark_results.json not found.")
        return

    raw = load_json(bench_path)
    results = raw["results"] if isinstance(raw, dict) else raw

    mamba = [x for x in results if x["model"] == "mamba"]
    transformer = [x for x in results if x["model"] == "transformer"]
    seqs = [x["seq_len"] for x in mamba]

    print("plotting inference throughput...")
    plt.figure(figsize=(10, 6))
    plt.plot(seqs, [x["inference_tokens_per_sec"] for x in mamba], marker="o", linewidth=2, label="Mamba")
    plt.plot(seqs, [x["inference_tokens_per_sec"] for x in transformer], marker="o", linewidth=2, label="Transformer")
    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens / Second")
    plt.title("Inference Throughput")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "inference_throughput.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print("plotting training throughput...")
    plt.figure(figsize=(10, 6))
    plt.plot(seqs, [x["training_tokens_per_sec"] for x in mamba], marker="o", linewidth=2, label="Mamba")
    plt.plot(seqs, [x["training_tokens_per_sec"] for x in transformer], marker="o", linewidth=2, label="Transformer")
    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens / Second")
    plt.title("Training Throughput")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_throughput.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print("plotting memory usage...")
    plt.figure(figsize=(10, 6))
    plt.plot(seqs, [x["training_peak_memory_mb"] for x in mamba], marker="o", linewidth=2, label="Mamba")
    plt.plot(seqs, [x["training_peak_memory_mb"] for x in transformer], marker="o", linewidth=2, label="Transformer")
    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Training Memory (MB)")
    plt.title("Training Memory Scaling")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "memory_scaling.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print("plotting speedup...")
    speedups = [m["inference_tokens_per_sec"] / t["inference_tokens_per_sec"] for m, t in zip(mamba, transformer)]
    plt.figure(figsize=(10, 6))
    plt.plot(seqs, speedups, marker="o", linewidth=2)
    plt.xlabel("Sequence Length")
    plt.ylabel("Mamba Speedup Over Transformer")
    plt.title("Inference Speedup Scaling")
    plt.savefig(os.path.join(out_dir, "speedup.png"), dpi=220, bbox_inches="tight")
    plt.close()

def save_summary_table_image(summary, out_dir):
    print("saving summary table...")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    headers = ["Metric", "Mamba", "Transformer"]
    rows = [
        ["Perplexity", f"{summary['mamba'].get('perplexity', '—'):.2f}", f"{summary['transformer'].get('perplexity', '—'):.2f}"],
        ["Bits-per-byte", f"{summary['mamba'].get('bpb', '—'):.2f}", f"{summary['transformer'].get('bpb', '—'):.2f}"],
        ["Inference TPS @ 4096", f"{summary['mamba'].get('inf_4096', '—'):,.0f}", f"{summary['transformer'].get('inf_4096', '—'):,.0f}"],
        ["Training TPS @ 4096", f"{summary['mamba'].get('trn_4096', '—'):,.0f}", f"{summary['transformer'].get('trn_4096', '—'):,.0f}"],
        ["Training Memory @ 4096 (MB)", f"{summary['mamba'].get('mem_4096', '—'):,.1f}", f"{summary['transformer'].get('mem_4096', '—'):,.1f}"],
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for col in range(len(headers)):
        cell = table[(0, col)]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("navy")

    for row in range(1, len(rows) + 1):
        table[(row, 0)].set_facecolor("lightsteelblue")
        table[(row, 1)].set_facecolor("aliceblue")
        table[(row, 2)].set_facecolor("mistyrose")

    plt.title("Final Experiment Summary", fontsize=16, weight="bold", pad=18)
    fig.savefig(os.path.join(out_dir, "final_summary_table.png"), bbox_inches="tight", dpi=220)
    plt.close(fig)

def print_summary_table(log_dir, out_dir):
    print("\n--- FINAL SUMMARY ---")
    summary = {}

    for model in ["mamba", "transformer"]:
        path = os.path.join(log_dir, f"{model}_eval.json")
        if not os.path.exists(path):
            continue
        data = load_json(path)
        summary[model] = {"perplexity": data.get("perplexity"), "bpb": data.get("bpb")}

    bench_path = os.path.join(log_dir, "benchmark_results.json")
    if os.path.exists(bench_path):
        raw = load_json(bench_path)
        bench = raw["results"] if isinstance(raw, dict) else raw
        for row in bench:
            model = row["model"]
            seq = row["seq_len"]
            if model not in summary:
                summary[model] = {}
            if seq == 4096:
                summary[model]["inf_4096"] = row.get("inference_tokens_per_sec")
                summary[model]["trn_4096"] = row.get("training_tokens_per_sec")
                summary[model]["mem_4096"] = row.get("training_peak_memory_mb")

    print(f"{'Metric':<38}{'Mamba':>20}{'Transformer':>20}")
    print("-" * 80)
    
    rows = [
        ("Perplexity", "perplexity"),
        ("Bits-per-byte", "bpb"),
        ("Inference TPS @ 4096", "inf_4096"),
        ("Training TPS @ 4096", "trn_4096"),
        ("Training Memory @ 4096 MB", "mem_4096"),
    ]

    for label, key in rows:
        m = summary.get("mamba", {}).get(key, "—")
        t = summary.get("transformer", {}).get(key, "—")
        if isinstance(m, float): m = round(m, 2)
        if isinstance(t, float): t = round(t, 2)
        print(f"{label:<38}{str(m):>20}{str(t):>20}")

    try:
        m_inf = summary["mamba"]["inf_4096"]
        t_inf = summary["transformer"]["inf_4096"]
        m_mem = summary["mamba"]["mem_4096"]
        t_mem = summary["transformer"]["mem_4096"]

        print("\nOBSERVATIONS:")
        print(f"• Mamba inference is {m_inf/t_inf:.2f}x faster at seq=4096")
        print(f"• Transformer uses {t_mem/m_mem:.2f}x more memory at seq=4096\n")
    except Exception:
        pass

    save_summary_table_image(summary, out_dir)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_training_loss(LOG_DIR, OUT_DIR)
    plot_val_perplexity(LOG_DIR, OUT_DIR)
    plot_benchmarks(LOG_DIR, OUT_DIR)
    print_summary_table(LOG_DIR, OUT_DIR)