# Mamba: Paper Study & Reproduction

Academic reproduction of [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023) for a Generative AI course project.

## Objective

Reproduce the core claims of the original Mamba paper at small scale:
- Train a Mamba-130M language model on a subset of The Pile
- Compare against a same-size GPT-2-style Transformer baseline
- Measure perplexity, inference throughput, and GPU memory usage
- Benchmark scaling behavior across sequence lengths (512–8192)
- Validate that Mamba achieves competitive quality with better efficiency

All experiments run on a Lightning.ai instance with an NVIDIA A100 GPU (40GB VRAM, 312 BF16 TFLOPs) and 30 CPUs, using BF16 precision.

## Repository Structure

```
├── slides.md                # Presentation deck
├── paper.pdf                # Original Mamba paper
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   ├── prepare_data.py      # Download & tokenize Pile subset
│   ├── train_mamba.py       # Train Mamba or Transformer baseline
│   ├── evaluate_mamba.py    # Held-out perplexity & text generation
│   ├── benchmark_mamba.py   # Throughput & memory scaling benchmarks
│   └── plot_results.py      # Generate plots for the presentation
├── data/                    # Tokenized data (generated)
└── out/                     # Checkpoints, logs, plots (generated)
```

## Scripts

| Script | Purpose |
|---|---|
| `prepare_data.py` | Streams a subset of The Pile, tokenizes with GPT-NeoX tokenizer, writes `data/train.bin` and `data/val.bin` |
| `train_mamba.py` | Trains Mamba-130M or a GPT-2-style Transformer. Logs loss, perplexity, and throughput to JSON |
| `evaluate_mamba.py` | Loads a trained checkpoint, computes validation perplexity, generates text samples |
| `benchmark_mamba.py` | Sweeps sequence lengths (256–8192), measures inference throughput and peak GPU memory |
| `plot_results.py` | Produces training loss, perplexity, throughput, memory, and speedup ratio plots |

## Google Colab workflow

For the current course/demo workflow, use the single Colab notebook:

```
Mamba_Colab_Workflow.ipynb
```

The notebook keeps the original script pipeline intact, but orchestrates it end to
end in one place:

1. environment check and dependency installation
2. `mamba-ssm` / `causal-conv1d` installation from official prebuilt GitHub
   wheels for the pinned Colab `2025.07` runtime
3. streaming data preparation from `monology/pile-uncopyrighted`
4. Mamba-130M training
5. Transformer baseline training
6. validation perplexity and text generation
7. throughput and memory benchmarking
8. plot generation into `out/`

Colab defaults are intentionally smaller than the original A100 run
(`seq_len=512`, `batch_size=1`, short training, smaller streamed shard). The
model definitions, tokenizer, binary token shard format, evaluation intent, and
benchmark methodology remain the same.

Recommended install strategy in Colab:

- pin Colab to Runtime Version `2025.07`
- keep Colab's preinstalled Python 3.11 / PyTorch 2.6 / CUDA 12 stack
- install `requirements.txt`
- install official prebuilt GitHub release wheels for `causal-conv1d` and
  `mamba-ssm`
- choose the correct CXX11 ABI wheel dynamically from
  `torch._C._GLIBCXX_USE_CXX11_ABI`

The dependency cell uninstalls old `mamba-ssm` / `causal-conv1d` wheels before
reinstalling, because stale wheels can be linked against the wrong CUDA runtime.
It also runs a small Mamba model-construction smoke test before training starts.

Default data scale is `25_000` train examples and `2_000` validation examples,
roughly five times the original smoke-test shard. On a stable Colab Pro/L4
runtime, `50_000` / `5_000` is a reasonable next step. `100_000` / `10_000` is a
stretch setting for A100-class sessions. The full dataset is not Colab-practical:
Hugging Face lists `monology/pile-uncopyrighted` at about 335 GB raw and roughly
176M rows.

The Mamba project documents the same core requirement: install PyTorch first,
then install Mamba with `--no-build-isolation` so the build uses the active
CUDA-enabled PyTorch environment.

## Local/A100 script setup

For non-Colab script runs, Python 3.10 remains the safest target for stable
`mamba-ssm` builds. Colab currently controls the notebook Python runtime, so the
notebook instead installs against Colab's active PyTorch/CUDA stack and verifies
the Mamba CUDA extension before training.

Requires Linux, NVIDIA GPU (A100 recommended for BF16), and CUDA 11.6+.

```bash
# 1. Create and activate a fresh Python 3.10 environment
conda create -n mamba310 python=3.10 -y
conda activate mamba310

# 2. Install PyTorch (adjust to match your CUDA version)
pip install torch torchvision torchaudio

# 3. Install build basics and Mamba dependencies directly
pip install ninja
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

# 4. Install remaining project requirements
pip install -r requirements.txt
```

To verify your environment is correctly configured for the A100 and Mamba, run:
```bash
python scripts/check_env.py
```

## Execution

Run all commands from the repository root:

```bash
# 1. Prepare data (~10 min)
python scripts/prepare_data.py

# 2. Train Mamba (BF16 on A100)
python scripts/train_mamba.py --model mamba

# 3. Train Transformer baseline
python scripts/train_mamba.py --model transformer

# 4. Evaluate both
python scripts/evaluate_mamba.py --model mamba --ckpt out/mamba_final.pt
python scripts/evaluate_mamba.py --model transformer --ckpt out/transformer_final.pt

# 5. Benchmark throughput & memory across sequence lengths
python scripts/benchmark_mamba.py

# 6. Generate all plots
python scripts/plot_results.py
```

### Optional: longer sequence training

```bash
python scripts/train_mamba.py --model mamba --seq_len 2048
python scripts/train_mamba.py --model transformer --seq_len 2048
```

## Expected Outputs

After running the full pipeline, `out/` will contain:

- `mamba_final.pt`, `transformer_final.pt` — trained checkpoints
- `mamba_log.json`, `transformer_log.json` — training logs (with config metadata)
- `mamba_eval.json`, `transformer_eval.json` — evaluation results
- `benchmark_results.json` — throughput and memory across sequence lengths
- `training_loss.png`, `val_perplexity.png` — training curves
- `throughput.png`, `memory.png` — efficiency vs sequence length
- `speedup.png` — Mamba/Transformer throughput ratio per sequence length

## Model Configurations

| | Mamba | Transformer |
|---|---|---|
| Parameters | ~130M | ~130M |
| Layers | 24 | 12 |
| Hidden dim | 768 | 768 |
| SSM state dim | 16 | — |
| Attention heads | — | 12 |
| Precision | BF16 | BF16 |

Mamba config follows the paper (24 layers, d_model=768, d_state=16). The Transformer uses a GPT-2 architecture with comparable parameter count. Both are trained and benchmarked in BF16 on an A100 40GB GPU.

## Benchmark Methodology

- **Throughput:** Forward-pass tokens/sec with CUDA synchronization, measured after warmup
- **Memory:** Peak GPU allocation during timed iterations (reset after warmup)
- **Sequence lengths:** 256, 512, 1024, 2048, 4096, 8192
- **Fairness:** Same tokenizer, same vocab, same approximate parameter count, same dtype

## Limitations

- This is a small-scale reproduction, not a full replication of the paper's 300B-token runs.
- Results are directionally indicative, not publication-grade.
- `mamba-ssm` requires Linux + CUDA; the scripts will not run on CPU or macOS.
- The Transformer baseline uses fixed positional embeddings, which limits its max sequence length.

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Dao, T. & Gu, A. (2024). *Transformers are SSMs.* [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- Official implementation: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
