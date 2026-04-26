# Mamba: Paper Study & Reproduction

Academic reproduction of [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023) for a Generative AI course project.

---

# Objective

The goal of this project is to reproduce the core claims of the Mamba paper at a smaller, experimentally tractable scale:

- Train a Mamba-130M language model on a subset of The Pile
- Train a same-size GPT-2-style Transformer baseline on the same data
- Compare validation quality, generation quality, throughput, and memory usage
- Benchmark scaling behavior across long sequence lengths
- Analyze whether Mamba retains its efficiency advantage in practice

The final report and experimental analysis are summarized in `RESULTS.md`, while the presentation deck is assembled from `slides.md`.

---

# Final Results at a Glance

## Model Quality

| Model | Parameters | Validation Loss | Perplexity | Bits/Byte |
|---|---:|---:|---:|---:|
| Mamba-130M | 129.1M | 3.4404 | 31.20 | 4.9634 |
| Transformer-130M | 125.2M | 3.7236 | 41.41 | 5.3720 |

## Long-Context Benchmark at 4096 Tokens

| Metric | Mamba | Transformer |
|---|---:|---:|
| Inference Throughput (tok/s) | 387,062.4 | 72,327.2 |
| Training Throughput (tok/s) | 92,768.8 | 29,711.1 |
| Inference Peak Memory (MB) | 4,201.5 | 9,228.6 |
| Training Peak Memory (MB) | 12,436.6 | 33,790.2 |

## Key Takeaways

- Mamba converges faster during training
- Mamba achieves lower validation loss and perplexity
- Mamba scales significantly better at long sequence lengths
- Transformer throughput and memory usage degrade more sharply as context length grows
- The strongest gap appears at 4096 tokens, where Mamba is dramatically more efficient

---

# Repository Structure

```text
.
├── README.md
├── RESULTS.md
├── slides.md
├── paper.pdf
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── build_mamba_deps.py
│   ├── prepare_data.py
│   ├── train_mamba.py
│   ├── evaluate_mamba.py
│   ├── benchmark_mamba.py
│   └── plot_results.py
├── data/                    # Generated token shards
└── out/                     # Generated checkpoints, logs, plots
```

---

# Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | `monology/pile-uncopyrighted` |
| Tokenizer | `EleutherAI/gpt-neox-20b` |
| Hardware | NVIDIA A100-SXM4-80GB |
| Precision | BF16 |
| Training Sequence Length | 2048 |
| Benchmark Sequence Lengths | 512, 1024, 2048, 4096 |
| Mamba Parameters | 129.1M |
| Transformer Parameters | 125.2M |

The tokenized dataset shard produced by `prepare_data.py` contains:

- `data/train.bin` → 427,434,311 tokens
- `data/val.bin` → 6,573,390 tokens

The binary shards are stored as `uint16` memory-mapped arrays for efficient loading during training and evaluation.

---

# Environment Setup

The final runs were executed in a Linux GPU environment using a PyTorch 2.6 + CUDA 12.6 stack and prebuilt Mamba wheels matching the active CXX11 ABI.

## Recommended Installation Path

```bash
# Step 1: install CUDA-enabled PyTorch stack
pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 -q

# Step 2: detect ABI
python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')"

# Step 3: install causal-conv1d wheel
pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Step 4: install mamba-ssm wheel
pip install --no-deps \
  https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Step 5: pin transformers version
pip install "transformers==4.39.3" -q
```

If a source build is preferred, the repository also includes:

```bash
python scripts/build_mamba_deps.py
```

---

# Execution

Run all commands from the repository root.

## 1. Prepare Data

```bash
python scripts/prepare_data.py
```

This streams a subset of The Pile, tokenizes it with GPT-NeoX, and writes:

- `data/train.bin`
- `data/val.bin`

---

## 2. Train Mamba

```bash
python scripts/train_mamba.py --model mamba
```

---

## 3. Train Transformer Baseline

The Transformer baseline was retrained with a longer budget to make the comparison more defensible under the available hardware constraints.

```bash
python scripts/train_mamba.py --model transformer --batch_size 4 --max_steps 20000
```

The training script internally uses gradient accumulation (`grad_accum_steps=2`), so the comparison should be interpreted using effective token budget rather than raw physical batch size alone.

---

## 4. Evaluate Both Checkpoints

```bash
python scripts/evaluate_mamba.py --model mamba --ckpt out/mamba_final.pt

python scripts/evaluate_mamba.py --model transformer --ckpt out/transformer_final.pt
```

---

## 5. Run Throughput and Memory Benchmarks

```bash
python scripts/benchmark_mamba.py
```

---

## 6. Generate Plots and Summary Tables

```bash
python scripts/plot_results.py
```

This generates:

- training curves
- throughput plots
- memory scaling plots
- speedup plots
- saved summary table image

---

# Training Dynamics

The final training curves show a clear optimization gap between the two architectures.

| Model | Training Steps | Final Train Loss | Final Training-Run Val Loss |
|---|---:|---:|---:|
| Mamba | 5,000 | 3.54 | 3.28 |
| Transformer | 20,000 | 3.94 | 3.80 |

## Main Observations

- Mamba converges faster and more smoothly
- Transformer training is noisier and requires more steps
- Mamba reaches a strong validation regime much earlier

---

# Validation and Generation Quality

Final evaluation metrics:

| Metric | Mamba | Transformer |
|---|---:|---:|
| Validation Loss | 3.4404 | 3.7236 |
| Perplexity | 31.20 | 41.41 |
| Bits per Byte | 4.9634 | 5.3720 |

## Qualitative Observations

- Mamba generations are generally more coherent
- Transformer outputs are more repetitive
- Mamba preserves topic structure slightly better
- Neither model is production-grade at this scale, which is expected

---

# Throughput Analysis

## Inference Throughput

| Sequence Length | Mamba (tok/s) | Transformer (tok/s) | Speedup |
|---|---:|---:|---:|
| 512 | 168,054.0 | 217,505.8 | 0.77× |
| 1024 | 335,953.8 | 179,759.3 | 1.87× |
| 2048 | 379,297.9 | 102,904.6 | 3.69× |
| 4096 | 387,062.4 | 72,327.2 | 5.35× |

## Training Throughput

| Sequence Length | Mamba (tok/s) | Transformer (tok/s) | Speedup |
|---|---:|---:|---:|
| 512 | 25,644.5 | 57,350.9 | 0.45× |
| 1024 | 49,292.4 | 58,510.5 | 0.84× |
| 2048 | 88,047.3 | 42,795.4 | 2.06× |
| 4096 | 92,768.8 | 29,711.1 | 3.12× |

## Key Insight

- Transformer is competitive at short sequence lengths
- Mamba dominates once sequence length increases
- Long-context scaling is where Mamba shows its strongest advantage

---

# Memory Usage Analysis

## Inference Peak Memory

| Sequence Length | Mamba (MB) | Transformer (MB) | Reduction |
|---|---:|---:|---:|
| 512 | 653.6 | 1,073.7 | 39.1% |
| 1024 | 1,816.5 | 2,434.1 | 25.4% |
| 2048 | 2,606.9 | 3,682.5 | 29.2% |
| 4096 | 4,201.5 | 9,228.6 | 54.5% |

## Training Peak Memory

| Sequence Length | Mamba (MB) | Transformer (MB) | Reduction |
|---|---:|---:|---:|
| 512 | 2,235.0 | 3,047.5 | 26.7% |
| 1024 | 3,688.4 | 5,710.1 | 35.4% |
| 2048 | 6,613.1 | 12,768.1 | 48.2% |
| 4096 | 12,436.6 | 33,790.2 | 63.2% |

At 4096 tokens, Transformer training memory grows dramatically, while Mamba remains far more manageable.

---

# Results Summary

The final experiments support the central claim of the Mamba paper:

- Mamba is substantially more efficient at long sequence lengths
- Mamba converges faster
- Mamba achieves better validation quality
- Mamba uses significantly less GPU memory
- Mamba scales much better for long-context inference

---

# Outputs in `out/`

After running the pipeline, `out/` contains:

- `mamba_final.pt`
- `transformer_final.pt`
- `mamba_log.json`
- `transformer_log.json`
- `mamba_eval.json`
- `transformer_eval.json`
- `benchmark_results.json`
- `training_loss.png`
- `val_perplexity.png`
- `inference_throughput.png`
- `training_throughput.png`
- `memory_scaling.png`
- `speedup.png`
- `final_summary_table.png`

---

# Benchmark Methodology

- Throughput measured using CUDA synchronization and timed forward passes
- Peak memory measured after warmup iterations
- Sequence lengths benchmarked: 512, 1024, 2048, 4096
- Same tokenizer and vocabulary used for both models
- Same dataset shard used for training and evaluation

---

# Limitations

- Small-scale reproduction, not full-scale pretraining
- Single-run experiments without multiple seeds
- Transformer constrained by memory at long sequence lengths
- Only language modeling evaluated
- Models are tiny compared to modern frontier LLMs

---

# Conclusion

This reproduction validates the central empirical claims of the Mamba architecture:

- Linear-time scaling provides major practical benefits
- Mamba achieves significantly higher throughput at long sequence lengths
- Mamba uses dramatically less memory than Transformers
- Mamba achieves better validation quality in this setup

The experiments demonstrate that selective state space models are a strong long-context alternative to attention-heavy Transformer architectures.

---

# References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*  
  https://arxiv.org/abs/2312.00752

- Official implementation:  
  https://github.com/state-spaces/mamba

- Experimental analysis and benchmark discussion:  
  `RESULTS.md`