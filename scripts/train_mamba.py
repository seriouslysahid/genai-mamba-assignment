"""
Train Mamba-130M and a GPT-2-style Transformer baseline on a subset
of The Pile. Academic reproduction of Gu & Dao (2023).

Usage:
    python scripts/train_mamba.py --model mamba
    python scripts/train_mamba.py --model transformer
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

DEFAULTS = dict(
    seq_len=1024,
    batch_size=8,
    grad_accum_steps=4,
    lr=6e-4,
    weight_decay=0.1,
    warmup_steps=500,
    max_steps=10_000,
    eval_interval=500,
    eval_steps=50,
    log_interval=50,
    dtype="bfloat16",
    seed=42,
    data_dir="data",
    out_dir="out",
)

# Mamba-130M (paper: 24 layers, d_model=768, d_state=16)
MAMBA_CFG = dict(
    d_model=768,
    n_layer=24,
    vocab_size=50277,
    d_intermediate=0,
    ssm_cfg=dict(d_state=16),
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=8,
    tie_embeddings=True,
)

# GPT-2 baseline (~130M params)
# n_positions is set dynamically based on --seq_len
TRANSFORMER_CFG = dict(
    vocab_size=50277,
    n_embd=768,
    n_layer=12,
    n_head=12,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)


class PileShardDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset over pre-tokenized .bin files."""

    def __init__(self, data_dir, split, seq_len):
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run prepare_data.py first."
            )
        self.data = torch.from_numpy(
            np.memmap(path, dtype="uint16", mode="r")
        ).long()
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def build_mamba():
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    return MambaLMHeadModel(MambaConfig(**MAMBA_CFG))


def build_transformer(seq_len=1024):
    cfg = {**TRANSFORMER_CFG, "n_positions": seq_len}
    return GPT2LMHeadModel(GPT2Config(**cfg))


def get_lr(step, warmup, max_steps, base_lr):
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, loader, eval_steps, device):
    model.eval()
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x).logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(args):
    cfg = {**DEFAULTS, **vars(args)}
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}[cfg["dtype"]]

    is_mamba = cfg["model"] == "mamba"

    print(f"Building {cfg['model']} model (seq_len={cfg['seq_len']})...")
    model = build_mamba() if is_mamba else build_transformer(cfg["seq_len"])
    model = model.to(device=device, dtype=pt_dtype)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.1f}M")

    train_ds = PileShardDataset(cfg["data_dir"], "train", cfg["seq_len"])
    val_ds = PileShardDataset(cfg["data_dir"], "val", cfg["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["dtype"] == "float16"))

    log = {"train_loss": [], "val_loss": [], "step": [], "throughput": [],
           "config": {"model": cfg["model"], "seq_len": cfg["seq_len"],
                      "batch_size": cfg["batch_size"], "dtype": cfg["dtype"]}}
    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()

    for step in range(cfg["max_steps"]):
        lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg["grad_accum_steps"]):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=pt_dtype):
                logits = model(x).logits
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss = loss / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            tokens_per_sec = (cfg["batch_size"] * cfg["grad_accum_steps"]
                              * cfg["seq_len"] * cfg["log_interval"]) / max(dt, 1e-9)
            print(f"step {step:>6d} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
            log["train_loss"].append(accum_loss)
            log["step"].append(step)
            log["throughput"].append(tokens_per_sec)
            t0 = time.time()

        if step > 0 and step % cfg["eval_interval"] == 0:
            val_loss = estimate_loss(model, val_loader, cfg["eval_steps"], device)
            ppl = math.exp(min(val_loss, 20))
            print(f"  → val loss {val_loss:.4f} | perplexity {ppl:.2f}")
            log["val_loss"].append({"step": step, "loss": val_loss, "ppl": ppl})

    # Save checkpoint and log
    tag = cfg["model"]
    ckpt_path = os.path.join(cfg["out_dir"], f"{tag}_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    log_path = os.path.join(cfg["out_dir"], f"{tag}_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved checkpoint → {ckpt_path}")
    print(f"Saved log → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mamba", "transformer"], required=True)
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--out_dir", type=str, default=DEFAULTS["out_dir"])
    parser.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"])
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--dtype", type=str, default=DEFAULTS["dtype"])
    train(parser.parse_args())
