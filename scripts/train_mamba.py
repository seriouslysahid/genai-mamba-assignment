# train mamba and transformer
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from transformers import GPT2Config, GPT2LMHeadModel

DEFAULTS = {"batch_size": 16, "seq_len": 512, "grad_accum": 4, "max_steps": 2500, "eval_interval": 200, "eval_iters": 50, "lr": 6e-4, "min_lr": 6e-5, "warmup_steps": 100}

MAMBA_CFG = {"d_model": 768, "n_layer": 24, "vocab_size": 50277, "ssm_cfg": {"d_state": 16, "expand": 2}}

TRANSFORMER_CFG = {"n_embd": 768, "n_layer": 12, "n_head": 12, "vocab_size": 50277, "resid_pdrop": 0.1, "embd_pdrop": 0.1, "attn_pdrop": 0.1, "use_cache": False}

def build_mamba():
    return MambaLMHeadModel(MambaConfig(**MAMBA_CFG))

def build_transformer(seq_len):
    cfg = GPT2Config(**TRANSFORMER_CFG)
    cfg.n_positions = seq_len
    return GPT2LMHeadModel(cfg)

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+seq_len+1]).astype(np.int64)) for i in ix])
    return x.cuda(), y.cuda()

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, seq_len, eval_iters):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, seq_len)
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[k] = loss.item()
        avg_loss = losses.mean().item()
        out[split] = {"loss": avg_loss, "ppl": math.exp(avg_loss)}
    model.train()
    return out

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(model, model_name, train_data, val_data, args, pt_dtype):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    logs = {"train_loss": [], "val_loss": [], "step": []}
    
    print(f"\ntraining {model_name}...")
    t0 = time.time()
    
    for step in tqdm(range(args.max_steps)):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            res = estimate_loss(model, train_data, val_data, args.batch_size, args.seq_len, args.eval_iters)
            print(f"step {step}: train loss {res['train']['loss']:.4f}, val loss {res['val']['loss']:.4f}")
            
            logs["step"].append(step)
            logs["train_loss"].append(res["train"]["loss"])
            logs["val_loss"].append({"loss": res["val"]["loss"], "ppl": res["val"]["ppl"]})
            
            with open(os.path.join(args.out_dir, f"{model_name}_log.json"), "w") as f:
                json.dump(logs, f)
                
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{model_name}_model.pt"))

        model.train()
        for micro_step in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.seq_len)
            with torch.amp.autocast(device_type="cuda", dtype=pt_dtype):
                logits = model(x).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / args.grad_accum
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    dt = time.time() - t0
    print(f"done in {dt/60:.2f} mins")

def main(args):
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    pt_dtype = torch.bfloat16

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model in ["mamba", "both"]:
        mamba = build_mamba().cuda().to(pt_dtype)
        train(mamba, "mamba", train_data, val_data, args, pt_dtype)
        del mamba
        torch.cuda.empty_cache()

    if args.model in ["transformer", "both"]:
        transformer = build_transformer(args.seq_len).cuda().to(pt_dtype)
        train(transformer, "transformer", train_data, val_data, args, pt_dtype)
        del transformer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mamba", "transformer", "both"], default="both")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--eval_interval", type=int, default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval_iters", type=int, default=DEFAULTS["eval_iters"])
    
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--min_lr", type=float, default=DEFAULTS["min_lr"])
    parser.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
    
    args = parser.parse_args()
    main(args)
