# evaluate mamba model
import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from train_mamba import build_mamba, build_transformer, DEFAULTS, MAMBA_CFG, TRANSFORMER_CFG

@torch.no_grad()
def eval_ppl(model, data_arr, ctx_len, batch_size, max_batches):
    model.eval()
    tokens = len(data_arr)
    n_batches = min(max_batches, (tokens - 1) // (batch_size * ctx_len))
    if n_batches == 0: return 0.0

    total_loss = 0.0
    for i in tqdm(range(n_batches), desc="eval ppl"):
        start = i * batch_size * ctx_len
        end = start + batch_size * ctx_len + 1
        
        chunk = torch.from_numpy(data_arr[start:end].astype(np.int64)).cuda()
        
        x = chunk[:-1].view(batch_size, ctx_len)
        y = chunk[1:].view(batch_size, ctx_len)

        logits = model(x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    avg_loss = total_loss / n_batches
    return math.exp(avg_loss), avg_loss / math.log(2)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_gen_len, temperature=1.0, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device="cuda")

    for _ in range(max_gen_len):
        logits = model(x).logits[:, -1, :]
        logits = logits / temperature
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

    return tokenizer.decode(x[0].tolist())

def main(args):
    import numpy as np
    from transformers import AutoTokenizer
    
    val_path = os.path.join(args.data_dir, "val.bin")
    if not os.path.exists(val_path):
        print("no val.bin found")
        return
        
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    
    dtype = torch.bfloat16
    
    print(f"loading {args.model}...")
    if args.model == "mamba":
        model = build_mamba().cuda().to(dtype)
    else:
        model = build_transformer(args.seq_len).cuda().to(dtype)

    ckpt_path = os.path.join(args.ckpt, f"{args.model}_model.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("warning: no ckpt found, using random weights")

    print("running ppl eval...")
    ppl, bpb = eval_ppl(model, val_data, args.seq_len, args.batch_size, args.max_batches)
    print(f"PPL: {ppl:.2f}, BPB: {bpb:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    res = {"perplexity": ppl, "bpb": bpb}
    with open(os.path.join(args.out_dir, f"{args.model}_eval.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\ngenerating text...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    prompts = ["The future of artificial intelligence is", "In a shocking turn of events,"]
    
    for p in prompts:
        gen = generate(model, tokenizer, p, max_gen_len=100)
        print(f"\nprompt: {p}")
        print(f"gen: {gen}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mamba", "transformer"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    
    args = parser.parse_args()
    main(args)