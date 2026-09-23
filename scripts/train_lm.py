#!/usr/bin/env python3
"""
The extrinsic signal. One small GPT per tokenizer variant, same text,
same model, same schedule; report bits per byte and bits per character of
the FineWeb-2 npi_Deva test split. Cross-tokenizer perplexity is meaningless
(different vocabularies), bits per byte of the same raw text is not.

Protocol: every variant trains one epoch over the same --docs documents of
corpus/train_lm.txt (identical text, so the variants differ only in how it is
cut; a variant with higher fertility takes more optimizer steps -- both are
recorded). Evaluation is the total negative log-likelihood of the first
--test-docs documents of corpus/test.txt, the same documents fertility.jsonl
uses, in non-overlapping context windows, divided by the UTF-8 bytes /
code points of that raw text.

    python3 scripts/train_lm.py --variant fst --vocab 16000 --docs 50000
Appends one JSON line to results/lm_bpc.jsonl. The paper's protocol:
--docs 400000 --batch 128 --lr 1.2e-3 (results/lm_bpc_server.jsonl, 3 seeds);
--save keeps the weights for finetune_tok.py; the matched-steps control is
--epochs = fst tokens / fixed tokens: 1.116 at 8k, 1.164 at 16k, 1.21 at 32k.
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import common
from common import ROOT

SCRATCH = os.environ.get("LM_SCRATCH", f"{ROOT}/corpus/lm_cache")
EOT = "<|endoftext|>"


# ---- model: nanoGPT-sized -------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False, is_causal=True)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, ctx, n_layer, n_head, n_embd):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(ctx, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))
        for m in self.modules():        # GPT-2 init; the default N(0,1) embedding is tied to the head
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        return self.ln(x)

    def forward(self, idx):
        return self.head(self.hidden(idx))


# ---- data ----------------------------------------------------------------------
def encode(variant, vocab, path, n, tag):
    """token ids of the first n docs of path, EOT-joined, cached as uint16 (uint32 past 65k vocab)"""
    dt = np.uint16 if vocab < 65535 else np.uint32
    out = f"{SCRATCH}/{tag}_{variant}_{vocab}_{n}.bin"
    if os.path.exists(out):
        return np.memmap(out, dtype=dt, mode="r")
    tok, tier, eot, _ = common.load_tokenizer(variant, vocab)
    docs = lambda: itertools.islice(open(path, encoding="utf-8"), n)
    if hasattr(tier, "prime"):      # one flookup pass over every word type
        tier.prime(docs())
    os.makedirs(SCRATCH, exist_ok=True)
    # streamed twice over the file: 400k docs are 3 GB as Python strings, and a
    # 135M-token list of ints is 4 GB
    with open(out + ".part", "wb") as f:
        it = docs()
        while chunk := list(itertools.islice(it, 2000)):
            marked = [tier.mark(d) if tier else d for d in chunk]
            ids = [x for e in tok.encode_batch(marked) for x in (*e.ids, eot)]
            arr = np.array(ids, dtype=dt)
            assert arr.max() < vocab
            arr.tofile(f)
    os.rename(out + ".part", out)
    return np.memmap(out, dtype=dt, mode="r")


def batches(data, ctx, bs, steps, seed):
    g = np.random.default_rng(seed)
    for _ in range(steps):
        ix = g.integers(0, len(data) - ctx - 1, bs)
        x = np.stack([data[i:i + ctx] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + ctx + 1] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def nll(model, data, ctx, bs, dev):
    """total NLL in nats of data[1:] given data[:-1], non-overlapping windows"""
    model.eval()
    total, n = 0.0, 0
    T = (len(data) - 1) // ctx * ctx
    for i in range(0, T, ctx * bs):
        x = torch.from_numpy(np.stack([data[j:j + ctx] for j in range(i, min(i + ctx * bs, T), ctx)]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([data[j + 1:j + ctx + 1] for j in range(i, min(i + ctx * bs, T), ctx)]).astype(np.int64)).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(x)
        total += F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="sum").item()
        n += y.numel()
    model.train()
    return total, n


def selfcheck():
    """an untrained model must score ~log2(V) bits per token (catches a bad init
    or a broken tied head) and a trained-on-repetition one must beat it"""
    torch.manual_seed(0)
    V, ctx = 64, 16
    m = GPT(V, ctx, 1, 2, 32)
    data = np.tile(np.arange(V, dtype=np.uint16), 40)
    t, n = nll(m, data, ctx, 8, "cpu")
    assert abs(t / n / math.log(2) - math.log2(V)) < 0.3, t / n / math.log(2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for x, y in batches(data, ctx, 8, 150, 0):
        loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    t2, _ = nll(m, data, ctx, 8, "cpu")
    assert t2 < t / 4, (t, t2)
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--docs", type=int, default=50_000)
    ap.add_argument("--test-docs", type=int, default=5000)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=f"{ROOT}/results/lm_bpc.jsonl")
    ap.add_argument("--encode-only", action="store_true", help="build the token caches and exit")
    ap.add_argument("--save", action="store_true", help="keep the weights for finetune_tok.py")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation: batch is split in this many micro-batches (big vocabs)")
    ap.add_argument("--tag", default="", help="checkpoint name suffix, e.g. matched for the equal-steps baseline, so it "
                                             "does not overwrite the 1-epoch weights of the same variant")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    a.vocab = common.load_tokenizer(a.variant, a.vocab)[3]   # external tokenizers: their real size

    train = encode(a.variant, a.vocab, f"{ROOT}/corpus/train_lm.txt", a.docs, "train")
    test = encode(a.variant, a.vocab, f"{ROOT}/corpus/test.txt", a.test_docs, "test")
    if a.encode_only:
        return print(f"{a.variant}: {len(train):,} train tokens, {len(test):,} test tokens cached")
    raw = list(itertools.islice(open(f"{ROOT}/corpus/test.txt", encoding="utf-8"), a.test_docs))
    test_bytes, test_chars = sum(len(l.encode()) for l in raw), sum(len(l) for l in raw)
    steps = int(a.epochs * len(train) / (a.ctx * a.batch))
    print(f"{a.variant}: {len(train):,} train tokens, {len(test):,} test tokens, {steps} steps on {dev}")

    model = GPT(a.vocab, a.ctx, a.n_layer, a.n_head, a.n_embd).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, min(200, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    t0, loss_ema = time.time(), None
    for step, (x, y) in enumerate(batches(train, a.ctx, a.batch, steps, a.seed)):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        for xm, ym in zip(x.chunk(a.accum), y.chunk(a.accum)):
            with torch.autocast(dev, dtype=torch.bfloat16):
                logits = model(xm)
            loss = F.cross_entropy(logits.float().view(-1, a.vocab), ym.view(-1)) / a.accum
            loss.backward()
        loss = loss * a.accum          # the last micro-batch's loss, for the log
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if step % 200 == 0:
            print(f"  step {step}/{steps} loss {loss_ema:.3f} {time.time() - t0:.0f}s", flush=True)
    total, ntok = nll(model, test, a.ctx, a.batch // a.accum, dev)
    bits = total / math.log(2)
    rec = {"variant": a.variant, "vocab": a.vocab, "train_docs": a.docs, "train_tokens": len(train),
           "epochs": a.epochs, "steps": steps, "ctx": a.ctx, "batch": a.batch, "params": nparams,
           "n_layer": a.n_layer, "n_embd": a.n_embd, "test_docs": a.test_docs, "test_tokens": ntok,
           "test_bytes": test_bytes, "test_chars": test_chars, "loss_per_token": round(total / ntok, 4),
           "bits_per_byte": round(bits / test_bytes, 4), "bits_per_char": round(bits / test_chars, 4),
           "train_seconds": round(time.time() - t0), "seed": a.seed, "lr": a.lr,
           "host": os.uname().nodename}
    if a.tag:
        rec["tag"] = a.tag
    if a.save:
        os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)
        rec["checkpoint"] = f"checkpoints/{a.variant}{'_' + a.tag if a.tag else ''}_{a.vocab}_{a.docs}_s{a.seed}.pt"
        torch.save({"args": vars(a), "state": model.state_dict()}, f"{ROOT}/{rec['checkpoint']}")
    print(json.dumps(rec, ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
