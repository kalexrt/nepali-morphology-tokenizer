#!/usr/bin/env python3
"""
Downstream signal beyond bits per byte. Token classification
(Nep-gLUE NER or POS) on top of the per-variant GPTs from train_lm.py --save,
matched budget: same data, same split, same schedule; the variants differ only
in the tokenizer and the pretrained weights that go with it.

The sentence is joined with spaces, marked by the variant's morphological tier
exactly as in pretraining, BPE-encoded, and each token is assigned to the word
its offsets end in. The model is causal, so the word's label is predicted from
its LAST sub-token (which has seen the whole word). Loss only on those
positions; padding is on the right, so causal attention needs no pad mask.

Split: sentences shuffled with seed 0, 80/10/10 train/dev/test (Nep-gLUE ships
one file per task). Best dev epoch's test score is reported.

Task cc is sequence classification on running text (mteb/NepaliNewsClassification,
3 topics, news paragraphs, not pre-split at word level): the document is marked,
encoded, truncated to ctx-1 and closed with EOT; the label is predicted from
that EOT position. Official test split; dev = 20% of train (seed 0).

    python3 scripts/finetune_tok.py --task ner --variant fst --seed 1 [--dump]
Appends one JSON line to results/downstream.jsonl. Needs the checkpoint from
train_lm.py --save and the task data in downstream/ (fetch_downstream.py).
"""
import argparse
import csv
import collections
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import common
from common import ROOT
from train_lm import GPT

FILES = {"ner": "downstream/NER.csv", "pos": "downstream/POS.csv", "cc": "downstream/CC_{}.tsv"}


def read(task):
    """[(words, labels)] per sentence_id, in file order"""
    sents = collections.OrderedDict()
    with open(f"{ROOT}/{FILES[task]}", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w, l = row["words"].strip().replace("﻿", ""), row["labels"].strip()
            if w and l:
                sents.setdefault(row["sentence_id"], ([], []))
                sents[row["sentence_id"]][0].append(w)
                sents[row["sentence_id"]][1].append(l)
    return list(sents.values())


def read_cc():
    """train, test: [(text, label)]"""
    out = []
    for part in ("train", "test"):
        rows = [l.rstrip("\n").split("\t", 1) for l in open(f"{ROOT}/{FILES['cc'].format(part)}", encoding="utf-8")]
        out.append([(t, l) for l, t in rows])
    return out


def encode_doc(tok, tier, text, ctx, pad):
    ids = tok.encode(tier.mark(text) if tier else text).ids[:ctx - 1] + [pad]
    return ids, [len(ids) - 1]


def split(sents):
    idx = list(range(len(sents)))
    random.Random(0).shuffle(idx)
    a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
    return [[sents[i] for i in idx[lo:hi]] for lo, hi in ((0, a), (a, b), (b, len(idx)))]


def encode_sent(tok, tier, words, ctx):
    """token ids of the space-joined sentence, and the position of each word's last token
    (-1 when the word fell past ctx)"""
    text = " ".join(words)
    marked = tier.mark(text) if tier else text
    spans, pos = [], 0
    for w in marked.split(" "):          # the tier never touches spaces, so words stay aligned
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    enc = tok.encode(marked)
    last = [-1] * len(words)
    wi = 0
    for ti, (s, e) in enumerate(enc.offsets):
        if e == 0 or ti >= ctx:
            continue
        c = e - 1                          # a token like " को" starts on the space; its end is in the word
        while wi < len(spans) - 1 and c >= spans[wi][1]:
            wi += 1
        if spans[wi][0] <= c < spans[wi][1]:
            last[wi] = ti
    return enc.ids[:ctx], last


def batches(data, bs, shuffle, rng, pad):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        rows = [data[j] for j in order[i:i + bs]]
        T = max(len(r[0]) for r in rows)
        x = torch.full((len(rows), T), pad, dtype=torch.long)
        y = torch.full((len(rows), T), -100, dtype=torch.long)
        for k, (ids, last, labs) in enumerate(rows):
            x[k, :len(ids)] = torch.tensor(ids)
            for p, l in zip(last, labs):
                if p >= 0:
                    y[k, p] = l
        yield x, y


@torch.no_grad()
def predict(model, clf, data, bs, pad, dev):
    model.eval()
    out = []
    for x, y in batches(data, bs, False, None, pad):
        x = x.to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            h = model.hidden(x)
        pred = clf(h.float()).argmax(-1).cpu()
        for k in range(x.shape[0]):
            m = y[k] >= 0
            out.append((y[k][m].tolist(), pred[k][m].tolist()))
    model.train()
    return out


def spans(labels):
    """BIO -> {(type, start, end)}; an I- without a matching B- starts a span (conlleval)"""
    out, cur = set(), None
    for i, l in enumerate(labels + ["O"]):
        if cur and not (l.startswith("I-") and l[2:] == cur[0]):
            out.add((cur[0], cur[1], i)); cur = None
        if l.startswith("B-") or (l.startswith("I-") and cur is None):
            cur = (l[2:], i)
    return out


def metrics(pairs, names, task):
    gold = [g for gs, _ in pairs for g in gs]
    pred = [p for _, ps in pairs for p in ps]
    labs = sorted(set(gold))
    f1s = []
    for l in labs:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    m = {"acc": sum(g == p for g, p in zip(gold, pred)) / len(gold), "macro_f1": sum(f1s) / len(f1s), "n": len(gold)}
    if task == "ner":
        tp = fp = fn = 0
        for gs, ps in pairs:
            g, p = spans([names[i] for i in gs]), spans([names[i] for i in ps])
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
        m["span_p"] = tp / (tp + fp) if tp + fp else 0.0
        m["span_r"] = tp / (tp + fn) if tp + fn else 0.0
        m["span_f1"] = 2 * m["span_p"] * m["span_r"] / (m["span_p"] + m["span_r"]) if tp else 0.0
        o = names.index("O")
        m["macro_f1_no_o"] = sum(f for f, l in zip(f1s, labs) if l != o) / (len(labs) - 1)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def selfcheck():
    tok = Tokenizer.from_file(common.tokenizer_path("fixed", 16000))
    words = ["नेपाल", "सरकारले", "काठमाडौंमा", "।"]
    ids, last = encode_sent(tok, None, words, 512)
    assert all(p >= 0 for p in last) and last == sorted(last) and last[-1] == len(ids) - 1, last
    # every word's last token decodes to the end of that word
    for w, p in zip(words, last):
        assert w.endswith(tok.decode([ids[p]]).strip()), (w, tok.decode([ids[p]]))
    assert spans(["B-PER", "I-PER", "O", "I-LOC", "B-LOC"]) == {("PER", 0, 2), ("LOC", 3, 4), ("LOC", 4, 5)}
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(FILES), default="ner")
    ap.add_argument("--variant")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--docs", type=int, default=400_000, help="which pretrained checkpoint")
    ap.add_argument("--pre-seed", type=int, default=1, help="which pretrained checkpoint")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scratch", action="store_true", help="random init: no pretraining")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", default=f"{ROOT}/results/downstream.jsonl")
    ap.add_argument("--train-frac", type=float, default=1.0, help="use this share of the training sentences (data-efficiency curve)")
    ap.add_argument("--dump", action="store_true", help="save test predictions to results/preds/ for significance and slice analysis")
    ap.add_argument("--tag", default="", help="pretraining checkpoint suffix (train_lm.py --tag), e.g. matched")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    tok, tier, pad, a.vocab = common.load_tokenizer(a.variant, a.vocab)
    if a.task == "cc":
        tr, test = read_cc()
        tr, dv, extra = split(tr); dv += extra                  # 80/20 of train -> train/dev; test is official
        names = sorted({l for _, l in tr})
        lab = {l: i for i, l in enumerate(names)}
        if hasattr(tier, "prime"):
            tier.prime(t for part in (tr, dv, test) for t, _ in part)
        parts = [[(*encode_doc(tok, tier, t, a.ctx, pad), [lab[l]]) for t, l in part] for part in (tr, dv, test)]
        train, dv, test = parts
        lost = sum(len(tok.encode(tier.mark(t) if tier else t).ids) >= a.ctx for t, _ in tr)
        ntok = sum(len(ids) - 1 for ids, _, _ in train)
        nword = sum(len(t.split()) for t, _ in tr)
        raw_train = [t.split() for t, _ in tr]
    else:
        sents = read(a.task)
        if hasattr(tier, "prime"):
            tier.prime(" ".join(w) for w, _ in sents)
        names = sorted({l for _, ls in sents for l in ls})
        lab = {l: i for i, l in enumerate(names)}
        parts = []
        for part in split(sents):
            rows = []
            for words, labels in part:
                ids, last = encode_sent(tok, tier, words, a.ctx)
                rows.append((ids, last, [lab[l] for l in labels]))
            parts.append(rows)
        train, dv, test = parts
        lost = sum(p < 0 for rows in parts for _, last, _ in rows for p in last)
        ntok = sum(len(ids) for ids, _, _ in train)
        nword = sum(len(last) for _, last, _ in train)
        raw_train = [w for w, _ in split(sents)[0]]
        raw_test = [w for w, _ in split(sents)[2]]
    if a.train_frac < 1:
        keep = rng.sample(range(len(train)), max(1, int(a.train_frac * len(train))))
        train = [train[i] for i in sorted(keep)]
        raw_train = [raw_train[i] for i in sorted(keep)]
    print(f"{a.task}/{a.variant}: {len(train)}/{len(dv)}/{len(test)} sents, {len(names)} labels, "
          f"train fertility {ntok / nword:.3f}, {lost} {'docs truncated at' if a.task == 'cc' else 'words past'} ctx")

    tagged = f"{a.variant}_{a.tag}" if a.tag else a.variant
    ck = f"{ROOT}/checkpoints/{tagged}_{a.vocab}_{a.docs}_s{a.pre_seed}.pt"
    state = torch.load(ck, map_location="cpu")
    ar = state["args"]
    model = GPT(ar["vocab"], ar["ctx"], ar["n_layer"], ar["n_head"], ar["n_embd"])
    if not a.scratch:
        model.load_state_dict(state["state"])
    clf = nn.Linear(ar["n_embd"], len(names))
    model.to(dev); clf.to(dev)
    params = list(model.parameters()) + list(clf.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    best, t0 = None, time.time()
    for ep in range(a.epochs):
        for x, y in batches(train, a.batch, True, rng, pad):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                h = model.hidden(x)
            loss = F.cross_entropy(clf(h.float()).view(-1, len(names)), y.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        md = metrics(predict(model, clf, dv, 64, pad, dev), names, a.task)
        key = "span_f1" if a.task == "ner" else "macro_f1"
        print(f"  epoch {ep + 1} loss {loss.item():.3f} dev {key} {md[key]:.4f} {time.time() - t0:.0f}s", flush=True)
        if best is None or md[key] > best[1][key]:
            tp = predict(model, clf, test, 64, pad, dev)
            best = (ep + 1, md, metrics(tp, names, a.task), tp)
    rec = {"task": a.task, "variant": a.variant, "vocab": a.vocab, "pretrain_docs": a.docs, "pre_seed": a.pre_seed,
           "seed": a.seed, "scratch": a.scratch, "epochs": a.epochs, "best_epoch": best[0], "lr": a.lr,
           "batch": a.batch, "train_sents": len(train), "train_fertility": round(ntok / nword, 4),
           "train_frac": a.train_frac, "dev": best[1], "test": best[2], "seconds": round(time.time() - t0),
           "host": os.uname().nodename}
    if a.tag:
        rec["tag"] = a.tag
    if a.dump:   # per-sentence gold/pred (+ words, train word types) for paired tests and slices
        os.makedirs(f"{ROOT}/results/preds", exist_ok=True)
        rec["preds"] = f"results/preds/{a.task}_{tagged}_{a.vocab}_s{a.seed}{'_scratch' if a.scratch else ''}{'' if a.train_frac == 1 else f'_f{a.train_frac}'}.json"
        json.dump({"names": names, "pairs": best[3], "test_words": raw_test if a.task != "cc" else None,
                   "train_types": sorted({w for ws in raw_train for w in ws}) if a.task != "cc" else None},
                  open(f"{ROOT}/{rec['preds']}", "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(rec, ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
