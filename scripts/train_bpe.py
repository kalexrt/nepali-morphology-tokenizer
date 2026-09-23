#!/usr/bin/env python3
"""
Matched-budget byte-level BPE tokenizers, one per pre-tokenization variant
(see common.py for the ids), same corpus sample, same vocab size; then
fertility (tokens per whitespace word) on the FineWeb-2 test split.

    python3 scripts/train_bpe.py --variants fixed fst fst2 --vocab 16000 --docs 100000 --test-docs 5000
Writes tokenizers/{variant}_{vocab}.json and appends to results/fertility.jsonl.
"""
import argparse
import itertools
import json
import os
import time

import common
from common import ROOT
from papaya import bpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["broken", "fixed", "core", "full", "gated", "fst", "fst2"])
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--docs", type=int, default=100_000)
    ap.add_argument("--test-docs", type=int, default=5000)
    a = ap.parse_args()
    train, test = f"{common.CORPUS}/train_tok.txt", f"{common.CORPUS}/test.txt"
    # test-side counts are variant-independent: whitespace words and UTF-8 bytes of the raw text
    raw = list(itertools.islice(open(test, encoding="utf-8"), a.test_docs))
    words = sum(len(l.split()) for l in raw)
    nbytes = sum(len(l.encode()) for l in raw)
    os.makedirs(f"{ROOT}/tokenizers", exist_ok=True)
    for v in a.variants:
        tier = common.MORPH[v]() if v in common.MORPH else None
        t0 = time.time()
        if hasattr(tier, "prime"):      # one flookup pass over train and test word types
            tier.prime(itertools.chain(bpe.lines([train], a.docs), raw))
        model = "unigram" if v in common.UNIGRAM else "bpe"
        tok = bpe.train([train], a.vocab, tier, pattern="broken" if v == "broken" else "fixed", docs=a.docs, model=model)
        out = f"{ROOT}/tokenizers/{v}_{a.vocab}.json"
        tok.save(out)
        marked = [tier.mark(l) for l in raw] if tier else raw
        encs = tok.encode_batch(marked)
        ntok = sum(len(e.ids) for e in encs)
        unk = tok.token_to_id("<unk>")
        n_unk = sum(e.ids.count(unk) for e in encs) if unk is not None else 0
        if n_unk:
            print(f"WARNING {v}: {n_unk} <unk> tokens in the test docs -- bits per byte would be optimistic")
        assert all(tok.decode(e.ids) == l for e, l in zip(tok.encode_batch(marked[:200]), raw[:200])), "round-trip"
        rec = {"variant": v, "vocab": a.vocab, "train_docs": a.docs, "test_docs": a.test_docs,
               "test_words": words, "test_bytes": nbytes, "tokens": ntok,
               "tokens_per_word": round(ntok / words, 4), "bytes_per_token": round(nbytes / ntok, 3),
               "train_seconds": round(time.time() - t0), "tokenizer": os.path.relpath(out, ROOT),
               "model": model, "unk_tokens": n_unk,
               "regex_tier": json.load(open(common.regex_tier.OUT))["dataset_version"] if tier else None,
               "fst_tier": json.load(open(f"{common.FST_DIR[common.FST_VERSION[v]]}/fst_tier.json"))["stems"]
               if v in common.FST_VERSION else None}
        print(json.dumps(rec, ensure_ascii=False))
        common.append(f"{ROOT}/results/fertility.jsonl", rec)


if __name__ == "__main__":
    main()
