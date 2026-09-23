#!/usr/bin/env python3
"""
Morphological consistency (after MorphBPE). Does the stem of
an inflected word get the same tokens as the stem on its own? A tokenizer that
cuts सरकार|ले and सरकार|हरू both as [" सरकार", ...] is consistent; one that
gives " सरका", "रले" is not, and the model must relearn the stem per form.

Two samples of (form, stem):
  gold    data/eval/gold_sheet.tsv rows with a cut; stem = first piece (after a
          negative न). Annotator-defined, independent of the FST.
  corpus  the FST v1 lexicon's analyses (lex mode, no fallback) of every word
          type with count >= 20 in the training word counts (needs the full
          counts table, see common.word_counts); stem = first piece.
          Defined by the same lexicon the fst variant uses, so the fst rows are
          consistent by construction there -- the corpus sample says how far
          the OTHER tokenizers are from that, on 30k+ real forms.

Per form, in the mid-sentence context " " + word: the tokens lying inside the
stem span vs the tokens of " " + stem alone, as bags. P/R/F1 of that, averaged
over forms (a token straddling the stem's end is a wrong prediction). `exact`
is the share of forms whose stem tokens equal the stem-alone tokens.

    python3 scripts/morph_consistency.py [--vocab 16000 8000 32000]
Appends to results/morph_consistency.jsonl.
"""
import argparse
import collections
import csv
import json
import os

from tokenizers import Tokenizer

import common
from common import ROOT

MIN_COUNT = 20


def gold_sample():
    out = []
    for r in csv.DictReader(open(f"{common.EVAL}/gold_sheet.tsv", encoding="utf-8"), delimiter="\t"):
        p = r["GOLD"].split("|")
        if len(p) > 1:
            stem = p[1] if p[0] == "न" and len(p) > 2 else p[0]
            out.append((r["word"], stem))
    return out


def corpus_sample():
    counts = common.word_counts(full=True)
    words = [w for w, c in counts.items() if c >= MIN_COUNT]
    tier = common.fst("lex", 1)
    tier.lookup(words)
    out = []
    for w in words:
        p = tier.cache[w]
        if p and len(p) > 1:
            out.append((w, p[1] if p[0] == "न" and len(p) > 2 else p[0]))
    return out


class System:
    def __init__(self, variant, vocab, path=None):
        self.tok = Tokenizer.from_file(path or common.tokenizer_path(variant, vocab))
        self.tier = common.MORPH[variant]() if variant in common.MORPH else None

    def prime(self, words):
        if hasattr(self.tier, "prime"):
            self.tier.prime([" ".join(words)])

    def tokens(self, text, span=None):
        """token ids of text; with span=(s, e) in text, only tokens inside it, and
        None for a token that straddles e"""
        marked = self.tier.mark(text) if self.tier else text
        enc = self.tok.encode(marked)
        if span is None:
            return list(enc.ids)
        s, e = span
        if self.tier:                       # joiners shift offsets: map e onto the marked text
            e += marked[:e + marked.count("⁠", 0, e) + 1].count("⁠")
        out = []
        for i, (a, b) in zip(enc.ids, enc.offsets):
            if a < e and b > s:
                out.append(i if b <= e else None)
        return out


def score(system, sample):
    n = tp = fp = fn = exact = 0
    for word, stem in sample:
        if not word.startswith(stem) or word == stem:
            continue
        text = " " + word
        ref = collections.Counter(system.tokens(" " + stem))
        got = system.tokens(text, (0, 1 + len(stem)))
        bag = collections.Counter(got)
        exact += None not in got and bag == ref
        for t in set(bag) | set(ref):
            c = min(bag[t], ref[t]) if t is not None else 0
            tp += c; fp += bag[t] - c; fn += ref[t] - c
        n += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
            "exact": round(exact / n, 4), "n": n}


def selfcheck():
    sysm = System("full", 16000)                          # regex tier: " सरकार⁠ले", so the joiner offsets are exercised
    ids = sysm.tokens(" सरकारले", (0, 6))                 # " सरकार" is 6 chars
    assert None not in ids and sysm.tok.decode(ids) == " सरकार", sysm.tok.decode([i for i in ids if i is not None])
    ids = sysm.tokens(" सरकारले", (0, 3))                 # a cut inside the stem token straddles
    assert None in ids, ids
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, nargs="+", default=[16000])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--variants", nargs="+", default=["broken", "fixed", "core", "full", "gated", "fst", "fst2",
                                                      "fst3", "litchi", "morf", "uni", "fst_uni"])
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    samples = {"gold": gold_sample(), "corpus": corpus_sample()}
    print({k: len(v) for k, v in samples.items()}, "forms")
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    out = open(f"{ROOT}/results/morph_consistency.jsonl", "a", encoding="utf-8")
    for vocab in a.vocab:
        for v in a.variants:
            if not os.path.exists(common.tokenizer_path(v, vocab)):
                continue
            sysm = System(v, vocab)
            for name, sample in samples.items():
                sysm.prime([w for w, _ in sample] + [s for _, s in sample])
                rec = {"set": name, "variant": v, "vocab": vocab, **score(sysm, sample)}
                print(f"  {name:6s} {v:7s} {vocab:5d}  P {rec['P']:.3f}  R {rec['R']:.3f}  F1 {rec['F1']:.3f}  exact {rec['exact']:.3f}  n {rec['n']}")
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
