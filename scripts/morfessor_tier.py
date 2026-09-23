#!/usr/bin/env python3
"""
Morfessor Baseline as a pre-BPE tier: variant `morf`, the unsupervised
counterpart to Papaya (Luitel et al. 2025 pre-segment Nepali this way).

Trained on the same word counts the FST stem induction reads
(corpus/wordcounts_train_tok.json or data/corpus/wordcounts_train_tok.json.gz): every
Devanagari word type seen at least MIN_COUNT times, counts log-dampened
(Morfessor's `--dampening log`), default corpus weight, recursive batch
training with Python's RNG seeded (SEED). Nothing is tuned on the gold set.
The paper's model ships as data/morfessor_baseline.bin.gz and is used unless
--train writes tokenizers/morfessor_baseline.bin.

Atoms are akshara-level units -- a character plus the combining marks that
follow it -- so, like Papaya, a piece never starts with a matra or a halanta
and the gold set's akshara convention applies to both. A cut after a halanta
(गर्|नु) is still possible, as it is for the FST tier.

Same interface as the other tiers: split(word), prime(texts), mark(text).

    python3 scripts/morfessor_tier.py --train          # -> tokenizers/morfessor_baseline.bin (gitignored, overrides the shipped model)
    python3 scripts/morfessor_tier.py --selfcheck      # API + round-trip check on a small sample, seconds
    python3 scripts/morfessor_tier.py --split WORD...
Needs `pip install morfessor` (or `pip install -e ".[morfessor]"`).
"""
import argparse
import gzip
import math
import os
import pickle
import random
import re
import time
import unicodedata

import common
from common import ROOT, regex_tier

MODEL = f"{ROOT}/tokenizers/morfessor_baseline.bin"
SHIPPED = f"{ROOT}/data/morfessor_baseline.bin.gz"   # the model behind the paper's morf rows
MIN_COUNT = 5
SEED = 0      # Morfessor's batch training shuffles with the global random module
DEV = re.compile(r"^[ऀ-ॣ०-ॿ]+$")


def atoms(word):
    """akshara-level units: split in front of every character that is not a combining mark"""
    out = []
    for ch in word:
        if out and unicodedata.category(ch).startswith("M"):
            out[-1] += ch
        else:
            out.append(ch)
    return tuple(out)


def dampen(count):
    return max(1, int(round(math.log(count + 1, 2))))


def train(counts, path=MODEL, min_count=MIN_COUNT):
    import morfessor
    random.seed(SEED)
    data = [(c, atoms(w)) for w, c in counts.items() if c >= min_count and DEV.match(w) and len(atoms(w)) > 1]
    model = morfessor.BaselineModel()
    model.load_data(data, count_modifier=dampen)
    t0 = time.time()
    model.train_batch()
    morfessor.MorfessorIO().write_binary_model_file(path, model)
    print(f"morfessor: {len(data):,} word types (count >= {min_count}) -> {path} in {time.time() - t0:.0f}s")
    return model


class MorfessorTier:
    def __init__(self, model=None, path=MODEL):
        if model is None:
            import morfessor  # noqa: F401 -- the pickle needs its classes
            if path == MODEL and not os.path.exists(MODEL):
                model = pickle.load(gzip.open(SHIPPED, "rb"))
            else:
                model = morfessor.MorfessorIO().read_binary_model_file(path)
        self.model = model
        self.cache = {}

    def split(self, word):
        if word not in self.cache:
            if not DEV.match(word) or len(atoms(word)) < 2:
                pieces = [word]
            else:
                segs = self.model.viterbi_segment(atoms(word))[0]
                pieces = ["".join(s) if isinstance(s, tuple) else s for s in segs]
            assert "".join(pieces) == word, (word, pieces)
            self.cache[word] = pieces
        return self.cache[word]

    def prime(self, texts):
        for t in texts:
            for w in regex_tier.WORD.findall(t):
                self.split(w)

    def mark(self, text, sep="⁠"):
        out, last = [], 0
        for m in regex_tier.WORD.finditer(text):
            out.append(text[last:m.start()])
            out.append(sep.join(self.split(m.group())))
            last = m.end()
        out.append(text[last:])
        return "".join(out)


def load():
    return MorfessorTier()


def selfcheck():
    counts = common.word_counts(full=True)
    top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:3000])
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    tmp = f"{ROOT}/tokenizers/.morfessor_selfcheck.bin"
    try:
        tier = MorfessorTier(train(top, tmp, 1), tmp)
        tier = MorfessorTier(path=tmp)                       # binary round trip
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    assert atoms("गर्नुपर्ने") == ("ग", "र्", "नु", "प", "र्", "ने"), atoms("गर्नुपर्ने")
    text = "सरकारले भनेको छ, नेपालमा २०८१ सालमा। abc"
    marked = tier.mark(text)
    assert marked.replace("⁠", "") == text
    for w in ["सरकारले", "नेपालमा", "केटाहरूलाई", "गर्नुपर्ने"]:
        p = tier.split(w)
        assert all(not unicodedata.category(x[0]).startswith("M") for x in p), (w, p)
        print(f"  {w} -> {'|'.join(p)}")
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--split", nargs="+")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.train:
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        train(common.word_counts(full=True))
    if a.split:
        t = load()
        for w in a.split:
            print(w, "->", "|".join(t.split(w)))


if __name__ == "__main__":
    main()
