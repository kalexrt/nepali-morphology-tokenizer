#!/usr/bin/env python3
"""
Did showing the regex tier's proposal anchor the gold annotation?

data/eval/gold_sheet.tsv carries proposal_shown: the annotators saw the regex
tier's `full` split (PROPOSAL, from an earlier version of the tier) next to 422
of the 607 words and nothing next to the other 185, chosen at random (PROPOSAL
is empty there). The measurement:

  systems     boundary P/R/F1 of every segmenter and tokenizer in
              score_boundaries.py on blind vs shown rows. The number that
              matters is each system's F1 on the blind rows, which no
              suggestion could have pulled toward it.

For each, a permutation p for (shown - blind): the share of 10,000 random
relabellings of the 607 rows into 422 / 185 whose F1 gap is at least as
large as the observed one (one-sided). BPE rows use tokenizers/ (or the
shipped 16k copies); a missing tokenizer is skipped with a note.

    python3 scripts/gold_anchoring.py [--vocab 16000] [--perm 10000]
Appends one JSON line per system to results/gold_anchoring.jsonl.
"""
import argparse
import csv
import json
import os
import random

import common
from common import ROOT, regex_tier


def gold_bounds(seg, marks="|"):
    b, i = set(), 0
    for ch in seg:
        if ch in marks:
            b.add(i)
        else:
            i += 1
    return b


class Tier:
    """regex or FST tier as a segmenter (same as score_boundaries.TierSys/FstSys)"""
    def __init__(self, t):
        self.t = t

    def prime(self, words):
        if hasattr(self.t, "lookup"):
            self.t.lookup(words)

    def bounds(self, word):
        b, i = set(), 0
        for p in self.t.split(word)[:-1]:
            i += len(p); b.add(i)
        return b


def counts(system, rows):
    """per-row (tp, fp, fn, exact) of system against GOLD"""
    if hasattr(system, "prime"):
        system.prime([r["word"] for r in rows])
    out = []
    for r in rows:
        g, s = gold_bounds(r["GOLD"]), system.bounds(r["word"])
        out.append((len(g & s), len(s - g), len(g - s), g == s))
    return out


def prf(c, idx):
    tp = sum(c[i][0] for i in idx); fp = sum(c[i][1] for i in idx); fn = sum(c[i][2] for i in idx)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4),
            "exact": round(sum(c[i][3] for i in idx) / len(idx), 4), "n": len(idx)}


def split_scores(c, shown, perm, seed=0):
    blind = [i for i in range(len(c)) if i not in shown]
    shown = sorted(shown)
    f = lambda idx: prf(c, idx)["F1"]
    gap = f(shown) - f(blind)
    rng, idx, hits = random.Random(seed), list(range(len(c))), 0
    for _ in range(perm):
        rng.shuffle(idx)
        hits += f(idx[len(blind):]) - f(idx[:len(blind)]) >= gap - 1e-12
    return {"blind": prf(c, blind), "shown": prf(c, shown), "all": prf(c, range(len(c))),
            "gap_F1": round(gap, 4), "perm_p": round(hits / perm, 4)}


def systems(vocab):
    make = {"tier_core": lambda: Tier(regex_tier.load("core")),
            "tier_full": lambda: Tier(regex_tier.load("full")),
            "tier_strict": lambda: Tier(regex_tier.load("strict")),
            "tier_gated": lambda: Tier(regex_tier.load("full", common.counts())),
            "fst_lex": lambda: Tier(common.fst("lex", 1)),
            "fst_hybrid": lambda: Tier(common.fst("hybrid", 1)),
            "fst2_hybrid": lambda: Tier(common.fst("hybrid", 2))}
    try:
        import score_boundaries as sb          # needs the `tokenizers` library
        for v in ["broken", "fixed", "core", "full", "gated", "fst", "fst2"]:
            path = common.tokenizer_path(v, vocab)
            if not os.path.exists(path):
                print(f"  {f'bpe_{v}_{vocab}':18s} skipped: {path} missing")
                continue
            make[f"bpe_{v}_{vocab}"] = (lambda v=v, path=path: sb.TokSys(
                path, common.MORPH[v]() if v in common.MORPH else None))
    except ImportError as e:
        print(f"  BPE rows skipped: {e}")
    for name, fn in make.items():
        try:
            yield name, fn()
        except (FileNotFoundError, OSError) as e:
            print(f"  {name:18s} skipped: {e.__class__.__name__}: {getattr(e, 'filename', '') or e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--perm", type=int, default=10000)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(f"{common.EVAL}/gold_sheet.tsv", encoding="utf-8"), delimiter="\t"))
    shown = {i for i, r in enumerate(rows) if r["proposal_shown"] == "1"}
    print(f"gold_sheet: {len(rows)} words, proposal shown on {len(shown)}, blind {len(rows) - len(shown)}")
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    out = open(f"{ROOT}/results/gold_anchoring.jsonl", "a", encoding="utf-8")
    for name, system in systems(a.vocab):
        rec = {"system": name, **split_scores(counts(system, rows), shown, a.perm)}
        print(f"  {name:30s} blind F1 {rec['blind']['F1']:.3f}  shown F1 {rec['shown']['F1']:.3f}  "
              f"all {rec['all']['F1']:.3f}  gap {rec['gap_F1']:+.3f}  p {rec['perm_p']:.3f}")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
