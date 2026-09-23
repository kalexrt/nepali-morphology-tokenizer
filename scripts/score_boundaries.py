#!/usr/bin/env python3
"""
Morpheme-boundary precision / recall / F1 of each segmenter against the
eval sets. A boundary is a character index inside the word; gold boundaries
are the | in the gold column; system boundaries are where the segmenter's
pieces (regex tier) or a tokenizer's tokens start.

Sets (data/eval/): silver_ud.tsv (real lemmas, tiny), silver_hunspell.tsv
(synthetic, stem boundary only -> recall is the meaningful number, precision
is a lower bound because the suffix stack is unsplit in the gold),
gold_sheet.tsv (| = inflectional cut, + = derivational cut; derivational cuts
are scored separately -- the annotators used none).

    python3 scripts/score_boundaries.py [--vocab 16000]
Appends one JSON line per (set, system) to results/boundary_f1.jsonl.
"""
import argparse
import csv
import json
import os

from tokenizers import Tokenizer

import common
from common import ROOT, regex_tier

SEP = "\u2060"


def gold_bounds(seg, marks="|"):
    b, i = set(), 0
    for ch in seg:
        if ch in marks:
            b.add(i)
        else:
            i += 1
    return b


def load_set(name):
    path = f"{common.EVAL}/{name}.tsv"
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    col = "GOLD" if "GOLD" in rows[0] else "gold"
    return [(r["word"], r[col].strip()) for r in rows if r[col].strip()]


class TierSys:
    def __init__(self, profile, attested=None):
        self.t = regex_tier.load(profile, attested)

    def bounds(self, word):
        b, i = set(), 0
        for p in self.t.split(word)[:-1]:
            i += len(p); b.add(i)
        return b


class FstSys(TierSys):
    def __init__(self, mode, version=1, **kw):
        self.t = common.fst(mode, version, **kw)

    def prime(self, words):
        self.t.lookup(words)


class MorfSys(TierSys):
    def __init__(self):
        import morfessor_tier
        self.t = morfessor_tier.load()


class TokSys:
    def __init__(self, path, tier=None):
        self.tok = Tokenizer.from_file(path)
        self.tier = tier

    def prime(self, words):
        if hasattr(self.tier, "lookup"):
            self.tier.lookup(words)

    def bounds(self, word):
        text = " " + (self.tier.mark(word) if self.tier else word)
        enc = self.tok.encode(text)
        # offsets are over the marked text; map back by discarding SEP positions
        pos = [i for i, ch in enumerate(text) if ch != SEP]
        b = set()
        for s, _ in enc.offsets[1:] if enc.offsets[0][0] == 0 else enc.offsets:
            if s < len(text) and text[s] == SEP:
                s += 1
            if 0 < s < len(text):
                b.add(pos.index(s) - 1)
        b.discard(0)
        return b


def score(system, data, marks="|"):
    tp = fp = fn = exact = 0
    if hasattr(system, "prime"):
        system.prime([w for w, _ in data])
    for word, seg in data:
        g = gold_bounds(seg, marks)
        s = system.bounds(word)
        tp += len(g & s); fp += len(s - g); fn += len(g - s); exact += g == s
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4),
            "exact": round(exact / len(data), 4), "n": len(data)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--only", nargs="+", help="score only these systems (e.g. morfessor bpe_uni_16000); "
                                              "results/*.jsonl already holds the rest")
    a = ap.parse_args()
    counts = common.counts()
    systems = {"tier_core": TierSys("core"), "tier_full": TierSys("full"),
               "tier_gated": TierSys("full", counts),
               "fst_lex": FstSys("lex"), "fst_hybrid": FstSys("hybrid"),
               "fst2_lex": FstSys("lex", 2), "fst2_hybrid": FstSys("hybrid", 2),
               # strict fallback + tier ordering ablation; defaults above unchanged
               "tier_strict": TierSys("strict"),
               "fst_hybrid_strict": FstSys("hybrid", fallback_profile="strict"),
               "fst_regexfirst_full": FstSys("hybrid", order="regex-first"),
               "fst_regexfirst_strict": FstSys("hybrid", order="regex-first", fallback_profile="strict"),
               "fst2_hybrid_strict": FstSys("hybrid", 2, fallback_profile="strict")}
    if any(os.path.exists(f"{ROOT}/{m}") for m in ("tokenizers/morfessor_baseline.bin", "data/morfessor_baseline.bin.gz")):  # unsupervised segmenter
        systems["morfessor"] = MorfSys()
    for v in ["broken", "fixed", "core", "full", "gated", "fst", "fst2", "fst3", "litchi", "morf", "uni", "fst_uni"]:
        path = common.tokenizer_path(v, a.vocab)
        if os.path.exists(path):
            systems[f"bpe_{v}_{a.vocab}"] = TokSys(path, common.MORPH[v]() if v in common.MORPH else None)
    if a.only:
        systems = {k: v for k, v in systems.items() if k in a.only}
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    out = open(f"{ROOT}/results/boundary_f1.jsonl", "a", encoding="utf-8")
    for name in ["silver_ud", "silver_hunspell", "gold_sheet"]:
        data = load_set(name)
        if not data:
            print(f"{name}: empty, skipped"); continue
        print(f"\n{name} ({len(data)} words)" + ("  [stem boundary only: read R]" if name == "silver_hunspell" else ""))
        for sname, sys_ in systems.items():
            rec = {"set": name, "system": sname, **score(sys_, data)}
            if name == "gold_sheet":
                rec["deriv"] = score(sys_, data, marks="+")
            print(f"  {sname:18s} P {rec['P']:.3f}  R {rec['R']:.3f}  F1 {rec['F1']:.3f}  exact {rec['exact']:.3f}")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
