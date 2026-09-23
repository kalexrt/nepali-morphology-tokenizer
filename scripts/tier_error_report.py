#!/usr/bin/env python3
"""
Per-affix precision of the regex tier on data/eval/gold_sheet.tsv: for every suffix
the tier cuts off, how many of those cuts sit on a gold boundary. This is how
STRICT_DROP in regex_tier.py was derived; re-run it when the sheet grows.

    python3 scripts/tier_error_report.py [--profile full|strict] [--bucket natural] [--set gold_sheet]
    python3 scripts/tier_error_report.py --corpus corpus/wordcounts_train_tok.json [--profile full]
        gold-free view: per affix, the share of corpus word types the tier cuts
        whose stem is itself attested standalone (count >= 100); an affix whose
        stems are almost never words is cutting inside words

Caveat: STRICT_DROP was derived FROM the gold sheet, which is also the
evaluation set, and six of its 19 affixes rest on 1-2 observations. The
--corpus view is the gold-free check: 9 of the 19 are supported by corpus
evidence alone, 7 are too rare to judge, 3 (की भर खेर) rest on the sheet only.
Treat strict-profile numbers as tuned on the evaluation set.
"""
import argparse
import collections
import csv
import os

import common
from common import regex_tier
from score_boundaries import gold_bounds, load_set


def corpus_report(a):
    counts = common.fst_tier.load_counts(a.corpus)
    t = regex_tier.load(a.profile)
    stat = collections.defaultdict(lambda: [0, 0, []])   # affix -> [types, stem attested, unattested examples]
    for w, n in counts.items():
        if n < 100 or not regex_tier.WORD.fullmatch(w) or not regex_tier.STEM_OK.search(w):
            continue
        p = t.split(w)
        if len(p) < 2:
            continue
        suf, stem = p[-1], "".join(p[:-1])
        stat[suf][0] += 1
        # verbal cuts leave a halanta stem that is bound by definition: attested
        # means its infinitive / prospective form is a word (गर्+ने <- गर्नु)
        if t.group[suf] == "verbal":
            ok = any(counts.get(stem + x, 0) >= 100 for x in ("नु", "ने", "छ"))
        else:
            ok = counts.get(stem, 0) >= 100 or stem in regex_tier.BOUND_STEMS or bool(regex_tier.BOUND_RX.search(stem))
        if ok:
            stat[suf][1] += 1
        else:
            stat[suf][2].append((n, w))
    print(f"{a.corpus}: word types with count >= 100, profile {a.profile}; stem attested = standalone count >= 100 (verbal: stem+नु/ने/छ)")
    print(f"{'affix':10s} {'types':>6s} {'attest':>6s} {'rate':>5s}  most frequent unattested")
    for suf, (n, ok, ex) in sorted(stat.items(), key=lambda kv: kv[1][1] / kv[1][0]):
        ex.sort(reverse=True)
        print(f"{suf:10s} {n:6d} {ok:6d} {ok/n:5.2f}  {' '.join(w for _, w in ex[:a.top])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--bucket", help="restrict to one gold-sheet bucket (natural, verbal, postp ...)")
    ap.add_argument("--set", default="gold_sheet")
    ap.add_argument("--top", type=int, default=4, help="worst words shown per affix")
    ap.add_argument("--corpus", help="word-count JSON: report stem attestation per affix instead of gold precision")
    a = ap.parse_args()
    if a.corpus:
        return corpus_report(a)
    data = load_set(a.set)
    if a.bucket:
        rows = csv.DictReader(open(f"{common.EVAL}/{a.set}.tsv", encoding="utf-8"), delimiter="\t")
        keep = {r["word"] for r in rows if r.get("bucket") == a.bucket}
        data = [(w, g) for w, g in data if w in keep]
    t = regex_tier.load(a.profile)
    stat = collections.defaultdict(lambda: [0, 0, []])   # affix -> [cuts, wrong, words]
    tp = fp = fn = 0
    for word, seg in data:
        g = gold_bounds(seg)
        pieces = t.split(word)
        i, s = 0, set()
        for p, suf in zip(pieces, pieces[1:]):
            i += len(p); s.add(i)
            stat[suf][0] += 1
            if i not in g:
                stat[suf][1] += 1
                stat[suf][2].append(f"{word}→{'|'.join(pieces)}")
        tp += len(g & s); fp += len(s - g); fn += len(g - s)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    print(f"{a.set} {a.bucket or 'all buckets'}, profile {a.profile}: {len(data)} words, "
          f"{tp + fp} cuts, {fp} wrong, P {p:.3f} R {r:.3f} F1 {2*p*r/(p+r) if p+r else 0:.3f}")
    print(f"{'affix':10s} {'cuts':>5s} {'ok':>4s} {'wrong':>5s} {'prec':>6s}  worst")
    for suf, (n, bad, words) in sorted(stat.items(), key=lambda kv: (-kv[1][1], -kv[1][0])):
        print(f"{suf:10s} {n:5d} {n-bad:4d} {bad:5d} {(n-bad)/n:6.2f}  {' '.join(words[:a.top])}")


if __name__ == "__main__":
    main()
