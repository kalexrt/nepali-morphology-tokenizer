import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from papaya import foma  # noqa: E402

needs_foma = pytest.mark.skipif(not foma.available(), reason="foma/flookup not installed")


def gold_rows():
    with open(os.path.join(ROOT, "data", "eval", "gold_sheet.tsv"), encoding="utf-8") as f:
        return [(r["word"], r["GOLD"].strip()) for r in csv.DictReader(f, delimiter="\t") if r["GOLD"].strip()]


def boundaries(pieces):
    b, i = set(), 0
    for p in pieces[:-1]:
        i += len(p)
        b.add(i)
    return b


def gold_prf(split):
    """P, R, F1 of a word -> pieces function against the gold sheet"""
    tp = fp = fn = 0
    for word, seg in gold_rows():
        g = boundaries(seg.split("|"))
        s = boundaries(split(word))
        tp += len(g & s); fp += len(s - g); fn += len(g - s)
    p, r = tp / (tp + fp), tp / (tp + fn)
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4)
