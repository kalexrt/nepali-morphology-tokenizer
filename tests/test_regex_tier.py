import json

from conftest import gold_prf

from papaya import regex_tier

TEXT = "सरकारले विद्यार्थीहरूलाई छात्रवृत्ति दिने भएको छ। रामको घरमा ५ जना थिए।"


def test_rebuild_matches_shipped_table():
    shipped = json.load(open(regex_tier.OUT, encoding="utf-8"))["affixes"]
    assert regex_tier.build() == shipped


def test_selfcheck():
    n, _ = regex_tier.selfcheck(regex_tier.build())    # asserts inside
    assert n > 2000


def test_mark_is_lossless():
    for prof in ("core", "full", "strict"):
        t = regex_tier.load(prof)
        marked = t.mark(TEXT)
        assert marked.replace("⁠", "") == TEXT
    assert regex_tier.load("full").mark("रामले किताबहरू पढ्यो।") == "राम⁠ले किताब⁠हरू पढ्यो।"


def test_gold_sheet_scores_match_paper():
    # results/boundary_f1.jsonl: tier_core / tier_full / tier_strict
    assert gold_prf(regex_tier.load("core").split) == (0.7768, 0.4163, 0.5421)
    assert gold_prf(regex_tier.load("full").split) == (0.7902, 0.811, 0.8005)
    assert gold_prf(regex_tier.load("strict").split) == (0.9471, 0.7703, 0.8496)
