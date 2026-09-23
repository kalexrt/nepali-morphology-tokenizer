import filecmp
import os

from conftest import gold_prf, needs_foma

from papaya import fst_tier
from papaya._paths import pretrained_dir


def test_lexc_regenerates_from_stems(tmp_path):
    """the shipped grammar is exactly what write_lexc makes of the shipped stem list"""
    for name in ("papaya-v1-16k", "papaya-v2-16k"):
        d = pretrained_dir(name)
        out = tmp_path / f"{name}.lexc"
        fst_tier.write_lexc(fst_tier.read_stems(os.path.join(d, "stems.tsv")), str(out))
        assert filecmp.cmp(out, os.path.join(d, "segmenter.lexc"), shallow=False), name


@needs_foma
def test_selfcheck():
    for v in (1, 2):
        assert fst_tier.selfcheck(fst_tier.load("lex", v)) == {}


@needs_foma
def test_known_cuts():
    t = fst_tier.load("hybrid", 2)
    words = {"सरकारले": ["सरकार", "ले"], "गर्नुपर्ने": ["गर्नु", "पर्", "ने"], "भूमिका": ["भूमिका"],
             "विद्यार्थीहरूलाई": ["विद्यार्थी", "हरू", "लाई"], "मकै": ["मकै"]}
    t.prime([" ".join(words)])
    for w, want in words.items():
        assert t.split(w) == want, w


@needs_foma
def test_gold_sheet_scores_match_paper():
    # results/boundary_f1.jsonl: fst_lex, fst_hybrid, fst2_lex, fst2_hybrid, fst2_hybrid_strict
    cases = {("lex", 1, "full"): 0.9463, ("hybrid", 1, "full"): 0.9606,
             ("lex", 2, "full"): 0.951, ("hybrid", 2, "full"): 0.9652, ("hybrid", 2, "strict"): 0.9687}
    for (mode, v, fb), f1 in cases.items():
        t = fst_tier.load(mode, v, fallback_profile=fb)
        from conftest import gold_rows
        t.prime([" ".join(w for w, _ in gold_rows())])
        assert gold_prf(t.split)[2] == f1, (mode, v, fb)
