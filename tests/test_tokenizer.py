import os

import pytest
from conftest import needs_foma

from papaya import available
from papaya._paths import PRETRAINED_ROOT

pytest.importorskip("tokenizers")
from papaya import Papaya  # noqa: E402

TEXTS = ["सरकारले विद्यार्थीहरूलाई छात्रवृत्ति दिने भएको छ।",
         "नेपालको राजधानी काठमाडौं हो। Kathmandu 2026 — ठाउँमा 123 😀",
         "", "   ", "उनीहरूले गर्नुपर्ने कामहरू अहिलेसम्म सकिएका छैनन्"]


def shipped(needs_fst):
    names = [n for n in available() if ("papaya-v" in n) == needs_fst]
    return names


@pytest.mark.parametrize("name", shipped(False))
def test_roundtrip_no_foma(name):
    tok = Papaya.from_pretrained(name)
    for t in TEXTS:
        assert tok.decode(tok.encode(t)) == t


@needs_foma
@pytest.mark.parametrize("name", shipped(True))
def test_roundtrip_fst(name):
    tok = Papaya.from_pretrained(name)
    for t, ids in zip(TEXTS, tok.encode_batch(TEXTS)):
        assert tok.decode(ids) == t
        assert ids == tok.encode(t)
    assert tok.segment("सरकारले") == ["सरकार", "ले"]
    # no token crosses a morph cut
    assert "सरकार" in "".join(tok.tokenize("सरकारले")) and tok.tokenize(" सरकारले")[-1].endswith("ले")


def test_config_present():
    for name in available():
        assert os.path.exists(os.path.join(PRETRAINED_ROOT, name, "papaya_config.json"))
