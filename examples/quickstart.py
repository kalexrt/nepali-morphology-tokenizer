"""Papaya in a few lines: segment, encode, decode, and compare with plain BPE.

    python3 examples/quickstart.py            (needs foma; see README)
"""
from papaya import Papaya

text = "सरकारले विद्यार्थीहरूलाई छात्रवृत्ति दिने निर्णय गरेको छ।"

papaya = Papaya.from_pretrained("papaya-v2-16k")
plain = Papaya.from_pretrained("bpe-baseline-16k")

print("morphemes:", " ".join("|".join(papaya.segment(w)) for w in text.split()))
for name, tok in (("Papaya v2", papaya), ("plain BPE", plain)):
    ids = tok.encode(text)
    assert tok.decode(ids) == text          # lossless
    print(f"{name:10s} {len(ids):2d} tokens:", " · ".join(tok.tokenize(text)))
