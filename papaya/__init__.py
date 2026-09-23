"""Papaya: morphology-guided BPE tokenization for Nepali.

    from papaya import Papaya
    tok = Papaya.from_pretrained("papaya-v2-16k")
    tok.segment("सरकारले")          # ['सरकार', 'ले']
    ids = tok.encode("सरकारले विद्यार्थीहरूलाई छात्रवृत्ति दिने भएको छ")
    tok.decode(ids)                 # the original text, byte for byte
"""
from ._paths import available

__version__ = "1.0.0"
__all__ = ["Papaya", "load", "available"]


def __getattr__(name):          # the tiers work without importing `tokenizers`
    if name in ("Papaya", "load"):
        from . import tokenizer
        return getattr(tokenizer, name)
    raise AttributeError(name)
