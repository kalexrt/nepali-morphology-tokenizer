"""
Papaya = a morphological tier + a BPE vocabulary trained on its output, kept
together in one directory so they cannot drift apart:

    tokenizer.json        the BPE (Hugging Face `tokenizers` format)
    papaya_config.json    which tier produced the training text, and how
    segmenter.lexc        the FST grammar + stem lexicon (fst tier only)
    stems.tsv             the stem lexicon as a table (fst tier only)
    frequent_words.tsv    word types frequent enough to be kept whole (fst hybrid / gated regex)

tokenizer.json on its own is NOT Papaya: the tier's cuts are text
preprocessing, not part of the HF pipeline. A vocabulary trained on marked text
applied to unmarked text behaves like plain BPE with a mismatched vocabulary.
Always go through this class (or call .mark() yourself before encoding).
"""
import json
import os
import shutil

from tokenizers import Tokenizer

from . import bpe, fst_tier, regex_tier
from ._paths import pretrained_dir

TIER_FILES = ("segmenter.lexc", "stems.tsv", "frequent_words.tsv", "fst_tier.json")


def make_tier(config, directory):
    """the tier described by a papaya_config.json, reading its files from directory"""
    t = config.get("tier", "none")
    counts_path = os.path.join(directory, "frequent_words.tsv")
    if t == "fst":
        return fst_tier.load(config.get("mode", "hybrid"), config.get("fst_version", 2), tokenizer_dir=directory,
                             order=config.get("order", "fst-first"),
                             fallback_profile=config.get("fallback_profile", "full"))
    if t == "regex":
        att = fst_tier.load_counts(counts_path) if config.get("attested") else None
        return regex_tier.load(config.get("profile", "full"), att)
    if t == "none":
        return None
    raise ValueError(f"unknown tier {t!r}")


class Papaya:
    def __init__(self, tokenizer, tier, config, directory=None):
        self.backend = tokenizer          # tokenizers.Tokenizer
        self.tier = tier                  # FstTier | RegexTier | None
        self.config = config
        self.directory = directory
        self.eot_id = tokenizer.token_to_id(bpe.EOT)

    @classmethod
    def from_pretrained(cls, name_or_path="papaya-v2-16k"):
        """a shipped tokenizer (papaya-v2-16k, papaya-v1-16k, papaya-regex-16k,
        bpe-baseline-16k) or a directory written by `papaya train` / .save()"""
        d = pretrained_dir(name_or_path)
        config = json.load(open(os.path.join(d, "papaya_config.json"), encoding="utf-8"))
        return cls(Tokenizer.from_file(os.path.join(d, "tokenizer.json")), make_tier(config, d), config, d)

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    def segment(self, word):
        """morph pieces of one word: सरकारले -> ['सरकार', 'ले']"""
        return self.tier.split(word) if self.tier else [word]

    def mark(self, text):
        """text with U+2060 at every morph cut (what the vocabulary was trained on)"""
        return self.tier.mark(text) if self.tier else text

    def prime(self, texts):
        """analyse every word type of texts in one FST call (speeds up large batches)"""
        if hasattr(self.tier, "prime"):
            self.tier.prime(texts)

    def encode(self, text):
        """token ids of text (marked first)"""
        return self.backend.encode(self.mark(text)).ids

    def encode_batch(self, texts):
        texts = list(texts)
        self.prime(texts)
        return [e.ids for e in self.backend.encode_batch([self.mark(t) for t in texts])]

    def tokenize(self, text):
        """the tokens as readable strings (slices of the text; a token that
        ends inside a multi-byte character is shown with its whole character)"""
        marked = self.mark(text)
        out = []
        for s, e in self.backend.encode(marked).offsets:
            out.append(marked[s:e].replace(bpe.SEP, ""))
        return out

    def decode(self, ids):
        return self.backend.decode(ids)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.backend.save(os.path.join(directory, "tokenizer.json"))
        json.dump(self.config, open(os.path.join(directory, "papaya_config.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        if self.directory and os.path.abspath(self.directory) != os.path.abspath(directory):
            for f in TIER_FILES:
                if os.path.exists(os.path.join(self.directory, f)):
                    shutil.copy(os.path.join(self.directory, f), directory)

    def __repr__(self):
        return f"Papaya({self.config.get('name', self.directory)!r}, tier={self.config.get('tier')}, vocab={self.vocab_size})"


def load(name_or_path="papaya-v2-16k"):
    return Papaya.from_pretrained(name_or_path)
