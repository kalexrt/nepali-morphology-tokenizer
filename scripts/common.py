"""
Shared by the reproduction scripts: repo paths, the paper's variant ids, and
loading a (tokenizer, tier) pair by id.

Variant ids are the keys in results/*.jsonl, tokenizers/ and checkpoints/:

  broken   plain BPE with GPT-2's mark-splitting regex (control)
  fixed    plain byte-level BPE (baseline)
  core     regex tier, 20 affixes         full   regex tier, 164 affixes
  gated    full + attestation gate        fst    Papaya v1 (FST tier v1, hybrid)
  fst2     Papaya v2                      fst3   Papaya v2 with the strict regex fallback (ablation)
  litchi   fst3 with regex-first order (ablation)
  morf     Morfessor Baseline cuts + BPE (morfessor_tier.py; unsupervised baseline)
  uni      Unigram LM instead of BPE, fixed pre-tokenizer
  fst_uni  Papaya v1's tier + Unigram LM
  <id>@<tag>  in the tables: a second checkpoint of the same tokenizer (train_lm.py --tag),
           e.g. fixed@matched = plain BPE trained for Papaya's step count, with weights
  x_<name> a published tokenizer, tokenizers/external/<name>.json

Working files live at the repo root and are gitignored: corpus/ (fetch_corpus.py),
tokenizers/ (train_bpe.py), checkpoints/ (train_lm.py --save), downstream/
(fetch_downstream.py). The FST grammars default to the shipped ones; point
PAPAYA_FST_V1 / PAPAYA_FST_V2 at a `papaya build-fst` output dir to use your own.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # run from a checkout without installing

from tokenizers import Tokenizer  # noqa: E402

from papaya import fst_tier, regex_tier  # noqa: E402
from papaya._paths import pretrained_dir  # noqa: E402

EVAL = f"{ROOT}/data/eval"
CORPUS = f"{ROOT}/corpus"
COUNTS_JSON = f"{CORPUS}/wordcounts_train_tok.json"
SHIPPED = {"fst": "papaya-v1-16k", "fst2": "papaya-v2-16k", "full": "papaya-regex-16k", "fixed": "bpe-baseline-16k"}
FST_DIR = {1: os.environ.get("PAPAYA_FST_V1") or pretrained_dir("papaya-v1-16k"),
           2: os.environ.get("PAPAYA_FST_V2") or pretrained_dir("papaya-v2-16k")}
FST_VERSION = {"fst": 1, "fst2": 2, "fst3": 2, "litchi": 2, "fst_uni": 1}
UNIGRAM = {"uni", "fst_uni"}      # Unigram LM model instead of BPE; everything else as for BPE
EXTERNAL = {"nepberta": "[SEP]", "sakonii": "</s>", "llama3": "<|end_of_text|>", "qwen25": "<|endoftext|>",
            "gemma2": "<eos>", "xlmr": "</s>", "mbert": "[SEP]", "gpt4o": "<|endoftext|>", "arkios": None,
            "nepalibpe": None}   # published tokenizers and their document separator


def word_counts(full=False):
    """{word: count} of the tokenizer-training sample. The tiers only ask whether a
    word reaches 100, so the shipped frequent-word list is enough; full=True
    (stem induction, the consistency sample's count >= 20) needs
    corpus/wordcounts_train_tok.json (`papaya wordcounts corpus/train_tok.txt ...`)
    or its shipped copy data/corpus/wordcounts_train_tok.json.gz"""
    for p in (COUNTS_JSON, f"{ROOT}/data/corpus/wordcounts_train_tok.json.gz"):
        if os.path.exists(p):
            return fst_tier.load_counts(p)
    if full:
        sys.exit(f"{COUNTS_JSON} missing: run `papaya wordcounts corpus/train_tok.txt {COUNTS_JSON}`")
    return fst_tier.load_counts(os.path.join(FST_DIR[1], "frequent_words.tsv"))


_COUNTS = None


def counts():
    global _COUNTS
    if _COUNTS is None:
        _COUNTS = word_counts()
    return _COUNTS


def fst(mode="hybrid", version=1, **kw):
    return fst_tier.load(mode, version, tokenizer_dir=FST_DIR[version], counts=counts(), **kw)


MORPH = {"core": lambda: regex_tier.load("core"), "full": lambda: regex_tier.load("full"),
         "gated": lambda: regex_tier.load("full", counts()),
         "fst": lambda: fst("hybrid", 1), "fst2": lambda: fst("hybrid", 2),
         "fst3": lambda: fst("hybrid", 2, fallback_profile="strict"),
         "litchi": lambda: fst("hybrid", 2, fallback_profile="strict", order="regex-first"),
         "morf": lambda: __import__("morfessor_tier").load(),
         "fst_uni": lambda: fst("hybrid", 1)}


class Plain:
    """a published tokenizer used like ours: no special tokens added, no tier"""
    def __init__(self, path):
        self.t = Tokenizer.from_file(path)

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False)

    def encode_batch(self, texts):
        return self.t.encode_batch(texts, add_special_tokens=False)

    def __getattr__(self, k):
        return getattr(self.t, k)


def tokenizer_path(variant, vocab):
    """tokenizers/{variant}_{vocab}.json if trained here, else the shipped copy (16k)"""
    p = f"{ROOT}/tokenizers/{variant}_{vocab}.json"
    if not os.path.exists(p) and vocab == 16000 and variant in SHIPPED:
        p = os.path.join(pretrained_dir(SHIPPED[variant]), "tokenizer.json")
    return p


def load_tokenizer(variant, vocab):
    """(tokenizer, tier, eot id, vocab size). Variant x_<name> is tokenizers/external/<name>.json
    with its own separator token as EOT; vocab is then its real size, whatever was asked."""
    if variant.startswith("x_"):
        name = variant[2:]
        tok = Plain(f"{ROOT}/tokenizers/external/{name}.json")
        sep = EXTERNAL[name] or "</w>"
        eot = tok.token_to_id(sep)
        assert eot is not None, (variant, sep)
        return tok, None, eot, tok.get_vocab_size()
    tok = Tokenizer.from_file(tokenizer_path(variant, vocab))
    return tok, MORPH[variant]() if variant in MORPH else None, tok.token_to_id("<|endoftext|>"), vocab


def append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
