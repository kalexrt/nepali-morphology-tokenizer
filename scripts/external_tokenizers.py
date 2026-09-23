#!/usr/bin/env python3
"""
External rows: the same intrinsic numbers (fertility on the FineWeb-2 test
split, gold-sheet boundary F1, morphological consistency) for published
tokenizers that Nepali text actually meets -- three Nepali-specific ones and
the multilingual vocabularies of open LLMs. tokenizer.json files come from the
HF hub into tokenizers/external/ (`--fetch`; NepBERTa ships only vocab.txt and
is wrapped as WordPiece). bpb / downstream rows for x_<name> come from
train_lm.py / finetune_tok.py --variant x_<name>.

    python3 scripts/external_tokenizers.py --fetch      # download into tokenizers/external/
    python3 scripts/external_tokenizers.py [--test-docs 5000]
Appends to results/external.jsonl.
"""
import argparse
import itertools
import json
import os

from tokenizers import Tokenizer

import common
import morph_consistency as mc
import score_boundaries as sb
from common import ROOT
EXT = {  # name: (hub repo, family)
    "sakonii": ("Sakonii/distilbert-base-nepali", "nepali sentencepiece"),
    "nepberta": ("NepBERTa/NepBERTa", "nepali wordpiece"),
    "nepalibpe": ("Aananda-giri/NepaliBPE", "nepali bpe"),
    "arkios": ("sajalregmi4/arkios-tokenizer", "nepali byte bpe"),
    "mbert": ("bert-base-multilingual-cased", "multilingual wordpiece"),
    "xlmr": ("xlm-roberta-base", "multilingual sentencepiece"),
    "gemma2": ("unsloth/gemma-2-2b", "llm sentencepiece"),
    "llama3": ("unsloth/Llama-3.1-8B", "llm byte bpe"),
    "qwen25": ("Qwen/Qwen2.5-7B", "llm byte bpe"),
    "gpt4o": ("Xenova/gpt-4o", "llm byte bpe"),
}


class NoSpecial:
    """Tokenizer wrapper: never add [CLS]/[SEP]/<s>, so counts and offsets match ours."""
    def __init__(self, path):
        self.t = Tokenizer.from_file(path)
        self.unk = self.t.token_to_id("[UNK]") if self.t.token_to_id("[UNK]") is not None else self.t.token_to_id("<unk>")

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False)

    def encode_batch(self, texts):
        return self.t.encode_batch(texts, add_special_tokens=False)

    def decode(self, ids):
        return self.t.decode(ids)

    def get_vocab_size(self):
        return self.t.get_vocab_size()


def fetch():
    """tokenizer.json of every EXT repo -> tokenizers/external/<name>.json. Some repos
    need you to accept their licence on the Hub (and `huggingface-cli login`) first."""
    from huggingface_hub import hf_hub_download
    os.makedirs(f"{ROOT}/tokenizers/external", exist_ok=True)
    for name, (repo, _) in EXT.items():
        out = f"{ROOT}/tokenizers/external/{name}.json"
        if os.path.exists(out):
            continue
        try:
            if name == "nepberta":     # vocab.txt only: rebuild as a cased WordPiece that keeps vowel signs
                from tokenizers.implementations import BertWordPieceTokenizer
                vocab = hf_hub_download(repo, "vocab.txt")
                BertWordPieceTokenizer(vocab, lowercase=False, strip_accents=False).save(out)
            else:
                import shutil
                shutil.copy(hf_hub_download(repo, "tokenizer.json"), out)
            print(name, "->", out)
        except Exception as e:          # noqa: BLE001 -- report and go on to the next repo
            print(name, "FAILED:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-docs", type=int, default=5000)
    ap.add_argument("--fetch", action="store_true", help="download the tokenizers and exit")
    a = ap.parse_args()
    if a.fetch:
        return fetch()
    raw = list(itertools.islice(open(f"{common.CORPUS}/test.txt", encoding="utf-8"), a.test_docs))
    words = sum(len(l.split()) for l in raw)
    nbytes = sum(len(l.encode()) for l in raw)
    gold = sb.load_set("gold_sheet")
    samples = {"gold": mc.gold_sample(), "corpus": mc.corpus_sample()}
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    out = open(f"{ROOT}/results/external.jsonl", "a", encoding="utf-8")
    for name, (repo, family) in EXT.items():
        path = f"{ROOT}/tokenizers/external/{name}.json"
        if not os.path.exists(path):
            print(name, "missing", path); continue
        tok = NoSpecial(path)
        encs = tok.encode_batch(raw)
        ntok = sum(len(e.ids) for e in encs)
        nunk = sum(e.ids.count(tok.unk) for e in encs) if tok.unk is not None else 0
        rec = {"name": name, "repo": repo, "family": family, "vocab": tok.get_vocab_size(),
               "test_docs": a.test_docs, "tokens_per_word": round(ntok / words, 4),
               "bytes_per_token": round(nbytes / ntok, 3), "unk_rate": round(nunk / ntok, 5)}
        bsys = sb.TokSys(path); bsys.tok = tok
        rec["boundary_gold"] = sb.score(bsys, gold)
        csys = mc.System(None, None, path); csys.tok = tok
        for k, sample in samples.items():
            rec[f"consistency_{k}"] = mc.score(csys, sample)
        print(f"{name:10s} {rec['vocab']:7d}  fert {rec['tokens_per_word']:.3f}  unk {rec['unk_rate']:.4f}  "
              f"bound F1 {rec['boundary_gold']['F1']:.3f}  cons gold {rec['consistency_gold']['F1']:.3f} "
              f"corpus {rec['consistency_corpus']['F1']:.3f}")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
