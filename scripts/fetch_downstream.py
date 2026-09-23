#!/usr/bin/env python3
"""
Downstream task data for finetune_tok.py, into downstream/ (gitignored).

  cc    mteb/NepaliNewsClassification (Hugging Face): 3 news topics, 2,048 train /
        1,495 test paragraphs. Written as downstream/CC_{train,test}.tsv, one
        `label<TAB>text` row per document, whitespace collapsed. Fetched here.
  ner   Nep-gLUE NER and POS (the NepBERTa benchmark: https://nepberta.github.io/,
  pos   distributed as Google-Drive CSVs with columns sentence_id, words, labels).
        We do not redistribute them: download NER.csv and POS.csv from the
        Nep-gLUE page and put them in downstream/. finetune_tok.py makes its own
        seed-0 80/10/10 split.

    python3 scripts/fetch_downstream.py
"""
import os

from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/downstream"


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("mteb/NepaliNewsClassification")
    for part in ("train", "test"):
        with open(f"{OUT}/CC_{part}.tsv", "w", encoding="utf-8") as f:
            for r in ds[part]:
                f.write(f"{r['label']}\t{' '.join(r['text'].split())}\n")
        print(f"{OUT}/CC_{part}.tsv: {len(ds[part])} docs")
    for t in ("NER", "POS"):
        if not os.path.exists(f"{OUT}/{t}.csv"):
            print(f"missing {OUT}/{t}.csv -- download it from Nep-gLUE (see this script's docstring)")


if __name__ == "__main__":
    main()
