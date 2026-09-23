#!/usr/bin/env python3
"""
Fetch the corpus: FineWeb-2 npi_Deva, cut into the samples every script reads.

FineWeb-2 npi_Deva (HuggingFaceFW/fineweb-2, ODC-By). We take the designated
test split whole (49k docs, 69 MB -- nobody trains on it, and Regmi et al.
2026 report their bits-per-byte on it) and one train shard (001, 1.84 GB),
from which fixed-size text samples are cut for tokenizer training and the
small LM. Everything under corpus/ is gitignored; data/corpus/MANIFEST.json
holds the sha256 of every file the paper used, so --check tells you whether
your download is byte-identical to ours.

    python3 scripts/fetch_corpus.py            download (~1.9 GB) + cut samples (~3.5 GB)
    python3 scripts/fetch_corpus.py --check    re-hash against data/corpus/MANIFEST.json
"""
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = f"{ROOT}/corpus"
MANIFEST = f"{ROOT}/data/corpus/MANIFEST.json"           # the paper's hashes
LOCAL_MANIFEST = f"{CORPUS}/MANIFEST.json"              # what this download produced
REPO = "HuggingFaceFW/fineweb-2"
FILES = {"test": "data/npi_Deva/test/000_00000.parquet",
         "train": "data/npi_Deva/train/001_00000.parquet"}
# one train shard, first N docs -- shard order is crawl order, not a random sample
SAMPLES = {"train_tok": ("train", 200_000),   # tokenizer training
           "train_lm": ("train", 400_000),    # small LM (superset of train_tok)
           "test": ("test", None)}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def cut(src, n, out):
    """First n documents' text, one doc per line (newlines inside a doc -> space)."""
    docs = chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for batch in pq.ParquetFile(src).iter_batches(batch_size=2000, columns=["text"]):
            for t in batch.column(0).to_pylist():
                f.write(" ".join(t.split()) + "\n")
                docs += 1; chars += len(t)
                if n and docs >= n:
                    return docs, chars
    return docs, chars


def main():
    if sys.argv[1:] == ["--check"]:
        m = json.load(open(MANIFEST, encoding="utf-8"))
        bad = [p for p, e in m["files"].items() if sha(f"{CORPUS}/{p}") != e["sha256"]]
        print("corpus OK, no drift" if not bad else "DRIFT: " + ", ".join(bad))
        sys.exit(bool(bad))
    os.makedirs(CORPUS, exist_ok=True)
    files = {}
    for split, rel in FILES.items():
        p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=CORPUS)
        files[rel] = {"split": split, "sha256": sha(p), "bytes": os.path.getsize(p),
                      "rows": pq.ParquetFile(p).metadata.num_rows}
        print(rel, files[rel]["rows"], "rows")
    for name, (split, n) in SAMPLES.items():
        out = f"{CORPUS}/{name}.txt"
        docs, chars = cut(f"{CORPUS}/{FILES[split]}", n, out)
        files[f"{name}.txt"] = {"split": split, "docs": docs, "chars": chars,
                                "sha256": sha(out), "bytes": os.path.getsize(out)}
        print(name, docs, "docs", f"{chars/1e6:.1f}M chars")
    json.dump({"repo": REPO, "subset": "npi_Deva", "license": "ODC-By 1.0",
               "why": "Regmi et al. 2026 (arXiv 2608.26449) report bits-per-byte on this "
                      "test split; web-general rather than news-only (IRIIS).",
               "files": files},
              open(LOCAL_MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ref = json.load(open(MANIFEST, encoding="utf-8"))["files"]
    diff = [p for p, e in files.items() if p in ref and ref[p]["sha256"] != e["sha256"]]
    print("wrote", LOCAL_MANIFEST, "-", "identical to the paper's corpus" if not diff else "DIFFERS from the paper's: " + ", ".join(diff))


if __name__ == "__main__":
    main()
