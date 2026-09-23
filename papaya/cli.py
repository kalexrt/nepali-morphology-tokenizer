"""
papaya segment  WORD...                         morph cuts of words
papaya encode   "TEXT"                          tokens + ids
papaya wordcounts CORPUS.txt COUNTS.json        word-count table of a corpus (one document per line)
papaya build-fst --counts COUNTS.json --out DIR [--version 2]
                                                induce stems, write + compile the FST into DIR
papaya build-fst --from-stems STEMS.tsv --out DIR [--version 2]
                                                grammar from an existing stem list (no corpus)
papaya train --text CORPUS.txt --out DIR [--tier fst|regex|none] [--vocab 16000]
                                                train the BPE on tier-marked text -> a tokenizer dir
papaya build-regex                              rebuild the regex tier's suffix table from data/affix_inventory.json
papaya list                                     shipped tokenizers
"""
import argparse
import json
import os
import sys
import time

from . import _paths, fst_tier, regex_tier


def cmd_segment(a):
    from .tokenizer import Papaya
    if a.tier == "regex":
        t = regex_tier.load(a.profile)
    else:
        t = Papaya.from_pretrained(a.tokenizer).tier
        if t is None:
            sys.exit(f"{a.tokenizer} has no morphological tier")
        if hasattr(t, "prime"):
            t.prime([" ".join(a.words)])
    for w in a.words:
        print(w, "->", " | ".join(t.split(w)))


def cmd_encode(a):
    from .tokenizer import Papaya
    tok = Papaya.from_pretrained(a.tokenizer)
    text = " ".join(a.text)
    print("tokens:", " ".join(tok.tokenize(text)))
    ids = tok.encode(text)
    print("ids:   ", ids)
    assert tok.decode(ids) == text


def write_frequent(counts, path, min_count=fst_tier.FREQ_WHOLE):
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tcount\n")
        for w, c in sorted(((w, c) for w, c in counts.items() if c >= min_count), key=lambda x: (-x[1], x[0])):
            f.write(f"{w}\t{c}\n")


def cmd_build_fst(a):
    os.makedirs(a.out, exist_ok=True)
    if a.from_stems:
        rows = fst_tier.read_stems(a.from_stems)
        if os.path.abspath(a.from_stems) != os.path.abspath(os.path.join(a.out, "stems.tsv")):
            fst_tier.write_stems(rows, os.path.join(a.out, "stems.tsv"))
        bad = fst_tier.build_from_rows(rows, a.out, a.version)
    else:
        if not a.counts:
            sys.exit("need --counts (from `papaya wordcounts`) or --from-stems")
        counts = fst_tier.load_counts(a.counts)
        bad = fst_tier.build(counts, a.out, a.version, counts_label=os.path.basename(a.counts))
        write_frequent(counts, os.path.join(a.out, "frequent_words.tsv"))
    print(f"wrote {a.out}/segmenter.lexc (compiled into {_paths.CACHE_DIR})")
    if bad:
        print("WARNING: self-check cuts differ (expected on a small or unusual corpus):", bad)


def cmd_train(a):
    from . import bpe
    from .tokenizer import Papaya, make_tier
    config = {"name": os.path.basename(os.path.normpath(a.out)), "tier": a.tier, "vocab": a.vocab,
              "pretokenizer": "fixed", "trained_on": [os.path.basename(p) for p in a.text], "train_docs": a.docs}
    if a.tier == "fst":
        if not os.path.exists(os.path.join(a.out, "segmenter.lexc")):
            sys.exit(f"{a.out}/segmenter.lexc missing: run `papaya build-fst --out {a.out}` first "
                     f"(or copy segmenter.lexc/frequent_words.tsv from a shipped tokenizer)")
        config.update(fst_version=a.version, mode="hybrid", order="fst-first", fallback_profile=a.fallback_profile)
    elif a.tier == "regex":
        config.update(profile=a.profile, attested=False)
    os.makedirs(a.out, exist_ok=True)
    tier = make_tier(config, a.out)
    t0 = time.time()
    tok = bpe.train(a.text, a.vocab, tier, docs=a.docs)
    config["train_seconds"] = round(time.time() - t0)
    p = Papaya(tok, tier, config, a.out)
    p.save(a.out)
    # sanity: lossless round trip + tokens per word on the first lines
    sample = list(bpe.lines(a.text, 200))
    ids = p.encode_batch(sample)
    assert all(p.decode(i) == s for i, s in zip(ids, sample)), "round-trip failed"
    words = sum(len(s.split()) for s in sample)
    print(f"wrote {a.out}/tokenizer.json ({tok.get_vocab_size()} tokens, {config['train_seconds']} s); "
          f"round-trip ok; {sum(map(len, ids)) / max(words, 1):.2f} tokens/word on the first {len(sample)} lines")


def cmd_build_regex(a):
    affixes = regex_tier.build(a.inventory)
    n, split = regex_tier.selfcheck(affixes)
    regex_tier.write(affixes, a.inventory)
    print(f"self-check OK ({n} dataset forms round-trip, {split} split); wrote {regex_tier.OUT}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="papaya", description="Morphology-guided BPE tokenization for Nepali")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment", help="morph cuts of words")
    s.add_argument("words", nargs="+")
    s.add_argument("--tokenizer", default="papaya-v2-16k", help="shipped name or tokenizer dir")
    s.add_argument("--tier", choices=["tokenizer", "regex"], default="tokenizer",
                   help="regex: the regex tier alone (no foma needed)")
    s.add_argument("--profile", default="full", choices=["core", "full", "strict"])
    s.set_defaults(f=cmd_segment)

    s = sub.add_parser("encode", help="tokens and ids of a text")
    s.add_argument("text", nargs="+")
    s.add_argument("--tokenizer", default="papaya-v2-16k")
    s.set_defaults(f=cmd_encode)

    s = sub.add_parser("wordcounts", help="word -> count table of a corpus (count >= 5)")
    s.add_argument("corpus")
    s.add_argument("out")
    s.set_defaults(f=lambda a: regex_tier.wordcounts(a.corpus, a.out))

    s = sub.add_parser("build-fst", help="induce the stem lexicon and compile the FST")
    s.add_argument("--counts", help="word counts (.json from `papaya wordcounts`, .json.gz, or word<TAB>count .tsv)")
    s.add_argument("--from-stems", help="use this stems.tsv instead of inducing one")
    s.add_argument("--out", required=True)
    s.add_argument("--version", type=int, default=2, choices=[1, 2])
    s.set_defaults(f=cmd_build_fst)

    s = sub.add_parser("train", help="train a BPE vocabulary on tier-marked text")
    s.add_argument("--text", nargs="+", required=True, help="training text, one document per line")
    s.add_argument("--out", required=True, help="tokenizer dir (for --tier fst: the dir build-fst wrote)")
    s.add_argument("--tier", default="fst", choices=["fst", "regex", "none"])
    s.add_argument("--vocab", type=int, default=16000)
    s.add_argument("--docs", type=int, default=None, help="use only the first N lines")
    s.add_argument("--version", type=int, default=2, choices=[1, 2], help="fst tier version")
    s.add_argument("--fallback-profile", default="full", choices=["core", "full", "strict"])
    s.add_argument("--profile", default="full", choices=["core", "full", "strict"], help="regex tier profile")
    s.set_defaults(f=cmd_train)

    s = sub.add_parser("build-regex", help="rebuild the regex tier's suffix table")
    s.add_argument("--inventory", default=regex_tier.INV)
    s.set_defaults(f=cmd_build_regex)

    s = sub.add_parser("list", help="shipped tokenizers")
    s.set_defaults(f=lambda a: print("\n".join(_paths.available())))

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
