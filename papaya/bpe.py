"""
Byte-level BPE with the morphological tier as a hard pre-token boundary.

The tier runs as text preprocessing: tier.mark() puts a WORD JOINER (U+2060)
at each morph cut, and the tokenizer's first pre-tokenizer stage is
Split(U+2060, removed). No merge can cross a cut, and decoding reproduces the
original text byte for byte.

  broken   GPT-2 word regex \\p{L}+          -- splits Devanagari vowel signs off their base (control)
  fixed    [\\p{L}\\p{M}]+ (marks attached)  -- the plain-BPE baseline
  marked   fixed + Split(U+2060)            -- every Papaya variant

model="unigram" trains a Unigram LM on the same pre-tokenization instead of BPE
(the paper's `uni` and `fst_uni` rows).
"""
import itertools

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SEP = "⁠"
EOT = "<|endoftext|>"
# GPT-2's pattern with the letter class swapped; \p{M} keeps Devanagari vowel signs on their base
PATTERN = {
    "broken": r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    "fixed":  r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+",
}


def new_tokenizer(vocab, marked=True, pattern="fixed", model="bpe"):
    """(untrained tokenizer, trainer). marked: split on U+2060 first (a tier is used)"""
    tok = Tokenizer(models.Unigram() if model == "unigram" else models.BPE(byte_fallback=False))
    stages = []
    if marked:
        stages.append(pre_tokenizers.Split(SEP, behavior="removed"))
    stages.append(pre_tokenizers.Split(Regex(PATTERN[pattern]), behavior="isolated"))
    stages.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(stages)
    tok.decoder = decoders.ByteLevel()
    if model == "unigram":
        # every byte symbol is in the initial alphabet, so <unk> should never be emitted
        trainer = trainers.UnigramTrainer(vocab_size=vocab, special_tokens=[EOT, "<unk>"], unk_token="<unk>",
                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                          show_progress=False)
    else:
        trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
    return tok, trainer


def lines(paths, n=None, tier=None):
    """the first n lines (one document per line) of the given files, marked by tier"""
    it = itertools.chain.from_iterable(open(p, encoding="utf-8") for p in paths)
    for line in itertools.islice(it, n):
        yield tier.mark(line) if tier else line


def train(paths, vocab, tier=None, pattern="fixed", docs=None, model="bpe"):
    """train a tokenizer on the first `docs` lines of `paths`, marked by `tier`"""
    tok, trainer = new_tokenizer(vocab, marked=tier is not None, pattern=pattern, model=model)
    if hasattr(tier, "prime"):      # one flookup pass over every word type, not one per word
        tier.prime(lines(paths, docs))
    tok.train_from_iterator(lines(paths, docs, tier), trainer=trainer)
    return tok
