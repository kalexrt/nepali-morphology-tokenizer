# Papaya: morphology-guided tokenization for Nepali

Code, data and pretrained tokenizers for *Preserving Morphemes: Morphology-Guided
Pre-Tokenization for Nepali*.

Byte-level BPE knows nothing about Nepali morphology. It learns सरकारले ("by the
government") and सरकार ("government") as unrelated tokens, and it cuts other
words off the stem/suffix boundary: लेख्नु ("to write") becomes लेख|्नु, a
token starting with a halanta, where the morphemes are लेख्|नु. **Papaya** fixes this before
BPE runs. A rule-based segmenter, a finite-state transducer with a regex
fallback built from the affix tables of Prasain (2011), marks the morpheme
boundaries in the text. BPE is then trained with those boundaries as hard
pre-token boundaries, so no merge can cross a stem/affix cut.

```
सरकारले विद्यार्थीहरूलाई छात्रवृत्ति दिने भएको छ
सरकार|ले  विद्यार्थी|हरू|लाई  छात्रवृत्ति  दि|ने  भए|को  छ      ← morpheme cuts; BPE works inside each piece
```

Decoding is lossless: `decode(encode(x)) == x`, byte for byte.

## Results (16k vocabulary, same data, same model)

| | plain BPE | **Papaya v1** | **Papaya v2** |
|---|---|---|---|
| tokens per word | 1.40 | 1.63 | 1.63 |
| morpheme-boundary F1, gold set (607 words) | 0.22 | 0.86 | **0.86** |
| stem-token consistency F1 (gold inflected forms) | 0.31 | 0.90 | 0.90 |
| bits per byte, 17M GPT, equal epochs | 0.497 | 0.481 (−3.3 %) | **0.481** |
| bits per byte, equal optimizer steps | 0.486 | 0.481 (−1.1 %) | — |
| NER span-F1 (Nep-gLUE, 5 seeds) | 0.736 (0.755 at equal steps) | 0.764 | **0.766** |
| NER span-F1, entities with a word unseen in training | 0.508 (0.529 at equal steps) | 0.592 | **0.606** |
| POS macro-F1 | 0.929 | 0.930 | 0.930 |
| news-topic classification macro-F1 | 0.972 | 0.975 | 0.975 |

Papaya v1 is the variant in the paper's main tables. v2 is the same design with
a homograph list and a stricter stem lexicon. It is the recommended tokenizer.

How to read these numbers:
- **The cost is fertility.** Papaya uses 16 % more tokens per word, because hard
  boundaries forbid merges BPE would otherwise buy with frequency.
- **Most of the language-model gain is extra optimizer steps.** At equal steps
  the gain is 1–1.6 % across 8k–32k vocabularies; at equal epochs it is 2–5 %.
  An unsupervised Morfessor segmentation under the same BPE matches it at equal
  steps (0.4802 vs 0.4809 bits per byte): the grammar's own contribution is
  boundary accuracy (segmenter F1 0.96 vs 0.47), not language-model loss.
- **NER improves on unseen entities; the easier tasks do not move.** Against
  plain BPE trained for the same number of optimizer steps, overall NER gains
  +0.9 points, which is not significant (p = 0.07), but entities containing a
  word unseen in training gain +6.3 points (95 % interval [+1.7, +11.4]).
  POS and topic classification are at ceiling.

All numbers: [`results/ablation_table.md`](results/ablation_table.md),
[`results/significance.md`](results/significance.md).

## Install

```bash
git clone https://github.com/kalexrt/nepali-morphology-tokenizer && cd nepali-morphology-tokenizer
pip install -e .
```

The FST tier needs [foma](https://fomafst.github.io/) (`foma` and `flookup` on
the `PATH`):

```bash
sudo apt install foma-bin      # Debian / Ubuntu
brew install foma              # macOS
```

or build it from [source](https://github.com/mhulden/foma). Set
`PAPAYA_FOMA_BIN=/dir/with/binaries` if they are not on the `PATH`. Without
foma, `papaya-regex-16k` (regex tier only) and `bpe-baseline-16k` still work.
On the first use of an FST tokenizer the grammar is compiled once (a few
seconds) into `~/.cache/papaya`.

## Quickstart

```python
from papaya import Papaya

tok = Papaya.from_pretrained("papaya-v2-16k")
text = "सरकारले विद्यार्थीहरूलाई छात्रवृत्ति दिने भएको छ"

tok.segment("विद्यार्थीहरूलाई")   # ['विद्यार्थी', 'हरू', 'लाई']
tok.tokenize(text)                # tokens as strings; none crosses a morpheme cut
ids = tok.encode(text)            # marks morpheme cuts, then BPE
assert tok.decode(ids) == text
tok.encode_batch([text, text])   # faster for many documents: one FST call for all word types
tok.backend                       # the underlying tokenizers.Tokenizer
```

The shipped tokenizers: `papaya-v2-16k` (recommended), `papaya-v1-16k`,
`papaya-regex-16k` (no foma needed), and `bpe-baseline-16k` (plain BPE, for
comparison). All were trained on the same 100k FineWeb-2 Nepali documents.

> Use `Papaya.encode`, not `tok.backend.encode`. The morpheme marking is text
> preprocessing that `tokenizer.json` cannot store. Without it, the vocabulary
> sees text it was not trained on and behaves worse than plain BPE.

From the command line:

```bash
papaya segment गर्नुपर्ने उनीहरूलाई भूमिका       # गर्नु | पर् | ने   उनी | हरू | लाई   भूमिका
papaya encode "सरकारले निर्णय गरेको छ"
papaya segment --tier regex केटाहरूलाई           # regex tier only, no foma
```

## Train your own tokenizer

You need a Nepali text corpus with one document per line. The FST tier's stem
lexicon is learned from the corpus's word counts, so it adapts to your domain.

```bash
# 1. word counts of the corpus
papaya wordcounts my_corpus.txt counts.json

# 2. induce the stem lexicon, write + compile the transducer (needs foma)
papaya build-fst --counts counts.json --version 2 --out my_tokenizer/

# 3. train BPE on the marked text
papaya train --text my_corpus.txt --out my_tokenizer/ --vocab 32000
```

```python
tok = Papaya.from_pretrained("my_tokenizer/")
```

Options:
- To reuse our stem lexicon instead of inducing one, run
  `papaya build-fst --from-stems papaya/pretrained/papaya-v2-16k/stems.tsv --out my_tokenizer/`
  and copy `frequent_words.tsv` from the same directory.
- For a regex-only tokenizer (no foma), use `papaya train --tier regex`; for a
  plain-BPE baseline on the same data, use `--tier none`.

Stem induction uses absolute count thresholds tuned on 200k documents (about
400M characters). They are recorded in `fst_tier.json` in the output directory;
rescale them if your corpus is much larger or smaller. Training each shipped
16k tokenizer on 100k documents took 2–3 minutes.

## Repository

```
papaya/              the package: regex_tier.py, fst_tier.py (stem induction, lexc grammar, runtime),
                     bpe.py, tokenizer.py (Papaya), cli.py
papaya/pretrained/   the four shipped tokenizers (tokenizer.json, config, grammar, stem lexicon)
data/                the affix dataset, the gold morpheme-boundary set, silver sets, corpus manifest
scripts/             everything needed to reproduce the paper (see docs/reproduce.md)
results/             every number in the paper, as append-only JSON lines + the two generated tables
docs/                method.md (how the tiers work), reproduce.md (paper table -> command)
tests/               pytest; FST tests are skipped when foma is missing
```

## Data

Two resources are released with the paper; see [`data/README.md`](data/README.md):

- **Affix dataset** (`data/affix_inventory.*`): 556 Nepali affixes with function,
  IPA, segmentation tier (regex vs FST), provenance and review status. They are
  re-derived from the rule tables of Prasain (2011); the Devanagari was repaired
  by OCR and by a native speaker, because the source PDF's text layer corrupts it.
- **Gold morpheme boundaries** (`data/eval/gold_sheet.tsv`): 607 Nepali word
  types from web text, annotated by native speakers.

## Reproducing the paper

[`docs/reproduce.md`](docs/reproduce.md) lists the command behind every table:
- corpus download with hash check;
- tokenizer training, boundary F1 and consistency (CPU);
- the 17M-parameter LM sweep and fine-tuning (one GPU; a run peaks at ~24 GB at 16k
  vocab and ~34 GB at 32k);
- the ten published tokenizers.

## Limitations

- **Scale.** Everything is measured with a 17M-parameter model; whether the
  bits-per-byte gain survives at larger scale is untested.
- **Annotation.** The gold set is the adjudicated consensus of seven
  native-speaker volunteers, the authors among them; annotators conferred, so
  no inter-annotator agreement coefficient exists.
- **Segmentation errors.** The FST's errors are homographs (a word that is also
  stem + suffix), verbs too rare to induce, and compound verbs; derivational
  morphology is not segmented.
- **Nep-gLUE.** Its NER/POS text is pre-split at word level, which leaves the
  segmenter less to do than on running text.

## Citation

```bibtex
@article{shrestha2026preserving,
  title  = {Preserving Morphemes: Morphology-Guided Pre-Tokenization for Nepali},
  author = {Shrestha, Kalash and Pradhan, Nikhil},
  year   = {2026},
}
```

Please also cite the source of the affix tables:

```bibtex
@phdthesis{prasain2011,
  author = {Prasain, Balaram},
  title  = {A Computational Analysis of Nepali Morphology: A Model for Natural Language Processing},
  school = {Tribhuvan University, Kirtipur, Kathmandu},
  year   = {2011},
  url    = {https://ojs.ub.uni-konstanz.de/jsal/dissertations/diss-balaram.pdf},
}
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data, lexicons and vocabularies: CC BY 4.0,
except the third-party silver sets, which keep their own licences
([`data/LICENSE`](data/LICENSE)). The tokenizers and lexicons were trained on
FineWeb-2 (ODC-By 1.0).
