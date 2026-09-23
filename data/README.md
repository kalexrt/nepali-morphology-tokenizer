# Data

Two resources released with the paper, plus the silver sets and corpus
manifest used in its evaluation. Licences: [`LICENSE`](LICENSE) (CC BY 4.0,
except the third-party silver sets).

| file | what | rows |
|---|---|---|
| `affix_inventory.json` / `.tsv` | the affix dataset: Nepali inflectional and derivational affixes with segmentation tier, provenance and review status | 556 |
| `eval/gold_sheet.tsv` | gold morpheme boundaries, native-speaker annotated | 607 words |
| `eval/silver_ud.tsv` | stem boundaries from UD_Nepali-BK lemmas | 85 |
| `eval/silver_hunspell.tsv` | stem boundaries from Hunspell-generated form/lemma pairs | 3,223 |
| `corpus/MANIFEST.json` | sha256 of the FineWeb-2 files and samples the paper used | — |
| `corpus/wordcounts_train_tok.json.gz` | word-type counts of the 200k-document tokenizer corpus: the input to FST stem induction and Morfessor | 272,922 types |
| `morfessor_baseline.bin.gz` | the Morfessor Baseline model behind the paper's `morf` rows (pickled, gzip); `scripts/morfessor_tier.py` loads it | — |

## Affix dataset (`affix_inventory.*`)

The affix tables of Prasain (2011), *A Computational Analysis of Nepali
Morphology*, chapters 3–6 (pronouns, verbs, postpositions and case markers,
derivation), re-derived in our own notation and merged into one table, one row
per unique (affix, function, word class). The dissertation's PDF text layer
corrupts Devanagari: it drops the pre-base vowel sign ि, splits glyphs, and loses
conjuncts. Every affix form was therefore re-read from OCR of the rendered page
or corrected by a native speaker, using the IPA column (which survives intact) as
the anchor. `affix_source` records which channel each form came from.

Fields (TSV columns are a subset):

| field | meaning |
|---|---|
| `chapter`, `section_ref`, `paradigm_id`, `subgroup` | where the rule sits in the dissertation, and our paradigm id (100 paradigms) |
| `word_class` | verb 182 · derivational_affix 123 · postposition 99 · pronoun 61 · adverb 33 · particle 16 · … |
| `affix`, `affix_norm`, `affix_ipa` | the affix in Devanagari (bound vowel signs written with a dotted circle, `◌ी`), normalised, and IPA |
| `function` | gloss of what the affix marks (`PL/COLL`, `+ERG/INS`, `V->N`, …) |
| `surface_example`, `example_word`, `example_ipa` | an example form. Only partly repaired: not used by the tokenizer |
| `stem_alternation` | whether attaching the affix changes the stem |
| **`tier`** | `regex` (the affix concatenates without touching the stem: a string cut can find it) or `fst` (it needs a transducer: stem change, vowel deletion, agreement). 286 regex / 270 fst. Judges *generation*; see `papaya/regex_tier.py` for the nine case-marker rows that are `fst` by generation but cleanly segmentable |
| `confidence`, `needs_human_check` | our confidence in the row, and whether it was flagged for human review |
| `affix_source` | `pdf-text-layer` 400 · `chandra-ocr-2` 84 · `human` 72 |
| `devanagari_verified`, `human_confirmed` | form checked against IPA (548) / confirmed by the native-speaker reviewer (220) |
| `in_ch7_fst`, `ch7_fst_match`, `ch7_fst_files` | whether the dissertation's own chapter-7 transducer lists the affix (cross-check only; that code is not included here) |
| `in_apertium`, `apertium_classes` | whether apertium-ne lists it (cross-check only) |
| `risk` | our triage score used to order the human review |

`papaya build-regex` compiles this table into the regex tier's suffix list
(`papaya/data/regex_tier_v1.json`). The build is deterministic and reproduces
the shipped file exactly.

## Gold morpheme-boundary set (`eval/gold_sheet.tsv`)

607 word types sampled from the FineWeb-2 `npi_Deva` test split, drawn from
types with at least 3 occurrences in the first 20k test documents. Every
inflectional or clitic cut is marked with `|`. As far as we know, it is the
first human-annotated morpheme-boundary set for Nepali.

| column | meaning |
|---|---|
| `rank`, `word`, `freq` | row id, the word as it occurs in the corpus, its count in the 20k-document sample |
| `bucket` | why it was sampled: natural 249 (frequency-weighted random), verbal 119, gate_blocked 79, unsplit 60, postp 60, emphatic 40. Ignored in scoring |
| `proposal_shown` | 1 if the annotators saw `PROPOSAL` next to the word (422 rows), 0 if it was withheld (185 rows, chosen at random) |
| `PROPOSAL` | what the annotators were shown: the regex tier's (`full` profile, an earlier version) segmentation; empty where it was withheld |
| **`GOLD`** | the scored segmentation, normalised so every piece concatenates back to `word` and every cut falls on an akshara boundary (a piece never starts with a vowel sign or halanta): `गरे|को`, not `गर्|एको` |
| `GOLD_RAW` | the consensus as written, before normalisation (17 rows differ) |

The buckets over-represent hard cases (words the regex tier cut, left whole,
or was blocked from cutting), so F1 on this set is not an estimate for running
text.

**Annotation.** Seven native speakers of Nepali, unpaid volunteers who gave
consent; the authors are among them. The words were divided up. A word was
settled once two or three annotators agreed on it; harder words and
disagreements went to all seven; words that stayed ambiguous or had only a
narrow majority were removed (610 → 607; ranks 43, 72 and 233 are missing).
Annotators consulted each other, so there are no independent per-annotator
labels and no agreement coefficient.

Conventions: an infinitive before a vector verb stays one unit
(`गर्नु|पर्|ने`). So do the perfective and passive forms (`गरे|को`, `लिए|र`,
`देखिए|पछि`). The negative prefix is cut (`न|गर्|ने`), as is the first verb of a
compound (`आइ|पुग्|ने`). The sheet allowed a derivational `+` marker; nobody
used it.

**Anchoring check.** Because a proposal was shown on 70 % of rows,
`scripts/gold_anchoring.py` scores every system separately on the blind and
the shown rows. The regex tier's own agreement with the gold is 0.773 F1 on
blind rows and 0.812 on shown rows (permutation p = 0.11): at most a small
pull toward the baseline, which works against Papaya, not for it.

Limitations: the authors are among the annotators, and no inter-annotator
agreement was measured.

## Silver sets

- `silver_ud.tsv`: UD_Nepali-BK v2.18 (CC BY-SA 4.0), the 85 test tokens whose
  manually annotated lemma is a proper prefix of the form, giving one boundary,
  `lemma|rest`. Real annotation, but small and nominal only.
- `silver_hunspell.tsv`: 3,223 `form, lemma` pairs from dpakpdl/NepaliLemmatizer
  (MIT). They were generated from the `ne_NP` Hunspell affix file (the `rule`
  column is the Hunspell rule id), so the set is synthetic. It marks the stem
  boundary only and leaves the suffix stack unsplit. **Report recall on it, not
  precision.**

The raw sources are in `eval/external/` with their licences.

## Corpus

FineWeb-2 (`HuggingFaceFW/fineweb-2`, subset `npi_Deva`, ODC-By 1.0): the whole
test split plus train shard 001. `scripts/fetch_corpus.py` downloads it and cuts
three samples:

- `train_tok.txt`: 200k documents, for tokenizer training and word counts;
- `train_lm.txt`: 400k documents, for the LM;
- `test.txt`: the whole test split.

`--check` verifies the hashes against `corpus/MANIFEST.json`.
