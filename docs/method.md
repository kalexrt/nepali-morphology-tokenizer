# How Papaya works

Papaya decides morpheme boundaries with linguistic rules first, and only then
lets BPE merge inside the pieces. A byte-level BPE trained on frequency alone
knows nothing about Nepali morphology. It learns सरकारले ("by the government")
as one token and सरकार ("government") as another, so the model has to learn the
stem twice. For a rarer noun it produces cuts like सरका|रले, which straddle the
stem/suffix boundary.

The pipeline has three stages:

```
text ──► morphological tier ──► U+2060 at every cut ──► byte-level BPE (cuts are hard boundaries)
           FST (lexicon + grammar)                         tokenizers: Split(U+2060) → word regex → ByteLevel
           └─ regex tier for words the FST cannot analyse
```

## 1. Marking

`tier.mark(text)` goes through the text word by word. A word is a run of
Devanagari letters and combining marks (`regex_tier.WORD`). The tier returns
each word's pieces, rejoined with **U+2060 WORD JOINER**; every other character
is left untouched.

```
सरकारले विद्यार्थीहरूलाई दिने भएको छ
सरकार⁠ले विद्यार्थी⁠हरू⁠लाई दि⁠ने भए⁠को छ        (⁠ = U+2060)
```

U+2060 is a zero-width format character that does not occur in the corpus. The
tier only inserts cuts and never rewrites a character; `"".join(pieces) == word`
is asserted.

## 2. Pre-tokenization and BPE (`papaya/bpe.py`)

The `tokenizers` pipeline is a `pre_tokenizers.Sequence` of three stages:

1. `Split(U+2060, behavior="removed")`. The marker becomes a hard pre-token
   boundary and is deleted, so no BPE merge can cross it and it never appears in
   the token stream.
2. `Split(regex, behavior="isolated")` with a GPT-2-style word regex,
   `[\p{L}\p{M}]+`. It keeps combining marks (matras, halanta, anusvara, Unicode
   category M) attached to their consonant. GPT-2's original `\p{L}+` treats
   them as separators and shatters a Nepali word into about 4.4 tokens; the
   `broken` control reproduces that bug.
3. `ByteLevel`: each pre-token is mapped to its UTF-8 bytes. The alphabet is
   bytes, so there is no unknown token and any string decomposes.

The vocabulary is trained with `BpeTrainer` (256-byte initial alphabet, one
special token `<|endoftext|>`) on tier-marked text. Decoding reproduces the
input byte for byte: `decode(encode(mark(x))) == x` is asserted at build time.

**The marking is not inside `tokenizer.json`.** A vocabulary trained on marked
text but applied to unmarked text behaves like plain BPE with a vocabulary fitted
to a different distribution, which is worse than either. Use the `Papaya` class,
which keeps the tier and the vocabulary together, or call `.mark()` yourself.

## 3. The regex tier (`papaya/regex_tier.py`)

It cuts a word right to left by longest match against a fixed suffix list. The
list comes from the affixes in `data/affix_inventory.json` whose `tier` is
`regex`, meaning they concatenate in the spelling without changing the stem.

**Profiles**:
- `core` (20 suffixes): -हरू; ले लाई को का की कै मा मै बाट बाटै सँग सँगै सित
  सितै तिर तिरै देखि; -ही -सुकै.
- `full` (164): core, plus 80 postpositions and their emphatic ◌ै forms (जस्तै,
  भित्रै), plus the halanta-stem verbal suffixes (-नु -ने -ना -न -नै, and the
  potential -ला -लान् -लास् -लिस् -लिन् -ली).
- `strict`: `full` minus 19 low-precision suffixes, with stem length counted in
  aksharas and ले/कै/का barred after a stem ending in ि ी े ै.

**Cut rules**, the part that keeps precision:
- The stem keeps at least one full letter.
- Nominal cuts never follow a halanta. In शर्मा, अर्को and जिल्ला the halanta
  forms a conjunct; these are not stem + मा/को/ला.
- Case markers need a stem of two or more letters, or the pronoun म.
- Verbal cuts come only after a halanta, and never across a geminate (विभिन्न and
  हल्ला stay whole).
- Suffix order, right to left, is `[stem] [हरू] [postposition | case]* [genitive | verbal]`,
  with at most 4 cuts.

The regex tier sees only the string to the left of a suffix. So it cannot tell a
stem from a look-alike (अहि+ले, भूमि+का, म+कै), and it cannot cut vowel-stem
verbs (दि+ने, हुँ+दा, बनाउ+न) at all. That is why the FST tier exists.

## 4. The FST tier (`papaya/fst_tier.py`)

The FST tier cuts a word **only where a known stem plus a licensed chain of
suffixes spells it exactly.**

### Stem lexicon (`stems.tsv`)

Stems are induced from word counts over 200k documents by paradigm attestation,
seeded from the annex lexicon of Prasain (2011) (`papaya/data/annex_seed_stems.tsv`):

- **Verb stem**: needs at least 4 distinct inflected slots, each seen at least 3
  times (20 for a two-letter stem). At least 3 of those slots must be ones nouns
  never fill (छ, छन्, यो, दा, दै, दैन, थे …), and the infinitive (-नु) must be
  attested. There are three stem shapes: consonant (गर्; the grammar adds the
  halanta), vowel (दि, हु, खा), and polysyllabic -उ (आउ, बनाउ).
- **Nominal stem**: must occur on its own, with at least 2 distinct case markers
  (stricter for 1–3-letter strings). A **strong noun** (at least 8 markers, at
  least 1000 standalone uses, and takes -हरू) outranks a verbal parse of the same
  string (मान्छे, खाना).
- Suppletive forms of हु and जा (भए, गए, हो, भयो …) are listed explicitly. न + a
  verb is the negative prefix, not a lexeme. Compound stems (गरिराख, आइपुग) are
  decomposed when both halves are stems.
- **Two passes.** The transducer is compiled once. Every nominal "stem" that the
  FST itself analyses as an inflected form of a more frequent, bound or verbal
  stem is then dropped (उनीहरू, अहिलेसम्म, हुँदा), and the transducer is
  recompiled.

The v1 lexicon has 720 verbal and 18,568 nominal stems; v2 has 718 and 18,537.
Every threshold is recorded in the tokenizer dir's `fst_tier.json`. The
thresholds are **absolute counts**, so a larger corpus loosens all of them;
rescale them if you induce from much more text.

### The grammar (`segmenter.lexc`)

A foma lexc grammar. Its lower side is the surface word; its upper side is the
same string with `|` at each cut and a tag after the stem.

```
Root → Prefix (न|) → VStem | NStem
VStem → consonant / vowel / -उ stem classes
   consonant: ्|S for each consonant-initial suffix; े perfective; ि passive; fused forms (ी, ें, ौं, ोस्…)
   vowel:     |S; |न्छ… (n-insertion); ँ|दा; |यो; ए perfective; इ
   -उ stem:   ँ|छ/दा/थे; the उ-less form hosts यो/ए/इ (बनायो = बनाउ+यो)
   perfective → |र |रै | inferential छ… | past negative न… | nominal chain (गरे|को)
   infinitive → end, or one unit before a vector verb / honorific / genitive (गर्नु|पर्|ने)
   compound: V1 form | vector verb
NStem → [ |हरू ] → [ case | postposition ]* → [ को का की कै ]
```

The suffix chains are the verbal, pronominal and postpositional rule tables
realised on the surface. Seven postpositions missing from those tables (भन्दा,
बिना, द्वारा, लगायत, सम्बन्धी, झैँ, झैं) are added and declared in
`fst_tier.json`.

**Cuts fall on akshara boundaries only.** A piece never starts with a vowel sign
or a halanta, because a token cannot begin with a combining mark. So गर्+एको
surfaces as गरे|को, and ठिक+ै stays whole as ठिकै.

foma compiles the grammar in a few seconds. `ensure_bin()` caches the binary
under `~/.cache/papaya`, keyed by the grammar's hash.

### Choosing an analysis

`flookup` returns every analysis of a word; `rank()` picks one:
1. A verbal analysis with a cut beats an uncut, weak nominal one (हुँ|दा), but
   not a strong noun (मान्छे, not मा+न्छे).
2. Then the longest stem wins (भूमिका, not भूमि+का).
3. Then the fewest cuts, then verbal over nominal.

A word with no analysis depends on the mode:
- `lex`: left whole.
- `hybrid`, the default and what every Papaya tokenizer uses: handed to the
  regex tier, unless the word occurs at least 100 times in the training corpus
  (`frequent_words.tsv`). A frequent word the FST cannot analyse is usually a
  lexicalised look-alike (निकै, अहिले), so it stays whole.

### Version 2

Version 2 is the recommended version. It was built from a scan of the v1 cuts
over the 3,000 most frequent word types and changes three things:
- a whole-word list of 19 lexicalised homographs (भर्ना, मकै, काका, किन …);
- single-akshara nominal stems only for pronouns and strong nouns;
- the truncated verb stems कि and चि dropped.

Gold-sheet precision rises from 0.965 to 0.975 at the same recall.

### Known limits

- **Homographs.** Beyond the v2 list, a form that is both a word and stem +
  suffix gets the more frequent analysis.
- **Rare verbs.** Verbs below the induction threshold go unanalysed (नसड्ने).
- **Compounds.** V1+V2 compounds whose V1 is not a listed form are missed
  (देखापर्न).
- **No derivation.** Derivational morphology is not cut. The gold sheet marks no
  derivational cuts, so there is nothing to evaluate it against.
- **Higher fertility.** Morphological pre-splitting raises tokens per word by
  about 16 % at matched vocabulary (1.40 → 1.63 at 16k), because hard boundaries
  forbid merges BPE would otherwise buy with frequency.
