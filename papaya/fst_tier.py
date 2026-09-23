"""
The FST tier. A foma transducer that segments a Nepali word into
[prefix] stem suffix* at akshara boundaries, driven by a stem lexicon and the
affix tables (data/affix_inventory.json), and a Python wrapper that ranks its analyses.

Why an FST and a lexicon, not more regex: the regex tier only sees the string
to the left of a suffix, so it cannot tell a stem from a look-alike (अहि+ले,
भूमि+का, म+कै) and cannot cut vowel-stem verbs at all (दि+ने, हुँ+दा, बनाउ+न)
because "-ने after a halanta" was the only thing separating गर्ने from a noun
like दिने. Here a word is cut only where a *known* stem plus a licensed chain
of allomorphs spells it exactly. The stem lexicon is induced from the training
corpus by paradigm attestation (a verb stem must show >= 4 distinct inflected
slots, a nominal stem must occur standalone and with >= 2 case markers) and
seeded from the Prasain annexes; the suffix chains are the ch3/ch4/ch5 rule
tables realised on the surface (halanta before a consonant-initial verbal
suffix, ँ before -दा on vowel stems, n-insertion in -न्छ, the perfective
X+ए form as the host of -र/-को/-पछि, ...).

Cuts are at akshara boundaries only: a piece never begins with a matra or a
halanta, because a token cannot begin with a combining mark. गर्+एको is
therefore गरे|को and ठिक+ै is ठिकै -- the same convention
data/eval/gold_sheet.tsv is normalised to.

    papaya build-fst --counts counts.json --version 2 --out DIR   # induce stems, write + compile, self-check
    papaya build-fst --from-stems DIR/stems.tsv --version 2 --out DIR   # grammar from a stem list, no corpus
    papaya segment गर्नुपर्ने दिनेहरूलाई ...
Writes DIR/{stems.tsv, segmenter.lexc, fst_tier.json}; the compiled transducer is
cached under ~/.cache/papaya keyed by the lexc's hash (see ensure_bin).

Ranking of the FST's analyses (fst_tier.json records it): a verbal
analysis with a cut or a strong noun wins first (हुँ|दा, but मान्छे not
मा+न्छे), then longest stem (भूमिका over भूमि+का, काका over का+का), then
fewest cuts, then verbal over nominal. Words the FST cannot analyse are left whole ("lex" mode) or
handed to the regex tier ("hybrid" mode) -- the hybrid is the tokenizer's
pre-tokenizer; "lex" is the precision row of the ablation.
"""
import collections
import hashlib
import json
import os
import re
import subprocess
import tempfile

from . import regex_tier
from ._paths import CACHE_DIR, pretrained_dir
from .foma import flookup_bin, foma_bin

H = "्"          # halanta
CB = "ँ"         # chandrabindu
LETTER = re.compile(r"[क-हक़-य़]")
VOWEL_END = re.compile(r"[ा-ौअ-औ]$")   # matra or independent vowel
# induction thresholds (recorded in fst_tier.json)
VERB_SLOTS, VERB_SLOT_MIN, VERB_DISTINCT, FREQ_WHOLE = 4, 3, 3, 100
VERB_SLOT_MIN_SHORT = 20     # a two-letter stem (कि from किन्छ/कियो typos) needs real frequency
# nominal: (min standalone count, min distinct markers) by stem length -- a
# two-letter string is a stem only on strong evidence (ठि+कै, टि+पर, धो+का)
NOUN_MIN = {1: (100, 3), 2: (100, 3), 3: (20, 3)}
NOUN_MIN_DEFAULT, NOUN_MARK_MIN = (3, 2), 2
# a declinable this well attested (मान्छे, खाना, भिडियो) outranks a verbal parse
# of the same string (मा+न्छे, खा+ना, भिडि+यो): lexical evidence over morphotactics.
# It must also pluralise -- a verbal noun takes case (बनाउनको) but not -हरू
STRONG_NOUN = (8, 1000)
# slots nouns never fill: a verb stem must show >= VERB_DISTINCT of them
# (किन/जुन/दिन would otherwise induce कि/जु/दि from -न/-ने alone)
C_DISTINCT = {H + s for s in ("छ", "छन्", "यो", "दा", "दै", "दैन", "थे", "थ्यो")} | {"ेको", "ेर", "िएको", "ियो", "िन्छ"}
V_DISTINCT = {"न्छ", "न्छन्", "यो", "एको", "एर", "ँदा", "ँदै", "ँदैन", "इन्छ", "इयो", "इएको", "ँछ", "ँछन्", "ँथे"}
U_FUSED = {"यो", "एको", "एर", "ए", "एका", "इएको", "इयो", "इन्छ", "इने"}
U_SUF = {CB + s for s in ("छ", "छन्", "छु", "छौं", "दा", "दै", "दैन", "थे", "थ्यो")}

# ---- surface suffix inventories -------------------------------------------
# verbal, from ch4 (paradigm ids in comments); consonant-initial ones attach
# as ्|S to a consonant stem and |S to a vowel stem
V_CONS = {
    "participle_infinitive": ["नु", "ना", "नै"],
    "participle_purposive": ["न"],
    "participle_prospective": ["ने"],
    "tense_nonpast_affirmative": ["छ", "छन्", "छु", "छौ", "छौं", "छस्", "छिन्", "छे", "छेस्", "छ्यौ"],
    "aspect_imperfect+participle_durative": ["दा", "दै", "दो", "दी"],
    "tense_nonpast_negative_set1": ["दैन", "दैनन्", "दैनौ", "दैनौं", "दैनस्", "दिन", "दिनन्", "दिनौ", "दिनौं", "दिनस्", "दैनँ"],
    "mood_potential": ["ला", "लान्", "लास्", "लिस्", "लिन्", "ली"],
    "aspect_past_habitual_affirmative": ["थे", "थें", "थ्यो", "थ्यौ", "थ्यौं", "थिन्", "थिस्", "थी"],
    "aspect_past_habitual_negative": ["दैनथे", "दैनथें", "दैनथ्यो", "दैनथ्यौ", "दैनथ्यौं", "दैनथिन्", "दैनथिस्", "दिनथे", "दिनथ्यौ", "दिनथिस्"],
}
V_CONS_ALL = [s for v in V_CONS.values() for s in v]
# after the perfective form Xे / Xए (participle_perfective + participle_conditional):
PERF_NEXT = {"participle_conjunctive": ["र", "रै"],
             "aspect_inferential_affirmative": ["छ", "छन्", "छु", "छौ", "छौं", "छस्", "छिन्"],
             "tense_past_negative": ["न", "नन्", "नौ", "नौं", "नस्", "नँ"]}
# fused forms that are whole words: absolutive Xी, 1SG past Xें, optative/imperative
C_WHOLE = ["ी", "ें", "ौं", "ौ", "ूँ", "ोस्", "ून्", "ेस्", "ऊ", "ो"]
V_WHOLE = ["ऊँ", "ओस्", "ऊन्", "ऊ", "ओ", "ई", "एँ", "एस्", "औं", "औ"]
# vowel stem: n-insertion in the non-past (Prasain rule 7.3.3), geminate negative set2
V_NPAST = ["न्छ", "न्छन्", "न्छु", "न्छौ", "न्छौं", "न्छस्", "न्छिन्", "न्छे", "न्छेस्", "न्छ्यौ",
           "न्न", "न्नन्", "न्नौ", "न्नौं", "न्नस्", "न्नँ"]
# suppletive हु / जा (verb_suppletive): forms that host the perfective chain or stand alone
SUPPLETIVE = {"हु": {"perf": ["भए"], "whole": ["भयो", "भएँ", "भइ", "भई", "भो", "हो", "होस्", "छ", "थियो", "थिए"],
                     "host": {"हो": ["इन", "इनँ", "इनन्", "इनस्", "इनौ", "इनौं"]}},
              "जा": {"perf": ["गए"], "whole": ["गयो", "गएँ", "गइ", "गई"]}}
# postpositions the ch5 tables lack (found via data/eval/silver_hunspell.tsv);
# not in the affix inventory -- recorded as "extra" in fst_tier.json
EXTRA_POSTP = ["भन्दा", "बिना", "द्वारा", "लगायत", "सम्बन्धी", "झैँ", "झैं"]
# honorific -नुहुन्छ etc: infinitive + a form of हु
HON = ["हुन्छ", "हुन्न", "होस्", "हुन्थ्यो", "हुनेछ", "हुने", "हुँदा", "हुँदै", "हुँदैन", "भयो", "भएको", "भएका", "भएकी", "भएन", "भएर", "भए", "हुन्", "हुनु", "हुन"]
GEN = ["को", "का", "की", "कै"]
BOUND_PRON = sorted(regex_tier.BOUND_STEMS | {"उही", "यही", "त्यही", "जही", "कही", "उनी", "उनै", "यिनी", "तिनी", "आफू", "आफै", "आफैँ"})
NOMINALISED = re.compile(r"(ने|नु|े?को|े?का|े?की|ेकै)$")

# ---- version 2 (Papaya v2) ---------------------------------------------------
# From a scan of the v1 cuts over the 3000 most frequent word types (6 wrong of
# 1005 cut) and the single-akshara stems the induction let in (441 nominal, most
# of them OCR/Latin-letter fragments: क, गी, ने, भो ...). None of these words is
# in data/eval/gold_sheet.tsv.
V2_WHOLE = {"भर्ना", "मकै", "काका", "किन", "खाली", "विकसित", "निको", "सवार", "खुल्ला", "पाली", "लाली",
            "नामा", "चिन", "भाका", "कुवर", "छिपाई", "भोका", "रेको", "एस"}   # lexicalised homographs of stem+suffix
V2_DROP_VERB = {"कि", "चि"}          # किन्छ / चिन्छ are किन् / चिन्; the 2-letter gate let the truncated stems in
V2_ONE_AKSHARA_OK = set(BOUND_PRON) | {"म", "ऊ", "यो", "त्यो", "तँ", "यी", "ती", "बा"}
ONE_AKSHARA = re.compile(LETTER.pattern + r"[\u093e-\u094c]?[\u0901\u0902]?")   # one consonant with at most a matra and a nasal


def load_counts(path):
    """{word: count} from a JSON table (regex_tier.wordcounts) or a two-column
    TSV (frequent_words.tsv, header `word<TAB>count`)"""
    if path.endswith(".json"):
        return json.load(open(path, encoding="utf-8"))
    if path.endswith(".json.gz"):
        import gzip
        return json.load(gzip.open(path, "rt", encoding="utf-8"))
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            out[w] = int(c)
    return out


def annex_stems(path=regex_tier.SEEDS):
    """seed stems from the Prasain 2011 annexes: verbs by type, and
    nouns/pronouns/adjectives/adverbs/numerals as nominal stems.
    papaya/data/annex_seed_stems.tsv is the output of this step over the
    OCR-repaired annex lexicon: an annex07 verb entry of class 1a got -उ
    appended (अघा -> अघाउ), other consonant-final verb bases lost their
    halanta (the grammar adds it back); nominal entries from annexes 2-6 lost a
    final halanta when longer than two code points. Entries with a space, a
    damaged Devanagari form, or no letter were skipped."""
    out = {"V": set(), "N": set()}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            out[c].add(w)
    return out


def induce(counts):
    """stems attested by paradigm in the training sample"""
    csuf = {H + s for s in V_CONS_ALL} | {"ेको", "ेर", "े", "ेका", "ेकी", "िएको", "ियो", "िन्छ", "िने", "िन", "्यो", "ी"}
    vsuf = set(V_CONS_ALL) | set(V_NPAST) | {"एको", "एर", "ए", "एका", "यो", "इएको", "इयो", "इन्छ", "इने", "ँदा", "ँदै", "ँदैन"}
    vc, vv = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    nom = collections.defaultdict(collections.Counter)
    markers = ["ले", "को", "का", "की", "मा", "लाई", "हरू", "हरु", "बाट", "सँग", "देखि", "सम्म", "कै", "मै"]
    for word, c in counts.items():
        if c < 2:
            continue
        for s in csuf:
            if word.endswith(s) and len(word) > len(s) + 1:
                st = word[:-len(s)]
                if LETTER.search(st[-1]):
                    vc[st][s] += c
        for s in vsuf:
            if word.endswith(s) and len(word) > len(s) + 1:
                st = word[:-len(s)]
                if VOWEL_END.search(st):
                    vv[st][s] += c
                    # polysyllabic -उ stem: बनायो/बनाएको/बनाइ are बनाउ + यो/एको/इ with the उ dropped
                    if st.endswith("ा") and len(st) >= 2 and s in U_FUSED:
                        vv[st + "उ"][s] += c
        for s in U_SUF:
            if word.endswith(s) and len(word) > len(s) + 1 and word[-len(s) - 1] == "उ":
                vv[word[:-len(s)]][s] += c
        for s in markers:
            if word.endswith(s) and len(word) > len(s):
                st = word[:-len(s)]
                if LETTER.search(st) and not st.endswith(H):
                    nom[st][s] += c
    verbs = {}
    for kind, table, distinct in (("C", vc, C_DISTINCT), ("V", vv, V_DISTINCT)):
        for st, slots in table.items():
            m = VERB_SLOT_MIN if len(st) > 2 else VERB_SLOT_MIN_SHORT
            n = sum(1 for c in slots.values() if c >= m)
            nd = sum(1 for s, c in slots.items() if c >= m and s in distinct)
            inf = slots.get(H + "नु" if kind == "C" else "नु", 0)
            if n >= VERB_SLOTS and nd >= VERB_DISTINCT and inf >= 2 and len(st) >= 2:
                verbs[st] = (kind, n)
    for f in SUPPLETIVE.values():          # हो, भए are forms of हु, not stems
        for s in f["whole"] + f["perf"] + list(f.get("host", {})):
            verbs.pop(s, None)
    # a compound stem (गरिराख, आइपुग, हुनसक) is V1 form + vector verb
    v1 = ("ि", "इ", "नु", "न", "ए", "े")
    for st in [s for s in verbs]:
        for i in range(2, len(st) - 1):
            a, b = st[:i], st[i:]
            if b in verbs and (a in verbs or any(a.endswith(e) and (a[:-len(e)] in verbs or a[:-len(e)] + "उ" in verbs) for e in v1)):
                del verbs[st]
                break
    nouns, plural = {}, set()
    for st, m in nom.items():
        n = sum(1 for c in m.values() if c >= NOUN_MARK_MIN)
        alone, markers = NOUN_MIN.get(len(st), NOUN_MIN_DEFAULT)
        if n >= markers and counts.get(st, 0) >= alone:
            nouns[st] = n
            if m.get("हरू", 0) + m.get("हरु", 0) >= NOUN_MARK_MIN:
                plural.add(st)
    return verbs, nouns, plural


def vkind(stem):
    """C: consonant base (halanta added by the grammar); U: polysyllabic
    -उ stem (आउ, बनाउ: ँ non-past, X[:-1]यो past); V: other vowel stems"""
    if not VOWEL_END.search(stem):
        return "C"
    if stem.endswith("उ"):        # independent उ after a vowel: आउ, बनाउ, पिउ (not हु, धु: matra ु)
        return "U"
    return "V"


def build_stems(counts, drop=frozenset(), version=1):
    """(stem, class V|N, kind, evidence) rows: annex seeds + corpus-induced stems"""
    seeds = annex_stems()
    verbs, nouns, plural = induce(counts)
    if version >= 2:
        for v in V2_DROP_VERB:
            verbs.pop(v, None)
    for s in seeds["V"]:
        verbs.setdefault(s, ("annex", 0))
    vstems = {s.rstrip(H) for s in verbs}
    # a stem that is न + another verb stem is the negative prefix, not a lexeme
    vstems = {s for s in vstems if not (s.startswith("न") and s[1:] in vstems)}
    # nominalised participles (गर्ने, गरेको, दिनु) chain with case markers and so
    # look declinable; they belong to the verb, whose analysis is preferred
    def is_participle(s):
        m = NOMINALISED.search(s)
        if not m:
            return False
        base = s[:m.start()]
        return base in vstems or base.rstrip(H) in vstems or (m.group().startswith("े") and base in vstems) \
            or (base.endswith("ए") and base[:-1] in vstems)
    nstems = ({s for s in nouns if not is_participle(s)} | seeds["N"] | set(BOUND_PRON)) - drop
    nstems = {s for s in nstems if s not in vstems or vkind(s) != "C"}
    if version >= 2:   # a lone akshara is a stem only as a pronoun or a strong noun (बा), never a fragment
        strong = {s for s in nstems if nouns.get(s, 0) >= STRONG_NOUN[0] and counts.get(s, 0) >= STRONG_NOUN[1] and s in plural}
        nstems = {s for s in nstems if not s.startswith(H) and (not ONE_AKSHARA.fullmatch(s) or s in V2_ONE_AKSHARA_OK or s in strong)}
    rows = [(s, "V", vkind(s), verbs.get(s, verbs.get(s + H, ("?", 0)))[1]) for s in sorted(vstems)]
    rows += [(s, "N", "S" if nouns.get(s, 0) >= STRONG_NOUN[0] and counts.get(s, 0) >= STRONG_NOUN[1] and s in plural else "",
              nouns.get(s, 0)) for s in sorted(nstems)]
    return rows


# ---- lexc ------------------------------------------------------------------
def esc(s):
    return s.replace("%", "%%").replace(":", "%:").replace(";", "%;").replace("!", "%!").replace("<", "%<").replace(">", "%>")


def write_lexc(rows, path):
    rt = regex_tier.load("full")
    core = [s for s in rt.order if rt.group[s] == "core" and s not in ("ही",)]
    postp = [s for s in rt.order if rt.group[s] == "postp"] + EXTRA_POSTP
    case = [s for s in core if s not in GEN and s != "हरू"]
    L = ["Multichar_Symbols +V +N +S", "",
         "LEXICON Root", " Prefix ;", " VStem ;", " NStem ;", "",
         "LEXICON Prefix", "न| VStem ;", ""]
    # ---- verbal
    L += ["LEXICON VStem"]
    for s, cls, kind, _ in rows:
        if cls == "V":
            L.append(f"{esc(s)}+V {'V' + kind} ;")
            if kind == "U":       # बनाउ drops its उ before यो/ए/इ: बनायो, बनाए, बनाइ
                L.append(f"{esc(s[:-1])}+V VUfused ;")
    L += ["", "LEXICON V2", " VStem ;", ""]      # compound: V1 form | vector verb
    # consonant stem X: X्|S, Xे (perfective), Xि (passive -> vowel stem), whole fused forms
    L += ["LEXICON VC"]
    for s in V_CONS_ALL:
        L.append(f"{H}|{s} {'Inf' if s in ('नु',) else 'Purp' if s == 'न' else 'Prosp' if s == 'ने' else 'VEnd'} ;")
    L += [f"{H}नु InfUnit ;", f"{H}ना InfUnit ;", f"{H}न| V2 ;", "े Perf ;", "ि VV ;", "ि| V2 ;"] + [f"{f} VEnd ;" for f in C_WHOLE] + [""]
    # vowel stem (खा दि हु गरि): |S, |न्छ, ँ|दा, |यो, Xए perfective, Xइ passive
    L += ["LEXICON VV"]
    for s in V_CONS_ALL:
        if s.startswith("द"):
            L.append(f"{CB}|{s} VEnd ;")
        else:
            L.append(f"|{s} {'Inf' if s == 'नु' else 'Purp' if s == 'न' else 'Prosp' if s == 'ने' else 'VEnd'} ;")
    L += [f"|{s} VEnd ;" for s in V_NPAST] + ["नु InfUnit ;", "ना InfUnit ;", "न| V2 ;", "|यो VEnd ;", "ए Perf ;", "इ VV ;", "इ| V2 ;"] + [f"{f} VEnd ;" for f in V_WHOLE] + [""]
    # polysyllabic -उ stem (आउ बनाउ): ँ before छ/दा/थे, and X[:-1] hosts यो/ए/इ
    L += ["LEXICON VU"]
    for s in V_CONS_ALL:
        if s[0] in "छदथ":
            L.append(f"{CB}|{s} VEnd ;")
        else:
            L.append(f"|{s} {'Inf' if s == 'नु' else 'Purp' if s == 'न' else 'Prosp' if s == 'ने' else 'VEnd'} ;")
    L += ["नु InfUnit ;", "ना InfUnit ;", "न| V2 ;"] + [f"{CB}{f} VEnd ;" for f in ("ूँ", "ौं", "ौ")] + ["", "LEXICON VUfused",
          "यो VEnd ;", "ए Perf ;", "इ VV ;", "इ| V2 ;", "ऊ VEnd ;", "ओस् VEnd ;", "ऊन् VEnd ;", "ई VEnd ;", ""]
    L += ["LEXICON Perf", " VEnd ;", " NEnd ;"] + [f"|{s} VEnd ;" for v in PERF_NEXT.values() for s in v] + [""]
    # a word-final infinitive is cut (लड्|नु); before a vector verb, an
    # honorific हु form or a genitive it is one unit with its stem (गर्नु|पर्|ने,
    # गर्नु|हुन्छ, गर्नु|को) -- the gold sheet's 14:2 verdict, and UD's lemma
    L += ["LEXICON Inf", " VEnd ;", ""]
    L += ["LEXICON InfUnit", "| V2 ;", "|ले VEnd ;"] + [f"|{g} GenEnd ;" for g in GEN] + [f"|{h} VEnd ;" for h in HON] + [""]
    L += ["LEXICON Purp", " VEnd ;"] + [f"|{g} GenEnd ;" for g in GEN] + [""]
    L += ["LEXICON Prosp", " NEnd ;"] + [f"|{s} VEnd ;" for s in V_CONS["tense_nonpast_affirmative"]] + [""]
    L += ["LEXICON VEnd", " # ;", ""]
    # suppletive हु / जा
    L += ["LEXICON SuppStem"]
    for stem, f in SUPPLETIVE.items():
        L += [f"{p}+V Perf ;" for p in f["perf"]] + [f"{w}+V VEnd ;" for w in f["whole"]]
        L += [f"{h}+V|{s} VEnd ;" for h, ss in f.get("host", {}).items() for s in ss]
    L[L.index("LEXICON VStem")] = "LEXICON VStem\n SuppStem ;"
    L.append("")
    # ---- nominal: [stem] [हरू] [postp | case]* [GEN]
    L += ["LEXICON NStem"]
    for s, cls, kind, _ in rows:
        if cls == "N":
            L.append(f"{esc(s)}+{kind or 'N'} NEnd ;")
    L += ["", "LEXICON NEnd", " # ;", " Plural ;", " CaseOrPostp ;", " Gen ;", "",
          "LEXICON Plural", "|हरू NEnd2 ;", "|हरु NEnd2 ;", "",
          "LEXICON NEnd2", " # ;", " CaseOrPostp ;", " Gen ;", "",
          "LEXICON CaseOrPostp"]
    L += [f"|{s} AfterCase ;" for s in case] + [f"|{s} AfterPostp ;" for s in postp]
    L += ["", "LEXICON AfterCase", " # ;", " Gen ;", " PostpOnly ;", "",
          "LEXICON PostpOnly"] + [f"|{s} AfterPostp ;" for s in postp]
    L += ["", "LEXICON AfterPostp", " # ;", " Gen ;", " CaseOnce ;", "",
          "LEXICON CaseOnce"] + [f"|{s} AfterCaseOnce ;" for s in case]
    L += ["", "LEXICON AfterCaseOnce", " # ;", " Gen ;", "",
          "LEXICON Gen"] + [f"|{g} GenEnd ;" for g in GEN]
    L += ["", "LEXICON GenEnd", " # ;", " PostpOnly ;", " CaseOnce ;", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


def compile_fst(lexc, binpath):
    """compile a segmenter.lexc into a foma binary (seconds)"""
    script = f"""read lexc {lexc}
define Seg;
regex Seg .o. ["|" -> 0, "+V" -> 0, "+N" -> 0, "+S" -> 0];
save stack {binpath}
quit
"""
    fp = binpath[:-len(".bin")] + ".foma"
    open(fp, "w", encoding="utf-8").write(script)
    out = subprocess.run([foma_bin(), "-f", fp], capture_output=True, text=True).stdout
    if not os.path.exists(binpath):
        raise RuntimeError(out[-2000:])
    return out


def ensure_bin(lexc):
    """path of the compiled transducer for `lexc`, compiling it on first use into
    CACHE_DIR under the lexc's content hash (so an edited lexc is never served a
    stale binary, and nothing is written next to the package)"""
    h = hashlib.sha256(open(lexc, "rb").read()).hexdigest()[:20]
    binpath = os.path.join(CACHE_DIR, f"segmenter-{h}.bin")
    if not os.path.exists(binpath):
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=CACHE_DIR)
        compile_fst(os.path.abspath(lexc), os.path.join(tmp, "s.bin"))
        os.replace(os.path.join(tmp, "s.bin"), binpath)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    return binpath


# ---- runtime -----------------------------------------------------------------
class FstTier:
    """segmenter over the compiled transducer; analyses are cached per word type"""

    def __init__(self, binpath, mode="hybrid", counts=None, fallback=None, whole=frozenset(),
                 order="fst-first", fallback_profile="full", gate_fallback=False):
        """binpath: compiled transducer (ensure_bin(lexc)). mode: hybrid (unanalysed
        words go to the regex tier unless frequent) or lex (left whole).
        counts: {word: count}; only whether a word reaches FREQ_WHOLE matters, so
        frequent_words.tsv is enough. None means no word counts as frequent.
        order: fst-first (shipped) or regex-first (the fallback's cut wins when it
        makes one); fallback_profile: regex profile of the hybrid fallback;
        gate_fallback: run the fallback's attestation gate over self.counts.
        Defaults are the shipped behaviour -- every reported result keys on them"""
        self.bin = binpath
        self.mode = mode
        self.whole = whole
        self.order = order
        self.counts = counts if counts is not None else {}
        if fallback is None and mode == "hybrid":
            fallback = regex_tier.RegexTier(json.load(open(regex_tier.OUT, encoding="utf-8"))["affixes"],
                                            fallback_profile, self.counts if gate_fallback else None)
        self.fallback = fallback
        self.cache = {}
        self.raw = {}

    def lookup(self, words):
        words = [w for w in dict.fromkeys(words) if w not in self.cache]
        if not words:
            return
        p = subprocess.run([flookup_bin(), self.bin], input="\n".join(words) + "\n",
                           capture_output=True, text=True, encoding="utf-8")
        res = collections.defaultdict(list)
        for line in p.stdout.splitlines():
            if "\t" in line:
                w, a = line.split("\t", 1)
                if a != "+?":
                    res[w].append(a)
        for w in words:
            self.raw[w] = res.get(w, [])
            self.cache[w] = [w] if w in self.whole else self.rank(w, self.raw[w])

    @staticmethod
    def parse(a):
        """'न|गर+V|ने' -> (tag, pieces); the tag sits right after the stem"""
        tag = "+V" if "+V" in a else "+S" if "+S" in a else "+N"
        return tag, a.replace(tag, "").split("|")

    @staticmethod
    def rank(word, analyses):
        """a verbal analysis with a cut beats an uncut weak nominal one (हुँ|दा,
        not the declinable look-alike हुँदा) but not a strong noun (मान्छे, not
        मा+न्छे); then longest stem (भूमिका, not भूमि+का), fewest cuts, verbal"""
        best = None
        for a in analyses:
            tag, pieces = FstTier.parse(a)
            stem_i = 1 if pieces[0] == "न" and len(pieces) > 1 else 0
            key = (not ((tag == "+V" and len(pieces) > 1) or tag == "+S"), -len(pieces[stem_i]), len(pieces), tag != "+V")
            if best is None or key < best[0]:
                best = (key, pieces)
        assert best is None or "".join(best[1]) == word, (word, best)   # cuts only, never rewrites
        return best[1] if best else None

    def split(self, word):
        if self.order == "regex-first" and self.fallback is not None:
            p = self.fallback.split(word)
            if len(p) > 1:
                return p
        if word not in self.cache:
            self.lookup([word])
        pieces = self.cache[word]
        if pieces:
            return pieces
        if self.fallback is not None and self.counts.get(word, 0) < FREQ_WHOLE:
            return self.fallback.split(word)
        return [word]

    def prime(self, texts):
        words = set()
        for t in texts:
            words.update(regex_tier.WORD.findall(t))
        self.lookup(list(words))

    def mark(self, text, sep="⁠"):
        out, last = [], 0
        for m in regex_tier.WORD.finditer(text):
            out.append(text[last:m.start()])
            out.append(sep.join(self.split(m.group())))
            last = m.end()
        out.append(text[last:])
        return "".join(out)


def load(mode="hybrid", version=2, tokenizer_dir=None, counts=None, **kw):
    """FstTier over a tokenizer dir's segmenter.lexc and frequent_words.tsv
    (default: the shipped papaya-v{version}-16k). version 2 adds the whole-word
    homograph list; kw go to FstTier (order, fallback_profile, gate_fallback)."""
    d = tokenizer_dir or pretrained_dir(f"papaya-v{version}-16k")
    if counts is None and os.path.exists(os.path.join(d, "frequent_words.tsv")):
        counts = load_counts(os.path.join(d, "frequent_words.tsv"))
    return FstTier(ensure_bin(os.path.join(d, "segmenter.lexc")), mode=mode, counts=counts,
                   whole=V2_WHOLE if version >= 2 else frozenset(), **kw)


def selfcheck(tier):
    """every dataset form must round-trip; a few known cuts must hold"""
    cases = {"गर्नुपर्ने": ["गर्नु", "पर्", "ने"], "लड्नु": ["लड्", "नु"], "दिने": ["दि", "ने"], "हुँदा": ["हुँ", "दा"],
             "नगर्ने": ["न", "गर्", "ने"], "गरेको": ["गरे", "को"], "भूमिका": ["भूमिका"],
             "सरकारले": ["सरकार", "ले"], "अहिलेसम्म": ["अहिले", "सम्म"], "बनाउन": ["बनाउ", "न"],
             "उनीहरूलाई": ["उनी", "हरू", "लाई"], "बस्छन्": ["बस्", "छन्"]}
    tier.prime([" ".join(cases)])
    bad = {w: tier.split(w) for w, want in cases.items() if tier.split(w) != want}
    for w in cases:
        assert "".join(tier.split(w)) == w
    return bad


def build(counts, outdir, version=2, counts_label=None):
    """induce the stem lexicon from word counts, write + compile the transducer
    into outdir (stems.tsv, segmenter.lexc, fst_tier.json); returns the
    self-check failures ({} when every check holds)"""
    os.makedirs(outdir, exist_ok=True)
    rows = build_stems(counts, version=version)
    # pass 1: an inflected form that itself takes markers looks declinable
    # (उनीहरू+लाई, अहिलेसम्म+को); it is dropped when the FST analyses it over a
    # nominal stem that is more frequent than it (भूमिका 16875 vs भूमि 4889 stays)
    with tempfile.TemporaryDirectory() as tmp:
        write_lexc(rows, f"{tmp}/pass1.lexc")
        compile_fst(f"{tmp}/pass1.lexc", f"{tmp}/pass1.bin")
        t = FstTier(binpath=f"{tmp}/pass1.bin", mode="lex", counts=counts)
        nom = [r[0] for r in rows if r[1] == "N"]
        t.lookup(nom)
    drop = set()
    bound = set(BOUND_PRON)
    strength = {r[0]: r[2] for r in rows if r[1] == "N"}
    strong = lambda w: strength.get(w) == "S"
    for w in nom:
        for a in t.raw.get(w, []):
            tag, pieces = t.parse(a)
            if len(pieces) < 2:
                continue
            stem = pieces[1] if pieces[0] == "न" else pieces[0]
            # a verb form (हुँदा, आएपछि, गर्नुपर्ने) -- except the purposive Xन,
            # whose noun homographs (दिन, हुन) are too common to give up
            if tag == "+V" and pieces[-1] != "न" and not strong(w):
                drop.add(w); break
            if tag != "+V" and (counts.get(stem, 0) > counts.get(w, 0) or stem in bound):
                drop.add(w); break
    print(f"pass 1: {len(drop)} declinable-looking inflected forms dropped ({', '.join(sorted(drop, key=lambda s: -counts.get(s, 0))[:8])} ...)")
    rows = build_stems(counts, drop, version)
    write_stems(rows, f"{outdir}/stems.tsv")
    nv = sum(r[1] == "V" for r in rows)
    print(f"stems: {nv} verbal, {len(rows) - nv} nominal -> {outdir}/stems.tsv")
    bad = build_from_rows(rows, outdir, version)
    json.dump({"version": version,
               "stems": {"verbal": nv, "nominal": len(rows) - nv},
               "induction": {"corpus_counts": counts_label, "verb_slots": VERB_SLOTS,
                             "verb_slot_min_count": VERB_SLOT_MIN, "verb_slot_min_count_2letter": VERB_SLOT_MIN_SHORT,
                             "verb_distinct_slots": VERB_DISTINCT,
                             "noun_min_by_stem_length": {str(k): v for k, v in NOUN_MIN.items()},
                             "noun_min_default": NOUN_MIN_DEFAULT, "noun_marker_min_count": NOUN_MARK_MIN,
                             "frequent_whole_min": FREQ_WHOLE, "strong_noun_min_markers_count": STRONG_NOUN},
               "extra_postpositions_not_in_dataset": EXTRA_POSTP,
               "ranking": "verbal analysis with a cut or strong noun first, then longest stem, then fewest cuts, then verbal over nominal",
               "modes": {"lex": "unanalysed words left whole", "hybrid": "unanalysed words with count < frequent_whole_min go to regex tier full"},
               "akshara_boundary_only": True,
               **({"v2": {"whole_words": sorted(V2_WHOLE), "dropped_verb_stems": sorted(V2_DROP_VERB),
                          "one_akshara_nominal_stems_allowed": "pronouns, strong nouns"}} if version >= 2 else {})},
              open(f"{outdir}/fst_tier.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return bad


def write_stems(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("stem\tclass\tkind\tevidence\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")


def read_stems(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return [(s, c, k, int(e)) for s, c, k, e in (l.rstrip("\n").split("\t") for l in f)]


def build_from_rows(rows, outdir, version=2):
    """write outdir/segmenter.lexc from stem rows, compile it, self-check"""
    os.makedirs(outdir, exist_ok=True)
    write_lexc(rows, f"{outdir}/segmenter.lexc")
    tier = FstTier(ensure_bin(f"{outdir}/segmenter.lexc"), mode="lex",
                   whole=V2_WHOLE if version >= 2 else frozenset())
    bad = selfcheck(tier)
    print("self-check:", "ok" if not bad else bad)
    return bad
