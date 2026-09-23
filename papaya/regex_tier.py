"""
The regex tier. Splits a Nepali word into [stem] + [suffix]* using only
affixes that concatenate in the orthography without touching the stem, taken
from the affix inventory (data/affix_inventory.json, tier=regex). The compiled
suffix table ships as papaya/data/regex_tier_v1.json; build() regenerates it.

    केटाहरूलाई -> केटा + हरू + लाई        longest match, right to left

Three profiles (the paper ablates all three):
  core    nominal clitics: -हरू, case markers (ले लाई को का की मा बाट सँग सित
          तिर देखि + their ◌ै emphatic forms), pronominal -ही / -सुकै
  full    core + the ch5 adverbial postpositions (घरभित्र, त्यसपछि ...) + the
          consonant-initial verbal suffixes that attach without stem change
          (participles -नु -ने -ना -न -नै, potential -ला -लान् -लास् -लिस् -लिन् -ली)
  strict  full minus the low-precision affixes (STRICT_DROP), stems counted in
          aksharas not code points, and ले/कै/का barred after a stem in ि ी े ै
          (STRICT_FRONT). Precision-first: in the fst variant the regex tier is
          the fallback, so every wrong cut it makes is baked into the vocabulary

The `tier` field judges GENERATION (can the allomorph be derived by
concatenation?). Segmentation is weaker: घरमै -> घर + मै is a clean cut even
though मै itself is मा + ◌ै with the ◌ा deleted, and रामको -> राम + को is a
clean cut even though को/का/की is selected by the head noun. So the nine
case-marker rows the affix inventory puts in the FST tier for those reasons are admitted
here as whole units (SEGMENTABLE_FST); `tier` itself is not changed.

What is deliberately NOT here, and why (these are tier judgements):
  * free words listed as tier=regex (adverbs, conjunctions, interjections,
    particles पनि/त/नै): written as separate words, nothing to split
  * ◌ै emphatic on its own and vowel-initial suffixes (औ, ए, इ, आलो, इया):
    fuse with the stem's final consonant as a matra -- not concatenative
  * tense_nonpast_negative_set2 (खा+न -> खान्न): geminates, not concatenative
  * ch6 prefixes (प्र-, वि-, अ-): stripping a prefix by regex is a coin flip
  * ch6 derivational suffixes (-वान, -हट, -ली): lexicalised, high collision
  * aux paradigms (छ-/थि-/हो- forms): closed class, BPE learns them whole

    papaya segment --tier regex WORD...         show splits
    papaya build-regex                          rebuild papaya/data/regex_tier_v1.json + self-check
    papaya wordcounts corpus.txt counts.json    word-count table (attestation gate, FST induction)
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INV = os.path.join(REPO, "data", "affix_inventory.json")          # present in a source checkout
OUT = os.path.join(HERE, "data", "regex_tier_v1.json")             # shipped with the package
SEEDS = os.path.join(HERE, "data", "annex_seed_stems.tsv")

PROFILES = {
    "core": ["postp_plural_hʌru", "case_markers_no_emph", "case_markers_with_emph",
             "pron_indef_from_interrog", "pron_indef_from_rel", "pron_dem_remaining"],
    "full": ["adverbial_postp_with_emph", "adverbial_postp_no_emph",
             "participle_infinitive", "participle_purposive", "participle_prospective",
             "mood_potential"],
}
PROFILES["full"] = PROFILES["core"] + PROFILES["full"]
PROFILES["strict"] = PROFILES["full"]   # same paradigms; STRICT_DROP / STEM_RULE_STRICT do the rest
# tier=fst rows admitted as whole-unit suffixes: generation is non-concatenative
# (emphatic ◌ै deletes the base vowel; GEN agrees with the head) but the cut
# between host and marker is.
SEGMENTABLE_FST = {"case_markers_with_emph": ["बाटै", "मै", "सँगै", "सितै", "तिरै",
                                              "को", "का", "की", "कै"]}
# a suffix must start with a full letter; matra-initial ones (◌ै, ◌ी) fuse with the stem
LETTER = re.compile(r"^[ऄ-हक़-य़]")
# a stem must keep at least one full letter; matras/halant alone are not a word
STEM_OK = re.compile(r"[ऄ-हक़-य़]")
# per-group cut conditions on the stem, tuned on FineWeb-2 npi_Deva text:
#  * a halanta at the cut means the marker's consonant forms a conjunct with
#    the stem (शर्-मा, अर्-को, जिल्-ला, बाध्-यता) -- word-internal, so nominal
#    clitics and postpositions never cut there
#  * verbal suffixes (-न -ने -ला) are the reverse: they attach to halanta stems
#    (गर्-ने, बस्-नु) and collide with nouns otherwise (तीन, कुन, काला); a
#    geminate (बन्-न, सम्पन्-न, जिल्-ला) is not cut because विभिन्न/अन्न/हल्ला
#    outnumber भन्न/बन्न/चल्ला -- known ceiling
#  * postpositions are content words too (साथ, बीच, पार, वार) and need a stem
#    of two or more code points; two-letter nouns ending in a marker (ढोका,
#    काका) still split -- the tier's other known ceiling
# the halanta rule drops vowel-stem verbs (खानु, जाने, हुने); the FST tier
# (fst_tier.py) is what handles them
HALANT = "\u094d"
GEN = {"को", "का", "की", "कै"}
GROUP = {"participle_infinitive": "verbal", "participle_purposive": "verbal",
         "participle_prospective": "verbal", "mood_potential": "verbal",
         "adverbial_postp_with_emph": "postp", "adverbial_postp_no_emph": "postp"}


# stems the attestation gate must not veto because they never stand alone:
# ch3 oblique/emphatic pronoun stems (उसले, कसको, त्यही) and perfective
# participle stems in -ए (गरेको, लागिएको, गर्नुभएको)
BOUND_STEMS = {"उस", "उसै", "उन", "उनै", "यस", "यसै", "यिन", "यिनै", "त्यस", "त्यसै",
               "तिन", "तिनै", "जस", "जसै", "कस", "कसै", "तैँ", "मै", "त्य", "सो", "के",
               "को", "जो", "जे", "जुन", "कुन", "आफै", "आफ्"}
BOUND_RX = re.compile(r"[ए\u0947]$")   # independent ए or matra े


def _geminate(stem, suf):
    return stem.endswith(HALANT) and len(stem) > 1 and stem[-2] == suf[0]


# core clitics take any non-halanta stem of two or more code points; the single
# one-letter host that really occurs is 1SG म (मलाई, मसँग) -- otherwise आ+मा, क+मा
STEM_RULE = {"core": lambda st, suf: not st.endswith(HALANT) and (len(st) >= 2 or st == "म"),
             "verbal": lambda st, suf: st.endswith(HALANT) and not _geminate(st, suf),
             "postp": lambda st, suf: not st.endswith(HALANT) and len(st) >= 2}

# strict profile, derived from the per-affix error table on the gold sheet
# (data/eval/gold_sheet.tsv; scripts/tier_error_report.py -- re-derive when the sheet grows; six of these
# rest on 1-2 observations, see the caveat there)
STRICT_DROP = {"वार", "पालि", "ही", "ना", "यता", "खेर", "खेरै", "भर", "पार", "पर",
               "परि", "पल्टै", "की", "ला", "ली", "लान्", "लास्", "लिन्", "लिस्"}
# ले/कै/का after a stem ending in ि ी े ै is nearly always lexical (अहि+ले,
# त्यसै+ले, भूमि+का, गाउँपालि+का); सरकार+ले, नेपाल+का, केटा+का survive
STRICT_FRONT = {"ले", "कै", "का"}
FRONT_BAR = "\u093f\u0940\u0947\u0948"
AKSHARA = re.compile("[\u0904-\u0939\u0958-\u095f\u0978-\u097f]")   # escapes on purpose: see the NFD guard in selfcheck


def aksharas(s):
    """base letters not preceded by a halanta: ढो = 1 (ढ + ो), so len() >= 2
    passes ढो+का, काका, ठिकै and this does not"""
    return sum(1 for i, c in enumerate(s) if AKSHARA.match(c) and not (i and s[i - 1] == HALANT))


def _front(st, suf):
    return suf in STRICT_FRONT and st[-1] in FRONT_BAR


STEM_RULE_STRICT = {"core": lambda st, suf: not st.endswith(HALANT) and (aksharas(st) >= 2 or st == "म") and not _front(st, suf),
                    "verbal": STEM_RULE["verbal"],
                    "postp": lambda st, suf: not st.endswith(HALANT) and aksharas(st) >= 2}
MAX_SUFFIXES = 4
# Devanagari runs (danda ।॥ excluded), or runs of anything else non-space
WORD = re.compile(r"[\u0900-\u0963\u0966-\u097f]+|[^\s\u0900-\u097f]+|[\u0964\u0965]")


def build(inventory=INV):
    """suffix table from the affix inventory (data/affix_inventory.json)"""
    rows = json.load(open(inventory, encoding="utf-8"))["rows"]
    affixes = {}
    for r in rows:
        a = r["affix"].strip("-").strip()
        if r["paradigm_id"] not in PROFILES["full"]:
            continue
        if r["tier"] != "regex" and a not in SEGMENTABLE_FST.get(r["paradigm_id"], ()):
            continue
        if not LETTER.match(a):
            continue
        prof = "core" if r["paradigm_id"] in PROFILES["core"] else "full"
        grp = GROUP.get(r["paradigm_id"], "core")
        e = affixes.setdefault(a, {"profile": prof, "group": grp, "functions": [], "paradigms": []})
        if prof == "core":
            e["profile"], e["group"] = "core", "core"
        e["functions"].append(r["function"])
        e["paradigms"].append(r["paradigm_id"])
        if r["tier"] != "regex":
            e["tier_override"] = "tier=fst (generation); admitted as a whole-unit suffix for segmentation"
    # PR 5.1 (ch5): emphatic ◌ै on a postposition deletes a final ◌ा/◌ो/◌्
    # (ताका -> ताकै, जस्तो -> जस्तै, सहित -> सहितै). The rule is tier=fst as a
    # rewrite, but its output over the closed `with_emph` list is a closed list
    # too, so those forms are admitted as whole units like मै/कै.
    for a, e in list(affixes.items()):
        if "adverbial_postp_with_emph" in e["paradigms"]:
            em = re.sub("[\u093e\u094b\u094d]$", "", a) + "\u0948"
            affixes.setdefault(em, {"profile": "full", "group": "postp", "functions": [],
                                    "paradigms": ["pr_5_1_emph_truncation_on_postp"],
                                    "tier_override": "generated from " + a + " by PR 5.1 (tier=fst rewrite); closed list"})
            affixes[em]["functions"].append("+POSTP+EMPH")
    for e in affixes.values():
        e["functions"] = sorted(set(e["functions"]))
        e["paradigms"] = sorted(set(e["paradigms"]))
    return affixes


class RegexTier:
    def __init__(self, affixes, profile="full", attested=None, min_count=100):
        keep = [a for a, e in affixes.items()
                if (profile != "core" or e["profile"] == "core")
                and (profile != "strict" or a not in STRICT_DROP)]
        rule = STEM_RULE_STRICT if profile == "strict" else STEM_RULE
        # core clitics first, then the rest, longest first within each: a rare
        # postposition must not outbid a case marker (प्रतियोगिता+का, not
        # प्रतियोगि+ताका), and a suffix whose stem rule fails falls through
        self.order = sorted(keep, key=lambda a: (affixes[a]["group"] != "core", -len(a)))
        self.rule = {a: rule[affixes[a]["group"]] for a in keep}
        self.group = {a: affixes[a]["group"] for a in keep}
        # optional Koehn-style gate for nominal cuts: the stem must itself occur
        # as a word in the training sample (kills अहि+ले, नगरपालि+का, उम्मेद+वार).
        # Not applied to verbal cuts -- halanta stems are bound by definition.
        # Cannot fix lexicalised words whose stem is a word (भूमि+का, नि+कै,
        # पा+वर): that needs a lexicon (the FST tier). None = off
        self.attested = attested
        self.min_count = min_count
        self.gated = {a for a in keep if affixes[a]["group"] != "verbal"}

    def _cut(self, word):
        for suf in self.order:
            if len(word) > len(suf) and word.endswith(suf):
                stem = word[:-len(suf)]
                if STEM_OK.search(stem) and self.rule[suf](stem, suf) and (
                        self.attested is None or suf not in self.gated
                        or stem in BOUND_STEMS or BOUND_RX.search(stem)
                        or self.attested.get(stem, 0) >= self.min_count):
                    return stem, suf
        return None

    def split(self, word):
        # right to left: [stem] [हरू] [postp | case]* [GEN | verbal]; nothing is
        # cut left of हरू; the adnominal genitive को/का/की/कै may follow any
        # postposition or case marker (घर+सम्म+को, साथी+हरू+सँग+को), a
        # postposition may follow a case marker, but a non-genitive case marker
        # never follows another case marker (काकाले stays काका + ले)
        pieces = []
        while len(pieces) < MAX_SUFFIXES:
            if pieces and pieces[0] == "हरू":
                break
            c = self._cut(word)
            if not c:
                break
            g = self.group[c[1]]
            if pieces and c[1] != "हरू":
                prev = pieces[0]
                if self.group[prev] == "verbal" or not (g == "postp" or prev in GEN):
                    break
            word, suf = c
            pieces.insert(0, suf)
        return [word] + pieces

    def mark(self, text, sep="\u2060"):
        """text with a WORD JOINER (U+2060) at every morph cut; everything else
        byte-identical, so an HF Split(sep, 'removed') pre-tokenizer restores it"""
        out, last = [], 0
        for m in WORD.finditer(text):
            out.append(text[last:m.start()])
            out.append(sep.join(self.split(m.group())))
            last = m.end()
        out.append(text[last:])
        return "".join(out)

    def pretokenize(self, text):
        out = []
        for w in WORD.findall(text):
            out.extend(self.split(w))
        return out


def load(profile="full", attested=None):
    """profile: core | full | strict. attested: a {word: count} dict, a path to a
    JSON {word: count} table (see wordcounts()), or None (no attestation gate)."""
    att = json.load(open(attested, encoding="utf-8")) if isinstance(attested, str) else attested
    return RegexTier(json.load(open(OUT, encoding="utf-8"))["affixes"], profile, att)


def wordcounts(path, out, min_count=5):
    """Word -> count over a text file, for the attestation gate."""
    from collections import Counter
    c = Counter()
    for line in open(path, encoding="utf-8"):
        c.update(WORD.findall(line))
    c = {w: n for w, n in c.items() if n >= min_count}
    json.dump(c, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{out}: {len(c)} words with count >= {min_count}")


def check_file(path, tier):
    """round-trip every word of a text file through the tier; print the split rate"""
    n = split_n = 0
    for line in open(path, encoding="utf-8"):
        for w in WORD.findall(line):
            p = tier.split(w)
            assert "".join(p) == w, (w, p)
            n += 1
            split_n += len(p) > 1
    print(f"{path}: {n} words, round-trip 100%, {split_n} split ({100*split_n/max(n,1):.1f}%), "
          f"{sum(1 for _ in open(path, encoding='utf-8'))} lines")


def selfcheck(affixes):
    t = RegexTier(affixes, "full")
    assert t.split("केटाहरूलाई") == ["केटा", "हरू", "लाई"]
    assert t.split("घरभित्र") == ["घर", "भित्र"]
    assert t.split("गर्ने") == ["गर्", "ने"]
    assert t.split("तीन") == ["तीन"]          # verbal -न needs a halanta stem
    assert t.split("कुन") == ["कुन"]
    assert t.split("रवार") == ["रवार"]        # postposition needs a 2+ letter stem
    assert t.split("मलाई") == ["म", "लाई"]     # core clitics do not
    assert t.split("काकाले") == ["काका", "ले"]  # one case marker, then only हरू
    assert t.split("शर्मा") == ["शर्मा"]        # halanta before a marker = conjunct
    assert t.split("अर्को") == ["अर्को"]
    assert t.split("विभिन्न") == ["विभिन्न"]    # geminate is not a verbal cut
    assert t.split("जिल्ला") == ["जिल्ला"]
    assert t.split("गर्न") == ["गर्", "न"]
    assert t.split("प्रतियोगिताका") == ["प्रतियोगिता", "का"]  # falls through ताका
    assert RegexTier(affixes, "full", {"अहिले": 999}).split("अहिले") == ["अहिले"]
    assert RegexTier(affixes, "full", {"घर": 999}).split("घरमा") == ["घर", "मा"]
    assert t.mark("रामले किताबहरू पढ्यो।") == "राम\u2060ले किताब\u2060हरू पढ्यो।"
    assert RegexTier(affixes, "full", {}).split("उसले") == ["उस", "ले"]      # bound stems pass the gate
    assert RegexTier(affixes, "full", {}).split("गर्नुभएको") == ["गर्नुभए", "को"]
    assert t.split("केटाहरू") == ["केटा", "हरू"]
    assert t.split("घरसम्मको") == ["घर", "सम्म", "को"]      # postposition before a case marker
    assert t.split("साथीहरूसँगको") == ["साथी", "हरू", "सँग", "को"]
    assert t.split("जस्तै") == ["जस्तै"] and t.split("घरजस्तै") == ["घर", "जस्तै"]   # PR 5.1 form
    assert t.split("भित्रै") == ["भित्रै"] and t.split("घरभित्रै") == ["घर", "भित्रै"]
    assert t.split("कोही") == ["को", "ही"]
    assert t.split("घरमै") == ["घर", "मै"]
    assert t.split("रामको") == ["राम", "को"]
    assert t.split("मा") == ["मा"]            # whole word is never an affix
    assert t.split("हरू") == ["हरू"]
    assert RegexTier(affixes, "core").split("घरभित्र") == ["घरभित्र"]
    s = RegexTier(affixes, "strict")
    assert s.split("अहिले") == ["अहिले"]        # was अहि|ले
    assert s.split("भूमिका") == ["भूमिका"]       # was भूमि|का
    assert s.split("ढोका") == ["ढोका"]         # akshara rule
    assert s.split("काका") == ["काका"]
    assert s.split("दरवार") == ["दरवार"]        # वार dropped
    assert s.split("त्यसैले") == ["त्यसैले"]
    assert s.split("सरकारले") == ["सरकार", "ले"]  # legit cuts survive
    assert s.split("नेपालका") == ["नेपाल", "का"]
    assert s.split("केटाहरूलाई") == ["केटा", "हरू", "लाई"]
    assert s.split("गर्ने") == ["गर्", "ने"]
    # NFD guard: the literal क़ in LETTER/STEM_OK is precomposed U+0958; if the
    # source is ever NFD-normalised it becomes क + U+093C and the range
    # silently matches every matra
    assert not LETTER.match("ा") and not STEM_OK.search("े") and not AKSHARA.match("े")
    # round-trip over every Devanagari form in the affix inventory and the seed lexicon
    words = set()
    if os.path.exists(INV):
        for r in json.load(open(INV, encoding="utf-8"))["rows"]:
            words.update(WORD.findall(r.get("surface_example", "") + " " + r["affix"]))
    for line in open(SEEDS, encoding="utf-8"):
        words.update(WORD.findall(line.split("\t")[0]))
    bad = [w for w in words if "".join(t.split(w)) != w]
    assert not bad, bad[:10]
    return len(words), sum(len(t.split(w)) > 1 for w in words)


def write(affixes, inventory=INV, out=OUT):
    """write the compiled suffix table (regex_tier_v1.json) from build()'s output"""
    import hashlib
    json.dump({
        "dataset_version": "v1",
        "inventory_sha256": hashlib.sha256(open(inventory, "rb").read()).hexdigest(),
        "profiles": {k: sorted(v) for k, v in PROFILES.items()},
        "segmentable_fst": SEGMENTABLE_FST,
        "strict_drop": sorted(STRICT_DROP),
        "strict_front": {"suffixes": sorted(STRICT_FRONT), "barred_after": list(FRONT_BAR)},
        "n_affixes": {"core": sum(e["profile"] == "core" for e in affixes.values()),
                      "full": len(affixes)},
        "affixes": dict(sorted(affixes.items())),
    }, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
