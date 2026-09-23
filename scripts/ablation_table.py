#!/usr/bin/env python3
"""
One table from the results/*.jsonl files -> results/ablation_table.md.
Per (variant, vocab): fertility, gold-sheet boundary F1, consistency F1 (gold /
corpus), bpb mean±sd over seeds, NER span-F1 and POS macro-F1 mean±sd; plus the
matched-steps control rows (epochs > 1) and the external tokenizers (intrinsic
columns only). Latest row wins where a jsonl has duplicates.

    python3 scripts/ablation_table.py
"""
import collections
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["broken", "fixed", "uni", "morf", "core", "full", "gated", "fst", "fst_uni", "fst2", "fst3", "litchi"]
# Papaya is the name of our tokenizer (FST tier + byte BPE). Variant ids in results/*.jsonl and
# checkpoints keep their short names; every table shows the display name.
NAMES = {"fst": "**Papaya** (fst)", "fst2": "**Papaya v2** (fst2)", "fst3": "**Papaya v3** (fst3)", "litchi": "**Litchi v1** (litchi)", "fixed": "plain BPE (fixed)",
         "broken": "plain BPE, mark-splitting (broken)", "core": "Papaya, regex core only",
         "full": "Papaya, regex full only", "gated": "Papaya, regex full + gate",
         "uni": "Unigram LM (uni)", "morf": "Morfessor + BPE (morf)", "fst_uni": "Papaya + Unigram (fst_uni)"}
show = lambda v: NAMES.get(v, v)


def rows(name):
    p = f"{ROOT}/results/{name}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def msd(xs, d=4):
    if not xs:
        return "–"
    m = st.mean(xs)
    return f"{m:.{d}f} ±{st.stdev(xs):.{d}f} (n={len(xs)})" if len(xs) > 1 else f"{m:.{d}f} (n=1)"


def main():
    fert = {(r["variant"], r["vocab"]): r["tokens_per_word"] for r in rows("fertility")}
    bound = {r["system"]: r["F1"] for r in rows("boundary_f1") if r["set"] == "gold_sheet"}
    cons = {(r["set"], r["variant"], r["vocab"]): r["F1"] for r in rows("morph_consistency")}
    # The main 16k configs were run twice per seed (with and without --save), and the matched-steps
    # control with weights repeats seeds 1-3 again. Repeats of one seed are not independent samples: average them
    # within the seed, so n counts seeds.
    bpb_seed = collections.defaultdict(list)
    for r in rows("lm_bpc_server"):
        bpb_seed[(r["variant"], r["vocab"], r.get("epochs", 1.0), r["seed"])].append(r["bits_per_byte"])
    bpb = collections.defaultdict(list)
    for (v, voc, ep, _), xs in sorted(bpb_seed.items()):
        bpb[(v, voc, ep)].append(sum(xs) / len(xs))
    # Two downstream sweeps exist: downstream_server (3 seeds, covers 8k/32k and the
    # no-pretraining rows) and downstream_sig (5 seeds, 16k only, no scratch, plus the
    # x_* published tokenizers and the data-curve rows). The 5-seed run supersedes the
    # 3-seed one where both have a cell, so overlay it and fall back to the server sweep
    # for the vocab sizes and scratch rows it never covered. Keep results/significance.md
    # and this table telling the same story.
    metric = lambda t: "span_f1" if t == "ner" else "macro_f1"
    down = collections.defaultdict(list)
    for r in rows("downstream_server"):
        down[(r["task"], r["variant"], r["vocab"], r["scratch"])].append(r["test"][metric(r["task"])])
    sig = collections.defaultdict(list)
    for r in rows("downstream_sig"):
        if r.get("train_frac", 1.0) != 1.0:      # 10/25/50 % data-curve rows: significance.md only
            continue
        v = f"{r['variant']}@{r['tag']}" if r.get("tag") else r["variant"]   # tagged = another checkpoint
        sig[(r["task"], v, r["vocab"], r.get("scratch", False))].append(r["test"][metric(r["task"])])
    down.update(sig)
    keys = sorted({(v, k) for v, k in fert}, key=lambda x: (x[1], ORDER.index(x[0]) if x[0] in ORDER else 9))

    L = ["# Ablation table", "",
         "**Papaya** is our tokenizer: the FST tier (hybrid: lexicon transducer, regex fallback) marking morpheme "
         "boundaries, byte-level BPE on top. The regex-only rows are Papaya with the FST removed (ablation); "
         "`fixed` is plain byte BPE on the same data at the same vocab; `broken` is plain BPE with GPT-2's "
         "mark-splitting regex. Variant ids in parentheses are the keys used in results/*.jsonl and scripts/.", "",
         "Fertility = tokens per whitespace word on 5k FineWeb-2 test docs. Boundary F1 = gold-sheet "
         "morpheme boundaries recovered by the tokenizer's token starts (607 words). Consistency F1 = "
         "stem tokens inside the inflected form vs stem alone (gold 344 forms / corpus 51k FST-analysed "
         "types); read it together with fertility — a character-level split is trivially consistent. "
         "bpb = 17M GPT, 400k docs, 1 epoch, identical raw test text. NER = Nep-gLUE span-F1, "
         "POS = macro-F1 over 39 tags; last-subtoken labelling, full fine-tune, best-dev epoch. CC = news topic "
         "classification on running text (mteb/NepaliNewsClassification, 3 classes, 1,495 test docs), macro-F1.", "",
         "Downstream cells show n=5 where they come from the evidence run (`results/downstream_sig.jsonl`, 16k) "
         "and n=3 where they fall back to the earlier sweep (`results/downstream_server.jsonl`: 8k/32k and the "
         "no-pretraining rows). The two disagree by up to 0.006 on NER with the same ordering; only the 5-seed "
         "run has intervals and a paired test behind it — those are in `results/significance.md`.", "",
         "## Our variants", "",
         "| variant | vocab | fertility | boundary F1 | cons. gold | cons. corpus | bpb | NER span-F1 | POS macro-F1 | CC macro-F1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v, k in keys:
        L.append(f"| {show(v)} | {k} | {fert[(v, k)]:.3f} | {bound.get(f'bpe_{v}_{k}', float('nan')):.3f} | "
                 f"{cons.get(('gold', v, k), float('nan')):.3f} | {cons.get(('corpus', v, k), float('nan')):.3f} | "
                 f"{msd(bpb.get((v, k, 1.0)))} | {msd(down.get(('ner', v, k, False)), 3)} | {msd(down.get(('pos', v, k, False)), 3)} | "
                 f"{msd(down.get(('cc', v, k, False)), 3)} |")
    for v, k in sorted({(v, k) for (_, v, k, s) in down if s}):
        L.append(f"| {show(v)} (no pretraining) | {k} | | | | | | {msd(down.get(('ner', v, k, True)), 3)} | "
                 f"{msd(down.get(('pos', v, k, True)), 3)} | {msd(down.get(('cc', v, k, True)), 3)} |")
    L += ["", "## Matched-steps control", "",
          "plain BPE trained for as many optimizer steps as Papaya sees at the same vocab (Papaya/plain fertility ratio as epochs).", "",
          "| variant | vocab | epochs | bpb | plain BPE bpb (1 epoch) | Papaya bpb (1 epoch) | gap left |", "|---|---|---|---|---|---|---|"]
    for (v, k, e), xs in sorted(bpb.items()):
        if e != 1.0:
            m, f1, fs = st.mean(xs), st.mean(bpb[(v, k, 1.0)]), st.mean(bpb[("fst", k, 1.0)])
            L.append(f"| {show(v)} | {k} | {e} | {msd(xs)} | {f1:.4f} | {fs:.4f} | {(m - fs) / m * 100:.1f}% of {(f1 - fs) / f1 * 100:.1f}% |")
    L += ["", "## External tokenizers", "",
          "Intrinsic columns are measured on the 5k test docs. The last three are the same tokenizer put "
          "through our own 17M GPT on the same 400k documents (`train_lm.py --variant x_<name>`), so they are "
          "comparable to the rows above — but not at matched capacity: at 128k/152k vocab the embedding table "
          "alone is roughly 3× our whole model. Blank = not run through the LM.", "",
          "| tokenizer | family | vocab | fertility | unk rate | boundary F1 | cons. gold | cons. corpus | bpb | NER span-F1 | CC macro-F1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    seen = {}
    for r in rows("external"):
        seen[r["name"]] = r
    for r in sorted(seen.values(), key=lambda r: r["vocab"]):
        x = f"x_{r['name']}"                      # LM/fine-tune rows key on the x_ prefix
        xb = [v for (vv, _, e), xs in bpb.items() if vv == x and e == 1.0 for v in xs]
        xn = [v for (t, vv, _, s), xs in down.items() if vv == x and t == "ner" and not s for v in xs]
        xc = [v for (t, vv, _, s), xs in down.items() if vv == x and t == "cc" and not s for v in xs]
        L.append(f"| {r['name']} ({r['repo']}) | {r['family']} | {r['vocab']} | {r['tokens_per_word']:.3f} | "
                 f"{r['unk_rate']:.4f} | {r['boundary_gold']['F1']:.3f} | {r['consistency_gold']['F1']:.3f} | "
                 f"{r['consistency_corpus']['F1']:.3f} | {msd(xb)} | {msd(xn, 3)} | {msd(xc, 3)} |")
    L += ["", "Note: NepaliBPE marks word ends (`</w>`), so a stem alone and a stem inside a word never share "
          "the final token — its consistency is low by construction, not by segmentation quality.", ""]
    open(f"{ROOT}/results/ablation_table.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
