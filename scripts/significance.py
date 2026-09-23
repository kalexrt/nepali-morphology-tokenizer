#!/usr/bin/env python3
"""
Downstream evidence: seeds, paired significance, slices, data curve.

Reads results/downstream_sig.jsonl (finetune_tok.py --dump rows: 5 seeds per
task x variant, plus --train-frac rows) and the per-sentence predictions in
results/preds/ (written by finetune_tok.py --dump; not shipped, because they
contain the task's test sentences). Writes results/significance.md.

  means      mean +- sd and a 95 % t-interval over seeds, per (task, variant).
  paired     A vs B on the same test sentences and the same seed: paired
             bootstrap over sentences (2000 resamples), the seed-averaged
             difference in the task metric, its 95 % interval and the share of
             resamples where B >= A (one-sided p).
  slices     NER only: span P/R/F1 on (a) gold spans that contain a word never
             seen in the NER training sentences, (b) otherwise spans containing a
             word rare in the pretraining corpus (count < 50), (c) the rest.
  curve      the metric at 10/25/50/100 % of the training sentences.

    python3 scripts/significance.py [--ref fst full] [--boot 2000]
"""
import argparse
import collections
import json
import os
import statistics as st
import numpy as np

import common
from common import ROOT
from finetune_tok import spans

KEY = {"ner": "span_f1", "pos": "macro_f1", "cc": "macro_f1"}
ORDER = ["fixed", "fixed@matched", "broken", "uni", "morf", "core", "full", "gated", "fst", "fst_uni", "fst2", "fst3", "litchi",
         "x_nepberta", "x_sakonii", "x_llama3", "x_qwen25"]
NAMES = {"fst": "**Papaya** (fst)", "fst2": "**Papaya v2** (fst2)", "fst3": "**Papaya v3** (fst3)", "litchi": "**Litchi v1** (litchi)", "fixed": "plain BPE (fixed)",
         "broken": "plain BPE, mark-splitting (broken)", "core": "Papaya, regex core only",
         "full": "Papaya, regex full only", "gated": "Papaya, regex full + gate",
         "x_nepberta": "NepBERTa tokenizer", "x_sakonii": "Sakonii tokenizer", "x_llama3": "Llama 3.1 tokenizer",
         "x_qwen25": "Qwen 2.5 tokenizer",
         # equal-steps baseline weights, and the unsupervised / Unigram baselines
         "fixed@matched": "plain BPE, matched steps (fixed@matched)", "uni": "Unigram LM (uni)",
         "morf": "Morfessor + BPE (morf)", "fst_uni": "Papaya + Unigram (fst_uni)"}
show = lambda v: NAMES.get(v, v)


def load():
    rows = [json.loads(l) for l in open(f"{ROOT}/results/downstream_sig.jsonl", encoding="utf-8")]
    for r in rows:              # finetune_tok.py --tag: a different checkpoint of the same tokenizer
        if r.get("tag"):
            r["variant"] = f"{r['variant']}@{r['tag']}"
    return [r for r in rows if not r.get("scratch")]


def counts(task, names, pairs):
    """per-sentence sufficient statistics: NER -> (tp, fp, fn) of spans; else per-label (tp, fp, fn)"""
    out = []
    for gs, ps in pairs:
        if task == "ner":
            g, p = spans([names[i] for i in gs]), spans([names[i] for i in ps])
            out.append((len(g & p), len(p - g), len(g - p)))
        else:
            c = np.zeros((len(names), 3), dtype=np.int64)
            for a, b in zip(gs, ps):
                if a == b:
                    c[a, 0] += 1
                else:
                    c[b, 1] += 1; c[a, 2] += 1
            out.append(c)
    return out


def metric(task, cs, idx):
    if task == "ner":
        tp = sum(cs[i][0] for i in idx); fp = sum(cs[i][1] for i in idx); fn = sum(cs[i][2] for i in idx)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    c = sum(cs[i] for i in idx)
    f = [2 * a / (2 * a + b + d) if a else 0.0 for a, b, d in c if a + d > 0]   # labels present in gold
    return sum(f) / len(f)


def ci(xs):
    if len(xs) < 2:
        return f"{xs[0]:.3f}"
    m, s = st.mean(xs), st.stdev(xs)
    t = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(xs), 2.0)
    return f"{m:.3f} ± {s:.3f} [{m - t * s / len(xs) ** .5:.3f}, {m + t * s / len(xs) ** .5:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", default=["fst", "full"])   # the paper's table; one RNG is shared, so other refs shift every p
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if not os.path.isdir(f"{ROOT}/results/preds"):
        raise SystemExit("results/preds/ missing: it is written by the fine-tunes (scripts/run_finetune.sh, "
                         "finetune_tok.py --dump) and is not shipped")
    rows = load()
    full = collections.defaultdict(dict)      # (task, variant) -> seed -> row
    curve = collections.defaultdict(list)     # (task, variant, frac) -> metrics
    for r in rows:
        if r["train_frac"] == 1.0:
            full[(r["task"], r["variant"])][r["seed"]] = r
        curve[(r["task"], r["variant"], r["train_frac"])].append(r["test"][KEY[r["task"]]])
    L = ["# Downstream evidence: seeds, significance, slices, data curve", "",
         f"Source: `results/downstream_sig.jsonl` ({len(rows)} rows), per-sentence predictions in `results/preds/`. "
         "Metric: NER span-F1, POS/CC macro-F1, test split, best-dev epoch. Seeds 1–5 (pretraining seeds 1–3 reused for 4–5). "
         "**Papaya** = our tokenizer (FST tier + byte BPE); variant ids in parentheses.", ""]

    # ---- means --------------------------------------------------------------
    L += ["## Means over seeds (mean ± sd [95 % t-interval])", "", "| variant | NER | POS | CC |", "|---|---|---|---|"]
    variants = [v for v in ORDER if any((t, v) in full for t in KEY)]
    for v in variants:
        cells = [ci([r["test"][KEY[t]] for r in full[(t, v)].values()]) if (t, v) in full else "–" for t in KEY]
        L.append(f"| {show(v)} | " + " | ".join(cells) + " |")

    # ---- paired bootstrap -------------------------------------------------------
    rng = np.random.default_rng(0)
    cache = {}
    def stats(task, v, s):
        if (task, v, s) not in cache:
            d = json.load(open(f"{ROOT}/{full[(task, v)][s]['preds']}", encoding="utf-8"))
            cache[(task, v, s)] = (counts(task, d["names"], d["pairs"]), d)
        return cache[(task, v, s)]
    L += ["", "## Paired comparison (same sentences, same seed; bootstrap over sentences, seed-averaged)", "",
          "Δ = ref − baseline in the task metric; p = share of resamples with Δ ≤ 0 (one-sided). "
          "The bootstrap resamples test sentences only; the last column is Δ on the full test set per "
          "pretraining seed (fine-tuning seeds sharing a pretraining checkpoint averaged), which is where the "
          "pretraining variance shows.", "",
          "| task | ref | baseline | Δ mean | 95 % interval | p | seeds | Δ per pretraining seed |",
          "|---|---|---|---|---|---|---|---|"]
    for task in KEY:
        for ref in a.ref:
            for v in variants:
                if v == ref or (task, v) not in full or (task, ref) not in full:
                    continue
                seeds = sorted(set(full[(task, v)]) & set(full[(task, ref)]))
                if not seeds:
                    continue
                n = len(stats(task, ref, seeds[0])[0])
                deltas = []
                for _ in range(a.boot):
                    idx = rng.integers(0, n, n)
                    deltas.append(st.mean(metric(task, stats(task, ref, s)[0], idx) - metric(task, stats(task, v, s)[0], idx) for s in seeds))
                deltas = np.array(deltas)
                by_pre = collections.defaultdict(list)
                for s in seeds:
                    by_pre[full[(task, ref)][s]["pre_seed"]].append(
                        full[(task, ref)][s]["test"][KEY[task]] - full[(task, v)][s]["test"][KEY[task]])
                per_pre = ", ".join(f"{st.mean(x):+.3f}" for _, x in sorted(by_pre.items()))
                L.append(f"| {task} | {show(ref)} | {show(v)} | {deltas.mean():+.4f} | [{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}] | "
                         f"{(deltas <= 0).mean():.3f} | {len(seeds)} | {per_pre} |")

    # ---- NER slices ---------------------------------------------------------------
    if any(t == "ner" for t, _ in full):
        cnt = common.word_counts(full=True)
        L += ["", "## NER slices (span P / R / F1, summed over seeds)", "",
              "unseen = gold span contains a word absent from the NER training sentences; rare = otherwise contains a word "
              "with count < 50 in the pretraining word counts (where BPE fragments most); frequent = neither. "
              "Predicted spans are assigned to a slice the same way; n = gold spans per seed.", "",
              "| variant | unseen P/R/F1 (n) | rare P/R/F1 (n) | frequent P/R/F1 (n) |", "|---|---|---|---|"]
        unseen = {}      # (v, seed) -> per-sentence (tp, fp, fn) on unseen-slice spans
        for v in variants:
            if ("ner", v) not in full:
                continue
            acc = {k: np.zeros(3) for k in ("unseen", "rare", "frequent")}
            nspan = {k: 0 for k in acc}
            for s, r in sorted(full[("ner", v)].items()):
                cs, d = stats("ner", v, s)
                train_types = set(d["train_types"])
                per_sent = unseen[(v, s)] = []
                for (gs, ps), words in zip(d["pairs"], d["test_words"]):
                    words = words[:len(gs)]
                    def slice_of(span):
                        ws = words[span[1]:span[2]]
                        if any(w not in train_types for w in ws):
                            return "unseen"
                        if any(cnt.get(w, 0) < 50 for w in ws):
                            return "rare"
                        return "frequent"
                    g, p = spans([d["names"][i] for i in gs]), spans([d["names"][i] for i in ps])
                    u = [0, 0, 0]
                    for sp in g:
                        k = slice_of(sp); nspan[k] += 1
                        acc[k][2] += sp not in p; acc[k][0] += sp in p
                        if k == "unseen":
                            u[0] += sp in p; u[2] += sp not in p
                    for sp in p - g:
                        k = slice_of(sp)
                        acc[k][1] += 1
                        u[1] += k == "unseen"
                    per_sent.append(tuple(u))
            cells = []
            for k in acc:
                tp, fp, fn = acc[k]
                P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
                cells.append(f"{P:.3f} / {R:.3f} / {2 * P * R / (P + R) if P + R else 0:.3f} ({nspan[k] // len(full[('ner', v)])})")
            L.append(f"| {show(v)} | " + " | ".join(cells) + " |")
        L += ["", "### Unseen slice, paired bootstrap", "",
              "Span F1 on the unseen slice only, same resampling as above (sentences, seed-averaged).", "",
              "| ref | baseline | Δ mean | 95 % interval | p |", "|---|---|---|---|---|"]
        for ref in a.ref:
            for v in variants:
                if v == ref or ("ner", v) not in full or ("ner", ref) not in full:
                    continue
                seeds = sorted(set(full[("ner", v)]) & set(full[("ner", ref)]))
                if not seeds:
                    continue
                n = len(unseen[(ref, seeds[0])])
                deltas = []
                for _ in range(a.boot):
                    idx = rng.integers(0, n, n)
                    deltas.append(st.mean(metric("ner", unseen[(ref, s)], idx) - metric("ner", unseen[(v, s)], idx)
                                          for s in seeds))
                deltas = np.array(deltas)
                L.append(f"| {show(ref)} | {show(v)} | {deltas.mean():+.4f} | [{np.percentile(deltas, 2.5):+.4f}, "
                         f"{np.percentile(deltas, 97.5):+.4f}] | {(deltas <= 0).mean():.3f} |")

    # ---- curve ----------------------------------------------------------------------
    L += ["", "## Data-efficiency curve (mean ± sd over seeds)", ""]
    for task in ("ner", "cc"):
        fracs = sorted({f for (t, _, f) in curve if t == task})
        if not fracs:
            continue
        L += [f"### {task}", "", "| variant | " + " | ".join(f"{int(f * 100)} %" for f in fracs) + " |", "|---|" + "---|" * len(fracs)]
        for v in variants:
            cells = []
            for f in fracs:
                xs = curve.get((task, v, f), [])
                cells.append(f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f} ({len(xs)})" if len(xs) > 1 else (f"{xs[0]:.3f}" if xs else "–"))
            if any(c != "–" for c in cells):
                L.append(f"| {show(v)} | " + " | ".join(cells) + " |")
        L.append("")
    open(f"{ROOT}/results/significance.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
