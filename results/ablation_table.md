# Ablation table

**Papaya** is our tokenizer: the FST tier (hybrid: lexicon transducer, regex fallback) marking morpheme boundaries, byte-level BPE on top. The regex-only rows are Papaya with the FST removed (ablation); `fixed` is plain byte BPE on the same data at the same vocab; `broken` is plain BPE with GPT-2's mark-splitting regex. Variant ids in parentheses are the keys used in results/*.jsonl and scripts/.

Fertility = tokens per whitespace word on 5k FineWeb-2 test docs. Boundary F1 = gold-sheet morpheme boundaries recovered by the tokenizer's token starts (607 words). Consistency F1 = stem tokens inside the inflected form vs stem alone (gold 344 forms / corpus 51k FST-analysed types); read it together with fertility — a character-level split is trivially consistent. bpb = 17M GPT, 400k docs, 1 epoch, identical raw test text. NER = Nep-gLUE span-F1, POS = macro-F1 over 39 tags; last-subtoken labelling, full fine-tune, best-dev epoch. CC = news topic classification on running text (mteb/NepaliNewsClassification, 3 classes, 1,495 test docs), macro-F1.

Downstream cells show n=5 where they come from the evidence run (`results/downstream_sig.jsonl`, 16k) and n=3 where they fall back to the earlier sweep (`results/downstream_server.jsonl`: 8k/32k and the no-pretraining rows). The two disagree by up to 0.006 on NER with the same ordering; only the 5-seed run has intervals and a paired test behind it — those are in `results/significance.md`.

## Our variants

| variant | vocab | fertility | boundary F1 | cons. gold | cons. corpus | bpb | NER span-F1 | POS macro-F1 | CC macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| plain BPE (fixed) | 8000 | 1.584 | 0.215 | 0.416 | 0.644 | 0.4987 ±0.0011 (n=3) | 0.745 ±0.008 (n=3) | 0.922 ±0.002 (n=3) | 0.963 ±0.003 (n=3) |
| **Papaya** (fst) | 8000 | 1.768 | 0.771 | 0.911 | 0.976 | 0.4879 ±0.0001 (n=3) | 0.769 ±0.010 (n=3) | 0.925 ±0.006 (n=3) | 0.965 ±0.005 (n=3) |
| plain BPE, mark-splitting (broken) | 16000 | 4.400 | 0.177 | 0.902 | 0.887 | 0.5124 ±0.0010 (n=3) | 0.765 ±0.010 (n=5) | 0.926 ±0.004 (n=5) | 0.964 ±0.006 (n=5) |
| plain BPE (fixed) | 16000 | 1.400 | 0.218 | 0.305 | 0.536 | 0.4971 ±0.0013 (n=3) | 0.736 ±0.009 (n=5) | 0.929 ±0.003 (n=5) | 0.972 ±0.004 (n=5) |
| Unigram LM (uni) | 16000 | 1.778 | 0.525 | 0.613 | 0.873 | 0.4849 ±0.0003 (n=3) | 0.758 ±0.005 (n=5) | 0.929 ±0.004 (n=5) | 0.971 ±0.003 (n=5) |
| Morfessor + BPE (morf) | 16000 | 1.456 | 0.457 | 0.443 | 0.780 | 0.4871 ±0.0002 (n=3) | 0.747 ±0.005 (n=5) | 0.930 ±0.002 (n=5) | 0.974 ±0.002 (n=5) |
| Papaya, regex core only | 16000 | 1.551 | 0.494 | 0.510 | 0.809 | 0.4839 ±0.0006 (n=3) | 0.759 ±0.009 (n=5) | 0.931 ±0.001 (n=5) | 0.975 ±0.002 (n=5) |
| Papaya, regex full only | 16000 | 1.581 | 0.709 | 0.823 | 0.844 | 0.4820 ±0.0009 (n=3) | 0.761 ±0.015 (n=5) | 0.930 ±0.002 (n=5) | 0.974 ±0.002 (n=5) |
| Papaya, regex full + gate | 16000 | 1.565 | 0.700 | 0.782 | 0.827 | 0.4836 ±0.0002 (n=3) | 0.762 ±0.003 (n=5) | 0.931 ±0.002 (n=5) | 0.973 ±0.005 (n=5) |
| **Papaya** (fst) | 16000 | 1.629 | 0.859 | 0.903 | 0.970 | 0.4809 ±0.0007 (n=3) | 0.764 ±0.006 (n=5) | 0.930 ±0.002 (n=5) | 0.975 ±0.004 (n=5) |
| Papaya + Unigram (fst_uni) | 16000 | 1.908 | 0.783 | 0.936 | 0.986 | 0.4807 ±0.0017 (n=3) | 0.756 ±0.009 (n=5) | 0.928 ±0.001 (n=5) | 0.974 ±0.003 (n=5) |
| **Papaya v2** (fst2) | 16000 | 1.628 | 0.863 | 0.903 | 0.968 | 0.4805 ±0.0009 (n=3) | 0.766 ±0.008 (n=5) | 0.930 ±0.004 (n=5) | 0.975 ±0.004 (n=5) |
| **Papaya v3** (fst3) | 16000 | 1.627 | 0.863 | 0.903 | 0.966 | 0.4807 ±0.0006 (n=3) | 0.759 ±0.003 (n=5) | 0.931 ±0.002 (n=5) | 0.977 ±0.002 (n=5) |
| **Litchi v1** (litchi) | 16000 | 1.628 | 0.804 | 0.879 | 0.931 | 0.4808 ±0.0008 (n=3) | 0.769 ±0.006 (n=5) | 0.931 ±0.004 (n=5) | 0.973 ±0.005 (n=5) |
| plain BPE (fixed) | 32000 | 1.279 | 0.174 | 0.197 | 0.415 | 0.4970 ±0.0006 (n=3) | 0.747 ±0.012 (n=3) | 0.933 ±0.002 (n=3) | 0.977 ±0.004 (n=3) |
| **Papaya** (fst) | 32000 | 1.550 | 0.920 | 0.891 | 0.965 | 0.4735 ±0.0003 (n=3) | 0.763 ±0.002 (n=3) | 0.932 ±0.004 (n=3) | 0.974 ±0.001 (n=3) |
| plain BPE (fixed) (no pretraining) | 16000 | | | | | | 0.519 ±0.019 (n=3) | 0.839 ±0.019 (n=3) | 0.920 ±0.001 (n=3) |
| **Papaya** (fst) (no pretraining) | 16000 | | | | | | 0.512 ±0.025 (n=3) | 0.847 ±0.006 (n=3) | 0.934 ±0.005 (n=3) |

## Matched-steps control

plain BPE trained for as many optimizer steps as Papaya sees at the same vocab (Papaya/plain fertility ratio as epochs).

| variant | vocab | epochs | bpb | plain BPE bpb (1 epoch) | Papaya bpb (1 epoch) | gap left |
|---|---|---|---|---|---|---|
| plain BPE (fixed) | 8000 | 1.116 | 0.4924 ±0.0010 (n=3) | 0.4987 | 0.4879 | 0.9% of 2.2% |
| plain BPE (fixed) | 16000 | 1.164 | 0.4863 ±0.0001 (n=3) | 0.4971 | 0.4809 | 1.1% of 3.3% |
| plain BPE (fixed) | 32000 | 1.21 | 0.4811 ±0.0004 (n=3) | 0.4970 | 0.4735 | 1.6% of 4.7% |
| Papaya + Unigram (fst_uni) | 16000 | 0.854 | 0.4897 ±0.0022 (n=3) | 0.4807 | 0.4809 | 1.8% of -0.1% |
| Morfessor + BPE (morf) | 16000 | 1.119 | 0.4802 ±0.0003 (n=3) | 0.4871 | 0.4809 | -0.1% of 1.3% |
| Unigram LM (uni) | 16000 | 0.917 | 0.4907 ±0.0014 (n=3) | 0.4849 | 0.4809 | 2.0% of 0.8% |

## External tokenizers

Intrinsic columns are measured on the 5k test docs. The last three are the same tokenizer put through our own 17M GPT on the same 400k documents (`train_lm.py --variant x_<name>`), so they are comparable to the rows above — but not at matched capacity: at 128k/152k vocab the embedding table alone is roughly 3× our whole model. Blank = not run through the LM.

| tokenizer | family | vocab | fertility | unk rate | boundary F1 | cons. gold | cons. corpus | bpb | NER span-F1 | CC macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| sakonii (Sakonii/distilbert-base-nepali) | nepali sentencepiece | 24581 | 1.340 | 0.0000 | 0.448 | 0.417 | 0.804 | 0.4867 ±0.0014 (n=3) | 0.757 ±0.007 (n=5) | 0.973 ±0.004 (n=5) |
| nepberta (NepBERTa/NepBERTa) | nepali wordpiece | 30523 | 1.285 | 0.0016 | 0.266 | 0.290 | 0.688 | 0.4866 ±0.0026 (n=3) | 0.762 ±0.010 (n=5) | 0.978 ±0.004 (n=5) |
| nepalibpe (Aananda-giri/NepaliBPE) | nepali bpe | 50006 | 1.238 | 0.0000 | 0.197 | 0.061 | 0.098 | – | – | – |
| arkios (sajalregmi4/arkios-tokenizer) | nepali byte bpe | 65536 | 1.420 | 0.0000 | 0.225 | 0.317 | 0.545 | – | – | – |
| mbert (bert-base-multilingual-cased) | multilingual wordpiece | 119547 | 2.820 | 0.0053 | 0.248 | 0.846 | 0.931 | – | – | – |
| llama3 (unsloth/Llama-3.1-8B) | llm byte bpe | 128256 | 3.565 | 0.0000 | 0.119 | 0.795 | 0.859 | 0.5059 (n=1) | 0.758 (n=1) | 0.962 (n=1) |
| qwen25 (Qwen/Qwen2.5-7B) | llm byte bpe | 151665 | 6.192 | 0.0000 | 0.222 | 0.913 | 0.956 | 0.5169 (n=1) | 0.769 (n=1) | 0.956 (n=1) |
| gpt4o (Xenova/gpt-4o) | llm byte bpe | 200000 | 2.114 | 0.0000 | 0.248 | 0.642 | 0.823 | – | – | – |
| xlmr (xlm-roberta-base) | multilingual sentencepiece | 250002 | 1.675 | 0.0000 | 0.428 | 0.622 | 0.844 | – | – | – |
| gemma2 (unsloth/gemma-2-2b) | llm sentencepiece | 256000 | 2.888 | 0.0000 | 0.360 | 0.905 | 0.957 | – | – | – |

Note: NepaliBPE marks word ends (`</w>`), so a stem alone and a stem inside a word never share the final token — its consistency is low by construction, not by segmentation quality.
