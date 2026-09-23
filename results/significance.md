# Downstream evidence: seeds, significance, slices, data curve

Source: `results/downstream_sig.jsonl` (426 rows), per-sentence predictions in `results/preds/`. Metric: NER span-F1, POS/CC macro-F1, test split, best-dev epoch. Seeds 1–5 (pretraining seeds 1–3 reused for 4–5). **Papaya** = our tokenizer (FST tier + byte BPE); variant ids in parentheses.

## Means over seeds (mean ± sd [95 % t-interval])

| variant | NER | POS | CC |
|---|---|---|---|
| plain BPE (fixed) | 0.736 ± 0.009 [0.725, 0.748] | 0.929 ± 0.003 [0.925, 0.933] | 0.972 ± 0.004 [0.967, 0.977] |
| plain BPE, matched steps (fixed@matched) | 0.755 ± 0.011 [0.741, 0.768] | 0.929 ± 0.003 [0.925, 0.933] | 0.974 ± 0.005 [0.968, 0.981] |
| plain BPE, mark-splitting (broken) | 0.765 ± 0.010 [0.752, 0.778] | 0.926 ± 0.004 [0.921, 0.931] | 0.964 ± 0.006 [0.956, 0.971] |
| Unigram LM (uni) | 0.758 ± 0.005 [0.752, 0.764] | 0.929 ± 0.004 [0.925, 0.934] | 0.971 ± 0.003 [0.968, 0.975] |
| Morfessor + BPE (morf) | 0.747 ± 0.005 [0.741, 0.753] | 0.930 ± 0.002 [0.928, 0.932] | 0.974 ± 0.002 [0.971, 0.977] |
| Papaya, regex core only | 0.759 ± 0.009 [0.748, 0.770] | 0.931 ± 0.001 [0.930, 0.932] | 0.975 ± 0.002 [0.973, 0.977] |
| Papaya, regex full only | 0.761 ± 0.015 [0.743, 0.780] | 0.930 ± 0.002 [0.928, 0.932] | 0.974 ± 0.002 [0.971, 0.978] |
| Papaya, regex full + gate | 0.762 ± 0.003 [0.758, 0.766] | 0.931 ± 0.002 [0.929, 0.933] | 0.973 ± 0.005 [0.966, 0.980] |
| **Papaya** (fst) | 0.764 ± 0.006 [0.756, 0.772] | 0.930 ± 0.002 [0.927, 0.933] | 0.975 ± 0.004 [0.969, 0.980] |
| Papaya + Unigram (fst_uni) | 0.756 ± 0.009 [0.745, 0.766] | 0.928 ± 0.001 [0.927, 0.929] | 0.974 ± 0.003 [0.970, 0.979] |
| **Papaya v2** (fst2) | 0.766 ± 0.008 [0.756, 0.776] | 0.930 ± 0.004 [0.925, 0.935] | 0.975 ± 0.004 [0.971, 0.980] |
| **Papaya v3** (fst3) | 0.759 ± 0.003 [0.756, 0.762] | 0.931 ± 0.002 [0.928, 0.934] | 0.977 ± 0.002 [0.974, 0.980] |
| **Litchi v1** (litchi) | 0.769 ± 0.006 [0.761, 0.777] | 0.931 ± 0.004 [0.926, 0.935] | 0.973 ± 0.005 [0.967, 0.979] |
| NepBERTa tokenizer | 0.762 ± 0.010 [0.750, 0.774] | 0.928 ± 0.005 [0.922, 0.935] | 0.978 ± 0.004 [0.974, 0.983] |
| Sakonii tokenizer | 0.757 ± 0.007 [0.748, 0.767] | 0.932 ± 0.003 [0.929, 0.935] | 0.973 ± 0.004 [0.968, 0.978] |
| Llama 3.1 tokenizer | 0.758 | 0.923 | 0.962 |
| Qwen 2.5 tokenizer | 0.769 | 0.922 | 0.956 |

## Paired comparison (same sentences, same seed; bootstrap over sentences, seed-averaged)

Δ = ref − baseline in the task metric; p = share of resamples with Δ ≤ 0 (one-sided). The bootstrap resamples test sentences only; the last column is Δ on the full test set per pretraining seed (fine-tuning seeds sharing a pretraining checkpoint averaged), which is where the pretraining variance shows.

| task | ref | baseline | Δ mean | 95 % interval | p | seeds | Δ per pretraining seed |
|---|---|---|---|---|---|---|---|
| ner | **Papaya** (fst) | plain BPE (fixed) | +0.0274 | [+0.0135, +0.0410] | 0.000 | 5 | +0.032, +0.028, +0.016 |
| ner | **Papaya** (fst) | plain BPE, matched steps (fixed@matched) | +0.0092 | [-0.0028, +0.0215] | 0.073 | 5 | +0.005, +0.006, +0.024 |
| ner | **Papaya** (fst) | plain BPE, mark-splitting (broken) | -0.0014 | [-0.0170, +0.0142] | 0.574 | 5 | -0.010, +0.012, -0.009 |
| ner | **Papaya** (fst) | Unigram LM (uni) | +0.0059 | [-0.0085, +0.0204] | 0.207 | 5 | +0.004, +0.011, -0.001 |
| ner | **Papaya** (fst) | Morfessor + BPE (morf) | +0.0172 | [+0.0043, +0.0302] | 0.004 | 5 | +0.021, +0.016, +0.012 |
| ner | **Papaya** (fst) | Papaya, regex core only | +0.0049 | [-0.0071, +0.0167] | 0.211 | 5 | +0.001, +0.013, -0.005 |
| ner | **Papaya** (fst) | Papaya, regex full only | +0.0027 | [-0.0096, +0.0153] | 0.332 | 5 | +0.000, +0.018, -0.023 |
| ner | **Papaya** (fst) | Papaya, regex full + gate | +0.0020 | [-0.0108, +0.0147] | 0.368 | 5 | +0.002, +0.002, +0.001 |
| ner | **Papaya** (fst) | Papaya + Unigram (fst_uni) | +0.0078 | [-0.0086, +0.0239] | 0.179 | 5 | +0.003, +0.018, -0.001 |
| ner | **Papaya** (fst) | **Papaya v2** (fst2) | -0.0021 | [-0.0127, +0.0084] | 0.637 | 5 | -0.006, +0.006, -0.011 |
| ner | **Papaya** (fst) | **Papaya v3** (fst3) | +0.0050 | [-0.0056, +0.0153] | 0.173 | 5 | +0.004, +0.006, +0.005 |
| ner | **Papaya** (fst) | **Litchi v1** (litchi) | -0.0055 | [-0.0175, +0.0054] | 0.819 | 5 | -0.009, +0.001, -0.011 |
| ner | **Papaya** (fst) | NepBERTa tokenizer | +0.0022 | [-0.0143, +0.0191] | 0.400 | 5 | +0.003, +0.010, -0.016 |
| ner | **Papaya** (fst) | Sakonii tokenizer | +0.0062 | [-0.0091, +0.0208] | 0.201 | 5 | +0.011, +0.006, -0.003 |
| ner | **Papaya** (fst) | Llama 3.1 tokenizer | -0.0021 | [-0.0268, +0.0218] | 0.565 | 1 | -0.002 |
| ner | **Papaya** (fst) | Qwen 2.5 tokenizer | -0.0126 | [-0.0348, +0.0102] | 0.863 | 1 | -0.013 |
| ner | Papaya, regex full only | plain BPE (fixed) | +0.0246 | [+0.0107, +0.0379] | 0.000 | 5 | +0.032, +0.010, +0.040 |
| ner | Papaya, regex full only | plain BPE, matched steps (fixed@matched) | +0.0065 | [-0.0071, +0.0188] | 0.171 | 5 | +0.005, -0.012, +0.047 |
| ner | Papaya, regex full only | plain BPE, mark-splitting (broken) | -0.0040 | [-0.0199, +0.0118] | 0.687 | 5 | -0.010, -0.007, +0.015 |
| ner | Papaya, regex full only | Unigram LM (uni) | +0.0030 | [-0.0121, +0.0176] | 0.330 | 5 | +0.004, -0.007, +0.022 |
| ner | Papaya, regex full only | Morfessor + BPE (morf) | +0.0145 | [+0.0006, +0.0298] | 0.021 | 5 | +0.021, -0.003, +0.035 |
| ner | Papaya, regex full only | Papaya, regex core only | +0.0020 | [-0.0095, +0.0138] | 0.359 | 5 | +0.001, -0.005, +0.018 |
| ner | Papaya, regex full only | Papaya, regex full + gate | -0.0008 | [-0.0118, +0.0105] | 0.556 | 5 | +0.002, -0.016, +0.024 |
| ner | Papaya, regex full only | **Papaya** (fst) | -0.0030 | [-0.0162, +0.0095] | 0.673 | 5 | -0.000, -0.018, +0.023 |
| ner | Papaya, regex full only | Papaya + Unigram (fst_uni) | +0.0055 | [-0.0104, +0.0207] | 0.240 | 5 | +0.003, -0.000, +0.022 |
| ner | Papaya, regex full only | **Papaya v2** (fst2) | -0.0046 | [-0.0155, +0.0063] | 0.785 | 5 | -0.006, -0.012, +0.013 |
| ner | Papaya, regex full only | **Papaya v3** (fst3) | +0.0024 | [-0.0091, +0.0137] | 0.337 | 5 | +0.004, -0.012, +0.028 |
| ner | Papaya, regex full only | **Litchi v1** (litchi) | -0.0081 | [-0.0199, +0.0034] | 0.915 | 5 | -0.009, -0.017, +0.013 |
| ner | Papaya, regex full only | NepBERTa tokenizer | -0.0006 | [-0.0172, +0.0167] | 0.533 | 5 | +0.003, -0.008, +0.007 |
| ner | Papaya, regex full only | Sakonii tokenizer | +0.0040 | [-0.0125, +0.0200] | 0.307 | 5 | +0.011, -0.012, +0.021 |
| ner | Papaya, regex full only | Llama 3.1 tokenizer | +0.0099 | [-0.0135, +0.0328] | 0.199 | 1 | +0.010 |
| ner | Papaya, regex full only | Qwen 2.5 tokenizer | -0.0008 | [-0.0219, +0.0207] | 0.531 | 1 | -0.001 |
| pos | **Papaya** (fst) | plain BPE (fixed) | +0.0010 | [-0.0034, +0.0055] | 0.334 | 5 | +0.001, -0.001, +0.005 |
| pos | **Papaya** (fst) | plain BPE, matched steps (fixed@matched) | +0.0008 | [-0.0030, +0.0043] | 0.337 | 5 | +0.003, -0.001, +0.002 |
| pos | **Papaya** (fst) | plain BPE, mark-splitting (broken) | +0.0027 | [-0.0054, +0.0129] | 0.318 | 5 | +0.008, -0.001, +0.004 |
| pos | **Papaya** (fst) | Unigram LM (uni) | +0.0005 | [-0.0037, +0.0046] | 0.398 | 5 | +0.001, -0.001, +0.003 |
| pos | **Papaya** (fst) | Morfessor + BPE (morf) | -0.0005 | [-0.0035, +0.0026] | 0.619 | 5 | +0.001, -0.001, -0.002 |
| pos | **Papaya** (fst) | Papaya, regex core only | -0.0012 | [-0.0056, +0.0035] | 0.716 | 5 | -0.001, -0.002, -0.001 |
| pos | **Papaya** (fst) | Papaya, regex full only | +0.0001 | [-0.0029, +0.0030] | 0.473 | 5 | +0.002, -0.002, -0.000 |
| pos | **Papaya** (fst) | Papaya, regex full + gate | -0.0013 | [-0.0046, +0.0019] | 0.772 | 5 | -0.002, -0.001, -0.001 |
| pos | **Papaya** (fst) | Papaya + Unigram (fst_uni) | +0.0016 | [-0.0034, +0.0066] | 0.261 | 5 | +0.003, +0.001, +0.001 |
| pos | **Papaya** (fst) | **Papaya v2** (fst2) | -0.0001 | [-0.0037, +0.0033] | 0.504 | 5 | +0.005, -0.004, -0.002 |
| pos | **Papaya** (fst) | **Papaya v3** (fst3) | -0.0011 | [-0.0037, +0.0013] | 0.809 | 5 | +0.001, -0.005, +0.002 |
| pos | **Papaya** (fst) | **Litchi v1** (litchi) | -0.0007 | [-0.0042, +0.0024] | 0.667 | 5 | +0.001, +0.000, -0.006 |
| pos | **Papaya** (fst) | NepBERTa tokenizer | +0.0015 | [-0.0044, +0.0074] | 0.298 | 5 | +0.001, -0.002, +0.009 |
| pos | **Papaya** (fst) | Sakonii tokenizer | -0.0023 | [-0.0065, +0.0012] | 0.895 | 5 | -0.002, -0.004, +0.002 |
| pos | **Papaya** (fst) | Llama 3.1 tokenizer | +0.0098 | [-0.0028, +0.0250] | 0.073 | 1 | +0.011 |
| pos | **Papaya** (fst) | Qwen 2.5 tokenizer | +0.0107 | [-0.0005, +0.0267] | 0.036 | 1 | +0.012 |
| pos | Papaya, regex full only | plain BPE (fixed) | +0.0008 | [-0.0028, +0.0045] | 0.346 | 5 | -0.002, +0.001, +0.005 |
| pos | Papaya, regex full only | plain BPE, matched steps (fixed@matched) | +0.0007 | [-0.0024, +0.0039] | 0.340 | 5 | +0.000, +0.001, +0.003 |
| pos | Papaya, regex full only | plain BPE, mark-splitting (broken) | +0.0023 | [-0.0058, +0.0123] | 0.344 | 5 | +0.006, +0.001, +0.004 |
| pos | Papaya, regex full only | Unigram LM (uni) | +0.0003 | [-0.0041, +0.0047] | 0.432 | 5 | -0.001, +0.000, +0.003 |
| pos | Papaya, regex full only | Morfessor + BPE (morf) | -0.0006 | [-0.0039, +0.0026] | 0.637 | 5 | -0.002, +0.001, -0.002 |
| pos | Papaya, regex full only | Papaya, regex core only | -0.0013 | [-0.0049, +0.0027] | 0.779 | 5 | -0.003, -0.001, -0.000 |
| pos | Papaya, regex full only | Papaya, regex full + gate | -0.0014 | [-0.0049, +0.0017] | 0.798 | 5 | -0.004, +0.001, -0.000 |
| pos | Papaya, regex full only | **Papaya** (fst) | -0.0001 | [-0.0031, +0.0029] | 0.528 | 5 | -0.002, +0.002, +0.000 |
| pos | Papaya, regex full only | Papaya + Unigram (fst_uni) | +0.0015 | [-0.0037, +0.0073] | 0.282 | 5 | +0.001, +0.003, +0.001 |
| pos | Papaya, regex full only | **Papaya v2** (fst2) | -0.0002 | [-0.0036, +0.0034] | 0.529 | 5 | +0.003, -0.002, -0.002 |
| pos | Papaya, regex full only | **Papaya v3** (fst3) | -0.0013 | [-0.0044, +0.0016] | 0.811 | 5 | -0.001, -0.003, +0.002 |
| pos | Papaya, regex full only | **Litchi v1** (litchi) | -0.0008 | [-0.0038, +0.0023] | 0.709 | 5 | -0.001, +0.002, -0.005 |
| pos | Papaya, regex full only | NepBERTa tokenizer | +0.0013 | [-0.0045, +0.0071] | 0.323 | 5 | -0.001, -0.001, +0.010 |
| pos | Papaya, regex full only | Sakonii tokenizer | -0.0023 | [-0.0062, +0.0009] | 0.914 | 5 | -0.004, -0.003, +0.003 |
| pos | Papaya, regex full only | Llama 3.1 tokenizer | +0.0024 | [-0.0108, +0.0174] | 0.382 | 1 | +0.004 |
| pos | Papaya, regex full only | Qwen 2.5 tokenizer | +0.0036 | [-0.0101, +0.0187] | 0.326 | 1 | +0.005 |
| cc | **Papaya** (fst) | plain BPE (fixed) | +0.0028 | [-0.0010, +0.0068] | 0.086 | 5 | +0.006, +0.002, -0.001 |
| cc | **Papaya** (fst) | plain BPE, matched steps (fixed@matched) | +0.0001 | [-0.0041, +0.0043] | 0.498 | 5 | +0.005, -0.002, -0.004 |
| cc | **Papaya** (fst) | plain BPE, mark-splitting (broken) | +0.0110 | [+0.0055, +0.0166] | 0.001 | 5 | +0.014, +0.006, +0.015 |
| cc | **Papaya** (fst) | Unigram LM (uni) | +0.0035 | [-0.0003, +0.0074] | 0.037 | 5 | +0.002, +0.004, +0.006 |
| cc | **Papaya** (fst) | Morfessor + BPE (morf) | +0.0006 | [-0.0026, +0.0039] | 0.362 | 5 | -0.002, +0.002, +0.003 |
| cc | **Papaya** (fst) | Papaya, regex core only | -0.0007 | [-0.0043, +0.0027] | 0.655 | 5 | -0.001, -0.001, +0.001 |
| cc | **Papaya** (fst) | Papaya, regex full only | +0.0001 | [-0.0038, +0.0040] | 0.479 | 5 | -0.001, -0.001, +0.005 |
| cc | **Papaya** (fst) | Papaya, regex full + gate | +0.0019 | [-0.0016, +0.0054] | 0.135 | 5 | +0.004, -0.000, +0.002 |
| cc | **Papaya** (fst) | Papaya + Unigram (fst_uni) | +0.0005 | [-0.0033, +0.0041] | 0.394 | 5 | -0.001, -0.002, +0.007 |
| cc | **Papaya** (fst) | **Papaya v2** (fst2) | -0.0005 | [-0.0043, +0.0031] | 0.613 | 5 | +0.002, -0.003, -0.000 |
| cc | **Papaya** (fst) | **Papaya v3** (fst3) | -0.0021 | [-0.0051, +0.0009] | 0.907 | 5 | -0.001, -0.005, -0.000 |
| cc | **Papaya** (fst) | **Litchi v1** (litchi) | +0.0021 | [-0.0012, +0.0054] | 0.105 | 5 | +0.002, +0.002, +0.003 |
| cc | **Papaya** (fst) | NepBERTa tokenizer | -0.0039 | [-0.0076, -0.0001] | 0.976 | 5 | -0.007, -0.001, -0.003 |
| cc | **Papaya** (fst) | Sakonii tokenizer | +0.0015 | [-0.0025, +0.0056] | 0.232 | 5 | +0.000, +0.003, +0.001 |
| cc | **Papaya** (fst) | Llama 3.1 tokenizer | +0.0162 | [+0.0075, +0.0255] | 0.001 | 1 | +0.016 |
| cc | **Papaya** (fst) | Qwen 2.5 tokenizer | +0.0224 | [+0.0123, +0.0326] | 0.000 | 1 | +0.022 |
| cc | Papaya, regex full only | plain BPE (fixed) | +0.0027 | [-0.0008, +0.0065] | 0.067 | 5 | +0.007, +0.002, -0.006 |
| cc | Papaya, regex full only | plain BPE, matched steps (fixed@matched) | -0.0001 | [-0.0041, +0.0041] | 0.515 | 5 | +0.006, -0.002, -0.009 |
| cc | Papaya, regex full only | plain BPE, mark-splitting (broken) | +0.0108 | [+0.0058, +0.0161] | 0.000 | 5 | +0.015, +0.007, +0.010 |
| cc | Papaya, regex full only | Unigram LM (uni) | +0.0033 | [-0.0014, +0.0079] | 0.081 | 5 | +0.004, +0.004, +0.001 |
| cc | Papaya, regex full only | Morfessor + BPE (morf) | +0.0004 | [-0.0038, +0.0044] | 0.410 | 5 | -0.001, +0.003, -0.002 |
| cc | Papaya, regex full only | Papaya, regex core only | -0.0009 | [-0.0039, +0.0022] | 0.717 | 5 | -0.000, -0.000, -0.004 |
| cc | Papaya, regex full only | Papaya, regex full + gate | +0.0016 | [-0.0026, +0.0060] | 0.223 | 5 | +0.005, +0.000, -0.003 |
| cc | Papaya, regex full only | **Papaya** (fst) | -0.0002 | [-0.0041, +0.0037] | 0.541 | 5 | +0.001, +0.001, -0.005 |
| cc | Papaya, regex full only | Papaya + Unigram (fst_uni) | +0.0002 | [-0.0038, +0.0039] | 0.469 | 5 | +0.001, -0.002, +0.002 |
| cc | Papaya, regex full only | **Papaya v2** (fst2) | -0.0008 | [-0.0036, +0.0019] | 0.700 | 5 | +0.003, -0.003, -0.005 |
| cc | Papaya, regex full only | **Papaya v3** (fst3) | -0.0022 | [-0.0055, +0.0011] | 0.913 | 5 | +0.001, -0.004, -0.005 |
| cc | Papaya, regex full only | **Litchi v1** (litchi) | +0.0019 | [-0.0017, +0.0053] | 0.137 | 5 | +0.003, +0.003, -0.002 |
| cc | Papaya, regex full only | NepBERTa tokenizer | -0.0040 | [-0.0077, -0.0004] | 0.982 | 5 | -0.006, -0.001, -0.008 |
| cc | Papaya, regex full only | Sakonii tokenizer | +0.0012 | [-0.0026, +0.0049] | 0.260 | 5 | +0.001, +0.004, -0.004 |
| cc | Papaya, regex full only | Llama 3.1 tokenizer | +0.0155 | [+0.0073, +0.0243] | 0.001 | 1 | +0.015 |
| cc | Papaya, regex full only | Qwen 2.5 tokenizer | +0.0215 | [+0.0127, +0.0308] | 0.000 | 1 | +0.021 |

## NER slices (span P / R / F1, summed over seeds)

unseen = gold span contains a word absent from the NER training sentences; rare = otherwise contains a word with count < 50 in the pretraining word counts (where BPE fragments most); frequent = neither. Predicted spans are assigned to a slice the same way; n = gold spans per seed.

| variant | unseen P/R/F1 (n) | rare P/R/F1 (n) | frequent P/R/F1 (n) |
|---|---|---|---|
| plain BPE (fixed) | 0.507 / 0.509 / 0.508 (117) | 0.719 / 0.681 / 0.699 (57) | 0.738 / 0.834 / 0.783 (585) |
| plain BPE, matched steps (fixed@matched) | 0.546 / 0.513 / 0.529 (117) | 0.802 / 0.723 / 0.760 (57) | 0.750 / 0.846 / 0.795 (585) |
| plain BPE, mark-splitting (broken) | 0.608 / 0.591 / 0.600 (117) | 0.769 / 0.723 / 0.745 (57) | 0.754 / 0.847 / 0.797 (585) |
| Unigram LM (uni) | 0.581 / 0.588 / 0.585 (117) | 0.776 / 0.695 / 0.733 (57) | 0.753 / 0.839 / 0.793 (585) |
| Morfessor + BPE (morf) | 0.579 / 0.556 / 0.567 (117) | 0.722 / 0.656 / 0.688 (57) | 0.741 / 0.835 / 0.785 (585) |
| Papaya, regex core only | 0.584 / 0.583 / 0.583 (117) | 0.731 / 0.677 / 0.703 (57) | 0.748 / 0.853 / 0.797 (585) |
| Papaya, regex full only | 0.608 / 0.588 / 0.598 (117) | 0.773 / 0.705 / 0.738 (57) | 0.750 / 0.843 / 0.794 (585) |
| Papaya, regex full + gate | 0.589 / 0.588 / 0.589 (117) | 0.765 / 0.719 / 0.741 (57) | 0.751 / 0.847 / 0.796 (585) |
| **Papaya** (fst) | 0.603 / 0.581 / 0.592 (117) | 0.773 / 0.705 / 0.738 (57) | 0.757 / 0.845 / 0.798 (585) |
| Papaya + Unigram (fst_uni) | 0.579 / 0.598 / 0.588 (117) | 0.752 / 0.712 / 0.732 (57) | 0.748 / 0.838 / 0.790 (585) |
| **Papaya v2** (fst2) | 0.625 / 0.588 / 0.606 (117) | 0.784 / 0.688 / 0.733 (57) | 0.759 / 0.841 / 0.798 (585) |
| **Papaya v3** (fst3) | 0.591 / 0.569 / 0.580 (117) | 0.761 / 0.670 / 0.713 (57) | 0.752 / 0.846 / 0.796 (585) |
| **Litchi v1** (litchi) | 0.609 / 0.579 / 0.594 (117) | 0.748 / 0.656 / 0.699 (57) | 0.763 / 0.858 / 0.808 (585) |
| NepBERTa tokenizer | 0.565 / 0.598 / 0.581 (117) | 0.769 / 0.737 / 0.753 (57) | 0.758 / 0.842 / 0.798 (585) |
| Sakonii tokenizer | 0.595 / 0.581 / 0.588 (117) | 0.759 / 0.719 / 0.739 (57) | 0.754 / 0.832 / 0.791 (585) |
| Llama 3.1 tokenizer | 0.619 / 0.598 / 0.609 (117) | 0.709 / 0.684 / 0.696 (57) | 0.749 / 0.838 / 0.791 (585) |
| Qwen 2.5 tokenizer | 0.574 / 0.598 / 0.586 (117) | 0.839 / 0.825 / 0.832 (57) | 0.746 / 0.858 / 0.798 (584) |

### Unseen slice, paired bootstrap

Span F1 on the unseen slice only, same resampling as above (sentences, seed-averaged).

| ref | baseline | Δ mean | 95 % interval | p |
|---|---|---|---|---|
| **Papaya** (fst) | plain BPE (fixed) | +0.0842 | [+0.0395, +0.1333] | 0.000 |
| **Papaya** (fst) | plain BPE, matched steps (fixed@matched) | +0.0633 | [+0.0172, +0.1140] | 0.001 |
| **Papaya** (fst) | plain BPE, mark-splitting (broken) | -0.0080 | [-0.0624, +0.0541] | 0.615 |
| **Papaya** (fst) | Unigram LM (uni) | +0.0071 | [-0.0494, +0.0649] | 0.404 |
| **Papaya** (fst) | Morfessor + BPE (morf) | +0.0236 | [-0.0182, +0.0677] | 0.135 |
| **Papaya** (fst) | Papaya, regex core only | +0.0074 | [-0.0291, +0.0402] | 0.329 |
| **Papaya** (fst) | Papaya, regex full only | -0.0064 | [-0.0583, +0.0512] | 0.606 |
| **Papaya** (fst) | Papaya, regex full + gate | +0.0039 | [-0.0372, +0.0433] | 0.425 |
| **Papaya** (fst) | Papaya + Unigram (fst_uni) | +0.0048 | [-0.0485, +0.0640] | 0.442 |
| **Papaya** (fst) | **Papaya v2** (fst2) | -0.0147 | [-0.0533, +0.0249] | 0.771 |
| **Papaya** (fst) | **Papaya v3** (fst3) | +0.0119 | [-0.0209, +0.0467] | 0.246 |
| **Papaya** (fst) | **Litchi v1** (litchi) | -0.0023 | [-0.0388, +0.0313] | 0.533 |
| **Papaya** (fst) | NepBERTa tokenizer | +0.0113 | [-0.0534, +0.0775] | 0.372 |
| **Papaya** (fst) | Sakonii tokenizer | +0.0039 | [-0.0404, +0.0479] | 0.426 |
| **Papaya** (fst) | Llama 3.1 tokenizer | -0.0248 | [-0.1126, +0.0632] | 0.718 |
| **Papaya** (fst) | Qwen 2.5 tokenizer | -0.0016 | [-0.0801, +0.0823] | 0.520 |
| Papaya, regex full only | plain BPE (fixed) | +0.0906 | [+0.0401, +0.1403] | 0.000 |
| Papaya, regex full only | plain BPE, matched steps (fixed@matched) | +0.0691 | [+0.0169, +0.1230] | 0.004 |
| Papaya, regex full only | plain BPE, mark-splitting (broken) | -0.0029 | [-0.0642, +0.0547] | 0.533 |
| Papaya, regex full only | Unigram LM (uni) | +0.0143 | [-0.0423, +0.0698] | 0.303 |
| Papaya, regex full only | Morfessor + BPE (morf) | +0.0302 | [-0.0194, +0.0823] | 0.121 |
| Papaya, regex full only | Papaya, regex core only | +0.0153 | [-0.0349, +0.0641] | 0.280 |
| Papaya, regex full only | Papaya, regex full + gate | +0.0106 | [-0.0300, +0.0477] | 0.295 |
| Papaya, regex full only | **Papaya** (fst) | +0.0067 | [-0.0517, +0.0590] | 0.399 |
| Papaya, regex full only | Papaya + Unigram (fst_uni) | +0.0113 | [-0.0419, +0.0681] | 0.343 |
| Papaya, regex full only | **Papaya v2** (fst2) | -0.0084 | [-0.0480, +0.0352] | 0.651 |
| Papaya, regex full only | **Papaya v3** (fst3) | +0.0177 | [-0.0295, +0.0612] | 0.215 |
| Papaya, regex full only | **Litchi v1** (litchi) | +0.0048 | [-0.0455, +0.0479] | 0.406 |
| Papaya, regex full only | NepBERTa tokenizer | +0.0178 | [-0.0462, +0.0813] | 0.284 |
| Papaya, regex full only | Sakonii tokenizer | +0.0115 | [-0.0483, +0.0688] | 0.351 |
| Papaya, regex full only | Llama 3.1 tokenizer | +0.0321 | [-0.0402, +0.1000] | 0.181 |
| Papaya, regex full only | Qwen 2.5 tokenizer | +0.0549 | [-0.0205, +0.1317] | 0.078 |

## Data-efficiency curve (mean ± sd over seeds)

### ner

| variant | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|
| plain BPE (fixed) | 0.653 ± 0.020 (5) | 0.686 ± 0.012 (5) | 0.714 ± 0.013 (5) | 0.736 ± 0.009 (5) |
| plain BPE, matched steps (fixed@matched) | – | – | – | 0.755 ± 0.011 (5) |
| plain BPE, mark-splitting (broken) | – | – | – | 0.765 ± 0.010 (5) |
| Unigram LM (uni) | – | – | – | 0.758 ± 0.005 (5) |
| Morfessor + BPE (morf) | – | – | – | 0.747 ± 0.005 (5) |
| Papaya, regex core only | – | – | – | 0.759 ± 0.009 (5) |
| Papaya, regex full only | 0.678 ± 0.014 (5) | 0.706 ± 0.006 (5) | 0.727 ± 0.008 (5) | 0.761 ± 0.015 (5) |
| Papaya, regex full + gate | – | – | – | 0.762 ± 0.003 (5) |
| **Papaya** (fst) | 0.677 ± 0.017 (5) | 0.715 ± 0.005 (5) | 0.735 ± 0.007 (5) | 0.764 ± 0.006 (5) |
| Papaya + Unigram (fst_uni) | – | – | – | 0.756 ± 0.009 (5) |
| **Papaya v2** (fst2) | – | – | – | 0.766 ± 0.008 (5) |
| **Papaya v3** (fst3) | – | – | – | 0.759 ± 0.003 (5) |
| **Litchi v1** (litchi) | – | – | – | 0.769 ± 0.006 (5) |
| NepBERTa tokenizer | 0.686 ± 0.016 (5) | 0.714 ± 0.006 (5) | 0.741 ± 0.011 (5) | 0.762 ± 0.010 (5) |
| Sakonii tokenizer | 0.685 ± 0.010 (5) | 0.710 ± 0.009 (5) | 0.741 ± 0.005 (5) | 0.757 ± 0.007 (5) |
| Llama 3.1 tokenizer | – | – | – | 0.758 |
| Qwen 2.5 tokenizer | – | – | – | 0.769 |

### cc

| variant | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|
| plain BPE (fixed) | 0.958 ± 0.005 (5) | 0.966 ± 0.003 (5) | 0.971 ± 0.003 (5) | 0.972 ± 0.004 (5) |
| plain BPE, matched steps (fixed@matched) | – | – | – | 0.974 ± 0.005 (5) |
| plain BPE, mark-splitting (broken) | – | – | – | 0.964 ± 0.006 (5) |
| Unigram LM (uni) | – | – | – | 0.971 ± 0.003 (5) |
| Morfessor + BPE (morf) | – | – | – | 0.974 ± 0.002 (5) |
| Papaya, regex core only | – | – | – | 0.975 ± 0.002 (5) |
| Papaya, regex full only | 0.963 ± 0.005 (5) | 0.970 ± 0.003 (5) | 0.972 ± 0.005 (5) | 0.974 ± 0.002 (5) |
| Papaya, regex full + gate | – | – | – | 0.973 ± 0.005 (5) |
| **Papaya** (fst) | 0.965 ± 0.003 (5) | 0.965 ± 0.008 (5) | 0.973 ± 0.002 (5) | 0.975 ± 0.004 (5) |
| Papaya + Unigram (fst_uni) | – | – | – | 0.974 ± 0.003 (5) |
| **Papaya v2** (fst2) | – | – | – | 0.975 ± 0.004 (5) |
| **Papaya v3** (fst3) | – | – | – | 0.977 ± 0.002 (5) |
| **Litchi v1** (litchi) | – | – | – | 0.973 ± 0.005 (5) |
| NepBERTa tokenizer | 0.962 ± 0.008 (5) | 0.969 ± 0.004 (5) | 0.972 ± 0.008 (5) | 0.978 ± 0.004 (5) |
| Sakonii tokenizer | 0.958 ± 0.014 (5) | 0.969 ± 0.007 (5) | 0.975 ± 0.003 (5) | 0.973 ± 0.004 (5) |
| Llama 3.1 tokenizer | – | – | – | 0.962 |
| Qwen 2.5 tokenizer | – | – | – | 0.956 |
