# Results

Every number in the paper, one JSON object per line. The files are
**append-only**: a re-run adds rows, and the table scripts take the latest row
per cell (or pool all seeds). Variant ids (`fixed`, `fst`, `fst2`, …) are
explained in `scripts/common.py`. In the prose, `fst` is Papaya v1 and `fst2`
is Papaya v2.

| file | written by | what |
|---|---|---|
| `fertility.jsonl` | `scripts/train_bpe.py` | tokens per word and bytes per token on 5k test documents |
| `boundary_f1.jsonl` | `scripts/score_boundaries.py` | morpheme-boundary P/R/F1 of every tier and tokenizer on the gold and silver sets |
| `morph_consistency.jsonl` | `scripts/morph_consistency.py` | stem-token consistency (gold sample and 51k-form corpus sample) |
| `lm_bpc_server.jsonl` | `scripts/train_lm.py` via `run_lm.sh` | the paper's LM protocol: bits per byte / per char, 400k documents, batch 128, 3 seeds; `epochs > 1` rows are the matched-steps control |
| `lm_bpc.jsonl` | `scripts/train_lm.py` | an earlier, smaller protocol (100k documents, batch 32, 1 seed) |
| `downstream_sig.jsonl` | `scripts/finetune_tok.py` via `run_finetune.sh` | **the downstream numbers to quote**: NER / POS / news classification, 5 seeds, 16k; `train_frac < 1` rows are the data-efficiency curve |
| `downstream_server.jsonl` | `scripts/finetune_tok.py` | the earlier 3-seed sweep (8k / 32k and the no-pretraining rows are only here) |
| `external.jsonl` | `scripts/external_tokenizers.py` | fertility, boundary F1 and consistency of 10 published tokenizers |
| `ablation_table.md` | `scripts/ablation_table.py` | the merged table |
| `significance.md` | `scripts/significance.py` | seed means, paired bootstrap, NER slices, data curve |

The `preds` field in `downstream_sig.jsonl` points to per-sentence predictions
under `results/preds/`. Those files are not shipped, because they contain the
downstream test sentences. `significance.py` therefore needs a re-run of the
fine-tunes (`--dump`) before it can regenerate `significance.md`. The `host`
field says which protocol produced a row: `server` is the main protocol, and
`laptop` the earlier, smaller one.
