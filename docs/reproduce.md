# Reproducing the paper

Every number in the paper is a row in `results/*.jsonl`. The two generated
tables are `results/ablation_table.md` and `results/significance.md`. This page
gives the command that produces each result. The scripts take the paper's
variant ids (`fixed`, `fst`, `fst2`, …; see `scripts/common.py`) and write
working files to gitignored folders at the repo root:

- `corpus/`
- `tokenizers/`
- `checkpoints/`
- `downstream/`
- `logs/`

Result files are **append-only**: re-running a script adds rows, and the table
scripts use the latest row for each cell.

```bash
pip install -e ".[corpus,lm,morfessor]"          # + foma for the FST tier (see README)
```

| paper | what | needs |
|---|---|---|
| Tab. 1, 3, 4 (intrinsic columns) | fertility, boundary F1, consistency | CPU; 2–3 min per tokenizer |
| Tab. 1 bpb column, Tab. 2 (vocab sweep, matched steps) | 17M GPT per tokenizer, 3 seeds | GPU; ~15–20 min per run, batch 128, ~24 GB per run at 16k vocab |
| Tab. 1 downstream columns, Tab. 5 (data curve) | fine-tuning, 5 seeds | the LM checkpoints, GPU; under a minute per run |
| Tab. 1, 3 (published tokenizers) | 10 published tokenizers | HF Hub access |

## 0. Corpus and word counts

```bash
python3 scripts/fetch_corpus.py              # FineWeb-2 npi_Deva: test split + train shard 001 -> corpus/
python3 scripts/fetch_corpus.py --check      # sha256 against data/corpus/MANIFEST.json (the paper's files)
papaya wordcounts corpus/train_tok.txt corpus/wordcounts_train_tok.json
```

## 1. Tiers

Two things are shipped and deterministic:
- the regex tier's suffix table, which `papaya build-regex` rebuilds from
  `data/affix_inventory.json`;
- the FST grammars (`papaya/pretrained/papaya-v{1,2}-16k/`).

To re-induce the stem lexicons from the word counts:

```bash
papaya build-fst --counts corpus/wordcounts_train_tok.json --version 1 --out work/fst_v1
papaya build-fst --counts corpus/wordcounts_train_tok.json --version 2 --out work/fst_v2
export PAPAYA_FST_V1=work/fst_v1 PAPAYA_FST_V2=work/fst_v2     # make the scripts use them
diff work/fst_v2/stems.tsv papaya/pretrained/papaya-v2-16k/stems.tsv   # should print nothing
```

## 2. Tokenizers and fertility (Tab. intrinsic, col. 1)

```bash
python3 scripts/train_bpe.py --variants broken fixed core full gated fst fst2 fst3 litchi --vocab 16000 --docs 100000
python3 scripts/train_bpe.py --variants fixed fst --vocab 8000       # vocabulary sweep
python3 scripts/train_bpe.py --variants fixed fst --vocab 32000
```
→ `tokenizers/{variant}_{vocab}.json`, `results/fertility.jsonl`. The shipped
`papaya-v1-16k`, `papaya-v2-16k`, `papaya-regex-16k` and `bpe-baseline-16k` are
the `fst`, `fst2`, `full` and `fixed` outputs of the first command.

## 3. Boundary F1 and consistency (Tab. intrinsic, cols. 2–3)

```bash
python3 scripts/score_boundaries.py --vocab 16000          # also 8000, 32000
python3 scripts/morph_consistency.py --vocab 16000 8000 32000
python3 scripts/tier_error_report.py --profile full        # per-affix precision behind the strict profile
```
→ `results/boundary_f1.jsonl`, `results/morph_consistency.jsonl`. Tier-level
gold-sheet F1 (regex full 0.800, FST v1 hybrid 0.961, v2 0.965) is also checked
by `pytest tests/`.

## 4. Language modelling (Tab. bpb)

17M-parameter GPT (6 layers, 384 wide, context 512), one epoch over the same
400k documents for every tokenizer, batch 128, learning rate 1.2e-3.

```bash
SAVE=1 VARIANTS="broken fixed core full gated fst fst2" bash scripts/run_lm.sh        # 16k, seeds 1-3
VOCAB=8000  VARIANTS="fixed fst" bash scripts/run_lm.sh
VOCAB=32000 PAR=2 VARIANTS="fixed fst" bash scripts/run_lm.sh
# matched-steps control: plain BPE for as many optimizer steps as Papaya takes
EPOCHS=1.116 VOCAB=8000  VARIANTS=fixed bash scripts/run_lm.sh
EPOCHS=1.164 VOCAB=16000 VARIANTS=fixed bash scripts/run_lm.sh
EPOCHS=1.21  VOCAB=32000 PAR=2 VARIANTS=fixed bash scripts/run_lm.sh
```
→ `results/lm_bpc_server.jsonl`. A single run is
`python3 scripts/train_lm.py --variant fst --vocab 16000 --docs 400000 --batch 128 --lr 1.2e-3 --seed 1`.
`python3 scripts/train_lm.py --selfcheck` tests the model code on the CPU in
seconds. `results/lm_bpc.jsonl` holds an earlier, smaller single-GPU protocol
(100k documents, batch 32, lr 6e-4, seed 1).

## 5. Downstream (Tab. downstream, slices, data curve)

```bash
python3 scripts/fetch_downstream.py      # news classification from HF; Nep-gLUE NER/POS by hand (see the script)
bash scripts/run_finetune.sh             # ner pos cc x 7 variants x 5 seeds, --dump
TASKS="ner cc" FRACS="0.1 0.25 0.5" VARIANTS="fixed full fst x_nepberta x_sakonii" bash scripts/run_finetune.sh
python3 scripts/significance.py          # -> results/significance.md
```
→ `results/downstream_sig.jsonl` (the 5-seed rows the paper quotes) and
`results/preds/`. `significance.py` needs the per-sentence predictions in
`results/preds/`. They are not shipped because they contain the task's test
sentences, so the paper's confidence intervals can be recomputed only after
re-running the fine-tunes. `results/downstream_server.jsonl` is the earlier
3-seed sweep; the ablation table uses it only for 8k/32k and the
no-pretraining rows.

## 6. Published tokenizers (Tab. external)

```bash
python3 scripts/external_tokenizers.py --fetch    # tokenizer.json of 10 HF repos -> tokenizers/external/
python3 scripts/external_tokenizers.py            # fertility, boundary F1, consistency -> results/external.jsonl
# the same LM / fine-tunes, with the published vocabulary (grad accumulation for 128k/152k vocabularies)
SAVE=1 PAR=2 VOCAB=30523 VARIANTS=x_nepberta bash scripts/run_lm.sh
SAVE=1 PAR=2 VOCAB=24581 VARIANTS=x_sakonii  bash scripts/run_lm.sh
SAVE=1 PAR=1 ACCUM=4 SEEDS=1 VOCAB=128256 VARIANTS=x_llama3 bash scripts/run_lm.sh
SAVE=1 PAR=1 ACCUM=4 SEEDS=1 VOCAB=151665 VARIANTS=x_qwen25 bash scripts/run_lm.sh
VARIANTS="x_nepberta x_sakonii" bash scripts/run_finetune.sh
```
Some repos require accepting a licence on the Hub first. NepBERTa ships only a
`vocab.txt`, which `--fetch` wraps as a cased WordPiece tokenizer that keeps
vowel signs. Check its fertility against the paper's 1.285 tokens/word before
comparing numbers.

## 7. Baselines: Morfessor, Unigram, equal-steps weights

```bash
pip install -e ".[morfessor]"   # Morfessor is not in the base install
python3 scripts/morfessor_tier.py --selfcheck
python3 scripts/morfessor_tier.py --train     # optional: retrain (seeded); otherwise the shipped data/morfessor_baseline.bin.gz is used
python3 scripts/train_bpe.py --variants uni fst_uni morf --vocab 16000 --docs 100000
python3 scripts/score_boundaries.py --vocab 16000 --only morfessor bpe_morf_16000 bpe_uni_16000 bpe_fst_uni_16000
python3 scripts/morph_consistency.py --vocab 16000 --variants morf uni fst_uni
SAVE=1 VARIANTS="uni morf fst_uni" bash scripts/run_lm.sh
EPOCHS=1.164 SAVE=1 CKPT_TAG=matched VARIANTS=fixed OUT=results/lm_bpc_rerun.jsonl bash scripts/run_lm.sh
VARIANTS="uni morf fst_uni" bash scripts/run_finetune.sh
TAG=matched VARIANTS=fixed bash scripts/run_finetune.sh
```
`morf` marks the cuts of a Morfessor Baseline model trained on the same word
counts (akshara atoms, log-dampened counts, nothing tuned on the gold set).
`uni` is a Unigram LM on plain BPE's pre-tokenizer, `fst_uni` a Unigram LM on
Papaya's marked text. `fixed@matched` in the tables is plain BPE trained for
Papaya's step count, with weights kept, so its fine-tunes are step-matched too.

## 8. Is the gold set anchored on the proposal?

```bash
python3 scripts/gold_anchoring.py --vocab 16000      # -> results/gold_anchoring.jsonl
```
The annotators saw the regex tier's proposal on 422 of the 607 words and
nothing on the other 185 (`proposal_shown` in `data/eval/gold_sheet.tsv`).
The script scores every tier and tokenizer on the blind and the shown rows
separately, with a permutation test for the gap.

## 9. Tables

```bash
python3 scripts/ablation_table.py        # results/*.jsonl -> results/ablation_table.md
```
