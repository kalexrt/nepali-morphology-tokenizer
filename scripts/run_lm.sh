#!/bin/bash
# LM sweep: encode every variant's token cache in parallel, then train every
# variant x seed, PAR runs at a time on one GPU. A
# batch-128 run peaks at ~24 GB at 16k vocab and ~34 GB at 32k (use PAR=2).
#
#   VARIANTS="fixed fst fst2" SEEDS="1 2 3" VOCAB=16000 SAVE=1 bash scripts/run_lm.sh
#   EPOCHS=1.164 VARIANTS=fixed bash scripts/run_lm.sh        # matched-steps control at 16k
#   VOCAB=128256 ACCUM=4 PAR=1 SEEDS=1 VARIANTS=x_llama3 bash scripts/run_lm.sh
#   EPOCHS=1.164 SAVE=1 CKPT_TAG=matched VARIANTS=fixed bash scripts/run_lm.sh   # same, with weights (fixed@matched)
#
# Results -> results/lm_bpc_server.jsonl (OUT=...), logs -> logs/.
set -u
cd "$(dirname "$0")/.." || exit 1
BATCH=${BATCH:-128}; LR=${LR:-1.2e-3}; DOCS=${DOCS:-400000}; PAR=${PAR:-3}; VOCAB=${VOCAB:-16000}
SAVE=${SAVE:+--save}                                # SAVE=1 keeps checkpoints/ for finetune_tok.py
TAG=${EPOCHS:+_e$EPOCHS}; EPOCHS=${EPOCHS:-1.0}
ACCUM=${ACCUM:-1}                                   # gradient accumulation for 100k+ vocabularies
CKTAG=${CKPT_TAG:+--tag $CKPT_TAG}                  # checkpoint name suffix, so it does not overwrite the 1-epoch weights
VARIANTS=${VARIANTS:-fixed core full gated fst fst2 broken}
OUT=${OUT:-results/lm_bpc_server.jsonl}
mkdir -p logs
for v in $VARIANTS; do
  python -u scripts/train_lm.py --variant "$v" --vocab "$VOCAB" --docs "$DOCS" --encode-only > "logs/encode_${v}_${VOCAB}_${DOCS}.log" 2>&1 &
done
wait
for s in ${SEEDS:-1 2 3}; do
  for v in $VARIANTS; do echo "$v $s"; done
done | xargs -P "$PAR" -n 2 sh -c 'python -u scripts/train_lm.py --variant "$0" --vocab '"$VOCAB"' --docs '"$DOCS"' --batch '"$BATCH"' --lr '"$LR"' --seed "$1" --epochs '"$EPOCHS"' --accum '"$ACCUM"' '"$SAVE"' '"$CKTAG"' --out '"$OUT"' > "logs/lm_$0_'"$VOCAB"'_s$1'"$TAG"'.log" 2>&1 || echo "FAIL $0 seed $1"'
echo "LM sweep done: $VARIANTS at $VOCAB"
