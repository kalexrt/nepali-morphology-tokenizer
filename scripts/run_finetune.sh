#!/bin/bash
# Fine-tuning sweep on the checkpoints from `SAVE=1 bash scripts/run_lm.sh`:
# every task x variant x seed (x training fraction), PAR at a time, predictions
# dumped for significance.py. Fine-tune seeds 4 and 5 reuse pretraining seeds 1 and 2.
#
#   bash scripts/run_finetune.sh                                   # the paper's 5-seed run
#   TASKS="ner cc" FRACS="0.1 0.25 0.5" VARIANTS="fixed full fst" bash scripts/run_finetune.sh   # data curve
#   TAG=matched VARIANTS=fixed bash scripts/run_finetune.sh        # on the CKPT_TAG=matched checkpoints
#
# Results -> results/downstream_sig.jsonl (OUT=...), predictions -> results/preds/.
set -u
cd "$(dirname "$0")/.." || exit 1
TASKS=${TASKS:-ner pos cc}; SEEDS=${SEEDS:-1 2 3 4 5}; FRACS=${FRACS:-1.0}; PAR=${PAR:-6}
VARIANTS=${VARIANTS:-fixed core full gated fst fst2 broken}
export OUT=${OUT:-results/downstream_sig.jsonl} VOCAB=${VOCAB:-16000} TAG=${TAG:-}
mkdir -p logs
ft() {  # task variant seed frac
  local ps=$(( (($3 - 1) % 3) + 1 ))
  python -u scripts/finetune_tok.py --task "$1" --variant "$2" --vocab "$VOCAB" --pre-seed "$ps" --seed "$3" \
    --train-frac "$4" ${TAG:+--tag $TAG} --dump --out "$OUT" > "logs/ft_$1_$2${TAG:+_$TAG}_s$3_f$4.log" 2>&1 || echo "FAIL $*"
}
export -f ft
for t in $TASKS; do for f in $FRACS; do for s in $SEEDS; do for v in $VARIANTS; do
  echo "$t $v $s $f"
done; done; done; done | xargs -P "$PAR" -L 1 bash -c 'ft "$@"' _
echo "fine-tuning done"
