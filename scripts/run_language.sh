#!/usr/bin/env bash
# One (language, model, split) pass: gate -> corpus -> occurrences ->
# extraction -> probe -> capped baselines. Every step is idempotent or
# resumable, so re-running after an interruption continues where it stopped.
#
#   bash scripts/run_language.sh Java Qwen/Qwen2.5-Coder-1.5B train
set -euo pipefail
LANGUAGE=${1:?usage: run_language.sh <Language> <model-id> [split]}
MODEL_ID=${2:?usage: run_language.sh <Language> <model-id> [split]}
SPLIT=${3:-train}

slug=$(python3 -c "import sys; print(sys.argv[1].lower().replace('++','pp').replace('#','sharp'))" "$LANGUAGE")
mslug=$(python3 -c "import sys; print(sys.argv[1].split('/')[-1].lower().replace('.','').replace('-',''))" "$MODEL_ID")
CANON=data/xlcost/${slug}_${SPLIT}.jsonl
OCC=outputs/xlcost_occ/${slug}_${SPLIT}.jsonl
STORE=outputs/activations_xlcost/${slug}_${SPLIT}_${mslug}
PROBE=outputs/probe_results/${slug}_${SPLIT}_${mslug}_problem.json
BASE=outputs/probe_results/${slug}_${SPLIT}_${mslug}_baselines_capped.json

echo "=== [1/6] tokenizer gate: $MODEL_ID"
python scripts/tokenizer_gate.py run --models "$MODEL_ID" --strict-version

echo "=== [2/6] corpus: $LANGUAGE/$SPLIT"
[ -s "$CANON" ] || python scripts/xlcost_data.py build \
  --language "$LANGUAGE" --split "$SPLIT" --out-dir data/xlcost

echo "=== [3/6] occurrences"
[ -s "$OCC" ] || python scripts/xlcost_occurrences.py extract \
  --input "$CANON" --output "$OCC"

echo "=== [4/6] activation extraction -> $STORE"
python scripts/extract_activations.py run \
  --canonical "$CANON" --occurrences "$OCC" \
  --model-id "$MODEL_ID" --out-dir "$STORE" --log-every 1000

echo "=== [5/6] probe (problem-grouped, control task, cap 2000)"
python scripts/probe.py run --store "$STORE" \
  --split-policy repo --allow-class-drop --control-task \
  --output "$PROBE"

echo "=== [6/6] baselines on the probe's exact sample"
python scripts/baselines.py run \
  --occurrences "$OCC" --canonical "$CANON" \
  --sample-ids "${PROBE}.sample_ids.json" \
  --split-policy repo --output "$BASE"

echo "=== done: $PROBE"
echo "===       $BASE"
