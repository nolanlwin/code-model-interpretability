#!/usr/bin/env bash
# One (role, language, model) causal pass: occurrences -> patch -> ablate ->
# steer, each with its controls. Mirrors run_language.sh, and every step is
# idempotent, so re-running after an interruption continues where it stopped.
#
#   bash scripts/run_causal.sh accumulator Python Qwen/Qwen2.5-Coder-1.5B train
#
# Roles: index_key | accumulator | iterator | boolean | class_struct
#
# NOTE on `boolean`: pipeline/roles.py's boolean extractor is far stricter
# than the boolean workstream's, yielding only a handful of occurrences per
# few hundred programs, so causal runs for that role will be thin until
# issue #16 is settled. The other four roles are unaffected.
set -euo pipefail
ROLE=${1:?usage: run_causal.sh <role> <Language> <model-id> [split]}
LANGUAGE=${2:?usage: run_causal.sh <role> <Language> <model-id> [split]}
MODEL_ID=${3:?usage: run_causal.sh <role> <Language> <model-id> [split]}
SPLIT=${4:-train}

slug=$(python3 -c "import sys; print(sys.argv[1].lower().replace('++','pp').replace('#','sharp'))" "$LANGUAGE")
mslug=$(python3 -c "import sys; print(sys.argv[1].split('/')[-1].lower().replace('.','').replace('-',''))" "$MODEL_ID")
CANON=data/xlcost/${slug}_${SPLIT}.jsonl
# ALL roles in one file: the distractor must hold a different role than the
# target, so a single-role file can only produce zero cases.
OCC=outputs/role_occ/all_${slug}_${SPLIT}.jsonl
BOOL_OCC=outputs/xlcost_occ/${slug}_${SPLIT}.jsonl
OUT=outputs/causal

echo "=== [1/5] self-check (no GPU needed)"
python scripts/causal.py verify

echo "=== [1b] GPU gate: hooks must write where they claim"
python scripts/causal.py sanity --model-id "$MODEL_ID"

echo "=== [2/5] corpus: $LANGUAGE/$SPLIT"
[ -s "${CANON}.stats.json" ] || python scripts/xlcost_data.py build \
  --language "$LANGUAGE" --split "$SPLIT" --out-dir data/xlcost

echo "=== [3/5] occurrences (all roles, shared across target roles)"
mkdir -p outputs/role_occ
# Guard on the COMPLETION MARKER, not on file size: [ -s ] is true for a
# half-written file, so an interrupted run would be silently reused.
[ -s "${OCC}.stats.json" ] || python scripts/role_occurrences.py extract \
  --input "$CANON" --role all --output "$OCC"

mkdir -p "$OUT"
for MODE in patch ablate steer; do
  RESULT=${OUT}/${ROLE}_${slug}_${SPLIT}_${mslug}_${MODE}.json
  echo "=== [$MODE] -> $RESULT"
  [ -s "$RESULT" ] || python scripts/causal.py run \
    --occurrences "$OCC" --canonical "$CANON" \
    --model-id "$MODEL_ID" --intervention "$MODE" \
    --target-role "$ROLE" \
    --output "$RESULT"
done

echo "=== done. results in $OUT/${ROLE}_${slug}_${SPLIT}_${mslug}_*.json"
