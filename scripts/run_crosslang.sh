#!/usr/bin/env bash
# Cross-lingual surface-baseline matrix: fit on one language, evaluate on
# another, over every ordered pair of the languages whose problems actually
# align. No GPU — this is a character n-gram model.
#
#   bash scripts/run_crosslang.sh accumulator iterator index_key
#
# Only Python / JavaScript / PHP share problem ids at usable rates (2953 /
# 1529 / 1145 pairwise); every other XLCoST pair shares 11-175 of ~9000, so
# matched transfer is not available for them and --matched will refuse.
set -euo pipefail
ROLES=${@:-accumulator iterator index_key}
LANGS="python javascript php"
SPLIT=${SPLIT:-train}
OUT=outputs/crosslang
mkdir -p "$OUT" outputs/role_occ

for slug in $LANGS; do
  case "$slug" in
    python) L=Python ;; javascript) L=Javascript ;; php) L=PHP ;;
  esac
  CANON=data/xlcost/${slug}_${SPLIT}.jsonl
  OCC=outputs/role_occ/all_${slug}_${SPLIT}.jsonl
  [ -s "$CANON" ] || python scripts/xlcost_data.py build \
    --language "$L" --split "$SPLIT" --out-dir data/xlcost
  echo "=== occurrences: $L (all roles)"
  [ -s "$OCC" ] || python scripts/role_occurrences.py extract \
    --input "$CANON" --role all --output "$OCC"
done

for ROLE in $ROLES; do
  for A in $LANGS; do for B in $LANGS; do
    [ "$A" = "$B" ] && continue
    RESULT=${OUT}/out_${ROLE}_${A}_to_${B}.json
    echo "=== $ROLE: $A -> $B"
    [ -s "$RESULT" ] || python scripts/baselines.py transfer \
      --train-occurrences outputs/role_occ/all_${A}_${SPLIT}.jsonl \
      --train-canonical  data/xlcost/${A}_${SPLIT}.jsonl \
      --test-occurrences  outputs/role_occ/all_${B}_${SPLIT}.jsonl \
      --test-canonical   data/xlcost/${B}_${SPLIT}.jsonl \
      --label-field role --role "$ROLE" --matched \
      --output "$RESULT"
  done; done
done

python scripts/export_crosslang.py --in "$OUT" --out results/lp4fm
echo "=== done. results/lp4fm/SUMMARY.md"
