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
  [ -s "${CANON}.stats.json" ] || python scripts/xlcost_data.py build \
    --language "$L" --split "$SPLIT" --out-dir data/xlcost
  echo "=== occurrences: $L (all roles)"
  # NOT [ -s "$OCC" ]: that is true for a half-written file, and a run
  # interrupted mid-extraction would then be skipped and silently reused.
  # cmd_extract writes the .stats.json only after the loop completes.
  [ -s "${OCC}.stats.json" ] || python scripts/role_occurrences.py extract \
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

# Probe transfer, if role-labelled activation stores exist. They are built by
#   extract_activations.py --label-field role
# and are the only part of this workflow that needs a GPU. Without them the
# baseline matrix above still stands on its own.
MODEL_SLUG=${MODEL_SLUG:-}
if [ -n "$MODEL_SLUG" ]; then
  for ROLE in $ROLES; do
    for A in $LANGS; do for B in $LANGS; do
      [ "$A" = "$B" ] && continue
      SA=outputs/activations_xlcost/${A}_${SPLIT}_${MODEL_SLUG}
      SB=outputs/activations_xlcost/${B}_${SPLIT}_${MODEL_SLUG}
      if [ -d "$SA" ] && [ -d "$SB" ]; then
        # MODEL_SLUG belongs in the name: a probe result depends on the model,
        # unlike the baselines. Without it, rerunning with a different model
        # finds the previous run's file, skips on existence, and the exporter
        # publishes the old model's scores.
        RESULT=${OUT}/probe_${ROLE}_${A}_to_${B}_${MODEL_SLUG}.json
        echo "=== probe $ROLE: $A -> $B"
        [ -s "$RESULT" ] || python scripts/crosslang.py run \
          --train-store "$SA" --test-store "$SB" --role "$ROLE" --output "$RESULT"
      else
        echo "    skip probe $ROLE $A->$B: no store (build with "
        echo "    extract_activations.py --label-field role)"
      fi
    done; done
  done
fi

# --model so the table cannot mix models, and so a legacy model-free probe
# file from an earlier run is not mistaken for this one.
EXPORT_MODEL=""
[ -n "$MODEL_SLUG" ] && EXPORT_MODEL="--model $MODEL_SLUG"
python scripts/export_crosslang.py --in "$OUT" --out results/lp4fm $EXPORT_MODEL
echo "=== done. results/lp4fm/SUMMARY.md"
