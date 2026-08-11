#!/usr/bin/env bash
# The renaming experiment, end to end: for each condition C1-C5, build the
# renamed corpus (restricted to the C0 probe's sampled problems), extract
# activations, probe, and compute the paired delta vs C0. Prints a final
# summary table. Requires a completed run_language.sh pass for the same
# (language, model, split) — its probe results + sample file are the anchor.
#
#   bash scripts/run_renaming.sh Python Qwen/Qwen2.5-Coder-1.5B train
set -euo pipefail
LANGUAGE=${1:?usage: run_renaming.sh <Language> <model-id> [split]}
MODEL_ID=${2:?usage: run_renaming.sh <Language> <model-id> [split]}
SPLIT=${3:-train}

if [ "$LANGUAGE" != "Python" ]; then
  echo "renamer v1 supports Python only (PROTOCOL SS2 rollout order)"; exit 1
fi

slug=$(python3 -c "import sys; print(sys.argv[1].lower().replace('++','pp').replace('#','sharp'))" "$LANGUAGE")
mslug=$(python3 -c "import sys; print(sys.argv[1].split('/')[-1].lower().replace('.','').replace('-',''))" "$MODEL_ID")
CANON=data/xlcost/${slug}_${SPLIT}.jsonl
OCC=outputs/xlcost_occ/${slug}_${SPLIT}.jsonl
C0_PROBE=outputs/probe_results/${slug}_${SPLIT}_${mslug}_problem.json
SAMPLE=${C0_PROBE}.sample_ids.json

C0_STORE=outputs/activations_xlcost/${slug}_${SPLIT}_${mslug}
for f in "$CANON" "$OCC" "$C0_STORE/index.jsonl"; do
  [ -s "$f" ] || { echo "MISSING $f — run: bash scripts/run_language.sh $LANGUAGE $MODEL_ID $SPLIT"; exit 1; }
done

echo "=== [C0] re-probe under the frozen hash split (anchors every delta)"
python scripts/probe.py run --store "$C0_STORE" \
  --split-policy repo --allow-class-drop --control-task --output "$C0_PROBE"

for c in C1 C2 C3 C4 C5; do
  RC=data/xlcost_renamed/$c/${slug}_${SPLIT}.jsonl
  RO=outputs/xlcost_occ_renamed/$c/${slug}_${SPLIT}.jsonl
  STORE=outputs/activations_xlcost/${slug}_${SPLIT}_${c}_${mslug}
  PR=outputs/probe_results/${slug}_${SPLIT}_${c}_${mslug}_problem.json
  DELTA=outputs/probe_results/${slug}_${SPLIT}_${c}_${mslug}_delta_vs_C0.json

  echo "=== [$c 1/4] rename"
  [ -s "$RO" ] || python scripts/rename_corpus.py run --condition "$c" \
    --canonical "$CANON" --occurrences "$OCC" --sample-ids "$SAMPLE" \
    --out-canonical "$RC" --out-occurrences "$RO"

  echo "=== [$c 2/4] extract -> $STORE"
  python scripts/extract_activations.py run \
    --canonical "$RC" --occurrences "$RO" \
    --model-id "$MODEL_ID" --out-dir "$STORE" --log-every 500

  echo "=== [$c 3/4] probe"
  python scripts/probe.py run --store "$STORE" \
    --split-policy repo --allow-class-drop --output "$PR"

  echo "=== [$c 4/4] paired delta vs C0"
  python scripts/bootstrap_ci.py delta "$PR" "$C0_PROBE" --n-boot 2000 > "$DELTA"
done

echo
echo "==================== RENAMING SUMMARY ===================="
python3 - "$slug" "$SPLIT" "$mslug" <<'EOF'
import json, sys
slug, split, mslug = sys.argv[1:4]
c0 = json.load(open(f"outputs/probe_results/{slug}_{split}_{mslug}_problem.json"))
print(f"C0 baseline probe: macroF1 {c0['aggregate']['test_macro_f1_mean']:.4f}")
print(f"{'cond':<6}{'probe F1':>10}{'delta vs C0':>13}{'95% CI':>22}{'n_shared':>10}")
for c in ["C1", "C2", "C3", "C4", "C5"]:
    try:
        p = json.load(open(f"outputs/probe_results/{slug}_{split}_{c}_{mslug}_problem.json"))
        d = json.load(open(f"outputs/probe_results/{slug}_{split}_{c}_{mslug}_delta_vs_C0.json"))
        star = " *" if d.get("excludes_zero") else ""
        print(f"{c:<6}{p['aggregate']['test_macro_f1_mean']:>10.4f}"
              f"{d['delta']:>+13.4f}"
              f"  [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]{star}"
              f"{d.get('n_shared', 0):>8}")
    except FileNotFoundError:
        print(f"{c:<6}  (missing)")
print("* = CI excludes zero. delta = condition minus C0 on shared test occurrences.")
EOF
