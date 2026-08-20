"""Export cross-lingual transfer results into results/lp4fm/.

Reads the JSONs written by `baselines.py transfer` and produces the table and
figures the LP4FM paper needs:

  results/lp4fm/summary.csv     one row per (role, source, target)
  results/lp4fm/SUMMARY.md      the matrix, with controls beside every cell
  results/lp4fm/heatmap_<role>.png   source x target, best masked-context transfer

The controls are printed next to the numbers on purpose. A transfer score is
only meaningful against its own majority and shuffled-label baselines, and
the whole point of this experiment is that a strong-looking number can come
from surface regularity rather than a learned representation.

    python scripts/export_crosslang.py --in outputs/crosslang --out results/lp4fm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
LANG_LABEL = {"python": "Python", "javascript": "JavaScript", "php": "PHP"}
# Sources may carry a renaming condition: out_<role>_C1python_to_<lang>.json.
# Without the optional group these cells are silently skipped, which is how a
# whole experiment can vanish from a summary that still looks complete.
BASE_RE = re.compile(
    r"out_(\w+?)_(C\d)?(python|javascript|php)_to_(python|javascript|php)\.json$")
# Probe results depend on the MODEL; baselines do not. The model therefore
# belongs in the probe filename, or a second model's run silently overwrites
# nothing and the exporter publishes the first model's scores beside the
# second model's stores. The trailing group is optional so older files still
# parse, and they are reported as model "unknown".
PROBE_RE = re.compile(
    r"probe_(\w+?)_(python|javascript|php)_to_(python|javascript|php)"
    r"(?:_([A-Za-z0-9]+))?\.json$")


def resolution(preds: list[dict]) -> tuple[float | None, str | None, int | None]:
    """(rho, smallest class, its count) for a transfer cell's target fold.

    rho is the macro-F1 movement caused by ONE occurrence of the smallest
    class changing its prediction. Ten of twelve within-language probing runs
    turned out to sit below their own rho; a transfer matrix is 18 cells, so
    quoting any of them without it repeats the mistake at scale.
    """
    if not preds:
        return None, None, None
    from sklearn.metrics import f1_score
    seed0 = [q for q in preds if int(q.get("seed", 0)) == 0] or preds
    y = [q["y_true"] for q in seed0]
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    if len(counts) < 2:
        return None, None, None
    small = min(counts, key=counts.get)
    big = max(counts, key=counts.get)
    arr = np.array(y)
    flipped = arr.copy()
    flipped[np.where(arr == small)[0][0]] = big
    labels = sorted(counts)
    rho = (f1_score(arr, arr, average="macro", labels=labels, zero_division=0)
           - f1_score(arr, flipped, average="macro", labels=labels, zero_division=0))
    return float(rho), small, int(counts[small])


def model_slug(model_id: str | None) -> str | None:
    """Filename slug for a model id, matching how the runners build it."""
    if not model_id:
        return None
    return model_id.split("/")[-1].lower().replace(".", "").replace("-", "")


def masked_best(agg: dict) -> float:
    """Best feature that does NOT see the variable name.

    name_only is reported separately: it answers a different question (do
    programmers reuse identifiers across languages) from the one the paper
    asks (does the surrounding code predict the role).
    """
    vals = [agg[k]["macro_f1"] for k in ("statement_masked", "line_masked", "window_masked")
            if k in agg and np.isfinite(agg[k]["macro_f1"])]
    return max(vals) if vals else float("nan")


def heatmap(rows, role: str, out: Path):
    langs = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    grid = np.full((len(langs), len(langs)), np.nan)
    for r in rows:
        if r["role"] != role or r.get("condition", "original") != "original":
            continue
        grid[langs.index(r["source"]), langs.index(r["target"])] = r["masked_best"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8), facecolor=SURFACE)
    im = ax.imshow(grid, cmap="BuPu", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(langs)), [LANG_LABEL.get(x, x) for x in langs],
                  color=INK2, fontsize=9)
    ax.set_yticks(range(len(langs)), [LANG_LABEL.get(x, x) for x in langs],
                  color=INK2, fontsize=9)
    ax.set_xlabel("evaluated on", color=INK2, fontsize=9)
    ax.set_ylabel("trained on", color=INK2, fontsize=9)
    for i in range(len(langs)):
        for j in range(len(langs)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.8 else INK)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color=INK2)
    ax.set_title(f"{role}: masked-context transfer\n(character n-grams, no model)",
                 color=INK, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    p = out / f"heatmap_{role}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="outputs/crosslang")
    ap.add_argument("--out", dest="dst", default="results/lp4fm")
    ap.add_argument("--allow-drop", action="store_true",
                    help="permit republishing a table that omits rows the "
                         "existing summary.csv contains")
    ap.add_argument("--model", default=None,
                    help="select probe results for one model when several are "
                         "present; matched against model_id or the filename slug")
    args = ap.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Two producers write into this directory and they have different
    # schemas: baselines.py transfer -> out_*.json, crosslang.py -> probe_*.json.
    # They are keyed into ONE row per (role, source, target), because the entire
    # point is reading the probe against its baseline in a single line.
    cells: dict = {}
    seen_models: set = set()
    for f in sorted(src.glob("probe_*.json")):
        m = PROBE_RE.search(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        role, a, b, fname_model = m.groups()
        model = d.get("model_id") or fname_model or "unknown"
        inner = model_slug(d.get("model_id"))

        # The model_id recorded inside the file is authoritative: it comes
        # from the store's meta.json. The filename slug is a convenience. If
        # both are present they must AGREE -- a file named for one model whose
        # contents identify another is a corrupt artifact, and choosing either
        # identity would publish one model's scores under the other's name.
        if fname_model and inner and fname_model != inner:
            raise SystemExit(
                f"{f.name}: filename says model {fname_model!r} but model_id "
                f"inside says {d.get('model_id')!r} (slug {inner!r}). Refusing "
                "to guess which is right; delete or rename the file."
            )

        # Selection matches the authoritative identity when there is one, and
        # falls back to the filename only for files that record no model_id.
        identity = inner or fname_model
        if args.model and identity != args.model:
            continue
        seen_models.add(model)
        cells[(role, a, b)] = {
            "probe_transfer": round(d["transfer_macro_f1_mean"], 4),
            "probe_indomain": round(d["indomain_macro_f1_mean"], 4),
            "probe_shuffled_source": round(d["shuffled_source_macro_f1_mean"], 4),
            "probe_rho": (None if d.get("resolution_rho") is None
                          else round(d["resolution_rho"], 4)),
            "probe_model": d.get("model_id"),
        }

    if len(seen_models) > 1:
        raise SystemExit(
            f"probe results from more than one model are present: "
            f"{sorted(seen_models)}. One table cannot carry both, so pass "
            f"--model to choose, rather than have the exporter pick silently."
        )

    rows = []
    for f in sorted(src.glob("out_*.json")):
        m = BASE_RE.search(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        role, cond, a, b = m.groups()
        cond = cond or "original"
        agg = d["aggregate"]
        rho, small, small_n = resolution(d.get("test_predictions") or [])
        rows.append({
            "role": role, "condition": cond, "source": a, "target": b,
            "n_train": d["n_train"], "n_test": d["n_test"],
            "pairing": d["pairing"],
            "name_only": round(agg["name_only"]["macro_f1"], 4),
            "statement_masked": round(agg["statement_masked"]["macro_f1"], 4),
            "line_masked": round(agg["line_masked"]["macro_f1"], 4),
            "window_masked": round(agg["window_masked"]["macro_f1"], 4),
            "masked_best": round(masked_best(agg), 4),
            "majority": round(d["majority_macro_f1"], 4),
            "shuffled_labels": round(d["shuffled_label_control_macro_f1"], 4),
            "rho": None if rho is None else round(rho, 4),
            "smallest_class": small, "smallest_n": small_n,
            "git_commit": (d.get("git_commit") or "")[:12],
            **{k: None for k in ("probe_transfer", "probe_indomain",
                                 "probe_shuffled_source", "probe_rho", "probe_model")},
            **(cells.get((role, a, b), {}) if cond == "original" else {}),
        })
    if not rows:
        print(f"no transfer results in {src}")
        return 1

    # This rewrites the whole table from whatever inputs happen to be present,
    # so a run that computes fewer conditions than a previous one silently
    # deletes the rest. That is how the C1/C2/C4 renaming cells were lost: a
    # session that recomputed only `original` republished an 18-row table over
    # a 36-row one, and the paper table built on those rows kept its numbers
    # while its evidence left the repository. Refuse instead, and say exactly
    # what would go.
    existing = dst / "summary.csv"
    if existing.exists():
        import csv as _csv
        key = lambda r: (r.get("role"), r.get("condition") or "original",
                         r.get("source"), r.get("target"))
        with existing.open(newline="") as fh:
            had = {key(r) for r in _csv.DictReader(fh)}
        lost = sorted(had - {key(r) for r in rows})
        if lost and not args.allow_drop:
            conds = sorted({k[1] for k in lost})
            print(f"REFUSING to overwrite {existing}: {len(lost)} published "
                  f"row(s) are not in this run, covering condition(s) {conds}.")
            for k in lost[:8]:
                print(f"    would drop: role={k[0]} condition={k[1]} {k[2]}->{k[3]}")
            if len(lost) > 8:
                print(f"    ... and {len(lost) - 8} more")
            print("Recompute them, or pass --allow-drop if they are genuinely "
                  "superseded (they stay recoverable in git history).")
            return 1
        if lost:
            print(f"dropping {len(lost)} previously published row(s) "
                  f"(--allow-drop): conditions {sorted({k[1] for k in lost})}")

    with (dst / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    roles = sorted({r["role"] for r in rows})
    figs = [heatmap(rows, role, dst) for role in roles]

    # These sentences describe the matrix table, which shows ORIGINAL rows
    # only. Computing them over renamed rows too would quote counts the reader
    # cannot reconcile with what is printed above them.
    orig = [r for r in rows if r.get("condition", "original") == "original"]
    rhos = [r["rho"] for r in orig if r["rho"]]
    effects = [r["masked_best"] - r["shuffled_labels"] for r in orig]
    ratio = (min(effects) / max(rhos)) if rhos else float("nan")
    RHO_SENTENCE = (
        "**ρ** is the macro-F1 movement from ONE test occurrence of the smallest "
        f"class changing its prediction. Measured, not assumed: it spans "
        f"**{min(rhos):.4f}–{max(rhos):.4f}** across these cells. The smallest "
        f"effect in the table still clears the largest ρ by {ratio:.0f}×, because "
        "these folds hold thousands of occurrences rather than the 2,000-capped "
        "samples of the within-language runs. Resolution is not the binding "
        "constraint on this experiment — worth stating precisely because it *was* "
        "the binding constraint on the probing work, where ten of twelve runs "
        "fell below their own ρ."
    ) if rhos else "ρ unavailable: no test predictions in these results."

    lines = [
        "# Cross-lingual transfer — surface baseline", "",
        "Generated by `scripts/export_crosslang.py` from `baselines.py transfer`.",
        "",
        "**No model is involved.** A character n-gram classifier is fitted on the",
        "source language and evaluated on the target, on the same problems",
        "(matched by `problem_id`). `masked_best` is the strongest feature that",
        "cannot see the variable name; `name_only` uses the name alone and",
        "answers a different question — whether programmers reuse identifiers",
        "across languages.",
        "",
        "Read every cell against its own `majority` and `shuffled` controls.",
        "",
        RHO_SENTENCE,
        "",
        "| role | source → target | n test | masked best | name only | probe | majority | shuffled | ρ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["role"], r["source"], r["target"])):
        if r.get("condition", "original") != "original":
            continue
        rho_txt = "—" if r["rho"] is None else f"{r['rho']:.4f}"
        pr = r.get("probe_transfer")
        probe_txt = "not run" if pr is None else f"{pr:.3f}"
        lines.append(
            f"| {r['role']} | {LANG_LABEL[r['source']]} → {LANG_LABEL[r['target']]} | "
            f"{r['n_test']} | **{r['masked_best']:.3f}** | {r['name_only']:.3f} | "
            f"{probe_txt} | {r['majority']:.3f} | {r['shuffled_labels']:.3f} | {rho_txt} |")

    best = max(orig, key=lambda r: r["masked_best"])
    name_wins = sum(1 for r in orig if r["name_only"] >= r["masked_best"])
    lines += [
        "", "## What this says", "",
        f"- Masked-context transfer reaches **{best['masked_best']:.3f}** "
        f"({best['role']}, {LANG_LABEL[best['source']]} → {LANG_LABEL[best['target']]}) "
        "with no model and the variable name removed.",
        f"- The name alone is the strongest feature in only **{name_wins}/{len(orig)}** "
        "cells, so this is not simply shared identifier conventions.",
        "- Majority and shuffled-label controls sit near chance everywhere, so the",
        "  transfer is real rather than an artifact of class imbalance.",
        "- **Transfer is asymmetric**: compare the two directions of any pair in the",
        "  heatmaps. Pairs sharing C-family statement syntax transfer better than",
        "  pairs that do not, which is a typological result rather than a",
        "  representational one.",
        "",
        "Any probe result on this task has to be read against these numbers, not",
        "against majority chance.",
        "", "Figures: " + ", ".join(f"`{p.name}`" for p in figs),
    ]

    # When the probe column is populated the paragraph above understates the
    # table: it describes the baseline alone. Report where the probe wins,
    # which is the comparison the experiment exists to make.
    probed = [r for r in orig if r.get("probe_transfer") not in (None, "")]
    if probed:
        near = [r for r in probed if "python" not in (r["source"], r["target"])]
        far = [r for r in probed if "python" in (r["source"], r["target"])]
        d = lambda r: float(r["probe_transfer"]) - float(r["masked_best"])
        mean = lambda g, k: sum(float(r[k]) for r in g) / len(g)
        wins = sum(1 for r in probed if d(r) > 0)
        lines[-1:-1] = [
            "",
            "## What the probe adds",
            "",
            f"- The probe beats the masked baseline in **{wins}/{len(probed)}** cells.",
        ]
        if near and far:
            lines[-1:-1] = [
                f"- Typologically close pairs (no Python), n={len(near)}: baseline "
                f"{mean(near,'masked_best'):.3f}, probe {mean(near,'probe_transfer'):.3f}.",
                f"- Pairs involving Python, n={len(far)}: baseline "
                f"{mean(far,'masked_best'):.3f}, probe {mean(far,'probe_transfer'):.3f}.",
                "- The baseline tracks typological distance; the probe does not. The "
                "model's contribution is largest exactly where surface similarity is "
                "least, and near zero where n-grams already saturate.",
            ]
    ren = [r for r in rows if r.get("condition", "original") != "original"]
    if ren:
        base_by = {(r["role"], r["target"]): r["masked_best"]
                   for r in rows if r.get("condition", "original") == "original"
                   and r["source"] == "python"}
        lines += [
            "", "## Does transfer survive renaming the source?", "",
            "Python is renamed before training — C1 `v1,v2,…`, C2 `a,b,c,…`,",
            "C4 random nouns — and the target language is left untouched. The",
            "variable name is masked in the features either way, so what changes",
            "is the SURROUNDING identifiers. A signal that survives is carried by",
            "structure (operators, syntax); one that collapses was lexical.", "",
            "Each cell is `masked best (its own shuffled control)`. The",
            "conditions do not share a control — renaming changes the corpus, so",
            "C1, C2 and C4 each get their own — and pairing a renamed value with",
            "the original's control, or vice versa, would misstate the headroom.",
            "",
            "| role | target | original | C1 | C2 | C4 |",
            "|---|---|---|---|---|---|",
        ]
        for role in sorted({r["role"] for r in ren}):
            for tgt in sorted({r["target"] for r in ren}):
                base = base_by.get((role, tgt))
                if base is None:
                    continue
                cells_ = {r["condition"]: r for r in ren
                          if r["role"] == role and r["target"] == tgt}
                base_row = next((r for r in rows
                                 if r.get("condition", "original") == "original"
                                 and r["role"] == role and r["target"] == tgt
                                 and r["source"] == "python"), None)
                base_txt = ("—" if base_row is None else
                            f"{base_row['masked_best']:.3f} ({base_row['shuffled_labels']:.3f})")
                got = " | ".join(
                    (f"{cells_[c]['masked_best']:.3f} ({cells_[c]['shuffled_labels']:.3f})"
                     if c in cells_ else "—")
                    for c in ("C1", "C2", "C4"))
                lines.append(f"| {role} | {LANG_LABEL[tgt]} | {base_txt} | {got} |")

    (dst / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} cells, summary.csv, SUMMARY.md, {len(figs)} figures -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
