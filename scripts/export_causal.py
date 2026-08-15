"""Export causal results into the tracked tree: results/boolean/causal/.

outputs/ is gitignored, so causal runs currently survive only in Drive. This
writes the durable, reviewable part into the repo:

  results/boolean/causal/summary.csv          one row per language/model/mode/layer
  results/boolean/causal/SUMMARY.md           cross-language specificity table
  results/boolean/causal/<run>.json           per-run metadata + summary_by_layer
  results/boolean/causal/*.png                layer profiles, effect vs controls

The per-case arrays are deliberately DROPPED. They are the bulk of the file
(161 cases x 9 layers x 3 models x 3 modes per language) and nothing in the
paper cites an individual case; the summaries and figures are what get read.
Raw runs stay in Drive if anyone needs to go back to them.

    python scripts/export_causal.py --in outputs/causal --out results/boolean/causal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb"
MODEL_LABELS = {
    "qwen2515b": "Qwen2.5-1.5B",
    "qwen25coder15b": "Qwen2.5-Coder-1.5B",
    "starcoder27b": "StarCoder2-7B",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d8d4")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def parse_name(p: Path):
    """<role>_<lang>_<split>_<model>_<mode>.json"""
    stem = p.stem.split("_")
    if len(stem) < 5:
        return None
    return {"role": stem[0], "lang": stem[1], "split": stem[2],
            "model": stem[-2], "mode": stem[-1]}


def specificity(row):
    """|clean - intervened| / |clean - random-position control|.

    The headline number: how much of the effect is specific to the variable's
    own positions rather than to editing anything at all. Undefined when the
    control is missing, which is reported rather than silently treated as 1.
    """
    ctrl = row.get("control_random_position_mean")
    if ctrl is None:
        return None
    d_int = abs(row["clean_mean"] - row["intervened_mean"])
    d_ctl = abs(row["clean_mean"] - ctrl)
    return None if d_ctl < 1e-9 else d_int / d_ctl


def fig_layer_profile(runs, lang, mode, out: Path):
    """Intervened value and both controls against layer, one panel per model."""
    models = sorted(runs)
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6),
                             squeeze=False, facecolor=SURFACE)
    for ax, ms in zip(axes[0], models):
        d = runs[ms]
        rows = d["summary_by_layer"]
        xs = [r["layer"] for r in rows]
        _style(ax)
        ax.plot(xs, [r["clean_mean"] for r in rows], color=MUTED, lw=1.2,
                ls="--", label="clean (no edit)")
        ax.plot(xs, [r["intervened_mean"] for r in rows], color=PALETTE[0],
                lw=2.0, marker="o", ms=3.5, label=mode)
        if any("control_random_position_mean" in r for r in rows):
            ax.plot(xs, [r.get("control_random_position_mean") for r in rows],
                    color=PALETTE[1], lw=1.6, marker="s", ms=3, label="random position")
        if any("control_random_direction_mean" in r for r in rows):
            ax.plot(xs, [r.get("control_random_direction_mean") for r in rows],
                    color=PALETTE[2], lw=1.6, marker="^", ms=3, label="random direction")
        if any("control_same_role_mean" in r for r in rows):
            ax.plot(xs, [r.get("control_same_role_mean") for r in rows],
                    color=INK, lw=1.6, ls=":", marker="d", ms=3, label="same role")
        ax.set_title(f"{MODEL_LABELS.get(ms, ms)}  (n={rows[0]['n']})",
                     color=INK, fontsize=10)
        ax.set_xlabel("layer  (0 = embeddings)", color=INK2, fontsize=9)
    axes[0][0].set_ylabel("logit difference", color=INK2, fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"{lang} — {mode}: an effect must clear its own controls",
                 color=INK, fontsize=11)
    fig.tight_layout()
    p = out / f"layer_profile_{lang}_{mode}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="outputs/causal")
    ap.add_argument("--out", dest="dst", default="results/boolean/causal")
    ap.add_argument("--role", default="boolean")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(f"{args.role}_*.json"))
    if not files:
        print(f"no {args.role} results in {src}")
        return 1

    by_lang_mode = defaultdict(dict)
    csv_rows = []
    for f in files:
        meta = parse_name(f)
        if meta is None:
            continue
        d = json.loads(f.read_text())
        by_lang_mode[(meta["lang"], meta["mode"])][meta["model"]] = d
        # summary-only copy: the per-case array is the bulk and nothing cites it
        slim = {k: v for k, v in d.items() if k != "cases"}
        (dst / f.name).write_text(json.dumps(slim, indent=1))
        for r in d["summary_by_layer"]:
            csv_rows.append({
                "language": meta["lang"], "model": meta["model"], "mode": meta["mode"],
                "layer": r["layer"], "n": r["n"],
                "clean": round(r["clean_mean"], 4),
                "intervened": round(r["intervened_mean"], 4),
                "ctrl_random_position": (None if r.get("control_random_position_mean") is None
                                         else round(r["control_random_position_mean"], 4)),
                "ctrl_random_direction": (None if r.get("control_random_direction_mean") is None
                                          else round(r["control_random_direction_mean"], 4)),
                "ctrl_same_role": (None if r.get("control_same_role_mean") is None
                                   else round(r["control_same_role_mean"], 4)),
                "specificity": (None if specificity(r) is None else round(specificity(r), 3)),
                "cases_scored": d.get("n_cases_scored"),
                "cases_available": d.get("n_cases_available"),
                "git_commit": (d.get("git_commit") or "")[:12],
            })

    with (dst / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    figs = []
    for (lang, mode), runs in sorted(by_lang_mode.items()):
        p = fig_layer_profile(runs, lang, mode, dst)
        if p:
            figs.append(p)

    # Cross-language specificity at each run's best layer.
    lines = ["# Causal interventions — boolean", "",
             "Generated by `scripts/export_causal.py`. Specificity is",
             "`|clean − intervened| / |clean − random-position control|` at the layer",
             "where the intervention is strongest: how much of the effect is specific",
             "to the variable's own token positions rather than to editing anything.",
             "",
             "TWO columns on purpose. The layer with the LARGEST effect is not",
             "generally the layer with the BEST specificity -- on C++/patch the",
             "peak-effect layer scores 3.6x while another layer reaches 6.9x.",
             "Quoting only the first understates the result; quoting only the",
             "second is cherry-picking. Both are shown, with the layer named.",
             "",
             "| language | mode | model | n | clean | spec. @ peak effect | best spec. (layer) |",
             "|---|---|---|---|---|---|---|"]
    for (lang, mode), runs in sorted(by_lang_mode.items()):
        for ms, d in sorted(runs.items()):
            rows_ = d["summary_by_layer"]
            peak = max(rows_, key=lambda r: abs(r["clean_mean"] - r["intervened_mean"]))
            sp_peak = specificity(peak)
            scored = [(specificity(r), r["layer"]) for r in rows_
                      if specificity(r) is not None]
            sp_best, sp_layer = max(scored) if scored else (None, None)
            peak_txt = "—" if sp_peak is None else f"{sp_peak:.1f}× (L{peak['layer']})"
            best_txt = "—" if sp_best is None else f"{sp_best:.1f}× (L{sp_layer})"
            lines.append(
                f"| {lang} | {mode} | {MODEL_LABELS.get(ms, ms)} | "
                f"{peak['n']} | {peak['clean_mean']:.2f} | {peak_txt} | {best_txt} |")
    lines += ["", "Figures: " + ", ".join(f"`{p.name}`" for p in figs)]
    (dst / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {len(files)} run summaries, summary.csv, SUMMARY.md, "
          f"{len(figs)} figures -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
