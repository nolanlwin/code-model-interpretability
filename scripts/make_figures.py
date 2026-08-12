"""Publication figures from probe_results artifacts — never hand-drawn numbers.

Reads the results.json / baselines / delta files that scripts/probe.py,
baselines.py and bootstrap_ci.py emit, and renders the boolean workstream's
figures in the style of results/:

  layer_curves_<lang>_<split>.png     test macro-F1 vs layer, all models,
                                      seed band, strongest-baseline reference
  renaming_deltas_<lang>_<split>.png  paired deltas C1-C5 with 95% CI whiskers
  probe_vs_baselines_<lang>_<split>_<model>.png
                                      probe against the model-free battery

Every figure is regenerated from artifacts, so it inherits their provenance
(git_commit inside the JSONs). Usage:

    uv run python scripts/make_figures.py --results-dir outputs/probe_results \
        --lang python --split valid --out results/boolean
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (dataviz reference instance, slots 1-3 pass
# all-pairs in light mode; aqua carries the relief rule -> direct labels).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"

MODEL_LABELS = {
    "qwen2515b": "Qwen2.5-1.5B",
    "qwen25coder15b": "Qwen2.5-Coder-1.5B",
    "starcoder27b": "StarCoder2-7B",
    "qwen34bbase": "Qwen3-4B-Base",
    "granite3bcodebase2k": "Granite-3B-Code",
}
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]
COND_LABELS = {
    "C1": "C1 neutral\nnumeric", "C2": "C2 single\nchar", "C3": "C3 all-\nsame",
    "C4": "C4 random\nnouns", "C5": "C5 misleading\n(index-style)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover(results_dir: Path, lang: str, split: str):
    """{model_slug: {c0, baselines, deltas{cond}}} from artifact filenames."""
    out: dict = {}
    for f in sorted(glob.glob(str(results_dir / f"{lang}_{split}_*_problem.json"))):
        m = re.match(rf".*{lang}_{split}_(?!C\d)(\w+)_problem\.json$", f)
        if not m:
            continue
        slug = m.group(1)
        entry = {"c0": _load(f), "deltas": {}, "baselines": None}
        bl = results_dir / f"{lang}_{split}_{slug}_baselines_capped.json"
        if bl.is_file():
            entry["baselines"] = _load(bl)
        for c in CONDITIONS:
            d = results_dir / f"{lang}_{split}_{c}_{slug}_delta_vs_C0.json"
            if d.is_file():
                entry["deltas"][c] = _load(d)
        out[slug] = entry
    return out


def strongest_baseline(entry):
    if not entry["baselines"]:
        return None, None
    agg = entry["baselines"]["aggregate"]
    name = max(agg, key=lambda k: agg[k]["macro_f1"] if np.isfinite(agg[k]["macro_f1"]) else -1)
    return name, agg[name]["macro_f1"]


def fig_layer_curves(models: dict, lang: str, split: str, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n_layers = 0
    for i, (slug, entry) in enumerate(models.items()):
        curves = np.array([s["test_macro_f1_curve"] for s in entry["c0"]["per_seed"]])
        mean, std = curves.mean(0), curves.std(0)
        # Models differ in depth (e.g. 28 vs 32 layers); the axis must span the
        # LONGEST curve or the deepest model's tail falls past the last tick.
        n_layers = max(n_layers, curves.shape[1])
        x = np.arange(curves.shape[1])   # this model's depth, not the running max
        color = SERIES[i % len(SERIES)]
        ax.plot(x, mean, color=color, linewidth=2, zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15,
                        linewidth=0, zorder=2)
        ax.annotate(MODEL_LABELS.get(slug, slug), xy=(x[-1], mean[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
    # Shared strongest-baseline reference — drawn ONLY if every model's
    # baseline is the same measurement. Baselines are model-independent in
    # this pipeline (same frozen occurrence sample), but a mismatched sample
    # or class set would make one model's score a false shared reference.
    seen = [(strongest_baseline(e), e["baselines"].get("sample_ids"),
             tuple(e["baselines"].get("classes_used", [])))
            for e in models.values() if e["baselines"]]
    if seen:
        (name, val), sample, classes = seen[0]
        comparable = all(
            abs(v - val) < 1e-9 and n == name and sm == sample and cl == classes
            for (n, v), sm, cl in seen
        )
        if comparable and val is not None:
            ax.axhline(val, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"strongest surface baseline ({name.replace('_', ' ')}) = {val:.3f}",
                        xy=(0.01, val), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=INK2)
        else:
            print("note: models' baselines differ (sample, classes, or value) — "
                  "no shared baseline line drawn; see per-model baseline figures")
    ax.set_xlabel("layer (0 = embedding)", color=INK2, fontsize=10)
    ax.set_ylabel("test macro F1 (mean ± sd over seeds)", color=INK2, fontsize=10)
    ticks = [0] + list(range(4, n_layers, 4))
    ax.set_xticks(ticks, ["emb"] + [str(t) for t in ticks[1:]])
    ax.set_title(f"Boolean occurrence-type probes — {lang}/{split} "
                 f"(problem-grouped splits, {len(models)} model{'s' * (len(models) > 1)})",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    p = out / f"layer_curves_{lang}_{split}.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_renaming_deltas(models: dict, lang: str, split: str, out: Path,
                        ref_delta: float | None = None, ref_label: str = ""):
    with_deltas = {s: e for s, e in models.items() if e["deltas"]}
    if not with_deltas:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n_m = len(with_deltas)
    width = 0.8 / n_m
    for i, (slug, entry) in enumerate(with_deltas.items()):
        xs, ys, lo, hi = [], [], [], []
        for j, c in enumerate(CONDITIONS):
            d = entry["deltas"].get(c)
            if not d:
                continue
            xs.append(j + (i - (n_m - 1) / 2) * width)
            ys.append(d["delta"])
            lo.append(d["delta"] - d["ci_low"])
            hi.append(d["ci_high"] - d["delta"])
        color = SERIES[i % len(SERIES)]
        ax.bar(xs, ys, width=width * 0.9, color=color, zorder=3,
               edgecolor=SURFACE, linewidth=1,
               label=MODEL_LABELS.get(slug, slug))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2,
                    elinewidth=1.2, capsize=2.5, zorder=4)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_xticks(range(len(CONDITIONS)), [COND_LABELS[c] for c in CONDITIONS],
                  fontsize=8.5)
    ax.set_ylabel("paired ΔF1 vs baseline (95% CI)", color=INK2, fontsize=10)
    ax.set_title(f"Renaming does not move the boolean probe — {lang}/{split}",
                 fontsize=11, color=INK, loc="left")
    # Optional cross-role reference. Off by default: a hardcoded constant
    # would be drawn beside artifacts from a different language, split, or
    # model set with no provenance of its own. The caller must pass both the
    # value and a label naming its configuration.
    if ref_delta is not None:
        ax.axhline(ref_delta, color=MUTED, linewidth=1.2, linestyle=(0, (2, 3)), zorder=1)
        ax.annotate(f"{ref_label} ({ref_delta:+.3f})",
                    xy=(0.99, ref_delta), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=INK2, ha="right")
    if len(with_deltas) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    p = out / f"renaming_deltas_{lang}_{split}.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_probe_vs_baselines(slug: str, entry: dict, lang: str, split: str, out: Path):
    if not entry["baselines"]:
        return None
    agg = entry["baselines"]["aggregate"]
    order = ["majority", "covariates_only", "name_only", "window_masked",
             "line_masked", "statement_masked"]
    rows = [(k.replace("_", " "), agg[k]["macro_f1"]) for k in order
            if k in agg and np.isfinite(agg[k]["macro_f1"])]
    c0 = entry["c0"]
    probe_f1 = c0["aggregate"]["test_macro_f1_mean"]
    rows.append((f"probe ({MODEL_LABELS.get(slug, slug)})", probe_f1))

    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(rows) + 1.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.yaxis.grid(False)
    y = np.arange(len(rows))
    vals = [v for _, v in rows]
    colors = [MUTED] * (len(rows) - 1) + [SERIES[0]]
    ax.barh(y, vals, height=0.62, color=colors, zorder=3,
            edgecolor=SURFACE, linewidth=1)
    std = c0["aggregate"].get("test_macro_f1_std")
    for yi, v in zip(y, vals):
        # keep the probe's label clear of its CI whisker
        pad = (std or 0) if yi == y[-1] else 0
        ax.annotate(f"{v:.3f}", xy=(v + pad, yi), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)
    if std:
        ax.errorbar([probe_f1], [y[-1]], xerr=[std], fmt="none",
                    ecolor=INK, elinewidth=1.2, capsize=2.5, zorder=4)
    ax.set_yticks(y, [r for r, _ in rows], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("macro F1 (identical occurrence sample and fold)",
                  color=INK2, fontsize=10)
    ax.set_title(f"Probe vs model-free baselines — {lang}/{split}",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    p = out / f"probe_vs_baselines_{lang}_{split}_{slug}.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="outputs/probe_results")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="results/boolean")
    ap.add_argument("--reference-delta", type=float,
                    help="optional cross-role reference line on the delta plot; "
                         "supply --reference-label naming its configuration")
    ap.add_argument("--reference-label", default="",
                    help="e.g. 'index role, Python/train, Qwen2.5-1.5B'")
    args = ap.parse_args(argv)
    if args.reference_delta is not None and not args.reference_label:
        ap.error("--reference-delta requires --reference-label naming the "
                 "configuration it came from (language, split, model)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = discover(Path(args.results_dir), args.lang, args.split)
    if not models:
        raise SystemExit(f"no {args.lang}_{args.split}_*_problem.json in {args.results_dir}")
    made = [fig_layer_curves(models, args.lang, args.split, out),
            fig_renaming_deltas(models, args.lang, args.split, out,
                                args.reference_delta, args.reference_label)]
    for slug, entry in models.items():
        made.append(fig_probe_vs_baselines(slug, entry, args.lang, args.split, out))
    for p in made:
        if p:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
