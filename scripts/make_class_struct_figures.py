"""Figures from unified class_struct CSVs (perturbation + crosslang).

Reads the Modal/unified layout:

    <root>/<model-dir>/class_struct/perturbation/{per_layer,summary,cosine_vs_baseline}.csv
    <root>/<model-dir>/class_struct/crosslang/crosslang.csv

and writes PNGs in the same style as scripts/make_figures.py.

    python scripts/make_class_struct_figures.py \\
        --results-dir results/modal/results \\
        --out results/class_struct
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"

MODELS = (
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-1.5B"),
    ("starcoder2-7b", "StarCoder2-7B"),
)
STRATEGIES = (
    "random_nouns",
    "single_chars",
    "all_same",
    "numeric_vars",
    "misleading_class_struct",
)
STRAT_LABELS = {
    "random_nouns": "random\nnouns",
    "single_chars": "single\nchars",
    "all_same": "all-same\n(layer-0 cheat)",
    "numeric_vars": "numeric\nvars",
    "misleading_class_struct": "misleading\nclass names",
}
LANG_ORDER = ("C++", "Javascript", "C")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def _legend_below(fig, handles=None, labels=None, ncol: int = 3):
    """Legend under the axes so it never covers lines or bars."""
    if handles is None or labels is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=ncol,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.4,
        handlelength=2.4,
    )


def _save(fig, path: Path, *, handles=None, labels=None, ncol: int = 3,
          caption: str | None = None) -> Path:
    fig.tight_layout()
    _legend_below(fig, handles, labels, ncol)
    if caption:
        fig.text(0.5, -0.12, caption, ha="center", fontsize=8, color=INK2)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return path


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_models(root: Path) -> list[dict]:
    out = []
    for folder, label in MODELS:
        pert = root / folder / "class_struct" / "perturbation"
        cross = root / folder / "class_struct" / "crosslang"
        needed = (
            pert / "per_layer.csv",
            pert / "summary.csv",
            pert / "cosine_vs_baseline.csv",
            cross / "crosslang.csv",
        )
        missing = [p for p in needed if not p.is_file()]
        if missing:
            raise SystemExit(
                "missing " + ", ".join(str(p) for p in missing)
            )
        out.append({
            "folder": folder,
            "label": label,
            "per_layer": _read(pert / "per_layer.csv"),
            "summary": {row["strategy"]: row for row in _read(pert / "summary.csv")},
            "cosine": _read(pert / "cosine_vs_baseline.csv"),
            "crosslang": _read(cross / "crosslang.csv"),
        })
    return out


def fig_layer_curves(models: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n_layers = 0
    for i, model in enumerate(models):
        rows = [r for r in model["per_layer"] if r["strategy"] == "baseline"]
        rows.sort(key=lambda r: int(r["layer"]))
        y = np.array([float(r["test_f1"]) for r in rows])
        x = np.arange(len(y))
        n_layers = max(n_layers, len(y))
        color = SERIES[i]
        ax.plot(x, y, color=color, linewidth=2, zorder=3, label=model["label"])
        layer = int(model["summary"]["baseline"]["selected_layer"])
        ax.scatter([layer], [y[layer]], color=color, s=22, zorder=4)
    control = float(models[0]["summary"]["baseline"]["control_f1"])
    ax.set_ylim(0.72, 1.02)
    ticks = [0] + list(range(4, n_layers, 4))
    ax.set_xticks(ticks, ["emb"] + [str(t) for t in ticks[1:]])
    ax.set_xlabel("layer (0 = embedding)", color=INK2, fontsize=10)
    ax.set_ylabel("test macro F1 (baseline names)", color=INK2, fontsize=10)
    ax.set_title(
        "class_struct probes stay high after the embedding",
        fontsize=11, color=INK, loc="left",
    )
    return _save(
        fig, out / "probe_f1_layer_class_struct.png",
        caption=(
            f"Dots mark the val-selected layer. "
            f"Hewitt control F1 = {control:.3f} (below this axis)."
        ),
    )


def fig_renaming_deltas(models: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n_m = len(models)
    width = 0.8 / n_m
    for i, model in enumerate(models):
        xs, ys = [], []
        for j, strat in enumerate(STRATEGIES):
            xs.append(j + (i - (n_m - 1) / 2) * width)
            ys.append(float(model["summary"][strat]["delta_f1_vs_baseline"]))
        ax.bar(
            xs, ys, width=width * 0.9, color=SERIES[i], zorder=3,
            edgecolor=SURFACE, linewidth=1, label=model["label"],
        )
    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_xticks(range(len(STRATEGIES)), [STRAT_LABELS[s] for s in STRATEGIES], fontsize=8)
    ax.set_ylabel("ΔF1 vs original-name baseline", color=INK2, fontsize=10)
    ax.set_title(
        "Renaming barely moves class_struct F1 (all-same is a layer-0 cheat)",
        fontsize=11, color=INK, loc="left",
    )
    return _save(fig, out / "delta_f1_class_struct.png")


def fig_crosslang(models: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n_m = len(models)
    width = 0.8 / n_m
    for i, model in enumerate(models):
        by_lang = {row["language"]: row for row in model["crosslang"]}
        xs, ys = [], []
        for j, lang in enumerate(LANG_ORDER):
            xs.append(j + (i - (n_m - 1) / 2) * width)
            ys.append(float(by_lang[lang]["transfer_f1_at_py_best"]))
        ax.bar(
            xs, ys, width=width * 0.9, color=SERIES[i], zorder=3,
            edgecolor=SURFACE, linewidth=1, label=model["label"],
        )
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.2f}", xy=(x, y), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=7.5, color=INK,
            )
    ax.set_xticks(range(len(LANG_ORDER)), ["C++ (n=606)", "JavaScript (n=285)", "C (n=97)"])
    ax.set_ylim(0.84, 1.02)
    ax.set_ylabel("transfer F1 at Python-selected layer", color=INK2, fontsize=10)
    ax.set_title(
        "Python class_struct probes transfer across languages",
        fontsize=11, color=INK, loc="left",
    )
    return _save(fig, out / "cross_language_class_struct.png")


COSINE_STYLE = {
    "random_nouns": ("random nouns", SERIES[0], "-"),
    "single_chars": ("single chars", SERIES[1], "--"),
    "all_same": ("all-same", MUTED, ":"),
    "numeric_vars": ("numeric", SERIES[2], "-."),
    "misleading_class_struct": ("misleading", INK2, (0, (3, 1, 1, 1))),
}


def fig_cosine(models: list[dict], out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    handles = []
    for ax, model in zip(axes, models):
        _style(ax)
        selected = int(model["summary"]["baseline"]["selected_layer"])
        n = 0
        for row in model["cosine"]:
            name, color, ls = COSINE_STYLE[row["strategy"]]
            ys = [float(row[k]) for k in row if k.startswith("layer_")]
            n = len(ys)
            line, = ax.plot(
                np.arange(n), ys, color=color, linewidth=1.6, linestyle=ls,
                label=name, zorder=3,
            )
            if ax is axes[0]:
                handles.append(line)
        ax.axvline(selected, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
        ax.set_title(model["label"], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("layer", color=INK2, fontsize=9)
        ax.set_ylim(-0.08, 0.62)
        ax.set_xticks([0] + list(range(8, n, 8)))
    axes[0].set_ylabel("cosine vs original-name probe", color=INK2, fontsize=9)
    fig.suptitle(
        "Renamed probes stay aligned; all-same is a different (identity) direction",
        fontsize=11, color=INK, x=0.02, ha="left", y=1.02,
    )
    labels = [h.get_label() for h in handles]
    return _save(
        fig, out / "cosine_vs_baseline_class_struct.png",
        handles=handles, labels=labels, ncol=5,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/modal/results")
    ap.add_argument("--out", default="results/class_struct")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(Path(args.results_dir))
    made = [
        fig_layer_curves(models, out),
        fig_renaming_deltas(models, out),
        fig_crosslang(models, out),
        fig_cosine(models, out),
    ]
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
