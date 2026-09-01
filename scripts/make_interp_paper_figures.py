"""Generate the Interp as a Science main-text figure from committed CSVs.

Three case panels in the appendix line-chart style: boolean layer curves
against a masked-line baseline, iterator baseline vs misleading rename, and
class-site patching recovery against the 0.05 gate.

    uv run python scripts/make_interp_paper_figures.py
"""
from __future__ import annotations

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/interp_science_short/figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e8e7e3"
GATE = 0.05

MODELS = (
    ("qwen2515b", "Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("qwen25coder15b", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-1.5B"),
    ("starcoder27b", "starcoder2-7b", "StarCoder2-7B"),
)
BOOLEAN_JSON = ROOT / "results/boolean/probe"
ITER_ROOT = ROOT / "results/modal/results"
PATCH = (
    ROOT / "results/modal/patching/class-struct-python-v1-20260819"
    / "summaries/Qwen--Qwen2.5-1.5B/float16/cb9960752d1df6cc/eval/summary.csv"
)


def rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def boolean_curves() -> tuple[list[np.ndarray], float]:
    curves = []
    for slug, _folder, _label in MODELS:
        payload = json.loads(
            (BOOLEAN_JSON / f"python_train_{slug}_problem.json").read_text()
        )
        stacked = np.array([seed["test_macro_f1_curve"] for seed in payload["per_seed"]])
        curves.append(stacked.mean(0))
    line = next(
        float(row["baseline_line_masked"])
        for row in rows(ROOT / "results/boolean/probe/summary.csv")
        if row["language"] == "python"
    )
    return curves, line


def iterator_curves() -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for _slug, folder, _label in MODELS:
        data = rows(
            ITER_ROOT / folder / "iterator/perturbation/per_layer.csv"
        )
        by = {}
        for row in data:
            by.setdefault(row["strategy"], []).append(
                (int(row["layer"]), float(row["test_f1"]))
            )
        def series(name: str) -> tuple[np.ndarray, np.ndarray]:
            pts = sorted(by[name])
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            return xs, ys
        out.append((series("baseline"), series("misleading_iterator")))
    return out


def patch_series() -> list[tuple[str, np.ndarray, np.ndarray]]:
    data = [
        row for row in rows(PATCH)
        if row["span"] == "query_name" and row["control"] == "target"
    ]
    out = []
    for direction in ("denoise", "noise"):
        pts = sorted(
            (int(row["layer"]), float(row["recovery"]))
            for row in data
            if row["direction"] == direction
        )
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        out.append((direction, xs, ys))
    return out


def fig_cases() -> None:
    boolean, masked = boolean_curves()
    iterator = iterator_curves()
    patch = patch_series()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.62), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    model_handles = []

    ax = axes[0]
    _style(ax)
    for (slug, folder, label), curve, color in zip(MODELS, boolean, SERIES):
        x = np.arange(len(curve))
        line, = ax.plot(x, curve, color=color, linewidth=1.35, label=label, zorder=3)
        model_handles.append(line)
    ax.axhline(masked, color=MUTED, linewidth=1.05, linestyle=(0, (4, 3)), zorder=2)
    ax.set_title("(a) Boolean, Python", fontsize=8, color=INK, loc="left", pad=3)
    ax.set_xlabel("layer", color=INK2, fontsize=7.5)
    ax.set_ylabel("macro-F1", color=INK2, fontsize=7.5)
    ax.set_ylim(0.84, 1.02)
    ax.set_xticks([0, 8, 16, 24])

    ax = axes[1]
    _style(ax)
    for (_slug, _folder, label), ((bx, by), (mx, my)), color in zip(MODELS, iterator, SERIES):
        ax.plot(bx, by, color=color, linewidth=1.35, linestyle="-", zorder=3)
        ax.plot(mx, my, color=color, linewidth=1.35, linestyle="--", zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.9, linestyle=(0, (2, 3)), zorder=2)
    ax.set_title("(b) Iterator", fontsize=8, color=INK, loc="left", pad=3)
    ax.set_xlabel("layer", color=INK2, fontsize=7.5)
    ax.set_ylabel("macro-F1", color=INK2, fontsize=7.5)
    ax.set_ylim(0.84, 1.02)
    ax.set_xticks([0, 8, 16, 24, 32])

    ax = axes[2]
    _style(ax)
    for (name, xs, ys), color, ls, marker in zip(
        patch, SERIES[:2], ("-", "--"), ("o", "s")
    ):
        ax.plot(
            xs, ys, color=color, linewidth=1.35, linestyle="None",
            marker=marker, markersize=5.0, zorder=3, label=name,
        )
        for x, y in zip(xs, ys):
            ax.hlines(y, x - 1.6, x + 1.6, color=color, linewidth=1.35, zorder=3)
    ax.axhline(GATE, color=MUTED, linewidth=1.05, linestyle=(0, (4, 3)), zorder=2)
    ax.set_title("(c) Class site", fontsize=8, color=INK, loc="left", pad=3)
    ax.set_xlabel("layer", color=INK2, fontsize=7.5)
    ax.set_ylabel("recovery", color=INK2, fontsize=7.5)
    ax.set_ylim(-0.004, 0.065)
    ax.set_xlim(-2, 28)
    ax.set_xticks([0, 18])
    ax.legend(
        loc="lower right", frameon=False, fontsize=6.2,
        handlelength=1.4, borderaxespad=0.05,
    )

    style_handles = [
        plt.Line2D([0], [0], color=INK2, lw=1.35, linestyle="-", label="original names"),
        plt.Line2D([0], [0], color=INK2, lw=1.35, linestyle="--", label="misleading rename"),
        plt.Line2D([0], [0], color=MUTED, lw=1.05, linestyle=(0, (4, 3)), label="reference"),
    ]
    fig.legend(
        model_handles + style_handles,
        [h.get_label() for h in model_handles + style_handles],
        loc="upper center", bbox_to_anchor=(0.5, -0.04),
        ncol=6, frameon=False, fontsize=6.2,
        columnspacing=0.7, handlelength=1.7, handletextpad=0.35,
    )
    fig.tight_layout(pad=0.25, w_pad=0.55)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "same_score.pdf", facecolor=SURFACE,
        bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    fig_cases()


if __name__ == "__main__":
    main()
