"""Generate the LP4FM short paper's figures from the committed result CSVs.

Separate from scripts/make_figures.py, which is the boolean workstream's
figure generator with its own CLI (--results-dir/--lang/--split/--out) and its
own notebook callers. These two do unrelated jobs on unrelated inputs and must
not share a filename.

Same discipline as the tables: every value is read from results/lp4fm/, never
typed in, so a figure cannot drift from the numbers the text reports.

Palette is the validated categorical set (slots 1-4), checked with the data-viz
validator: lightness band, chroma floor, CVD separation (worst adjacent pair
dE 9.2 deutan) and normal-vision floor all pass. Aqua sits at 2.74:1 against the
surface, below the 3:1 line, which obliges visible labels -- the slope chart
direct-labels every endpoint, so that relief is in place.

Colour is never the only channel. Each series also carries its own marker and
dash pattern, because these figures will be read in a printed, possibly
greyscale, PDF.

    uv run python scripts/make_paper_figures.py
"""
from __future__ import annotations

import csv
import glob
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path("results/lp4fm")
OUT = pathlib.Path("lp4fm_short/figures")
f = lambda r, k: float(r[k])

# Validated categorical slots 1-4, plus recessive ink for chrome.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 7.5,
    "figure.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def probed(path):
    return [r for r in csv.DictReader(pathlib.Path(path).open())
            if (r.get("probe_transfer") or "").strip()]


def groups(rows, key):
    near = [r for r in rows if "python" not in (r["source"], r["target"])]
    far = [r for r in rows if "python" in (r["source"], r["target"])]
    return st.mean(f(r, key) for r in near), st.mean(f(r, key) for r in far)


def fig_transfer():
    """The paper's spine: what tracks the boundary and what does not."""
    cap = probed(R / "summary.csv")
    tabs = {}
    for d in ["results/lp4fm"] + sorted(glob.glob("results/lp4fm_*")):
        p = pathlib.Path(d) / "summary.csv"
        if p.exists() and probed(p):
            rows = probed(p)
            tabs[rows[0]["probe_model"]] = rows
    rand = next(k for k in tabs if "random-init" in k)
    coder = next(k for k in tabs if "Coder" in k)
    base = next(k for k in tabs if k not in (rand, coder))

    # The identifier-alone row is a disclosure, not a fifth series: the probe
    # reads the variable's name and the masked baseline does not, so its score
    # bounds what the name alone supplies. Neutral ink rather than a fifth
    # categorical hue, which would both imply parity and fail the contrast
    # floor in print.
    series = [
        ("Surface $n$-gram (no model)", groups(cap, "masked_best"),   ORANGE, "o", (0, ())),
        ("Identifier alone (no model)", groups(cap, "name_only"),     MUTED,  "v", (0, (2.5, 1.6))),
        ("Probe, Qwen2.5-Coder-1.5B",   groups(tabs[coder], "probe_transfer"), BLUE, "s", (0, (5, 1.6))),
        ("Probe, Qwen2.5-1.5B (base)",  groups(tabs[base], "probe_transfer"),  AQUA, "^", (0, (1.6, 1.6))),
        ("Probe, untrained",            groups(tabs[rand], "probe_transfer"),  VIOLET, "D", (0, (4, 1.4, 1, 1.4))),
    ]

    FIG_H = 1.90
    fig, ax = plt.subplots(figsize=(4.7, FIG_H))
    x = [0, 1]
    # The two trained probes differ by 0.006, so their endpoint labels overlap
    # at both ends. Nudging them apart is the honest fix: the near-coincidence
    # is the finding, and hiding one label would hide it.
    # The minimum gap is in DATA units but the constraint is in points, so it
    # has to be derived from the axis height. A fixed 0.022 was right for a
    # 2.3in figure and let labels collide again when the figure shrank.
    def nudge(values, y_span, height_in, pts_needed=7.5):
        min_gap = y_span * (pts_needed / (height_in * 72.0))
        order = sorted(range(len(values)), key=lambda i: values[i])
        offs = [0.0] * len(values)
        for lo, hi in zip(order, order[1:]):
            gap = values[hi] + offs[hi] - (values[lo] + offs[lo])
            if gap < min_gap:
                offs[hi] += min_gap - gap
        return offs

    lefts = [sv[1][0] for sv in series]
    rights = [sv[1][1] for sv in series]
    Y_LO, Y_HI, H_IN = 0.50, 1.0, FIG_H
    loff = nudge(lefts, Y_HI - Y_LO, H_IN)
    roff = nudge(rights, Y_HI - Y_LO, H_IN)
    for i, (label, (a, b), colour, marker, dash) in enumerate(series):
        ax.plot(x, [a, b], color=colour, lw=1.4, ls=dash, marker=marker,
                ms=4.5, mfc=colour, mec="white", mew=0.9, zorder=3, label=label,
                clip_on=False)
        ax.annotate(f"{a:.3f}", (0, a + loff[i]), textcoords="offset points",
                    xytext=(-7, 0), ha="right", va="center", fontsize=7, color=INK)
        ax.annotate(f"{b:.3f}", (1, b + roff[i]), textcoords="offset points",
                    xytext=(7, 0), ha="left", va="center", fontsize=7, color=INK)

    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks(x)
    # Plain words: "pairs with Python" made the reader work out that it meant
    # every transfer where Python is one of the two languages.
    ax.set_xticklabels(["between JavaScript and PHP", "to or from Python"])
    ax.set_ylabel("cross-lingual macro-F1")
    ax.yaxis.grid(True, color=GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    # Wider figure fits the legend in three columns on one row, which is
    # shorter than two columns on two rows.
    ax.legend(frameon=False, fontsize=6.2, loc="lower left",
              bbox_to_anchor=(-0.09, -0.30), ncol=3, handlelength=2.0,
              columnspacing=0.7, labelspacing=0.25, borderpad=0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "transfer_slope.pdf")
    plt.close(fig)
    return "transfer_slope.pdf"


def fig_mechanism():
    """Which candidate explains the gap: presence, or agreement."""
    mech = list(csv.DictReader((R / "transfer_mechanism.csv").open()))

    def corr(xs, ys):
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return cov / den

    f1 = [f(r, "masked_best_macro_f1") for r in mech]
    # No in-axes captions: they collided with the top-left points in both
    # panels, and the reading belongs in the LaTeX caption where it has room.
    panels = [
        ("surviving mass", "surviving_mass"),
        ("share of mass whose sign flips", "sign_disagreement_mass"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for ax, (xlabel, col) in zip(axes, panels):
        xs = [f(r, col) for r in mech]
        for r, xv, yv in zip(mech, xs, f1):
            near = "python" not in (r["source"], r["target"])
            ax.plot(xv, yv, marker="o" if near else "^", ms=6,
                    mfc=ORANGE if near else BLUE, mec="white", mew=1.0,
                    ls="none", zorder=3)
        ax.set_xlabel(xlabel)
        ax.annotate(f"$r = {corr(xs, f1):+.2f}$", (0.96, 0.06),
                    xycoords="axes fraction", ha="right", fontsize=8.5, color=INK)
        ax.grid(True, color=GRID, lw=0.5, ls="-")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("iterator transfer macro-F1")
    handles = [plt.Line2D([], [], ls="none", marker=m, ms=6, mfc=c, mec="white",
                          mew=1.0, label=l)
               for m, c, l in (("o", ORANGE, "between JavaScript and PHP"),
                               ("^", BLUE, "to or from Python"))]
    axes[1].legend(handles=handles, frameon=False, fontsize=7,
                   loc="lower left", bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "mechanism_scatter.pdf")
    plt.close(fig)
    return "mechanism_scatter.pdf"


def fig_masked():
    """The paper's Table 1 as a slope chart: four conditions, two groups."""
    import csv
    mk = list(csv.DictReader((R / "masked_probe" / "conditions.csv").open()))
    def cond(c):
        g = [r for r in mk if r["condition"] == c]
        near = [float(r["transfer"]) for r in g
                if "python" not in (r["source"], r["target"])]
        far = [float(r["transfer"]) for r in g
               if "python" in (r["source"], r["target"])]
        return st.mean(near), st.mean(far)
    series = [
        ("Surface $n$-gram, name masked",   cond("surface_window_masked"), ORANGE, "o", (0, ())),
        ("Probe, span-pooled (includes occurrence)", cond("qwen25coder15b"), BLUE, "s", (0, (5, 1.6))),
        ("Probe, occurrence-excluded", cond("qwen25coder15bpoolcontext16"), AQUA, "^", (0, (1.6, 1.6))),
        ("Untrained, context-pooled",       cond("qwen25coder15brandominits0poolcontext16"),
                                            VIOLET, "D", (0, (4, 1.4, 1, 1.4))),
    ]
    FIG_H = 1.95
    fig, ax = plt.subplots(figsize=(4.7, FIG_H))
    x = [0, 1]
    Y_LO, Y_HI = 0.42, 0.97

    def nudge(values, y_span, height_in, pts_needed=7.5):
        min_gap = y_span * (pts_needed / (height_in * 72.0))
        order = sorted(range(len(values)), key=lambda i: values[i])
        offs = [0.0] * len(values)
        for lo, hi in zip(order, order[1:]):
            gap = values[hi] + offs[hi] - (values[lo] + offs[lo])
            if gap < min_gap:
                offs[hi] += min_gap - gap
        return offs

    lefts = [sv[1][0] for sv in series]
    rights = [sv[1][1] for sv in series]
    loff = nudge(lefts, Y_HI - Y_LO, FIG_H)
    roff = nudge(rights, Y_HI - Y_LO, FIG_H)
    for i, (label, (a, b), colour, marker, dash) in enumerate(series):
        ax.plot(x, [a, b], color=colour, lw=1.4, ls=dash, marker=marker,
                ms=4.5, mfc=colour, mec="white", mew=0.9, zorder=3, label=label,
                clip_on=False)
        ax.annotate(f"{a:.3f}", (0, a + loff[i]), textcoords="offset points",
                    xytext=(-7, 0), ha="right", va="center", fontsize=7, color=INK)
        ax.annotate(f"{b:.3f}", (1, b + roff[i]), textcoords="offset points",
                    xytext=(7, 0), ha="left", va="center", fontsize=7, color=INK)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks(x)
    ax.set_xticklabels(["between JavaScript and PHP", "to or from Python"])
    ax.set_ylabel("cross-lingual macro-F1")
    ax.yaxis.grid(True, color=GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left",
              bbox_to_anchor=(-0.09, -0.32), ncol=2, handlelength=2.0,
              columnspacing=0.8, labelspacing=0.3, borderpad=0)
    fig.savefig(OUT / "masked_slope.pdf")
    plt.close(fig)
    return "masked_slope.pdf"


if __name__ == "__main__":
    for name in (fig_transfer(), fig_mechanism(), fig_masked()):
        p = OUT / name
        print(f"  wrote {p} ({p.stat().st_size:,} bytes)")
