"""Generate the LP4FM appendix tables from the committed CSVs.

The appendix carries evidence that the LP4FM main text cannot accommodate.
Writing them by hand would reintroduce exactly the drift the figure test was
built to prevent, so every table here is emitted from the same files the test
checks.
"""
from __future__ import annotations

import csv
import glob
import pathlib
import statistics as st

R = pathlib.Path("results/lp4fm")
OUT = pathlib.Path("lp4fm_short/appendix_generated.tex")
LANG = {"python": "Python", "javascript": "JavaScript", "php": "PHP",
        "cpp": "C++", "csharp": "C\\#", "java": "Java", "c": "C"}
f = lambda r, k: float(r[k])


def rows(p, probe_only=False):
    rs = list(csv.DictReader(pathlib.Path(p).open()))
    return [r for r in rs if (r.get("probe_transfer") or "").strip()] if probe_only else rs


def overlap_table() -> str:
    ov = {(r["language_a"], r["language_b"]): int(r["shared_problem_ids"])
          for r in rows(R / "xlcost_problem_overlap.csv")}
    names = sorted({k for pair in ov for k in pair})
    head = " & ".join(LANG[n] for n in names)
    out = ["\\begin{table}[htbp]\\centering",
           "\\caption{Shared problem identifiers between every pair of XLCoST "
           "languages, train split. Matched cross-lingual transfer needs a pair to "
           "share enough problems to hold out a test fold. Only the three bold "
           "cells clear $500$. No pair outside Python/JavaScript/PHP does, "
           "including same-family pairs.}",
           "\\label{tab:overlap}",
           f"\\begin{{tabular}}{{l{'r' * len(names)}}}", "\\toprule",
           f" & {head} \\\\", "\\midrule"]
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("N/A")
            else:
                n = ov.get((a, b)) or ov.get((b, a)) or 0
                cells.append(f"\\textbf{{{n:,}}}" if n >= 500 else f"{n:,}")
        out.append(f"{LANG[a]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def full_transfer_table() -> str:
    cap = rows(R / "summary.csv", probe_only=True)
    out = ["\\begin{table}[htbp]\\centering\\small",
           "\\caption{All eighteen transfer cells for Qwen2.5-Coder-1.5B. "
           "\\emph{Surface} is the strongest masked-context $n$-gram feature. "
           "\\emph{Name} uses the variable's name alone. \\emph{Probe} is the "
           "residual-stream probe. Majority and shuffled-label controls sit near "
           "chance throughout. $\\rho$ is a descriptive one-instance score step, "
           "not a confidence interval or hypothesis test.}",
           "\\label{tab:full}",
           "\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "role & pair & $n$ & surface & name & probe & major. & shuf. & $\\rho$ \\\\",
           "\\midrule"]
    for r in sorted(cap, key=lambda r: (r["role"], r["source"], r["target"])):
        out.append(
            # Escaped outside the f-string expression: a backslash inside one
            # is a SyntaxError before Python 3.12, and this project supports
            # 3.11. A 3.12+ interpreter parses it happily, so the failure only
            # appears on the oldest supported version.
            r["role"].replace("_", "\\_") + " & "
            f"{LANG[r['source']]}$\\rightarrow${LANG[r['target']]} & {int(r['n_test']):,} & "
            f"{f(r,'masked_best'):.3f} & {f(r,'name_only'):.3f} & "
            f"{f(r,'probe_transfer'):.3f} & {f(r,'majority'):.3f} & "
            f"{f(r,'shuffled_labels'):.3f} & {f(r,'rho'):.4f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def model_table() -> str:
    out = ["\\begin{table}[htbp]\\centering",
           "\\caption{Probe transfer per model, averaged within each group. The "
           "untrained network shares Qwen2.5-1.5B's architecture and tokenizer "
           "with randomly initialised weights. It is the floor a representational "
           "claim must clear, and it is itself flat.}",
           "\\label{tab:models}",
           "\\begin{tabular}{lrrrr}", "\\toprule",
           "model & close pair & Python pairs & all & vs.\\ untrained \\\\", "\\midrule"]
    tabs = {}
    for d in ["results/lp4fm"] + sorted(glob.glob("results/lp4fm_*")):
        p = pathlib.Path(d) / "summary.csv"
        if not p.exists():
            continue
        rs = rows(p, probe_only=True)
        if rs:
            tabs[rs[0]["probe_model"]] = rs
    rand = next(k for k in tabs if "random-init" in k)
    base = st.mean(f(r, "probe_transfer") for r in tabs[rand])
    for k, rs in sorted(tabs.items(), key=lambda kv: -st.mean(f(r, "probe_transfer") for r in kv[1])):
        near = [r for r in rs if "python" not in (r["source"], r["target"])]
        far = [r for r in rs if "python" in (r["source"], r["target"])]
        m = lambda g: st.mean(f(r, "probe_transfer") for r in g)
        nm = k.split("/")[-1].replace("#random-init-s0", " (untrained)")
        lift = "N/A" if k == rand else f"$+{m(rs)-base:.3f}$"
        out.append(f"{nm} & {m(near):.3f} & {m(far):.3f} & {m(rs):.3f} & {lift} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def masked_probe_tables() -> str:
    """Emit only the per-cell evidence used by the two short papers."""
    conditions = rows(R / "masked_probe" / "conditions.csv")
    paired = rows(R / "masked_probe" / "paired_deltas.csv")
    contrasts = rows(R / "masked_probe" / "boundary_contrasts.csv")
    by_key = {
        (r["condition"], r["role"], r["source"], r["target"]): r
        for r in conditions
    }
    roles = ("accumulator", "index_key", "iterator")
    pairs = (
        ("javascript", "php"), ("php", "javascript"),
        ("javascript", "python"), ("php", "python"),
        ("python", "javascript"), ("python", "php"),
    )

    out = [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Per-cell macro-F1 for the four intersection-sample conditions. "
        "\\emph{Lift} subtracts the context-matched randomly initialized result "
        "from the context-pooled trained result.}",
        "\\label{tab:masked-full}",
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "role & pair & surface & span & context & random & lift \\\\", "\\midrule",
    ]
    for role in roles:
        for source, target in pairs:
            role_tex = role.replace("_", "\\_")

            def score(condition):
                return f(by_key[(condition, role, source, target)], "transfer")

            surface = score("surface_window_masked")
            span = score("qwen25coder15b")
            context = score("qwen25coder15bpoolcontext16")
            random = score("qwen25coder15brandominits0poolcontext16")
            out.append(
                f"{role_tex} & {LANG[source]}$\\rightarrow${LANG[target]} & "
                f"{surface:.3f} & {span:.3f} & {context:.3f} & {random:.3f} & "
                f"{context-random:+.3f} \\\\"
            )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
            "\\begin{table}[htbp]\\centering\\small",
            "\\caption{Problem-clustered paired differences between the "
            "context-pooled probe and the surface comparator. Negative values "
            "favor the surface comparator.}",
            "\\label{tab:paired-deltas}",
            "\\begin{tabular}{llrrr}", "\\toprule",
            "role & pair & $\\Delta$ & CI low & CI high \\\\", "\\midrule"]
    for r in paired:
        role_tex = r["role"].replace("_", "\\_")
        out.append(
            f"{role_tex} & "
            f"{LANG[r['source']]}$\\rightarrow${LANG[r['target']]} & "
            f"{f(r, 'delta'):+.3f} & {f(r, 'ci_low'):+.3f} & "
            f"{f(r, 'ci_high'):+.3f} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    out += [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Problem-clustered uncertainty for the aggregate boundary "
        "contrasts. The effect is Python-transfer minus "
        "JavaScript$\\leftrightarrow$PHP macro-F1. The difference-in-differences "
        "subtracts the untrained boundary from the trained boundary. Intervals "
        "resample problem identifiers and are conditional on the committed model "
        "runs, including one random weight initialization.}",
        "\\label{tab:boundary-contrasts}",
        "\\begin{tabular}{lrr}", "\\toprule",
        "estimand & estimate & 95\\% CI \\\\", "\\midrule",
    ]
    contrast_labels = {
        "span_trained_boundary": "span-pooled trained boundary",
        "context_trained_boundary": "occurrence-excluded trained boundary",
        "context_untrained_boundary": "occurrence-excluded untrained boundary",
        "context_boundary_difference_in_differences": "trained $-$ untrained boundary",
        "context_trained_minus_untrained_close": "training lift, close pair",
        "context_trained_minus_untrained_python": "training lift, Python pairs",
        "boundary_shift_after_excluding_occurrence": "boundary shift after exclusion",
    }
    for row in contrasts:
        out.append(
            f"{contrast_labels[row['estimand']]} & "
            f"{f(row, 'estimate'):+.3f} & "
            f"[{f(row, 'ci_low'):+.3f}, {f(row, 'ci_high'):+.3f}] \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    out += [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Rolewise occurrence-excluded effects and floor-adjusted "
        "transfer. Effects are Python minus close-pair macro-F1. Lift subtracts "
        "the matched untrained score. The final column is close-pair lift minus "
        "Python-transfer lift. These descriptive role summaries do not carry "
        "separate intervals.}",
        "\\label{tab:rolewise-floor-adjusted}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "role & trained effect & random effect & lift close & lift Python & "
        "$\\Delta$ lift \\\\", "\\midrule",
    ]
    for role in roles:
        def group_mean(condition, python_pairs):
            selected = [
                f(by_key[(condition, role, source, target)], "transfer")
                for source, target in pairs
                if ("python" in (source, target)) == python_pairs
            ]
            return st.mean(selected)

        trained_close = group_mean("qwen25coder15bpoolcontext16", False)
        trained_python = group_mean("qwen25coder15bpoolcontext16", True)
        random_close = group_mean(
            "qwen25coder15brandominits0poolcontext16", False
        )
        random_python = group_mean(
            "qwen25coder15brandominits0poolcontext16", True
        )
        lift_close = trained_close - random_close
        lift_python = trained_python - random_python
        role_tex = role.replace("_", "\\_")
        out.append(
            f"{role_tex} & "
            f"{trained_python - trained_close:+.3f} & "
            f"{random_python - random_close:+.3f} & "
            f"{lift_close:+.3f} & {lift_python:+.3f} & "
            f"{lift_close - lift_python:+.3f} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # Replication evidence: the second model's per-cell scores, and the
    # aggregate contrasts for both the second model and the second floor
    # initialization. Each headline number in the paper's replication
    # paragraph is one row here.
    out += [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Per-cell macro-F1 for the StarCoder2-7B replication, on the "
        "same prediction items as the Qwen run. \\emph{Lift} subtracts the "
        "context-matched randomly initialized result from the context-pooled "
        "trained result.}",
        "\\label{tab:masked-full-starcoder}",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "role & pair & span & context & random & lift \\\\", "\\midrule",
    ]
    for role in roles:
        for source, target in pairs:
            role_tex = role.replace("_", "\\_")

            def sc_score(condition):
                return f(by_key[(condition, role, source, target)], "transfer")

            span = sc_score("starcoder27b")
            context = sc_score("starcoder27bpoolcontext16")
            random = sc_score("starcoder27brandominits0poolcontext16")
            out.append(
                f"{role_tex} & {LANG[source]}$\\rightarrow${LANG[target]} & "
                f"{span:.3f} & {context:.3f} & {random:.3f} & "
                f"{context-random:+.3f} \\\\"
            )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    for csv_name, caption, label in (
        ("boundary_contrasts_starcoder27b.csv",
         "Boundary contrasts for the StarCoder2-7B replication, same estimator "
         "and resampling as the Qwen table. Conditional on one random weight "
         "initialization.",
         "tab:boundary-contrasts-starcoder"),
        ("boundary_contrasts_floors1.csv",
         "Qwen2.5-Coder-1.5B boundary contrasts recomputed against the second "
         "untrained weight initialization. Trained conditions are unchanged. "
         "Only the floor differs from the main table.",
         "tab:boundary-contrasts-floor1"),
    ):
        path = R / "masked_probe" / csv_name
        if not path.exists():
            continue
        out += [
            "\\begin{table}[htbp]\\centering\\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrr}", "\\toprule",
            "estimand & estimate & 95\\% CI \\\\", "\\midrule",
        ]
        for row in rows(path):
            out.append(
                f"{contrast_labels[row['estimand']]} & "
                f"{f(row, 'estimate'):+.3f} & "
                f"[{f(row, 'ci_low'):+.3f}, {f(row, 'ci_high'):+.3f}] \\\\"
            )
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    labels = {
        "surface_window_masked": "surface window",
        "qwen25coder15b": "span-pooled trained",
        "qwen25coder15bpoolcontext16": "context-pooled trained",
        "qwen25coder15brandominits0poolcontext16": "context-pooled untrained",
    }
    out += [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Aggregate diagnostics retained in the intersection-sample "
        "masked-probe export. Surface counts are test occurrences. Probe counts "
        "are shared source--target problems. Surface rows have no in-domain or "
        "seed fields. The untrained condition has two probe seeds versus five for "
        "the trained conditions.}",
        "\\label{tab:masked-diagnostics}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "condition & transfer & in-domain & shuffled source & seeds & "
        "cell population \\\\", "\\midrule",
    ]
    for condition in labels:
        group = [r for r in conditions if r["condition"] == condition]

        def optional_mean(key):
            values = [f(r, key) for r in group if r[key]]
            return "N/A" if not values else f"{st.mean(values):.3f}"

        seed_values = sorted({int(r["n_seeds"]) for r in group if r["n_seeds"]})
        seeds = "N/A" if not seed_values else "/".join(map(str, seed_values))
        shared = [int(r["n_shared_problems"]) for r in group if r["n_shared_problems"]]
        if not shared:
            shared_cell = "N/A"
        elif condition == "surface_window_masked":
            shared_cell = f"{min(shared):,}--{max(shared):,} occ."
        else:
            shared_cell = f"{min(shared)}--{max(shared)} problems"
        out.append(
            f"{labels[condition]} & {optional_mean('transfer')} & "
            f"{optional_mean('indomain')} & {optional_mean('shuffled_source')} & "
            f"{seeds} & {shared_cell} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def mechanism_table() -> str:
    m = rows(R / "transfer_mechanism.csv")
    a = rows(R / "transfer_mechanism_ablation.csv")
    out = ["\\begin{table}[htbp]\\centering\\small",
           "\\caption{Transfer mechanism for the iterator role. \\emph{Surviving} is "
           "the share of the source classifier's discriminative mass realised in the "
           "target. \\emph{Flip} is the share of that mass whose sign reverses. "
           "Surviving mass is uncorrelated with transfer. Flipped mass predicts it "
           "almost exactly. Source-weighted variants are given for robustness.}",
           "\\label{tab:mech}",
           "\\begin{tabular}{lrrrrr}", "\\toprule",
           "pair & F1 & surviving & flip & agree & flip (src-wt.) \\\\", "\\midrule"]
    for r in sorted(m, key=lambda r: -f(r, "masked_best_macro_f1")):
        out.append(
            f"{LANG[r['source']]}$\\rightarrow${LANG[r['target']]} & "
            f"{f(r,'masked_best_macro_f1'):.3f} & {f(r,'surviving_mass'):.3f} & "
            f"{f(r,'sign_disagreement_mass'):.3f} & {f(r,'coef_agreement'):+.3f} & "
            f"{f(r,'sign_disagreement_mass_source_weighted'):.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
            "\\begin{table}[htbp]\\centering",
            "\\caption{Masking an entire syntactic character class on both sides of "
            "the transfer. No class accounts for more than $0.021$ of the $0.39$ gap, "
            "so the realignment is distributed rather than carried by block "
            "delimiters or statement terminators.}",
            "\\label{tab:abl}", "\\begin{tabular}{llrr}", "\\toprule",
            "pair & masked class & macro-F1 & $\\Delta$ \\\\", "\\midrule"]
    for r in a:
        out.append(f"{LANG[r['source']]}$\\rightarrow${LANG[r['target']]} & "
                   f"{r['ablated']} & {f(r,'macro_f1'):.4f} & {f(r,'delta'):+.4f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def transfer_interval_table() -> str:
    rs = rows(R / "transfer_intervals.csv")
    out = [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Problem-clustered intervals for every surface transfer cell. "
        "The final row is the Python-pair minus close-pair boundary effect.}",
        "\\label{tab:transfer-intervals}",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "role & pair & macro-F1 & 95\\% CI & problems \\\\", "\\midrule",
    ]
    for r in rs:
        if r["role"] == "ALL":
            pair = "Python pairs $-$ close pair"
        else:
            pair = f"{LANG[r['source']]}$\\rightarrow${LANG[r['target']]}"
        role = r["role"].replace("_", "\\_")
        out.append(
            f"{role} & {pair} & {f(r, 'macro_f1'):.3f} & "
            f"[{f(r, 'ci_low'):.3f}, {f(r, 'ci_high'):.3f}] & "
            f"{int(r['n_problems'])} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def renaming_table() -> str:
    rs = [
        r for r in rows(R / "summary_renaming_uncapped.csv")
        if r["condition"] != "original"
    ]
    out = [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Uncapped renaming audit for Python-source transfer. "
        "C1, C2, and C4 are the committed identifier-renaming conditions. "
        "These samples differ slightly from the frozen main analysis and are "
        "reported as a robustness audit rather than pooled with it.}",
        "\\label{tab:renaming-uncapped}",
        "\\begin{tabular}{lllrrrr}", "\\toprule",
        "role & condition & target & $n_{test}$ & name & surface & majority \\\\",
        "\\midrule",
    ]
    for r in rs:
        role_tex = r["role"].replace("_", "\\_")
        out.append(
            f"{role_tex} & {r['condition']} & "
            f"{LANG[r['target']]} & {int(r['n_test']):,} & "
            f"{f(r, 'name_only'):.3f} & {f(r, 'masked_best'):.3f} & "
            f"{f(r, 'majority'):.3f} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def whitespace_table() -> str:
    rs = rows(R / "whitespace_normalisation_check.csv")
    out = [
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Whitespace-normalisation audit for the three decisive iterator "
        "surface transfers. Scores are unchanged at the displayed precision.}",
        "\\label{tab:whitespace-audit}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "pair & raw & normalised & $\\Delta$ \\\\", "\\midrule",
    ]
    for r in rs:
        out.append(
            f"{LANG[r['source']]}$\\rightarrow${LANG[r['target']]} & "
            f"{f(r, 'masked_best_raw'):.4f} & "
            f"{f(r, 'masked_best_normalised'):.4f} & "
            f"{f(r, 'delta'):+.4f} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def heatmap_figures() -> str:
    roles = [
        ("accumulator", "accumulator"),
        ("index_key", "index/key"),
        ("iterator", "iterator"),
    ]
    out = [
        "\\paragraph{Reading the heatmaps.} Rows are source languages and columns "
        "are target languages. These plots retain the complete available transfer "
        "matrices, including cells excluded from matched-pair inference because "
        "problem overlap is too small."
    ]
    for stem, role in roles:
        out += [
            "\\begin{figure}[htbp]\\centering",
            f"\\includegraphics[width=0.70\\linewidth]{{../results/lp4fm/heatmap_{stem}.png}}",
            f"\\caption{{Available {role} transfer matrix for "
            "Qwen2.5-Coder-1.5B. Cells without adequate matched problem overlap "
            "are descriptive only and are not used for the main boundary estimate.}",
            f"\\label{{fig:lp4fm-{stem}}}",
            "\\end{figure}",
        ]
    out += [
        "\\begin{figure}[htbp]\\centering",
        "\\includegraphics[width=0.72\\linewidth]{figures/transfer_slope.pdf}",
        "\\caption{Span-pooled transfer contrast for the two trained models and "
        "the architecture-matched random-initialization floor. This capped-sample "
        "diagnostic is distinct from the intersection-sample context-pooled analysis.}",
        "\\label{fig:lp4fm-transfer-slope}",
        "\\end{figure}",
    ]
    return "\n".join(out)


def geometry_tables() -> str:
    """Section 3's evidence. Emitted here so the claim that language identity is
    encoded, and the control showing the typing axis is not privileged, are both
    checkable rather than asserted."""
    G = R / "language_geometry"
    if not (G / "pca_by_layer.csv").exists():
        return "% language-geometry results not available"
    lay = rows(G / "pca_by_layer.csv")
    ctrl = {r["metric"]: float(r["value"]) for r in rows(G / "bucket_control.csv")}
    pcs = rows(G / "pc_correlations_layer4.csv")
    peak = max(lay, key=lambda r: f(r, "abs_r"))
    out = ["\\begin{table}[htbp]\\centering",
           "\\caption{Point-biserial correlation between PC1 of last-token residual "
           "activations and a static/dynamic typing label, by layer, with PC1's "
           "explained-variance ratio. The sign of $r$ is arbitrary (PCA fixes the axis "
           "only up to reflection, and each layer is refit independently). Magnitude is "
           "what carries meaning.}",
           "\\label{tab:geomlayers}", "\\begin{tabular}{rrr|rrr}", "\\toprule",
           "layer & $|r|$ & EVR & layer & $|r|$ & EVR \\\\", "\\midrule"]
    half = (len(lay) + 1) // 2
    for a, b in zip(lay[:half], lay[half:] + [None] * half):
        left = f"{int(a['layer'])} & {f(a,'abs_r'):.3f} & {f(a,'evr_pc1'):.3f}"
        right = ("&&" if b is None else
                 f"{int(b['layer'])} & {f(b,'abs_r'):.3f} & {f(b,'evr_pc1'):.3f}")
        out.append(f"{left} & {right} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out += ["\\paragraph{The typing axis is not privileged.} "
            f"Probing the same activations under every alternative $4$-versus-$3$ "
            f"partition of the seven languages "
            f"($\\binom{{7}}{{4}}={int(ctrl['num_combo_controls'])+1}$ groupings minus the "
            f"real one, so {int(ctrl['num_combo_controls'])} controls) gives best-layer "
            f"macro-F1 {ctrl['combo_control_mean_best_f1']:.4f} on average "
            f"(min {ctrl['combo_control_min_best_f1']:.4f}, "
            f"max {ctrl['combo_control_max_best_f1']:.4f}), against "
            f"{ctrl['real_static_dynamic_best_f1']:.4f} for the true static/dynamic "
            f"split. {int(ctrl['combo_controls_at_or_above_real'])} of the "
            f"{int(ctrl['num_combo_controls'])} controls match or exceed it. Any grouping "
            "of these languages is near-perfectly decodable, so the decodability of the "
            "typing split is evidence that language identity is encoded, not that type "
            "discipline is. At layer "
            f"{int(peak['layer'])}, where $|r|$ peaks at {f(peak,'abs_r'):.3f}, the "
            "remaining components carry almost none of the label. "
            + ", ".join(f"PC{int(r['pc'])} $r={f(r,'r'):+.3f}$" for r in pcs[1:4]) + "."]
    return "\n".join(out)


def causal_table() -> str:
    p = pathlib.Path("results/boolean/causal/summary.csv")
    if not p.exists():
        return "% causal summary not available"
    rs = list(csv.DictReader(p.open()))
    langs = sorted({r["language"] for r in rs})
    models = sorted({r["model"] for r in rs})
    modes = sorted({r["mode"] for r in rs})
    head = ("\\paragraph{Scope.} Causal interventions were run for the \\emph{boolean} "
            f"role over {len(langs)} languages "
            f"({', '.join(LANG.get(l,l) for l in langs)}), {len(models)} models, and "
            f"{len(modes)} intervention modes ({', '.join(modes)}), giving "
            f"{len(rs):,} layer-wise measurements. They are reported here rather than in "
            "the main text because they were run \\emph{within} languages and on a role "
            "outside the three this paper transfers, so they cannot support its central "
            "claim. Specificity is $|\\text{clean}-\\text{intervened}|$ divided by the "
            "same quantity for a random-position control. It measures how much of the "
            "effect is specific to the variable's own token positions rather than to "
            "editing anything at all. Two columns are given because the layer with the largest "
            "effect is generally not the layer with the best specificity, and quoting "
            "either alone misleads.")
    # Best specificity per (language, mode, model), and the peak-effect layer.
    best = {}
    for r in rs:
        k = (r["language"], r["mode"], r["model"])
        clean, interv = float(r["clean"]), float(r["intervened"])
        ctrl = float(r["ctrl_random_position"])
        eff = abs(clean - interv)
        den = abs(clean - ctrl)
        spec = eff / den if den else 0.0
        cur = best.setdefault(k, {"eff": -1, "spec": -1, "spec_layer": None})
        if eff > cur["eff"]:
            cur["eff"], cur["eff_spec"] = eff, spec
        if spec > cur["spec"]:
            cur["spec"], cur["spec_layer"] = spec, int(r["layer"])
    out = [head, "", "\\begin{table}[htbp]\\centering\\small",
           "\\caption{Causal interventions on boolean-flag occurrences. "
           "\\emph{Peak-effect spec.} is specificity at the largest-effect layer. "
           "\\emph{best spec.} is the maximum over layers, with that layer named.}",
           "\\label{tab:causal}", "\\begin{tabular}{llrrr}", "\\toprule",
           "language & mode & model & peak-effect spec. & best spec. (layer) \\\\",
           "\\midrule"]
    for (lang, mode, model), v in sorted(best.items()):
        out.append(f"{LANG.get(lang,lang)} & {mode} & {model} & "
                   f"{v.get('eff_spec',0):.1f}$\\times$ & "
                   f"{v['spec']:.1f}$\\times$ (L{v['spec_layer']}) \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


content = "\n\n".join([
    "% GENERATED by scripts/make_appendix.py -- do not edit by hand.",
    "\\FloatBarrier\n\\section{Complete transfer results}\n\\label{app:complete-transfer}",
    full_transfer_table(),
    transfer_interval_table(),
    model_table(),
    "\\FloatBarrier\n\\section{Pairwise problem overlap in XLCoST}\n\\label{app:overlap}",
    overlap_table(),
    "\\FloatBarrier\n\\section{Full masked-probe results}\n\\label{app:full}",
    masked_probe_tables(),
    "\\FloatBarrier\n\\section{Renaming and formatting audits}\n\\label{app:audits}",
    renaming_table(),
    whitespace_table(),
    "\\FloatBarrier\n\\section{Transfer-mechanism diagnostics}\n\\label{app:mechanism}",
    mechanism_table(),
    "\\FloatBarrier\n\\section{Language-geometry controls}\n\\label{app:geometry}",
    geometry_tables(),
    "\\FloatBarrier\n\\section{Complete transfer heatmaps}\n\\label{app:heatmaps}",
    heatmap_figures(),
]) + "\n"
OUT.write_text(content)
print(f"  wrote {OUT}")
