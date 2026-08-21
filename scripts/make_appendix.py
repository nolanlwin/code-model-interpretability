"""Generate the LP4FM appendix tables from the committed CSVs.

The appendices exist to carry the evidence the four-page main text cannot.
Writing them by hand would reintroduce exactly the drift the figure test was
built to prevent, so every table here is emitted from the same files the test
checks, and lp4fm_short/appendix_generated.tex is never edited directly.
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
    out = ["\\begin{table}[h]\\centering",
           "\\caption{Shared problem identifiers between every pair of XLCoST "
           "languages, train split. Matched cross-lingual transfer needs a pair to "
           "share enough problems to hold out a test fold; only the three bold "
           "cells clear $500$. No pair outside Python/JavaScript/PHP does, "
           "including same-family pairs.}",
           "\\label{tab:overlap}",
           f"\\begin{{tabular}}{{l{'r' * len(names)}}}", "\\toprule",
           f" & {head} \\\\", "\\midrule"]
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("---")
            else:
                n = ov.get((a, b)) or ov.get((b, a)) or 0
                cells.append(f"\\textbf{{{n:,}}}" if n >= 500 else f"{n:,}")
        out.append(f"{LANG[a]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def full_transfer_table() -> str:
    cap = rows(R / "summary.csv", probe_only=True)
    out = ["\\begin{table}[h]\\centering\\small",
           "\\caption{All eighteen transfer cells for Qwen2.5-Coder-1.5B. "
           "\\emph{Surface} is the strongest masked-context $n$-gram feature; "
           "\\emph{name} uses the variable's name alone; \\emph{probe} is the "
           "residual-stream probe. Majority and shuffled-label controls sit near "
           "chance throughout, and $\\rho$ is the macro-F1 movement from one test "
           "occurrence changing its prediction.}",
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
    out = ["\\begin{table}[h]\\centering",
           "\\caption{Probe transfer per model, averaged within each group. The "
           "untrained network shares Qwen2.5-1.5B's architecture and tokenizer "
           "with randomly initialised weights; it is the floor a representational "
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
        lift = "---" if k == rand else f"$+{m(rs)-base:.3f}$"
        out.append(f"{nm} & {m(near):.3f} & {m(far):.3f} & {m(rs):.3f} & {lift} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def mechanism_table() -> str:
    m = rows(R / "transfer_mechanism.csv")
    fig = ["\\begin{figure}[h]\\centering",
           "\\includegraphics{figures/mechanism_scatter.pdf}",
           "\\caption{Two candidate explanations for the transfer gap, over the six "
           "ordered pairs. Left: how much of the source classifier's discriminative "
           "mass the target realises, which is essentially unrelated to transfer. "
           "Right: the share of that mass whose sign reverses between a "
           "source-fitted and a target-fitted classifier, which tracks it closely. "
           "The cues are present in both groups; what changes is where they point.}",
           "\\label{fig:mech}", "\\end{figure}", ""]
    a = rows(R / "transfer_mechanism_ablation.csv")
    out = ["\\begin{table}[h]\\centering\\small",
           "\\caption{Transfer mechanism for the iterator role. \\emph{Surviving} is "
           "the share of the source classifier's discriminative mass realised in the "
           "target; \\emph{flip} is the share of that mass whose sign reverses. "
           "Surviving mass is uncorrelated with transfer; flipped mass predicts it "
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
    out = fig + out
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
            "\\begin{table}[h]\\centering",
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
    out = ["\\begin{table}[h]\\centering",
           "\\caption{Point-biserial correlation between PC1 of last-token residual "
           "activations and a static/dynamic typing label, by layer, with PC1's "
           "explained-variance ratio. The sign of $r$ is arbitrary (PCA fixes the axis "
           "only up to reflection, and each layer is refit independently); magnitude is "
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
            f"split; {int(ctrl['combo_controls_at_or_above_real'])} of the "
            f"{int(ctrl['num_combo_controls'])} controls match or exceed it. Any grouping "
            "of these languages is near-perfectly decodable, so the decodability of the "
            "typing split is evidence that language identity is encoded, not that type "
            "discipline is. At layer "
            f"{int(peak['layer'])}, where $|r|$ peaks at {f(peak,'abs_r'):.3f}, the "
            "remaining components carry almost none of the label: "
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
            "same quantity for a random-position control: how much of the effect is "
            "specific to the variable's own token positions rather than to editing "
            "anything at all. Two columns are given because the layer with the largest "
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
    out = [head, "", "\\begin{table}[h]\\centering\\small",
           "\\caption{Causal interventions on boolean-flag occurrences. "
           "\\emph{Peak-effect spec.} is specificity at the largest-effect layer; "
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


OUT.write_text("\n\n".join([
    "% GENERATED by scripts/make_appendix.py -- do not edit by hand.",
    "\\section{Pairwise problem overlap in XLCoST}\n\\label{app:overlap}",
    overlap_table(),
    "\\section{Full transfer table}\n\\label{app:full}",
    full_transfer_table(), model_table(),
    "\\section{Transfer mechanism}\n\\label{app:mech}",
    mechanism_table(),
    "\\section{Language geometry}\n\\label{app:geometry}",
    geometry_tables(),
    "\\section{Causal interventions}\n\\label{app:causal}",
    causal_table(),
]) + "\n")
print(f"  wrote {OUT}")
