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
            f"{r['role'].replace('_','\\_')} & "
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


def causal_table() -> str:
    p = pathlib.Path("results/boolean/causal/summary.csv")
    if not p.exists():
        return "% causal summary not available"
    rs = list(csv.DictReader(p.open()))
    langs = sorted({r["language"] for r in rs})
    models = sorted({r["model"] for r in rs})
    modes = sorted({r["mode"] for r in rs})
    return ("\\paragraph{Scope.} Causal interventions were run for the boolean role "
            f"over {len(langs)} languages ({', '.join(LANG.get(l,l) for l in langs)}), "
            f"{len(models)} models, and {len(modes)} intervention modes "
            f"({', '.join(modes)}), giving {len(rs):,} layer-wise measurements. "
            "They are reported here rather than in the main text because they were "
            "run within languages, not on the cross-lingual cells this paper is "
            "about, and so cannot support the paper's central claim.")


OUT.write_text("\n\n".join([
    "% GENERATED by scripts/make_appendix.py -- do not edit by hand.",
    "\\section{Pairwise problem overlap in XLCoST}\n\\label{app:overlap}",
    overlap_table(),
    "\\section{Full transfer table}\n\\label{app:full}",
    full_transfer_table(), model_table(),
    "\\section{Transfer mechanism}\n\\label{app:mech}",
    mechanism_table(),
    "\\section{Causal interventions}\n\\label{app:causal}",
    causal_table(),
]) + "\n")
print(f"  wrote {OUT}")
