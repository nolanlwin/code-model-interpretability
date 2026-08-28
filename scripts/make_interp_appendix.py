"""Generate the Interp as a Science appendix from committed result artifacts."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "interp4d_short" / "appendix_generated.tex"
MODEL = {
    "qwen2515b": "Qwen2.5-1.5B",
    "qwen25coder15b": "Qwen2.5-Coder-1.5B",
    "starcoder27b": "StarCoder2-7B",
}
LANG = {
    "c": "C",
    "csharp": "C\\#",
    "cpp": "C++",
    "java": "Java",
    "javascript": "JavaScript",
    "php": "PHP",
    "python": "Python",
}
LANG_DISPLAY = {
    "C": "C",
    "C#": "C\\#",
    "C++": "C++",
    "Java": "Java",
    "Javascript": "JavaScript",
    "JavaScript": "JavaScript",
    "PHP": "PHP",
    "Python": "Python",
}


def esc(value: str) -> str:
    return value.replace("_", "\\_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def boolean_table() -> str:
    rows = read_csv(ROOT / "results/boolean/probe/summary.csv")
    lines = [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Complete boolean occurrence-type probe and model-free baseline "
        "results. The best baseline is selected from name-only, masked statement, "
        "masked line, masked window, covariate-only, and majority features. "
        "$\\rho$ is a descriptive one-instance score step, not an interval or "
        "equivalence test; aligned probe/baseline predictions were not retained.}",
        "\\label{tab:boolean-full}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "language & model & problems & probe F1 & best baseline & difference & "
        "$\\rho$ & selectivity \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda r: (r["language"], r["model"])):
        lines.append(
            f"{LANG[row['language']]} & {MODEL[row['model']]} & "
            f"{int(row['n_problems']):,} & {float(row['macro_f1']):.3f} & "
            f"{float(row['best_baseline']):.3f} & "
            f"{float(row['probe_minus_baseline']):+.3f} & "
            f"{float(row['rho']):.4f} & {float(row['selectivity']):+.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    python = [row for row in rows if row["language"] == "python"]
    lines += [
        "\\begin{table}[h]\\centering\\small",
        "\\caption{Boolean Python identifier-renaming audit. C1--C5 are the five "
        "committed renaming conditions. Each condition refits a probe and reselects "
        "its layer, so deltas measure recoverability rather than fixed-direction "
        "invariance; intervals are problem-clustered. Empty "
        "renaming fields for other languages indicate that those runs were not "
        "committed, not a null result.}",
        "\\label{tab:boolean-renaming-full}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "model & C1 & C2 & C3 & C4 & C5 \\\\", "\\midrule",
    ]
    for row in python:
        lines.append(
            f"{MODEL[row['model']]} & "
            + " & ".join(f"{float(row[f'dC{i}']):+.3f}" for i in range(1, 6))
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    representative = {
        row["language"]: row for row in rows
    }
    lines += [
        "\\begin{table}[h]\\centering\\small",
        "\\caption{Complete model-free boolean baselines by language. Baselines "
        "are shared across the three neural models because they use the same "
        "frozen examples and surface features.}",
        "\\label{tab:boolean-baselines-full}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "language & majority & name & statement & line & window & covariates \\\\",
        "\\midrule",
    ]
    for language in sorted(representative):
        row = representative[language]
        lines.append(
            f"{LANG[language]} & {float(row['baseline_majority']):.3f} & "
            f"{float(row['baseline_name_only']):.3f} & "
            f"{float(row['baseline_statement_masked']):.3f} & "
            f"{float(row['baseline_line_masked']):.3f} & "
            f"{float(row['baseline_window_masked']):.3f} & "
            f"{float(row['baseline_covariates_only']):.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def class_struct_table() -> str:
    model_paths = [
        ("Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        ("Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-1.5B"),
        ("StarCoder2-7B", "starcoder2-7b"),
    ]
    lines = [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Complete class/structure perturbation results under the shared "
        "five-seed protocol. Each condition refits a probe and selects its own layer "
        "on validation data, so cross-condition deltas do not test a fixed direction.}",
        "\\label{tab:class-full}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "model & strategy & layer & F1 & 95\\% CI & control F1 & selectivity & "
        "$\\Delta$ \\\\",
        "\\midrule",
    ]
    for label, folder in model_paths:
        summary = read_csv(
            ROOT / "results/modal/results" / folder
            / "class_struct/perturbation/summary.csv"
        )
        for row in summary:
            delta = "---" if not row["delta_f1_vs_baseline"] else (
                f"{float(row['delta_f1_vs_baseline']):+.3f}"
            )
            lines.append(
                f"{label} & {esc(row['strategy'])} & "
                f"L{int(row['selected_layer'])} & "
                f"{float(row['test_f1_mean']):.3f} & "
                f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}] & "
                f"{float(row['control_f1']):.3f} & "
                f"{float(row['selectivity']):+.3f} & {delta} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Class/structure cross-language results. Transfer uses the "
        "Python-selected layer; in-domain layers are selected separately. "
        "Only languages present in the committed exports are shown.}",
        "\\label{tab:class-crosslang}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "model & language & programs & in-domain layer & in-domain F1 & "
        "transfer accuracy & transfer F1 \\\\", "\\midrule",
    ]
    for label, folder in model_paths:
        cross = read_csv(
            ROOT / "results/modal/results" / folder
            / "class_struct/crosslang/crosslang.csv"
        )
        for row in cross:
            transfer_acc = "---" if not row["transfer_acc_at_py_best"] else (
                f"{float(row['transfer_acc_at_py_best']):.3f}"
            )
            transfer_f1 = "---" if not row["transfer_f1_at_py_best"] else (
                f"{float(row['transfer_f1_at_py_best']):.3f}"
            )
            lines.append(
                f"{label} & {LANG_DISPLAY[row['language']]} & "
                f"{int(row['programs'])} & L{int(row['indomain_selected_layer'])} & "
                f"{float(row['indomain_test_f1_mean']):.3f} & "
                f"{transfer_acc} & {transfer_f1} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Layerwise class/structure robustness summaries. The F1 range "
        "is taken over all committed layers; maximum cosine similarity is measured "
        "against the baseline activation direction for each renamed condition.}",
        "\\label{tab:class-layerwise}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "model & strategy & minimum F1 & maximum F1 (layer) & "
        "maximum cosine (layer) \\\\", "\\midrule",
    ]
    for label, folder in model_paths:
        per_layer = read_csv(
            ROOT / "results/modal/results" / folder
            / "class_struct/perturbation/per_layer.csv"
        )
        cosine = read_csv(
            ROOT / "results/modal/results" / folder
            / "class_struct/perturbation/cosine_vs_baseline.csv"
        )
        strategies = sorted({row["strategy"] for row in per_layer})
        cosine_by_strategy = {row["strategy"]: row for row in cosine}
        for strategy in strategies:
            curve = [row for row in per_layer if row["strategy"] == strategy]
            low = min(float(row["test_f1"]) for row in curve)
            high = max(curve, key=lambda row: float(row["test_f1"]))
            if strategy == "baseline":
                cosine_cell = "---"
            else:
                c_row = cosine_by_strategy[strategy]
                layer_values = [
                    (int(key.removeprefix("layer_")), float(value))
                    for key, value in c_row.items() if key.startswith("layer_")
                ]
                c_layer, c_value = max(layer_values, key=lambda pair: pair[1])
                cosine_cell = f"{c_value:.3f} (L{c_layer})"
            lines.append(
                f"{label} & {esc(strategy)} & {low:.3f} & "
                f"{float(high['test_f1']):.3f} (L{int(high['layer'])}) & "
                f"{cosine_cell} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def iterator_table() -> str:
    configs = [
        (
            "Qwen2.5-Coder-1.5B",
            ROOT / "results/iterator/Qwen2.5-Coder-1.5B Results",
            "qwen_1.5B",
        ),
        (
            "StarCoder2-7B",
            ROOT / "results/iterator/Starcoder2-7B Results",
            "starcoder2",
        ),
    ]
    lines = [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Complete iterator perturbation results. Each condition refits a "
        "probe and selects its own best layer. These single-run summaries do not carry the "
        "five-seed intervals used for the class/structure role.}",
        "\\label{tab:iterator-full}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "model & strategy & best layer & accuracy & F1 & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for label, root, stem in configs:
        summary = read_csv(root / "perturbation" / f"{stem}_summary.csv")
        for row in summary:
            delta = "---" if not row["delta_f1_vs_baseline"] else (
                f"{float(row['delta_f1_vs_baseline']):+.3f}"
            )
            lines.append(
                f"{label} & {esc(row['strategy'])} & L{int(row['best_layer'])} & "
                f"{float(row['test_acc']):.3f} & {float(row['test_f1']):.3f} & "
                f"{delta} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Complete iterator cross-language transfer results. "
        "Transfer columns use the Python-selected layer.}",
        "\\label{tab:iterator-crosslang}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "model & language & programs & in-domain layer & in-domain F1 & "
        "transfer accuracy & transfer F1 \\\\", "\\midrule",
    ]
    for label, root, stem in configs:
        transfer = read_csv(root / "crosslang" / f"{stem}_crosslang.csv")
        for row in transfer:
            transfer_acc = "---" if not row["transfer_acc_at_py_best"] else (
                f"{float(row['transfer_acc_at_py_best']):.3f}"
            )
            transfer_f1 = "---" if not row["transfer_f1_at_py_best"] else (
                f"{float(row['transfer_f1_at_py_best']):.3f}"
            )
            lines.append(
                f"{label} & {LANG_DISPLAY[row['language']]} & "
                f"{int(row['programs'])} & L{int(row['indomain_best_layer'])} & "
                f"{float(row['indomain_test_f1']):.3f} & "
                f"{transfer_acc} & {transfer_f1} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Layerwise iterator robustness summaries. F1 ranges cover every "
        "committed layer; cosine similarity compares each perturbation direction "
        "with the baseline direction.}",
        "\\label{tab:iterator-layerwise}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrr}", "\\toprule",
        "model & strategy & minimum F1 & maximum F1 (layer) & "
        "maximum cosine (layer) \\\\", "\\midrule",
    ]
    for label, root, stem in configs:
        per_layer = read_csv(root / "perturbation" / f"{stem}_per_layer.csv")
        cosine = read_csv(
            root / "perturbation" / f"{stem}_cosine_vs_baseline.csv"
        )
        cosine_by_strategy = {row["strategy"]: row for row in cosine}
        for strategy in sorted({row["strategy"] for row in per_layer}):
            curve = [row for row in per_layer if row["strategy"] == strategy]
            low = min(float(row["test_f1"]) for row in curve)
            high = max(curve, key=lambda row: float(row["test_f1"]))
            if strategy == "baseline":
                cosine_cell = "---"
            else:
                c_row = cosine_by_strategy[strategy]
                layer_values = [
                    (int(key.removeprefix("layer_")), float(value))
                    for key, value in c_row.items() if key.startswith("layer_")
                ]
                c_layer, c_value = max(layer_values, key=lambda pair: pair[1])
                cosine_cell = f"{c_value:.3f} (L{c_layer})"
            lines.append(
                f"{label} & {esc(strategy)} & {low:.3f} & "
                f"{float(high['test_f1']):.3f} (L{int(high['layer'])}) & "
                f"{cosine_cell} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def iterator_patching_tables() -> str:
    configs = [
        (
            "Qwen2.5-Coder-1.5B",
            ROOT / "results/iterator/Qwen2.5-Coder-1.5B Results",
            "qwen_1.5B",
        ),
        (
            "StarCoder2-7B",
            ROOT / "results/iterator/Starcoder2-7B Results",
            "starcoder2",
        ),
    ]
    blocks = [
        "\\paragraph{Status.} The iterator recovery exports predate the "
        "gate-specified class/structure protocol. They report descriptive recovery "
        "curves without clustered confidence intervals or matched placebo/random "
        "controls. The terminal value is one by construction, and C-language "
        "cross-language rows with zero pairs are missing data rather than null effects."
    ]
    for label, root, stem in configs:
        for scope, path in (
            ("within-language perturbations",
             root / "perturbation" / f"{stem}_patching.csv"),
            ("cross-language transfer",
             root / "crosslang" / f"{stem}_crosslang_patching.csv"),
        ):
            data = read_csv(path)
            recovery_columns = [
                key for key in data[0] if key.startswith("recovery_layer_")
            ]
            lines = [
                "\\begin{table}[p]\\centering\\scriptsize",
                f"\\caption{{Exploratory iterator activation-patching recovery for "
                f"{label}, {scope}.}}",
                f"\\label{{tab:iterator-patching-{stem}-{scope.split('-')[0].split()[0]}}}",
                "\\resizebox{\\linewidth}{!}{%",
                f"\\begin{{tabular}}{{llrr{'r' * len(recovery_columns)}}}",
                "\\toprule",
                "condition & readout & pairs & clean/corrupt & "
                + " & ".join(
                    f"L{column.removeprefix('recovery_layer_')}"
                    for column in recovery_columns
                )
                + " \\\\",
                "\\midrule",
            ]
            for row in data:
                condition = LANG_DISPLAY.get(row["condition"], esc(row["condition"]))
                clean_corrupt = (
                    f"{float(row['m_clean']):.3f}/{float(row['m_corrupt']):.3f}"
                )
                recoveries = []
                for column in recovery_columns:
                    value = row[column]
                    recoveries.append(
                        "---" if value.lower() == "nan" else f"{float(value):.3f}"
                    )
                lines.append(
                    f"{condition} & L{int(row['readout_layer'])} & "
                    f"{int(row['n_pairs'])} & {clean_corrupt} & "
                    + " & ".join(recoveries) + " \\\\"
                )
            lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def class_causal_table() -> str:
    root = (
        ROOT / "results/modal/patching/class-struct-python-v1-20260819"
        / "summaries/Qwen--Qwen2.5-1.5B/float16/cb9960752d1df6cc"
        / "eval"
    )
    rows = read_csv(root / "summary.csv")
    lines = [
        "\\begin{table}[p]\\centering\\small",
        "\\caption{Complete gate-specified class/structure patching summary in "
        "Qwen2.5-1.5B. The causal gate required mean effect at least $0.10$, "
        "a positive interval, separation from controls, and recovery at least $0.05$ "
        "in both directions.}",
        "\\label{tab:causal-full}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{rllllrrrr}", "\\toprule",
        "layer & span & direction & control & $n$ & mean effect & 95\\% CI & "
        "recovery & minus random \\\\", "\\midrule",
    ]
    for row in rows:
        minus_random = (
            "---" if not row["minus_random"]
            else f"{float(row['minus_random']):+.4f}"
        )
        lines.append(
            f"{int(row['layer'])} & {esc(row['span'])} & {row['direction']} & "
            f"{esc(row['control'])} & {int(row['n'])} & "
            f"{float(row['mean_effect']):+.4f} & "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] & "
            f"{float(row['recovery']):+.4f} & {minus_random} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    linked = read_csv(root / "probe_link.csv")
    probe_movement = [abs(float(row["symmetric_probe_movement"])) for row in linked]
    behavior = [abs(float(row["symmetric_behavior"])) for row in linked]
    lines += [
        "\\begin{table}[h]\\centering\\small",
        "\\caption{Aggregate probe-link diagnostic for the class/structure "
        "evaluation pairs. Absolute values are reported because pair orientation "
        "is arbitrary. Large probe movement alongside small behavioral movement "
        "is descriptive of the failed causal transfer, not evidence that probe "
        "movement causes behavior.}",
        "\\label{tab:class-probe-link}",
        "\\begin{tabular}{lrr}", "\\toprule",
        "quantity & mean absolute & median absolute \\\\", "\\midrule",
        f"probe movement & {st.mean(probe_movement):.3f} & "
        f"{st.median(probe_movement):.3f} \\\\",
        f"behavioral movement & {st.mean(behavior):.3f} & "
        f"{st.median(behavior):.3f} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ]

    patch_root = (
        ROOT / "results/modal/patching/class-struct-python-v1-20260819"
    )
    gate = json.loads((patch_root / "gate_report.json").read_text())
    model_gate = gate["models"]["Qwen/Qwen2.5-1.5B"]
    behavior_gate = model_gate["behavior"]
    primary_gate = model_gate["primary"]
    link = json.loads((root / "probe_link_summary.json").read_text())
    completeness_primary = json.loads(
        (patch_root / "summaries/completeness/Qwen2.5-1.5B-primary.json").read_text()
    )
    completeness_behavior = json.loads(
        (patch_root / "summaries/completeness/Qwen2.5-1.5B-behavior.json").read_text()
    )
    status = json.loads((patch_root / "status.json").read_text())
    function_d = -float(behavior_gate["neg_function_D"]["point"])
    lines += [
        "\\begin{table}[h]\\centering\\small",
        "\\caption{Class/structure gate and readout audit. Both prompt classes "
        "favor \\texttt{True}; the pairwise gap remains separable, but the "
        "binary readout is not calibrated to distinguish function prompts.}",
        "\\label{tab:class-gate-audit}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "diagnostic & estimate & 95\\% CI & gate \\\\", "\\midrule",
        f"class logit difference & {behavior_gate['class_D']['point']:.3f} & "
        f"[{behavior_gate['class_D']['ci_low']:.3f}, "
        f"{behavior_gate['class_D']['ci_high']:.3f}] & pass \\\\",
        f"function logit difference & {function_d:.3f} & "
        f"[{-behavior_gate['neg_function_D']['ci_high']:.3f}, "
        f"{-behavior_gate['neg_function_D']['ci_low']:.3f}] & fail \\\\",
        f"class-minus-function gap & {behavior_gate['gap']['point']:.3f} & "
        f"[{behavior_gate['gap']['ci_low']:.3f}, "
        f"{behavior_gate['gap']['ci_high']:.3f}] & pass \\\\",
        f"pair-gap accuracy & {behavior_gate['pair_gap_acc']['point']:.3f} & "
        f"[{behavior_gate['pair_gap_acc']['ci_low']:.3f}, "
        f"{behavior_gate['pair_gap_acc']['ci_high']:.3f}] & pass \\\\",
        f"query/declaration probe AUC & "
        f"{primary_gate['probe_ood']['query_auc']:.3f}/"
        f"{primary_gate['probe_ood']['declaration_auc']:.3f} & --- & pass \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
        "\\begin{table}[h]\\centering\\small",
        "\\caption{Patching audit trail. Completeness establishes that the "
        "scheduled Qwen evaluation finished; it does not turn the failed causal "
        "gate into positive evidence. The controller stopped before Coder and "
        "StarCoder2 after the Qwen null.}",
        "\\label{tab:class-patching-audit}",
        "\\begin{tabular}{lr}", "\\toprule",
        "audit item & value \\\\", "\\midrule",
        f"behavior rows present/expected & "
        f"{completeness_behavior['n_present']}/"
        f"{completeness_behavior['n_expected']} \\\\",
        f"primary rows present/expected & "
        f"{completeness_primary['n_present']}/"
        f"{completeness_primary['n_expected']} \\\\",
        f"baseline probe-gap/behavior Spearman & "
        f"{link['baseline_probe_gap_vs_behavior_spearman']['point']:.3f} "
        f"[{link['baseline_probe_gap_vs_behavior_spearman']['ci_low']:.3f}, "
        f"{link['baseline_probe_gap_vs_behavior_spearman']['ci_high']:.3f}] \\\\",
        f"controller terminal state & \\texttt{{{esc(status['state'])}}} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)


def boolean_causal_tables() -> str:
    rows = read_csv(ROOT / "results/boolean/causal/summary.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["mode"], row["language"], row["model"]), []
        ).append(row)

    blocks = [
        "\\paragraph{Status and estimand.} These boolean interventions are "
        "exploratory diagnostics rather than evidence for the main causal claim. "
        "The committed export contains layer-wise means but not the per-case arrays "
        "needed to recompute clustered intervals, and the runs do not use the "
        "specified gate of the class/structure experiment. The exporter defines "
        "effect as $|\\mathrm{clean}-\\mathrm{intervened}|$ and specificity as that "
        "effect divided by the corresponding random-position-control effect. "
        "Specificity can become large when its denominator is small. We therefore "
        "report both specificity at the largest-effect layer and the maximum "
        "specificity over layers rather than selecting only the more favorable one."
    ]
    for mode in ("patch", "steer", "ablate"):
        lines = [
            "\\begin{table}[p]\\centering\\small",
            f"\\caption{{Exploratory boolean {mode} diagnostics. "
            "Effect and specificity use the exporter definitions above. "
            "No uncertainty interval is available from the committed summary.}",
            f"\\label{{tab:boolean-{mode}}}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{llrrrrr}", "\\toprule",
            "language & model & $n$ & peak layer & peak effect & "
            "spec. at peak & best spec. (layer) \\\\",
            "\\midrule",
        ]
        keys = sorted(
            (key for key in grouped if key[0] == mode),
            key=lambda key: (
                list(LANG).index(key[1]),
                list(MODEL).index(key[2]),
            ),
        )
        for _, language, model in keys:
            group = grouped[(mode, language, model)]

            def effect(row: dict[str, str]) -> float:
                return abs(float(row["clean"]) - float(row["intervened"]))

            peak = max(group, key=effect)
            with_specificity = [row for row in group if row["specificity"]]
            best = max(
                with_specificity, key=lambda row: float(row["specificity"])
            )
            lines.append(
                f"{LANG[language]} & {MODEL[model]} & {int(peak['n'])} & "
                f"L{int(peak['layer'])} & {effect(peak):.3f} & "
                f"{float(peak['specificity']):.1f}$\\times$ & "
                f"{float(best['specificity']):.1f}$\\times$ "
                f"(L{int(best['layer'])}) \\\\"
            )
        lines += [
            "\\bottomrule", "\\end{tabular}}", "\\end{table}"
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def boolean_causal_figures() -> str:
    blocks = [
        "\\paragraph{Full layer profiles.} Each panel plots the clean readout, "
        "the intervention, and the available random-position, random-direction, "
        "or same-role controls. These figures preserve the layerwise information "
        "that is compressed by Tables~\\ref{tab:boolean-patch}--"
        "\\ref{tab:boolean-ablate}."
    ]
    for language in ("cpp", "java", "javascript", "php", "python"):
        for mode in ("patch", "steer", "ablate"):
            blocks += [
                "\\begin{figure}[p]\\centering",
                f"\\includegraphics[width=\\linewidth]{{../results/boolean/causal/"
                f"layer_profile_{language}_{mode}.png}}",
                f"\\caption{{Boolean {mode} layer profiles for {LANG[language]} "
                "across all three models and available controls.}",
                f"\\label{{fig:boolean-{language}-{mode}}}",
                "\\end{figure}",
            ]
        blocks.append("\\clearpage")
    return "\n".join(blocks)


def boolean_probe_figures() -> str:
    blocks = [
        "\\paragraph{Boolean probe figures.} The baseline panels retain the "
        "individual surface comparators hidden by the best-baseline table. "
        "Renaming is available only for Python; the absence of corresponding "
        "figures in other languages is a coverage hole."
    ]
    for language in ("java", "javascript", "php", "python"):
        for model in ("qwen2515b", "qwen25coder15b", "starcoder27b"):
            blocks += [
                "\\begin{figure}[p]\\centering",
                f"\\includegraphics[width=0.9\\linewidth]{{../results/boolean/"
                f"probe_vs_baselines_{language}_train_{model}.png}}",
                f"\\caption{{Boolean probe and model-free baselines for "
                f"{LANG[language]}, {MODEL[model]}.}}",
                f"\\label{{fig:boolean-probe-{language}-{model}}}",
                "\\end{figure}",
            ]
        blocks += [
            "\\begin{figure}[p]\\centering",
            f"\\includegraphics[width=0.9\\linewidth]{{../results/boolean/"
            f"layer_curves_{language}_train.png}}",
            f"\\caption{{Boolean probe layer curves for {LANG[language]} across "
            "the three models.}",
            f"\\label{{fig:boolean-layers-{language}}}",
            "\\end{figure}",
        ]
        blocks.append("\\clearpage")
    blocks += [
        "\\begin{figure}[p]\\centering",
        "\\includegraphics[width=0.9\\linewidth]{../results/boolean/"
        "renaming_deltas_python_train.png}",
        "\\caption{Problem-paired boolean refit-performance deltas on Python. "
        "No corresponding committed renaming run exists for other languages.}",
        "\\label{fig:boolean-renaming-python}",
        "\\end{figure}",
        "\\clearpage",
    ]
    return "\n".join(blocks)


def role_figures() -> str:
    figures = [
        (
            "../sree_paper_ready/sree's experiments/accum_probe_f1.png",
            "Accumulator probe accuracy and macro-F1 across layers and all six "
            "identifier conditions.",
            "accumulator-probe-curves",
        ),
        (
            "../sree_paper_ready/sree's experiments/accum_delta_f1.png",
            "Accumulator macro-F1 changes relative to baseline naming.",
            "accumulator-renaming-deltas",
        ),
        (
            "../sree_paper_ready/sree's experiments/accum_cross_language.png",
            "Accumulator in-domain and Python-trained cross-language results for "
            "the available compiled languages.",
            "accumulator-crosslang",
        ),
        (
            "../sree_paper_ready/sree's experiments/index_probe_f1.png",
            "Index/key probe accuracy and macro-F1 across layers.",
            "index-probe-curves",
        ),
        (
            "../sree_paper_ready/sree's experiments/index_renamed_comparison.png",
            "Index/key probe comparison under original and misleading names.",
            "index-renaming-comparison",
        ),
        (
            "../sree_paper_ready/sree's experiments/index_cross_language.png",
            "Index/key in-domain and Python-trained cross-language results.",
            "index-crosslang",
        ),
        (
            "../results/accumulator/cosine_similarity_accumulator.png",
            "Legacy accumulator probe-direction cosine similarity under the "
            "identifier interventions.",
            "accumulator-cosine",
        ),
        (
            "../results/accumulator/best_layer_f1_accumulator.png",
            "Legacy accumulator best-layer macro-F1 by identifier condition.",
            "accumulator-best-layer",
        ),
        (
            "../results/accumulator/heatmap_accumulator.png",
            "Legacy accumulator transfer heatmap over the available languages.",
            "accumulator-heatmap",
        ),
        (
            "../results/accumulator/multi_model_accumulator.png",
            "Legacy accumulator multi-model comparison. This predates the "
            "shared modern protocol and is descriptive only.",
            "accumulator-multimodel",
        ),
        (
            "../results/baseline/probe_cosine_similarity_index_key.png",
            "Legacy index/key probe-direction cosine similarity across layers.",
            "index-cosine",
        ),
        (
            "../results/baseline/100_probe_cosine_similarity_index_key.png",
            "Index/key cosine-similarity sensitivity run using the smaller "
            "100-example configuration.",
            "index-cosine-100",
        ),
        (
            "../results/baseline/100_cross_language_index_key.png",
            "Index/key cross-language sensitivity run using the smaller "
            "100-example configuration.",
            "index-crosslang-100",
        ),
        (
            "../results/renamed/probe_accuracy_renamed.png",
            "Legacy index/key layer curves under the full renamed condition.",
            "index-renamed-probe",
        ),
        (
            "../results/renamed/cosine_similarity_renamed.png",
            "Legacy index/key cosine similarity under renaming.",
            "index-renamed-cosine",
        ),
        (
            "../results/renamed/cross_language_renamed.png",
            "Legacy index/key cross-language transfer under renaming.",
            "index-renamed-crosslang",
        ),
        (
            "../results/class_struct/probe_f1_layer_class_struct.png",
            "Class/structure probe macro-F1 across layers and perturbations.",
            "class-probe-curves",
        ),
        (
            "../results/class_struct/delta_f1_class_struct.png",
            "Class/structure perturbation deltas relative to baseline.",
            "class-renaming-deltas",
        ),
        (
            "../results/class_struct/cosine_vs_baseline_class_struct.png",
            "Class/structure cosine similarity to the baseline direction.",
            "class-cosine",
        ),
        (
            "../results/class_struct/cross_language_class_struct.png",
            "Available class/structure cross-language transfer results.",
            "class-crosslang",
        ),
    ]
    blocks = [
        "\\paragraph{Figure provenance.} These are committed analysis figures. "
        "They supplement the generated tables but do not add uncertainty estimates "
        "where the underlying export lacks them."
    ]
    for index, (path, caption, label) in enumerate(figures, start=1):
        blocks += [
            "\\begin{figure}[p]\\centering",
            f"\\includegraphics[width=0.9\\linewidth]{{\\detokenize{{{path}}}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{fig:interp-{label}}}",
            "\\end{figure}",
        ]
        if index % 4 == 0:
            blocks.append("\\clearpage")
    blocks.append("\\clearpage")
    return "\n".join(blocks)


def coverage_holes() -> str:
    return "\n".join([
        "\\begin{itemize}",
        "\\item Accumulator and index/key results use legacy exports; no committed "
        "problem-grouped, five-seed CSV reproduces those roles under the modern protocol.",
        "\\item Iterator results are available for Qwen2.5-Coder-1.5B and "
        "StarCoder2-7B, but not Qwen2.5-1.5B.",
        "\\item Class/structure transfer exports include Python, C++, JavaScript, "
        "and C, but not Java, C\\#, or PHP.",
        "\\item Boolean probe exports include Python, Java, JavaScript, and PHP; "
        "C, C\\#, and C++ are absent, and renaming is available only for Python.",
        "\\item Gate-specified class/structure patching stopped after the Qwen2.5-1.5B "
        "null; Coder, StarCoder2, and the all-layer core sweep were not attempted.",
        "\\item No matched model-free surface comparator is committed for the "
        "class/structure role.",
        "\\end{itemize}",
    ])


content = "\n\n".join([
    "% GENERATED by scripts/make_interp_appendix.py -- do not edit by hand.",
    "\\section{Per-role control results}\n\\label{app:controls}",
    boolean_table(),
    boolean_probe_figures(),
    class_struct_table(),
    iterator_table(),
    "\\section{Causal intervention details}\n\\label{app:causal}",
    class_causal_table(),
    iterator_patching_tables(),
    "\\section{Exploratory boolean interventions}\n\\label{app:boolean-causal}",
    boolean_causal_tables(),
    boolean_causal_figures(),
    "\\section{Committed role-analysis figures}\n\\label{app:role-figures}",
    role_figures(),
    "\\section{Explicit coverage holes}\n\\label{app:coverage-holes}",
    coverage_holes(),
]) + "\n"

OUT.write_text(content)
print(f"  wrote {OUT.relative_to(ROOT)}")
