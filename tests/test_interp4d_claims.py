"""Regression checks for the Interp as a Science short paper."""
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "interp4d_short/main.tex"
BIB = ROOT / "interp4d_short/refs.bib"
APPENDIX = ROOT / "interp4d_short/appendix_generated.tex"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    text = MAIN.read_text()
    appendix = APPENDIX.read_text()
    checks: list[tuple[str, bool]] = []

    checks.append((
        "scope states three models, five roles, and seven languages collectively",
        "three models, five roles, and all seven XLCoST languages" in text,
    ))
    checks.append((
        "LP4FM-only masked-probe story is absent",
        not any(term in text for term in (
            "masked_slope", "context-pooled", "span-pooled",
            "0.844", "one $1.5$B model, three languages, three roles",
        )),
    ))
    checks.append((
        "LP4FM appendix generator cannot overwrite Interp",
        "interp4d_short" not in (
            ROOT / "scripts/make_appendix.py"
        ).read_text(),
    ))
    checks.append((
        "boolean surface claim is scoped to Python",
        "On Python, boolean occurrence" in text
        and "Boolean exceeds Python local syntax" in text,
    ))
    checks.append((
        "class causal null states the True/True readout limitation",
        "Both prompt classes prefer" in text
        and "limited dynamic range" in text,
    ))

    boolean_causal = rows(ROOT / "results/boolean/causal/summary.csv")
    causal_cells = {
        (row["language"], row["model"], row["mode"])
        for row in boolean_causal
    }
    checks += [
        (
            "boolean appendix includes patch, steering, and ablation tables",
            all(
                f"\\label{{tab:boolean-{mode}}}" in appendix
                for mode in ("patch", "steer", "ablate")
            ),
        ),
        (
            "boolean appendix covers every available language-model-mode cell",
            len(causal_cells) == 5 * 3 * 3
            and all(name in appendix for name in (
                "C++", "Java", "JavaScript", "PHP", "Python",
                "Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B", "StarCoder2-7B",
            )),
        ),
        (
            "boolean causal results are labeled exploratory",
            "exploratory diagnostics rather than evidence for the main causal claim"
            in appendix
            and "not the per-case arrays needed to recompute clustered intervals"
            in appendix,
        ),
        (
            "Interp appendix covers all committed aggregate result families",
            all(label in appendix for label in (
                "tab:boolean-full", "tab:boolean-renaming-full",
                "tab:boolean-baselines-full",
                "tab:class-full", "tab:class-crosslang", "tab:class-layerwise",
                "tab:iterator-full", "tab:iterator-crosslang",
                "tab:iterator-layerwise", "tab:causal-full",
                "tab:class-probe-link", "tab:class-gate-audit",
                "tab:class-patching-audit", "app:coverage-holes",
            )),
        ),
        (
            "Interp appendix includes all iterator patching exports",
            all(fragment in appendix for fragment in (
                "tab:iterator-patching-qwen_1.5B-within",
                "tab:iterator-patching-qwen_1.5B-cross",
                "tab:iterator-patching-starcoder2-within",
                "tab:iterator-patching-starcoder2-cross",
            )),
        ),
        (
            "Interp appendix includes causal, probe, and role figure families",
            appendix.count("../results/boolean/causal/layer_profile_") == 15
            and appendix.count("\\label{fig:boolean-probe-") == 12
            and appendix.count("\\label{fig:boolean-layers-") == 4
            and "\\label{fig:boolean-renaming-python}" in appendix
            and appendix.count("\\label{fig:interp-") == 20,
        ),
    ]

    boolean = [
        row for row in rows(ROOT / "results/boolean/probe/summary.csv")
        if row["language"] == "python"
    ]
    boolean_f1 = [float(row["macro_f1"]) for row in boolean]
    boolean_diffs = [float(row["probe_minus_baseline"]) for row in boolean]
    boolean_rename = [
        abs(float(row[f"dC{i}"])) for row in boolean for i in range(1, 6)
    ]
    checks += [
        ("boolean has three model rows", len(boolean) == 3),
        (
            "boolean probe range is 0.981-0.988",
            f"{min(boolean_f1):.3f}" == "0.981"
            and f"{max(boolean_f1):.3f}" == "0.988",
        ),
        (
            "boolean body values use per-model rounding",
            "$0.982$, $0.981$, and $0.988$" in text,
        ),
        (
            "boolean masked-line baseline is 0.983",
            {f"{float(row['best_baseline']):.3f}" for row in boolean} == {"0.983"},
        ),
        (
            "boolean comparison does not use rho as inferential uncertainty",
            max(abs(value) for value in boolean_diffs) <= 0.005
            and "no paired interval" in text
            and "one resolvable test instance" not in text,
        ),
        (
            "boolean renaming maximum rounds to 0.013",
            f"{max(boolean_rename):.3f}" == "0.013",
        ),
    ]

    checks += [
        (
            "heterogeneous probe units are stated",
            "boolean mean-pools" in text
            and "token probes label all other tokens negative" in text,
        ),
        (
            "repository history is not called preregistration",
            "preregister" not in text.lower(),
        ),
        (
            "renaming is scoped to perturbation-specific refitting",
            "perturbation-specific refitting" in text
            and "fixed rename-invariant" in text,
        ),
    ]

    class_f1: list[float] = []
    class_selectivity: list[float] = []
    class_misleading: list[float] = []
    for folder in ("Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B", "starcoder2-7b"):
        data = rows(
            ROOT / "results/modal/results" / folder
            / "class_struct/perturbation/summary.csv"
        )
        baseline = next(row for row in data if row["strategy"] == "baseline")
        misleading = next(
            row for row in data if row["strategy"] == "misleading_class_struct"
        )
        class_f1.append(float(baseline["test_f1_mean"]))
        class_selectivity.append(float(baseline["selectivity"]))
        class_misleading.append(float(misleading["delta_f1_vs_baseline"]))
    checks += [
        (
            "class probe range is 0.979-0.982",
            f"{min(class_f1):.3f}" == "0.979"
            and f"{max(class_f1):.3f}" == "0.982",
        ),
        (
            "class selectivity range is 0.503-0.553",
            f"{min(class_selectivity):.3f}" == "0.503"
            and f"{max(class_selectivity):.3f}" == "0.553",
        ),
        (
            "class misleading deltas round to -0.004/-0.004/+0.004",
            [f"{value:+.3f}" for value in class_misleading]
            == ["-0.004", "-0.004", "+0.004"],
        ),
    ]

    iterator_configs = [
        (
            ROOT / "results/iterator/Qwen2.5-Coder-1.5B Results",
            "qwen_1.5B",
            "0.988",
            "+0.012",
            "0.936",
            "0.983",
        ),
        (
            ROOT / "results/iterator/Starcoder2-7B Results",
            "starcoder2",
            "0.990",
            "+0.009",
            "0.906",
            "0.964",
        ),
    ]
    for root, stem, expected_f1, expected_delta, expected_min, expected_max in iterator_configs:
        summary = rows(root / "perturbation" / f"{stem}_summary.csv")
        baseline = next(row for row in summary if row["strategy"] == "baseline")
        misleading = next(
            row for row in summary if row["strategy"] == "misleading_iterator"
        )
        transfer = rows(root / "crosslang" / f"{stem}_crosslang.csv")
        scores = [
            float(row["transfer_f1_at_py_best"])
            for row in transfer if row["transfer_f1_at_py_best"]
        ]
        checks.append((
            f"iterator {stem} claims trace to CSV",
            f"{float(baseline['test_f1']):.3f}" == expected_f1
            and f"{float(misleading['delta_f1_vs_baseline']):+.3f}" == expected_delta
            and f"{min(scores):.3f}" == expected_min
            and f"{max(scores):.3f}" == expected_max,
        ))

    causal_path = (
        ROOT / "results/modal/patching/class-struct-python-v1-20260819"
        / "summaries/Qwen--Qwen2.5-1.5B/float16/cb9960752d1df6cc"
        / "eval/summary.csv"
    )
    causal = [
        row for row in rows(causal_path)
        if row["layer"] == "18"
        and row["span"] == "query_name"
        and row["control"] == "target"
    ]
    denoise = next(row for row in causal if row["direction"] == "denoise")
    noise = next(row for row in causal if row["direction"] == "noise")
    checks.append((
        "causal null values trace to intervention summary",
        f"{float(denoise['mean_effect']):.4f}" == "0.0087"
        and f"{float(denoise['ci_low']):.4f}" == "-0.0007"
        and f"{float(noise['mean_effect']):.4f}" == "0.0199"
        and f"{float(noise['ci_high']):.4f}" == "0.0296",
    ))
    checks.append((
        "causal estimands and excluded effects are explicit",
        r"e_d=D_{\mathrm{patched\ function}}-D_{\mathrm{function}}" in text
        and r"e_n=D_{\mathrm{class}}-D_{\mathrm{patched\ class}}" in text
        and "template clusters" in text
        and "exclude mean effects above $0.0188$ and $0.0296$" in text,
    ))

    accumulator = (ROOT / "sree_paper_ready/section_4_1_accumulator.tex").read_text()
    index = (ROOT / "sree_paper_ready/section_4_2_index.tex").read_text()
    checks.append((
        "legacy dissociation values remain source-backed",
        "$+0.079$" in accumulator
        and "{-}0.272" in index
        and "$+0.079$" in text
        and "$-0.272$" in text,
    ))

    cited: set[str] = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", text):
        cited.update(key.strip() for key in group.split(","))
    entries = set(re.findall(
        r"^@(?!comment)\w+\{([^,]+),", BIB.read_text(), flags=re.MULTILINE
    ))
    checks.append(("every citation key resolves", cited <= entries))

    failures = 0
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        failures += int(not ok)
    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
