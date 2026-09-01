"""Regression checks for the Interp as a Science short paper."""
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/interp_science_short/main.tex"
BIB = ROOT / "paper/interp_science_short/refs.bib"
APPENDIX = ROOT / "paper/interp_science_short/appendix_generated.tex"
CHECKLIST = ROOT / "paper/interp_science_short/checklist.tex"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    text = MAIN.read_text()
    appendix = APPENDIX.read_text()
    checklist = CHECKLIST.read_text()
    generator = (ROOT / "scripts/make_interp_appendix.py").read_text()
    checks: list[tuple[str, bool]] = []
    prose = re.sub(r"\\(?:label|ref)\{[^}]+\}", "", text + "\n" + appendix)
    # The two-part title is a deliberate exception. The punctuation rule governs
    # running prose; a hook plus a descriptive second half is the naming
    # convention this venue's own literature uses.
    prose = re.sub(r"\\title\{[^}]*\}", "", prose)

    checks.append((
        "title leads with the finding rather than a claim taxonomy",
        r"\title{Same Score, Different Evidence:\\Decodability, Surface "
        r"Sufficiency, and Causal Relevance in Code Models}" in text
        and "Separating Decodability" not in text,
    ))
    checks.append((
        "manuscript prose avoids prohibited punctuation",
        "—" not in prose
        and r"\textemdash" not in prose
        and "---" not in prose
        and ";" not in prose
        and ":" not in prose,
    ))

    checks.append((
        "scope states three models, five heterogeneous targets, and seven languages",
        "three models" in text
        and "five identifier properties and roles" in text
        and "all seven XLCoST" in text,
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
        "interp_science_short" not in (
            ROOT / "scripts/make_appendix.py"
        ).read_text(),
    ))
    checks.append((
        "boolean surface claim is scoped to Python",
        "On Python," in text
        and "boolean occurrence type" in text
        and "selected masked local-syntax" in text,
    ))
    checks.append((
        "class causal null states the True/True readout limitation",
        "Both prompt classes prefer" in text
        and "limits the behavioral" in text
        and "effective decision range" in text,
    ))
    checks += [
        (
            "reporting matrix is operational rather than checkmark-only",
            all(term in text for term in (
                "target and prediction unit",
                "claim and estimand",
                "comparator and matching rule",
                "uncertainty",
                "falsifying next test",
            ))
            and "checkmark indicating that a control ran" in text,
        ),
        (
            "role-conditioned renaming is labeled as confounded",
            "The vocabulary therefore encodes the label" in text
            and "does not support\nsemantic robustness" in text
            and "confounded diagnostics rather than robustness tests" in appendix,
        ),
        (
            "iterator and class controls are described as within-program permutations",
            "Iterator and class or structure permute labels within\nprograms" in text
            and "Hewitt control" not in text,
        ),
        (
            "boolean cohort transition and retained evidence are explicit",
            "$1{,}301$ source problems" in text
            and "$1{,}410$ unique" in text
            and "$2{,}067$ seed-level predictions" in text
            and "aggregate paired summary" in text,
        ),
        (
            "paired cohort reduction is attributed to seed pooling, not overlap",
            "Pooling the five seed test folds" in text
            and "predictor overlap is complete on every fold" in text
            and "Requiring complete predictor overlap" not in text,
        ),
        (
            "causal scope is site-state rather than probe-direction use",
            "Full-residual patching tests site-state relevance" in text
            and "not use of the decoded probe" in text
            and "bounded negative at the tested site" in text,
        ),
        (
            "causal null is not excused by readout scale",
            "Recovery divides by the matched" in text
            and "does not explain a ratio" in text
            and "insensitive readout" not in text
            and "insensitive\nreadout" not in text,
        ),
        (
            "boolean bound reports comparator and language sensitivity",
            "masked enclosing statement at $0.970$" in text
            and "$+0.010$ to $+0.017$" in text
            and "$-0.004$ to $+0.021$" in text
            and "degenerate outside Python" in text,
        ),
        (
            "patching row count spans both schedules",
            "rows across its primary and behavior" in text
            and "primary rows" not in text,
        ),
        (
            "checklist does not overclaim reproducibility",
            checklist.count(r"\item[] Answer: \answerNo{}") >= 5
            and "aligned prediction inputs" in checklist
            and "complete environment and command record" in checklist,
        ),
    ]

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
                "tab:iterator-full", "tab:iterator-baselines",
                "tab:iterator-crosslang",
                "tab:iterator-layerwise", "tab:causal-full",
                "tab:class-probe-link", "tab:class-gate-audit",
                "tab:class-patching-audit", "app:coverage-holes",
            )),
        ),
        (
            "Interp appendix includes all iterator patching exports",
            all(fragment in appendix for fragment in (
                "tab:iterator-patching-qwen2.5-1.5B-within",
                "tab:iterator-patching-qwen2.5-1.5B-cross",
                "tab:iterator-patching-qwen_1.5B-within",
                "tab:iterator-patching-qwen_1.5B-cross",
                "tab:iterator-patching-starcoder2-within",
                "tab:iterator-patching-starcoder2-cross",
            )),
        ),
        (
            "Interp appendix keeps the probe figure families",
            appendix.count("\\label{fig:boolean-probe-") == 12
            and appendix.count("\\label{fig:boolean-layers-") == 4
            and "\\label{fig:boolean-renaming-python}" in appendix,
        ),
        # The submission build omits the supplementary figure dumps, which cost
        # about a page each and carry no number the tables lack. Dropping them
        # must stay a build choice rather than a deletion, so the generator has
        # to retain the code paths and the appendix has to say how to get them.
        (
            "omitted figure dumps stay regenerable and are pointed at",
            "full-figures" in appendix
            and all(
                fragment in generator
                for fragment in (
                    "--full-figures",
                    # exact signatures and call sites: a substring like
                    # "def role_figures" also matches a renamed-out stub
                    "def boolean_causal_figures() -> str:",
                    "def role_figures() -> str:",
                    "boolean_causal_figures(),",
                    "role_figures(),",
                    "layer_profile_",
                    "fig:interp-",
                )
            ),
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
            and "one resolvable test instance" not in text,
        ),
        (
            "boolean renaming maximum rounds to 0.013",
            f"{max(boolean_rename):.3f}" == "0.013",
        ),
    ]

    import csv as _csv
    paired = list(_csv.DictReader(
        (ROOT / "results" / "boolean" / "probe" / "paired_probe_vs_line.csv").open()))
    checks += [
        ("paired boolean CSV covers all three models",
         {r["slug"] for r in paired}
         == {"qwen2515b", "qwen25coder15b", "starcoder27b"}),
        ("every paired interval covers zero",
         all(float(r["ci_low"]) < 0 < float(r["ci_high"]) for r in paired)),
        ("paper's excluded-advantage bound matches the CSV",
         f"{min(float(r['ci_high']) for r in paired):.3f}" == "0.014"
         and f"{max(float(r['ci_high']) for r in paired):.3f}" == "0.017"
         and "$0.014$--$0.017$" in text),
        ("paper's cluster count matches the CSV",
         {r["n_clusters"] for r in paired} == {"915"} and "$915$" in text),
        ("bounded-negative language present, unresolved language gone",
         "bounded negative result" in text
         and "Not supported on Python." in text
         and "excluding probe advantages above" in text
         and "Not resolved. Retain paired" not in text),
    ]
    checks += [
        (
            "heterogeneous probe units are stated",
            "Boolean mean-pools" in text
            and "Binary token probes label all other tokens" in text,
        ),
        (
            "repository history is not called preregistration",
            "preregister" not in text.lower(),
        ),
        (
            "renaming is scoped to perturbation-specific refitting",
            "every condition refits its probe and\nlayer" in text
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

    import json as _json
    iterator_f1: list[float] = []
    iterator_selectivity: list[float] = []
    iterator_misleading: list[float] = []
    iterator_transfer: list[list[float]] = []
    for folder in ("Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B", "starcoder2-7b"):
        data = rows(
            ROOT / "results/modal/results" / folder
            / "iterator/perturbation/summary.csv"
        )
        baseline = next(row for row in data if row["strategy"] == "baseline")
        misleading = next(
            row for row in data if row["strategy"] == "misleading_iterator"
        )
        iterator_f1.append(float(baseline["test_f1_mean"]))
        iterator_selectivity.append(float(baseline["selectivity"]))
        iterator_misleading.append(float(misleading["delta_f1_vs_baseline"]))
        transfer = rows(
            ROOT / "results/modal/results" / folder
            / "iterator/crosslang/crosslang.csv"
        )
        iterator_transfer.append([
            float(row["transfer_f1_at_py_best"])
            for row in transfer if row["transfer_f1_at_py_best"]
        ])
    name_only = float(_json.loads(
        (
            ROOT / "results/modal/results/Qwen2.5-1.5B/iterator"
            / "surface_baseline/baselines.json"
        ).read_text(encoding="utf-8")
    )["aggregate"]["name_only"]["macro_f1"])
    checks += [
        (
            "iterator probe values are 0.976, 0.978, and 0.987",
            [f"{value:.3f}" for value in iterator_f1]
            == ["0.976", "0.978", "0.987"]
            and "$0.976$, $0.978$, and $0.987$" in text,
        ),
        (
            "iterator selectivity range is 0.526-0.552",
            f"{min(iterator_selectivity):.3f}" == "0.526"
            and f"{max(iterator_selectivity):.3f}" == "0.552"
            and "$+0.526$ to $+0.552$" in text,
        ),
        (
            "iterator misleading deltas round to +0.023/+0.020/+0.012",
            [f"{value:+.3f}" for value in iterator_misleading]
            == ["+0.023", "+0.020", "+0.012"],
        ),
        (
            "iterator name-only surface is 0.918",
            f"{name_only:.3f}" == "0.918" and "$0.918$" in text,
        ),
        (
            "iterator transfer ranges match the paper",
            [f"{min(scores):.3f}" for scores in iterator_transfer]
            == ["0.934", "0.898", "0.887"]
            and [f"{max(scores):.3f}" for scores in iterator_transfer]
            == ["0.981", "0.923", "0.938"]
            and "$0.934$--$0.981$" in text,
        ),
        (
            "iterator coverage hole is the unpaired surface gap",
            "probe-minus-surface gap is not a paired clustered interval"
            in appendix
            and "not Qwen2.5-1.5B" not in appendix,
        ),
        (
            "default appendix keeps the iterator figure family",
            "\\label{fig:interp-iterator-probe-curves}" in appendix
            and "\\label{fig:interp-iterator-patching-recovery}" in appendix,
        ),
    ]

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

    checks.append((
        "legacy dissociation values remain source-backed",
        "$+0.079$" in text
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
