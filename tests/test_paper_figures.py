"""Every figure quoted in the LP4FM section must match the committed results.

This exists because a derived statistic was published wrong: the paper said
the smallest effect clears the largest rho by 74x, computed against the
majority baseline, while the exporter that generates SUMMARY.md defines the
same sentence against the shuffled-label control and prints 57x. Both numbers
were "correct"; they answered different questions, and the paper and its own
generated summary contradicted each other.

So derived quantities here are computed with the EXPORTER's definitions rather
than re-derived, and literal figures are grepped out of the .tex and compared
to the CSV.
"""

import csv
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "lp4fm_paper" / "section_results.tex"
CAPPED = ROOT / "results" / "lp4fm" / "summary.csv"
UNCAPPED = ROOT / "results" / "lp4fm" / "summary_renaming_uncapped.csv"
NONCODE = ROOT / "results" / "lp4fm_qwen2515b" / "summary.csv"
RANDOM = ROOT / "results" / "lp4fm_qwen2515brandominits0" / "summary.csv"
OVERLAP = ROOT / "results" / "lp4fm" / "xlcost_problem_overlap.csv"
WSCHECK = ROOT / "results" / "lp4fm" / "whitespace_normalisation_check.csv"
MECH = ROOT / "results" / "lp4fm" / "transfer_mechanism.csv"
MECHABL = ROOT / "results" / "lp4fm" / "transfer_mechanism_ablation.csv"

f = lambda r, k: float(r[k])


def load(p, probe_only=False):
    rows = list(csv.DictReader(p.open()))
    if probe_only:
        rows = [r for r in rows if (r.get("probe_transfer") or "").strip()]
    return rows


def run() -> int:
    tex = TEX.read_text()
    cap = load(CAPPED, probe_only=True)
    orig = [r for r in load(CAPPED) if (r.get("condition") or "original") == "original"]
    unc = load(UNCAPPED)
    ov = {(r["language_a"], r["language_b"]): int(r["shared_problem_ids"])
          for r in load(OVERLAP)}

    g = lambda role, a, b: next(r for r in cap if r["role"] == role
                                and r["source"] == a and r["target"] == b)
    near = [r for r in cap if "python" not in (r["source"], r["target"])]
    far = [r for r in cap if "python" in (r["source"], r["target"])]
    mean = lambda gr, k: st.mean(f(r, k) for r in gr)
    allm = [f(r, "masked_best") for r in cap]
    allp = [f(r, "probe_transfer") for r in cap]
    pj, pp, yp = (g("iterator", "php", "javascript"), g("iterator", "php", "python"),
                  g("iterator", "python", "php"))

    # The exporter's definitions, not re-derived ones.
    rhos = [f(r, "rho") for r in orig if r["rho"]]
    effects = [f(r, "masked_best") - f(r, "shuffled_labels") for r in orig]
    rho_factor = min(effects) / max(rhos)

    checks = [
        ("max masked-context 0.965", f"{max(allm):.3f}" == "0.965"),
        ("rho factor is 57 (vs shuffled, as the exporter defines it)",
         f"{rho_factor:.0f}" == "57"),
        ("rho span 0.0003-0.0020",
         f"{min(rhos):.4f}" == "0.0003" and f"{max(rhos):.4f}" == "0.0020"),
        ("name_only strongest in 6 of 18",
         sum(1 for r in orig if f(r, "name_only") >= f(r, "masked_best")) == 6),
        ("iterator php->python 0.576", f"{f(pp,'masked_best'):.3f}" == "0.576"),
        ("iterator python->php 0.741", f"{f(yp,'masked_best'):.3f}" == "0.741"),
        ("same-source gap 0.39 at 2dp",
         f"{f(pj,'masked_best')-f(pp,'masked_best'):.2f}" == "0.39"),
        ("directional gap 0.165",
         f"{f(yp,'masked_best')-f(pp,'masked_best'):.3f}" == "0.165"),
        ("probe same-source -0.013",
         f"{f(pj,'probe_transfer')-f(pp,'probe_transfer'):+.3f}" == "-0.013"),
        ("probe directional +0.038",
         f"{f(yp,'probe_transfer')-f(pp,'probe_transfer'):+.3f}" == "+0.038"),
        ("close pair 0.952 / 0.893",
         f"{mean(near,'masked_best'):.3f}" == "0.952" and f"{mean(near,'probe_transfer'):.3f}" == "0.893"),
        ("python pairs 0.785 / 0.887",
         f"{mean(far,'masked_best'):.3f}" == "0.785" and f"{mean(far,'probe_transfer'):.3f}" == "0.887"),
        ("spreads 0.390 / 0.107",
         f"{max(allm)-min(allm):.3f}" == "0.390" and f"{max(allp)-min(allp):.3f}" == "0.107"),
        ("variance ratio 3.7x",
         f"{(max(allm)-min(allm))/(max(allp)-min(allp)):.1f}" == "3.7"),
        ("probe php->python 0.865", f"{f(pp,'probe_transfer'):.3f}" == "0.865"),
        ("split is 6 losses / 12 wins",
         sum(1 for r in near if f(r, "probe_transfer") <= f(r, "masked_best")) == 6
         and sum(1 for r in far if f(r, "probe_transfer") > f(r, "masked_best")) == 12),
        ("smallest gap 0.015 clears own rho by 3.8x",
         f"{min(abs(f(r,'probe_transfer')-f(r,'masked_best')) for r in cap):.3f}" == "0.015"
         and f"{min((abs(f(r,'probe_transfer')-f(r,'masked_best'))/f(r,'probe_rho')) for r in cap):.1f}" == "3.8"),
        ("shuffled-source control 0.434-0.512",
         f"{min(f(r,'probe_shuffled_source') for r in cap):.3f}" == "0.434"
         and f"{max(f(r,'probe_shuffled_source') for r in cap):.3f}" == "0.512"),
        ("in-domain 0.877-0.985",
         f"{min(f(r,'probe_indomain') for r in cap):.3f}" == "0.877"
         and f"{max(f(r,'probe_indomain') for r in cap):.3f}" == "0.985"),
        ("overlap: java/javascript 85", ov.get(("java", "javascript")) == 85),
        ("overlap: cpp/javascript 75", ov.get(("cpp", "javascript")) == 75),
        ("overlap: csharp/java 175", ov.get(("csharp", "java")) == 175),
        ("overlap: the three usable pairs 2953/1529/1145",
         ov.get(("javascript", "python")) == 2953 and ov.get(("javascript", "php")) == 1529
         and ov.get(("php", "python")) == 1145),
    ]

    # The three-way comparison: trained code, trained non-code, untrained.
    # These come from separate directories, so a run that quietly reused one
    # model's numbers for another would show up here.
    if NONCODE.exists() and RANDOM.exists():
        nc = load(NONCODE, probe_only=True)
        rnd = load(RANDOM, probe_only=True)
        pm = lambda rs: st.mean(f(r, "probe_transfer") for r in rs)
        key = lambda rs: {(r["role"], r["source"], r["target"]): f(r, "probe_transfer")
                          for r in rs}
        kr = key(rnd)
        lifts = {n: [key(rs)[k] - kr[k] for k in kr]
                 for n, rs in (("code", cap), ("noncode", nc))}
        checks += [
            ("three models present, each 18 cells",
             len(cap) == 18 and len(nc) == 18 and len(rnd) == 18),
            ("each table names a distinct model",
             len({cap[0]["probe_model"], nc[0]["probe_model"], rnd[0]["probe_model"]}) == 3),
            ("the untrained table is the random-init one",
             "random-init" in (rnd[0]["probe_model"] or "")),
            ("untrained floor 0.575", f"{pm(rnd):.3f}" == "0.575"),
            ("trained code 0.889 / non-code 0.892",
             f"{pm(cap):.3f}" == "0.889" and f"{pm(nc):.3f}" == "0.892"),
            ("lift over untrained 0.314 / 0.316",
             f"{st.mean(lifts['code']):.3f}" == "0.314"
             and f"{st.mean(lifts['noncode']):.3f}" == "0.316"),
            ("both trained models beat untrained in all 18 cells",
             all(x > 0 for x in lifts["code"]) and all(x > 0 for x in lifts["noncode"])),
            ("smallest per-cell lift 0.126", f"{min(lifts['code']):.3f}" == "0.126"),
            ("untrained clears its own shuffled control by 0.111",
             f"{pm(rnd) - st.mean(f(r,'probe_shuffled_source') for r in rnd):.3f}" == "0.111"),
            ("untrained drop across the boundary 0.021",
             f"{st.mean(f(r,'probe_transfer') for r in rnd if 'python' in (r['source'],r['target'])) - st.mean(f(r,'probe_transfer') for r in rnd if 'python' not in (r['source'],r['target'])):.3f}" == "-0.021"),
            ("the baseline is identical across all three runs",
             {f"{f(r,'masked_best'):.4f}" for r in cap}
             == {f"{f(r,'masked_best'):.4f}" for r in nc}
             == {f"{f(r,'masked_best'):.4f}" for r in rnd}),
        ]
    else:
        checks.append(("three-way tables present", False))

    # The section claims normalising whitespace moves transfer by less than
    # 0.001. That is a measurement, so it has to come from a committed file.
    if WSCHECK.exists():
        ws = list(csv.DictReader(WSCHECK.open()))
        checks += [
            ("whitespace check covers the decisive pairs",
             {(r["source"], r["target"]) for r in ws}
             >= {("php", "python"), ("php", "javascript")}),
            # Derived from the two score columns, never read from `delta`:
            # a stale or hand-edited delta would otherwise certify itself.
            ("normalising whitespace moves transfer by < 0.001",
             all(abs(float(r["masked_best_normalised"])
                     - float(r["masked_best_raw"])) < 0.001 for r in ws)),
            ("the delta column agrees with the scores beside it",
             all(abs(float(r["delta"]) - (float(r["masked_best_normalised"])
                                          - float(r["masked_best_raw"]))) < 1e-9
                 for r in ws)),
            # The bound is applied to the capped published table, so it has to
            # have been measured there. Uncapped iterator php->python is 0.609
            # and php->javascript 0.979; the capped ones are 0.576 and 0.965.
            ("measured on the capped sample the paper publishes",
             all(any(abs(float(r["masked_best_raw"]) - f(c, "masked_best")) < 5e-4
                     for c in orig
                     if c["role"] == r["role"] and c["source"] == r["source"]
                     and c["target"] == r["target"])
                 for r in ws)),
        ]
    else:
        checks.append(("whitespace normalisation check present", False))

    # The mechanism paragraph. Its whole argument rests on which correlation
    # is large, so all three are recomputed from the committed cells rather
    # than quoted.
    if MECH.exists() and MECHABL.exists():
        mech = list(csv.DictReader(MECH.open()))
        abl = list(csv.DictReader(MECHABL.open()))

        def corr(xs, ys):
            n = len(xs)
            mx, my = st.mean(xs), st.mean(ys)
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
            return cov / den if den else float("nan")

        f1s = [f(r, "masked_best_macro_f1") for r in mech]
        near_ = [r for r in mech if "python" not in (r["source"], r["target"])]
        far_ = [r for r in mech if "python" in (r["source"], r["target"])]
        checks += [
            ("mechanism covers all six ordered pairs", len(mech) == 6),
            ("surviving mass does NOT predict transfer (r = +0.09)",
             f'{corr([f(r,"surviving_mass") for r in mech], f1s):+.2f}' == "+0.09"),
            ("coefficient agreement DOES predict transfer (r = +0.87)",
             f'{corr([f(r,"coef_agreement") for r in mech], f1s):+.2f}' == "+0.87"),
            ("sign-flip mass predicts inversely (r = -0.91)",
             f'{corr([f(r,"sign_disagreement_mass") for r in mech], f1s):+.2f}' == "-0.91"),
            ("~70% of mass survives even in the worst pair",
             all(0.69 <= f(r, "surviving_mass") <= 0.87 for r in mech)),
            ("close pair agreement 0.89-0.94",
             all(0.89 <= f(r, "coef_agreement") <= 0.94 for r in near_)),
            ("python-crossing agreement 0.44-0.76",
             all(0.44 <= f(r, "coef_agreement") <= 0.76 for r in far_)),
            ("python-crossing sign-flip 21-37%",
             all(0.21 <= f(r, "sign_disagreement_mass") <= 0.37 for r in far_)),
            ("close-pair sign-flip about 7%",
             all(0.06 <= f(r, "sign_disagreement_mass") <= 0.075 for r in near_)),
            # The claim that no syntactic class carries the gap is the one a
            # reviewer will push on, so bound it from the committed numbers.
            ("no ablation moves transfer by more than 0.021",
             max(abs(f(r, "delta")) for r in abl) <= 0.0213),
            ("masking braces costs php->javascript at most 0.001",
             all(abs(f(r, "delta")) <= 0.001 for r in abl
                 if r["ablated"] == "brace" and r["source"] == "php"
                 and r["target"] == "javascript")),
        ]
    else:
        checks.append(("mechanism results present", False))

    # Renaming table: uncapped file, must still reproduce the published values.
    for role, pct, delta in (("index_key", 89, -0.032), ("accumulator", 83, -0.047),
                             ("iterator", 49, -0.195)):
        c0 = [r for r in unc if r["role"] == role and r["condition"] == "original"
              and r["source"] == "python" and r["target"] in ("javascript", "php")]
        rn = [r for r in unc if r["role"] == role and r["condition"] != "original"]
        h0 = st.mean(f(r, "masked_best") - f(r, "shuffled_labels") for r in c0)
        hr = st.mean(f(r, "masked_best") - f(r, "shuffled_labels") for r in rn)
        d = st.mean(f(r, "masked_best") for r in rn) - st.mean(f(r, "masked_best") for r in c0)
        checks.append((f"renaming {role}: {pct}% / {delta:+.3f}",
                       round(100 * hr / h0) == pct and f"{d:+.3f}" == f"{delta:+.3f}"))

    failures = 0
    for name, ok in checks:
        if not ok:
            failures += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    # Figures superseded by the capped rerun must not survive anywhere in the
    # prose -- a stale number reads as authoritative.
    stale = [x for x in ("0.979", "0.609", "$0.370$", "$0.230$", "0.0017",
                         "factor of $88$", "factor of $74$", "$2$ of $18$",
                         "is not yet run") if x in tex]
    if stale:
        failures += 1
    print(f"  {'OK  ' if not stale else 'FAIL'} no superseded figures in the .tex"
          + (f": {stale}" if stale else ""))

    # Every macro-F1-looking literal in the tex should appear in a CSV.
    known = set()
    for r in load(CAPPED) + unc:
        for k in ("masked_best", "name_only", "probe_transfer", "majority",
                  "shuffled_labels", "probe_indomain", "probe_shuffled_source"):
            if r.get(k):
                known.add(f"{float(r[k]):.3f}")
    derived = {"0.952", "0.893", "0.785", "0.887", "0.390", "0.107", "0.166",
               "0.006", "0.165", "0.013", "0.038", "0.015", "0.032", "0.047",
               "0.195", "0.272", "0.079", "0.039", "0.314", "0.126", "0.111",
               "0.021", "0.011", "0.575", "0.889", "0.892",
               # a stated bound rather than a measured cell; the whitespace
               # check above verifies every measured delta falls under it
               "0.001", "0.696", "0.735"}
    lits = {m for m in re.findall(r"0\.\d{3}(?!\d)", tex)}
    unknown = sorted(lits - known - derived)
    if unknown:
        failures += 1
    print(f"  {'OK  ' if not unknown else 'FAIL'} every 3dp figure traces to a CSV"
          + (f": unexplained {unknown}" if unknown else ""))

    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
