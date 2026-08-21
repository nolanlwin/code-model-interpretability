"""Which n-grams carry cross-lingual transfer, and which fail to cross.

The results section shows that the surface baseline transfers between
JavaScript and PHP at 0.965 and to Python at 0.576, and attributes the gap to
statement syntax. That is an attribution, not a mechanism. This script
measures the mechanism directly.

The baseline fits its vectorizer on the SOURCE language only and then calls
transform() on the target, so an n-gram the target never realises contributes
exactly zero to every target prediction. Transfer therefore succeeds to the
extent that the source classifier's discriminative weight sits on n-grams the
target also produces. Two questions follow, and this script answers both:

  survive   How much of the classifier's discriminative mass is realised in
            the target? Reported as a fraction, comparable across pairs.
  ablate    If the mass that fails to cross is concentrated in one syntactic
            class, then masking that class ON BOTH SIDES should cost the
            close pair a great deal and the distant pair almost nothing --
            because the distant pair could not use it anyway. That
            differential is a prediction the attribution makes and a
            correlation cannot.

    uv run python scripts/feature_transfer.py survive \
        --train-occurrences outputs/role_occ/capped_php_train.jsonl \
        --train-canonical   data/xlcost/php_train_shared.jsonl \
        --test-occurrences  outputs/role_occ/capped_python_train.jsonl \
        --test-canonical    data/xlcost/python_train_shared.jsonl \
        --role iterator --output outputs/mech/php_to_python.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from baselines import load_from_occurrences  # noqa: E402

#: Syntactic classes, tested in order; first match wins. An n-gram is assigned
#: by what it CONTAINS, because a char n-gram spanning "){ " is doing the work
#: of a brace even though it also holds a paren and a space.
FEATURE_CLASSES = [
    ("terminator", re.compile(r";")),
    ("brace", re.compile(r"[{}]")),
    ("bracket", re.compile(r"[\[\]]")),
    ("paren", re.compile(r"[()]")),
    ("operator", re.compile(r"[+\-*/%<>=!&|]")),
    ("keyword", re.compile(r"\b(for|while|in|range|let|var|const|foreach|do|if)\b")),
    ("alpha", re.compile(r"^[A-Za-z_ ]+$")),
]

#: Characters masked when a class is ablated. Applied to BOTH languages, so
#: neither side can use the class and the comparison stays fair.
ABLATE_CHARS = {
    "terminator": ";",
    "brace": "{}",
    "bracket": "[]",
    "paren": "()",
    "operator": "+-*/%<>=!&|",
}

MASKED_FIELDS = ("statement_masked", "line_masked", "window_masked")


def classify(ngram: str) -> str:
    for name, rx in FEATURE_CLASSES:
        if rx.search(ngram):
            return name
    return "other"


def ablate(texts, cls: str | None):
    """Replace every character of a class with a space, on one side."""
    if not cls or cls == "none":
        return texts
    chars = ABLATE_CHARS.get(cls)
    if chars is None:
        raise SystemExit(f"cannot ablate {cls!r}; known: {sorted(ABLATE_CHARS)}")
    table = str.maketrans({c: " " for c in chars})
    return [None if t is None else t.translate(table) for t in texts]


def fit_source(train_texts, y_tr, seed: int):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                          max_features=60000)
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                             random_state=seed).fit(Xtr, y_tr)
    return vec, clf, Xtr


def surviving_mass(vec, clf, Xtr, test_texts):
    """Fraction of the classifier's discriminative mass realised in the target.

    Mass of feature j is |coef_j| times its mean tf-idf. Computing it on both
    sides with the SAME vectorizer is what makes the ratio meaningful: the
    target's value is zero exactly when the target never produces that n-gram.
    """
    Xte = vec.transform(test_texts)
    coef = np.abs(clf.coef_).ravel()
    src = np.asarray(Xtr.mean(axis=0)).ravel()
    tgt = np.asarray(Xte.mean(axis=0)).ravel()
    m_src, m_tgt = coef * src, coef * tgt
    total = float(m_src.sum())
    return {
        "surviving_mass": float(m_tgt.sum() / total) if total else 0.0,
        "features": len(coef),
        "features_absent_in_target": int((tgt == 0).sum()),
        "_m_src": m_src, "_m_tgt": m_tgt,
        "_names": vec.get_feature_names_out(),
    }


def mass_by_class(names, m_src, m_tgt, top: int = 4000):
    """Where the surviving and lost mass sits, by syntactic class."""
    idx = np.argsort(m_src)[::-1][:top]
    out: dict = {}
    for j in idx:
        c = classify(names[j])
        d = out.setdefault(c, {"mass_source": 0.0, "mass_target": 0.0, "n": 0})
        d["mass_source"] += float(m_src[j])
        d["mass_target"] += float(m_tgt[j])
        d["n"] += 1
    tot_lost = sum(v["mass_source"] - v["mass_target"] for v in out.values()) or 1.0
    for v in out.values():
        lost = v["mass_source"] - v["mass_target"]
        v["retained"] = (v["mass_target"] / v["mass_source"]) if v["mass_source"] else 0.0
        v["share_of_lost_mass"] = lost / tot_lost
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mass_source"]))


def coefficient_agreement(vec, clf_src, Xtr, te_texts, y_te, seed):
    """Do the source's discriminative n-grams mean the same thing in the target?

    The survival measure answered the wrong question: roughly 70% of the
    source classifier's discriminative mass is realised in every target,
    including targets it transfers to badly, so transfer does not fail because
    the n-grams are missing. They are present and they point elsewhere.

    Fit a second classifier on the TARGET labels in the SAME feature space --
    the vectorizer stays the one fitted on the source, so feature j is the
    same n-gram on both sides -- and compare the two weight vectors. Agreement
    is their correlation, weighted by how much the source classifier leans on
    each feature. A feature the source ignores should not dominate the
    verdict.
    """
    Xte = vec.transform(te_texts)
    labels = sorted(set(y_te.tolist()))
    if len(labels) < 2:
        return None
    clf_tgt = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                 random_state=seed).fit(Xte, y_te)
    a = clf_src.coef_.ravel()
    b = clf_tgt.coef_.ravel()
    # Weight by the SAME quantity surviving_mass uses -- |beta| times the
    # feature's mean tf-idf in the source -- restricted to features the target
    # actually realises. Weighting by |beta| alone would let an n-gram the
    # source almost never emits count as much as one it leans on constantly,
    # and then "share of discriminative mass that flips sign" would not refer
    # to the mass surviving_mass reports. The two numbers are contrasted
    # directly in the paper, so they have to be the same measure.
    present = np.asarray(Xte.mean(axis=0)).ravel() > 0
    src_mass = np.asarray(Xtr.mean(axis=0)).ravel()
    w = np.abs(a) * src_mass * present
    if w.sum() == 0:
        return None
    ma = float((w * a).sum() / w.sum())
    mb = float((w * b).sum() / w.sum())
    cov = float((w * (a - ma) * (b - mb)).sum())
    va = float((w * (a - ma) ** 2).sum())
    vb = float((w * (b - mb) ** 2).sum())
    if va <= 0 or vb <= 0:
        return None
    # The n-grams the two languages read most oppositely, weighted by how much
    # the source leans on them. These are the concrete content of the number.
    flip = w * np.maximum(0.0, -(a * b))
    top = np.argsort(flip)[::-1][:25]
    names = vec.get_feature_names_out()
    flipped = [{"ngram": str(names[j]),
                "source_weight": round(float(a[j]), 3),
                "target_weight": round(float(b[j]), 3)}
               for j in top if flip[j] > 0]
    return {
        "agreement": cov / (va * vb) ** 0.5,
        "top_flipped": flipped,
        "features_compared": int(present.sum()),
        # Sign flips are the concrete failure: an n-gram the source reads as
        # evidence FOR the role that the target reads as evidence against.
        "sign_disagreement_mass": float(
            (w * ((np.sign(a) != np.sign(b)) & present)).sum() / w.sum()),
    }


def macro_f1(y_true, y_pred, labels) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def load_side(occ: str, canon: str, role: str, window: int, label_field: str):
    rows = load_from_occurrences(Path(occ), Path(canon), window, label_field=label_field)
    return rows


def matched(tr, te):
    """Keep only problems present on both sides, as the published table does."""
    shared = {r["repo"] for r in tr} & {r["repo"] for r in te}
    return ([r for r in tr if r["repo"] in shared],
            [r for r in te if r["repo"] in shared], len(shared))


def binarise(rows, role):
    return np.array([("target" if r["y"] == role else "other") for r in rows])


def evaluate(tr, te, role, field, seed, ablate_cls=None):
    """Transfer macro-F1 for one masked field, optionally with a class ablated
    on BOTH sides."""
    a = ablate([r[field] for r in tr], ablate_cls)
    b = ablate([r[field] for r in te], ablate_cls)
    if a and a[0] is None:
        return None, None
    y_tr, y_te = binarise(tr, role), binarise(te, role)
    labels = sorted(set(y_tr.tolist()) | set(y_te.tolist()))
    if len(set(y_tr.tolist())) < 2:
        return None, None
    vec, clf, Xtr = fit_source(a, y_tr, seed)
    pred = clf.predict(vec.transform(b))
    return macro_f1(y_te, pred, labels), (vec, clf, Xtr, b)


def cmd_survive(args) -> int:
    tr = load_side(args.train_occurrences, args.train_canonical, args.role,
                   args.window, args.label_field)
    te = load_side(args.test_occurrences, args.test_canonical, args.role,
                   args.window, args.label_field)
    tr, te, n_shared = matched(tr, te)
    if not tr or not te:
        raise SystemExit("no shared problems between these two languages")

    # Use the field that IS masked_best for this pair, so the mechanism is
    # measured on the same features the published number comes from.
    scored = {}
    for field in MASKED_FIELDS:
        f1, fitted = evaluate(tr, te, args.role, field, args.seed)
        if f1 is not None:
            scored[field] = (f1, fitted)
    if not scored:
        raise SystemExit("no usable masked field")
    best_field = max(scored, key=lambda k: scored[k][0])
    f1, (vec, clf, Xtr, b) = scored[best_field]

    surv = surviving_mass(vec, clf, Xtr, b)
    agree = coefficient_agreement(vec, clf, Xtr, b, binarise(te, args.role), args.seed)
    classes = mass_by_class(surv.pop("_names"), surv.pop("_m_src"), surv.pop("_m_tgt"))

    out = {
        "role": args.role, "source": args.source, "target": args.target,
        "n_shared_problems": n_shared, "n_train": len(tr), "n_test": len(te),
        "masked_best_field": best_field, "masked_best_macro_f1": round(f1, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in surv.items()},
        **({} if agree is None else
           {f"coef_{k}": (round(v, 4) if isinstance(v, float) else v)
            for k, v in agree.items()}),
        "by_class": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                         for kk, vv in v.items()} for k, v in classes.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "by_class"}))
    return 0


def cmd_ablate(args) -> int:
    tr = load_side(args.train_occurrences, args.train_canonical, args.role,
                   args.window, args.label_field)
    te = load_side(args.test_occurrences, args.test_canonical, args.role,
                   args.window, args.label_field)
    tr, te, n_shared = matched(tr, te)
    base_scores = {f: evaluate(tr, te, args.role, f, args.seed)[0] for f in MASKED_FIELDS}
    base_scores = {k: v for k, v in base_scores.items() if v is not None}
    best_field = max(base_scores, key=base_scores.get)
    baseline = base_scores[best_field]

    rows = []
    for cls in args.classes:
        f1, _ = evaluate(tr, te, args.role, best_field, args.seed, ablate_cls=cls)
        rows.append({"ablated": cls, "macro_f1": None if f1 is None else round(f1, 4),
                     "delta": None if f1 is None else round(f1 - baseline, 4)})
        print(f"  ablate {cls:<11} {f1:.4f}  ({f1 - baseline:+.4f})")
    out = {"role": args.role, "source": args.source, "target": args.target,
           "masked_best_field": best_field, "baseline_macro_f1": round(baseline, 4),
           "n_shared_problems": n_shared, "ablations": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    return 0


def cmd_verify(_args) -> int:
    checks = [
        ("';' is a terminator", classify(" ; ") == "terminator"),
        ("'){' is a brace", classify("){ ") == "brace"),
        ("'[i]' is a bracket", classify("[i]") == "bracket"),
        ("'for' is a keyword", classify("for") == "keyword"),
        ("plain letters are alpha", classify("tot") == "alpha"),
        ("terminator beats brace when both present", classify("};") == "terminator"),
        ("ablation replaces, never deletes",
         ablate(["a;b"], "terminator") == ["a b"]),
        ("ablation of none is identity", ablate(["a;b"], None) == ["a;b"]),
        ("ablation preserves length",
         len(ablate(["a{b}c"], "brace")[0]) == len("a{b}c")),
        ("ablation passes None through", ablate([None], "brace") == [None]),
    ]
    bad = 0
    for name, ok in checks:
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    try:
        ablate(["x"], "nonsense")
        print("  FAIL unknown class rejected"); bad += 1
    except SystemExit:
        print("  OK   unknown class rejected")
    print("\nALL PASS" if not bad else f"\n{bad} FAILURE(S)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("survive", "ablate"):
        p = sub.add_parser(name)
        p.add_argument("--train-occurrences", required=True)
        p.add_argument("--train-canonical", required=True)
        p.add_argument("--test-occurrences", required=True)
        p.add_argument("--test-canonical", required=True)
        p.add_argument("--role", required=True)
        p.add_argument("--source", default="?")
        p.add_argument("--target", default="?")
        p.add_argument("--window", type=int, default=120)
        p.add_argument("--label-field", default="role")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--output", required=True)
        if name == "ablate":
            p.add_argument("--classes", nargs="+",
                           default=["terminator", "brace", "bracket", "paren", "operator"])
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    return {"survive": cmd_survive, "ablate": cmd_ablate, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
