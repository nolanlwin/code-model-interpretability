"""The surface baseline cannot see the corpora's formatting difference.

XLCoST does not deliver these languages in the same shape. Python is drawn
from the formatted mirror and arrives with real newlines and indentation
(median 19 newlines per program, 0% flat); JavaScript and PHP come from the
tokenized mirror and arrive on a single line (100% and 98% zero-newline). The
typological boundary in the results section therefore coincides exactly with
a corpus-provenance boundary, which would make "transfer is worse across the
Python boundary" uninterpretable if the classifier keyed on formatting.

It cannot. scikit-learn's char analyzers collapse whitespace runs before
building n-grams, so line structure never becomes a feature. This test pins
that, because it is load-bearing for the typological claim and invisible in
the code -- nothing in baselines.py says "whitespace is discarded", it is a
property of the analyzer, and switching to a word analyzer or a custom
tokenizer would silently reintroduce the confound.
"""

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

MULTILINE = ("for i in range ( n ) :\n    total += a [ i ]\n"
             "    print ( total )\n    return total")
FLAT = "for i in range ( n ) : total += a [ i ] print ( total ) return total"


def run() -> int:
    failures = 0

    def check(name, ok):
        nonlocal failures
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    # The settings baselines.py actually uses for transfer.
    for analyzer in ("char_wb", "char"):
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=(2, 4))
        X = v.fit_transform([MULTILINE, FLAT])
        check(f"{analyzer}: multi-line and flattened vectorise identically",
              np.allclose(X[0].toarray(), X[1].toarray()))
        feats = v.get_feature_names_out()
        check(f"{analyzer}: no feature contains a newline",
              not any("\n" in f for f in feats))
        check(f"{analyzer}: no feature contains a double space",
              not any("  " in f for f in feats))

    # Negative control. The two samples DO differ -- otherwise the checks above
    # would pass on any input and prove nothing. A raw character n-gram
    # extractor, without scikit-learn's preprocessing, separates them. So the
    # blindness is a property of that preprocessing and would disappear under a
    # custom analyzer, which is exactly the change this test exists to catch.
    raw_ngrams = lambda t: [t[i:i + 3] for i in range(len(t) - 2)]
    v = TfidfVectorizer(analyzer=raw_ngrams)
    X = v.fit_transform([MULTILINE, FLAT])
    check("negative control: a raw char n-gram extractor DOES separate them",
          not np.allclose(X[0].toarray(), X[1].toarray()))
    check("negative control: it forms features containing a newline",
          any("\n" in f for f in v.get_feature_names_out()))

    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
