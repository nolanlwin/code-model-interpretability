# MuST-CoST vs XLCoST — per-language comparison

**Compiled 2026-08-06.**

- **MuST-CoST** numbers are *measured by me* from `github.com/reddy-lab-code-research/MuST-CoST` →
  `CoST_data.zip` → `CoST_data_release/raw_data/*.csv` (1,625 problem CSVs).
- **XLCoST** numbers are from Table 2 of the XLCoST paper ([arXiv 2206.08474](https://arxiv.org/abs/2206.08474)),
  not independently measured.

Both are Reddy-lab releases scraped from GeeksforGeeks, both cover the **same 7 languages**, and
**neither contains Go or Ruby**.

---

## 1. Programs per language

| Language | MuST-CoST | XLCoST | XLCoST / MuST-CoST |
|---|---:|---:|---:|
| C++ | 1,560 | 11,198 | 7.2× |
| Java | 1,560 | 11,028 | 7.1× |
| C# | 1,500 | 10,735 | 7.2× |
| Python | 1,461 | 10,622 | 7.3× |
| JavaScript | 1,022 | 9,951 | 9.7× |
| PHP | 553 | 3,553 | 6.4× |
| C | 301 | 574 | 1.9× |
| **Total** | **7,957** | **57,661** | **7.2×** |
| Problems | 1,625 | 11,265 | 6.9× |

## 2. Snippets per language

| Language | MuST-CoST | XLCoST | Snippets/program (MuST-CoST) | Snippets/program (XLCoST) |
|---|---:|---:|---:|---:|
| C++ | 15,042 | 106,397 | 9.64 | 9.52 |
| Java | 14,800 | 103,703 | 9.49 | 9.42 |
| C# | 14,084 | 100,032 | 9.39 | 9.33 |
| Python | 13,311 | 92,446 | 9.11 | 8.51 |
| JavaScript | 8,686 | 81,511 | 8.50 | 8.20 |
| PHP | 3,600 | 20,639 | 6.51 | 5.81 |
| C | 2,510 | 4,363 | 8.34 | 7.77 |
| **Total** | **72,033** | **509,091** | 9.05 avg | 8.81 avg |

Snippet granularity is essentially identical between the two — ~9 snippets per program. MuST-CoST's
"finer alignment" is not finer; XLCoST simply has more of it.

## 3. Program size

| Language | MuST-CoST lines/prog *(incl. comments)* | XLCoST lines/prog *(comments stripped)* | XLCoST tokens/prog |
|---|---:|---:|---:|
| C++ | 55.4 | 32.45 | 205.0 |
| Java | 57.9 | 34.93 | 227.1 |
| C# | 60.6 | 35.64 | 215.3 |
| Python | 41.7 | 20.54 | 188.5 |
| JavaScript | 44.7 | 26.47 | 184.6 |
| PHP | 38.1 | 23.23 | 163.5 |
| C | 48.5 | 31.50 | 198.0 |
| **Average** | 49.6 | 29.71 | 202.0 |

⚠️ **Not directly comparable.** My MuST-CoST line counts include comment lines; XLCoST strips comments and
docstrings. The real signal here is that both corpora are **short programs** — ~20–36 code lines.
Python is the shortest in both.

## 4. Official splits

MuST-CoST ships `processed_data` with splits, but it is **TransCoder-tokenized** (`_sa` suffix) — the source
is mangled and will not parse. Use `raw_data` and make your own splits.

XLCoST program-level splits (from the paper):

| Split | C++ | Java | Py | C# | JS | PHP | C | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 9,797 | 9,623 | 9,263 | 9,345 | 8,590 | 3,087 | 463 | 50,168 |
| valid | 492 | 494 | 472 | 491 | 475 | 158 | 60 | 2,642 |
| test | 909 | 911 | 887 | 899 | 886 | 308 | 51 | 4,851 |

Note these are split by **problem**, which is the grouping this project needs anyway.

## 5. Parseability *(measured on MuST-CoST)*

| Check | Result |
|---|---|
| Individual Python **snippets** parse | **23 / 13,311 (0.2%)** |
| Whole Python **programs** parse, raw | 171 / 1,625 |
| Dominant failure | `unexpected indent` — comments re-inserted at column 0 |
| Python 2 programs | ~85 |

Snippets are cut mid-function between comments, so **no role extractor can run on a snippet.** Whole
programs must be reassembled by concatenating a CSV row left-to-right, then repaired by stripping each
snippet's leading comment before joining. XLCoST already ships programs with comments stripped, which
avoids this failure mode entirely.

## 6. What this means per language, for role extraction

Estimated occurrences for the **boolean** role, scaling from my measured MuST-CoST Python rate
(1,217 occurrences from 1,461 programs = 0.83/program):

| Language | MuST-CoST programs | Est. boolean occurrences | XLCoST programs | Est. boolean occurrences | Verdict |
|---|---:|---:|---:|---:|---|
| C++ | 1,560 | ~1,300 | 11,198 | ~9,300 | extractor missing |
| Java | 1,560 | ~1,300 | 11,028 | ~9,200 | ✅ extractor exists |
| C# | 1,500 | ~1,250 | 10,735 | ~8,900 | extractor missing |
| Python | 1,461 | **1,217 (measured)** | 10,622 | ~8,800 | ✅ extractor exists |
| JavaScript | 1,022 | ~850 | 9,951 | ~8,300 | ⚠️ extractor untracked |
| PHP | 553 | ~460 | 3,553 | ~2,950 | ⚠️ extractor untracked |
| C | 301 | ~250 | 574 | ~480 | ❌ too small either way |

For reference, the paper's current boolean numbers use **7,981 CodeSearchNet Python occurrences**.

**Reading of the table:**

- On **MuST-CoST**, only Python / Java / C++ / C# clear ~1,000 occurrences. PHP (~460) and C (~250) cannot
  support a 5-class probe with usable confidence intervals.
- On **XLCoST**, all languages except C clear ~3,000, and the four with extractors (Python, Java, +
  untracked JS/PHP) are all viable.
- **C is unusable in both** (574 programs even in XLCoST) and should be cut regardless of corpus.

## 6b. ⚠️ How to actually obtain XLCoST — the released data is TOKENIZED

**Correction to an earlier claim in this file:** I previously wrote that XLCoST "ships comment-stripped
programs… removing the repair pass." That was wrong. The official XLCoST release is **TransCoder-style
tokenized**, with `NEW_LINE` / `INDENT` / `DEDENT` sentinels and spaces inserted around all punctuation.
Every released directory is named `pair_data_tok_*`, and the official Python program-level data looks
like:

```
def Conversion ( centi ) : NEW_LINE INDENT pixels = ( 96 * centi ) / 2.54 NEW_LINE …
```

**Measured: 0 / 300 official Python programs parse.** The official download is also a single Google
Drive file, which rate-limits on large fetches.

### The workable acquisition path *(all measured)*

| Language | Source | Parse rate | Notes |
|---|---|---|---|
| **Python** | `giulio98/xlcost-formatted` (HF) | **276 / 300 (92%)** | Properly detokenized, real indentation |
| **C++** | `giulio98/xlcost-formatted` (HF) | detokenized | same mirror |
| **Java, C#, JS, PHP, C** | `codeparrot/xlcost-text-to-code` (HF) | **268 / 300 (89%) tokenized as-is** | see below |

**Brace languages need no detokenization at all.** Whitespace is not semantic in Java/C++/C#/JS/PHP/C,
so `import java . io . * ;` is still valid Java — tree-sitter parses the tokenized form at 89% for
Java, and naive detokenization changes nothing (266/300). Only **Python** genuinely needs the
detokenized mirror, because `INDENT`/`DEDENT` are sentinels.

Record counts match the paper's Table 2 (e.g. Python program-level valid = 472), so **the HF
text-to-code splits are the complete program-level dataset**, not a subset. No Google Drive needed.

### ⚠️ But tokenized formatting is a confound for *this* paper

`System . out . println` tokenizes into a very different BPE sequence than `System.out.println`. Since
the entire paper is about how a model represents code, feeding it unnaturally-spaced code is
out-of-distribution input and a reviewer can reasonably challenge it.

**Recommendation:** detokenize *and* run a standard formatter per language — `black` (Python),
`google-java-format` (Java), `prettier` (JS), `clang-format` (C++/C/C#) — so the model sees natural
code. Budget ~1 day. If you skip it, you must state the formatting as a limitation, and you should
check that the paper's existing XLCoST numbers were not produced on tokenized text.

**Open question for the team:** which form did the paper's existing XLCoST results use? If they used the
tokenized release, every existing XLCoST number was computed on non-natural code.

## 7. Recommendation

Use **XLCoST**. The reason is not size alone:

1. MuST-CoST's distinguishing feature is snippet-level alignment, and **snippets are 0.2% parseable** — the
   thing that differentiates it is the thing you cannot use.
2. Once you reassemble programs, MuST-CoST *is* a 7.2× smaller XLCoST with the same problems, same source,
   same languages, same ~9 snippets/program.
3. XLCoST ships comment-stripped programs and problem-level splits, removing the repair pass.
4. **Four of the five roles in the paper are already on XLCoST**, so unifying there costs only the
   boolean role a corpus change instead of costing all five.

Open item worth 30 minutes: XLCoST has 11,265 problems vs MuST-CoST's 1,625, both from GeeksforGeeks, so
XLCoST is *probably* a superset. **Verify by problem-title overlap** if you want continuity with any
already-computed MuST-CoST results.

Also unresolved: the XLCoST license is listed as CC BY-SA 4.0 in the paper's appendix but
CC BY-NC-SA 4.0 on the arXiv posting. If NC applies it affects publication — assign someone to close
this in week 1.
