---
license: apache-2.0
task_categories:
  - token-classification
language:
  - en
tags:
  - code
  - interpretability
  - probing
  - variable-roles
pretty_name: XLCoST Variable Roles
size_categories:
  - 100K<n<1M
configs:
  - config_name: python_perturbations
    data_files:
      - split: train
        path: python_perturbations/train.jsonl
      - split: validation
        path: python_perturbations/valid.jsonl
      - split: test
        path: python_perturbations/test.jsonl
  - config_name: multilingual_baseline
    data_files:
      - split: train
        path: multilingual_baseline/train.jsonl
      - split: validation
        path: multilingual_baseline/valid.jsonl
      - split: test
        path: multilingual_baseline/test.jsonl
---

# XLCoST Variable Roles

Program-level code with **structurally derived variable-role labels** for
probing how code LLMs represent variables. Built from
[XLCoST](https://github.com/reddy-lab-code-research/XLCoST) (Zhu et al., 2022).

Five roles, labeled from AST/structural analysis — never from the variable's
name — so probes trained on these labels must rely on context:

| Role | Definition |
|---|---|
| `index_key` | used as an array index or dict key (`arr[i]`, `d[key]`) |
| `accumulator` | target of `+=`-style updates or `.append()`-style calls inside a loop |
| `iterator` | bound in a loop header (`for x in …`, `for (int i = …`) |
| `boolean` | assigned a boolean literal |
| `class_struct` | declared class/struct name |

## Configs

**`python_perturbations`** — every Python program under 10 naming strategies:
`baseline`, `random_nouns`, `single_chars` (a, b, c…), `all_same` (everything
→ `x`), `numeric_vars` (v1, v2…), and `misleading_<role>` for each role
(role variables get counter-role names, all other variables get role-looking
names). Role labels are re-extracted from the transformed code.

**`multilingual_baseline`** — original programs in all 7 XLCoST languages
(C++, Java, Python, C#, Javascript, PHP, C) with role labels, for
cross-language transfer experiments.

## Fields

```json
{
  "id": "10005:Python:baseline",
  "problem_id": 10005,
  "language": "Python",
  "split": "train",
  "strategy": "baseline",
  "code": "def maxPresum(a, b): ...",
  "roles": {
    "index_key": ["i"],
    "accumulator": ["X"],
    "iterator": ["i"],
    "boolean": [],
    "class_struct": []
  }
}
```

Rows store code plus role-name sets rather than token-level labels, so the
dataset is model-agnostic: map names to token labels with whatever tokenizer
you are probing (reference implementation in the companion pipeline's
`probing.label_tokens`).

## Labeling method

Python roles come from the `ast` module; the other six languages use regex
extractors (subscripts, augmented assignments/increments/collector calls,
loop headers, boolean assignments, class/struct declarations) with per-language
keyword exclusion. PHP identifiers are labeled without the `$` sigil.
Programs with no role-labeled variable are dropped.

## Provenance and credits

Source programs are the program-level `nl2code_search` release of **XLCoST**,
which pairs GeeksforGeeks solutions across 7 languages. All credit for the
underlying corpus goes to the XLCoST authors; this dataset adds only the
role labels and the renaming variants. XLCoST is released under the Apache
2.0 license, as is this derivative.

```bibtex
@article{zhu2022xlcost,
  title   = {XLCoST: A Benchmark Dataset for Cross-lingual Code Intelligence},
  author  = {Zhu, Ming and Jain, Aneesh and Suresh, Karthik and
             Ravindran, Roshan and Tipirneni, Sindhu and Reddy, Chandan K.},
  journal = {arXiv preprint arXiv:2206.08474},
  year    = {2022}
}
```

If you use the role labels or perturbations, please also cite this dataset.

## Known limitations

- Non-Python role labels are regex-derived and inherit that noise
  (e.g. `++` in a for-header counts toward `accumulator`, matching the
  original experimental protocol).
- XLCoST code is competitive-programming style; identifier names are already
  short and partially uninformative, which attenuates renaming effects.
- Renaming is whole-word textual substitution with keyword/builtin
  protection, not scope-aware alpha-renaming.
- Under `all_same`, distinct variables collapse to one name, so
  exclusion-based roles (notably `accumulator`, which excludes loop and index
  variables) survive in far fewer programs; this mirrors the original
  experimental protocol, where labels are re-extracted after renaming.
