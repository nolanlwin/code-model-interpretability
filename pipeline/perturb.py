"""Variable-naming perturbations (Python), ported from the probing notebooks.

Role-agnostic strategies rename every identifier; `misleading` is
role-parameterized: variables holding the role get counter-role names and all
other variables get role-looking names. Role labels are re-extracted from the
perturbed code, so labels always reflect structure after renaming.
"""

import ast
import random
import re
import string

from .roles import LANG_KEYWORDS, PYTHON_PROTECTED, extract_roles

RANDOM_NOUNS = sorted(set([
    "apple", "bag", "ball", "basil", "bear", "bed", "belt", "birch", "bird", "bolt",
    "bowl", "box", "broom", "brush", "cage", "cart", "cat", "cedar", "clip", "coat",
    "coin", "comb", "cord", "crow", "cube", "cup", "deer", "desk", "dial", "dish",
    "dog", "dome", "drop", "drum", "duck", "dust", "edge", "fern", "fish", "flame",
    "fork", "fox", "frog", "gate", "gear", "gem", "glove", "glow", "grain", "grid",
    "hat", "hawk", "heap", "hill", "hole", "hook", "hose", "hull", "jar", "kelp",
    "knob", "knot", "lamb", "lamp", "latch", "leaf", "lens", "lion", "lobe", "lock",
    "loop", "lump", "maple", "mark", "mast", "mesh", "mint", "mist", "mold", "mole",
    "moss", "nail", "net", "newt", "node", "oak", "orb", "palm", "path", "peak",
    "pen", "pile", "pine", "pipe", "plank", "plate", "plug", "pool", "port", "pot",
    "puma", "ramp", "reed", "reef", "ring", "rod", "root", "rope", "rose", "rug",
    "sage", "sand", "seal", "seed", "slab", "slot", "stem", "tree", "vine", "wolf",
]))

# Names that conventionally signal each role; misleading_<role> gives role
# variables names from another role's pool and non-role variables names from
# the role's own pool.
ROLE_LOOKING = {
    "index_key": ["i", "j", "k", "n", "m", "idx", "pos", "key", "ptr",
                  "cur", "row", "col", "x", "y", "p", "q", "r", "t", "u"],
    "accumulator": ["total", "count", "result", "output", "answer", "value",
                    "amount", "data", "temp", "buf", "res", "ret", "score",
                    "maximum", "minimum", "current", "previous", "running",
                    "accumulator", "product", "difference"],
    "iterator": ["i", "j", "k", "it", "item", "elem", "entry", "each",
                 "cursor", "iter", "idx", "element"],
    "boolean": ["flag", "done", "found", "ok", "valid", "seen", "active",
                "enabled", "is_valid", "has_next", "ready", "stop"],
    "class_struct": ["Node", "Item", "Point", "Pair", "Entry", "Record",
                     "Box", "Cell", "Graph", "Tree", "Stack", "Table"],
}

# Counter-pool used to rename the role's own variables under misleading_<role>.
MISLEADING_COUNTER = {
    "index_key": ROLE_LOOKING["accumulator"],
    "accumulator": ROLE_LOOKING["index_key"],
    "iterator": ROLE_LOOKING["accumulator"],
    "boolean": ROLE_LOOKING["accumulator"],
    "class_struct": ROLE_LOOKING["accumulator"],
}


# Languages whose renaming output has been validated. Regex whole-word
# substitution cannot be made safe language-by-language: keyword protection is
# not enough because stdlib/API names are ordinary words to a regex — Java
# renames `System`, `out`, `println`; C/C++ lose preprocessor and namespace
# tokens (`#include <bits/stdc++.h>` -> `#e <b/j++.c>`); PHP loses its
# `<?php` open tag. Python is validated because collect_all_identifiers uses
# the AST there and PYTHON_PROTECTED covers builtins. Correct multi-language
# renaming needs identifier nodes from a parser, not word matches (issue #6).
RENAME_SUPPORTED = {"Python"}


class UnsupportedRenameLanguage(ValueError):
    """Raised rather than silently emitting corrupted source."""


def _protected_for(language):
    if language == "Python":
        return PYTHON_PROTECTED
    return PYTHON_PROTECTED | LANG_KEYWORDS.get(language, set())


def collect_all_identifiers(code):
    names = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
    except SyntaxError:
        names = set(re.findall(r"\b[a-zA-Z_]\w*\b", code))
    return names


def _renameable(code, language="Python"):
    protected = _protected_for(language)
    return sorted(n for n in collect_all_identifiers(code) if n not in protected)


def apply_rename_map(code, rename_map):
    """Whole-word substitution, longest names first to avoid partial matches."""
    for orig in sorted(rename_map, key=len, reverse=True):
        code = re.sub(r"\b" + re.escape(orig) + r"\b", rename_map[orig], code)
    return code


def _cycle_pool(pool, n, rng):
    out = pool * (n // len(pool) + 2)
    rng.shuffle(out)
    return out


def perturb_random_nouns(code, seed=0, language="Python"):
    rng = random.Random(seed)
    names = _renameable(code, language)
    pool = rng.sample(RANDOM_NOUNS, min(len(RANDOM_NOUNS), len(names)))
    if len(names) > len(pool):
        pool += rng.choices(RANDOM_NOUNS, k=len(names) - len(pool))
    return apply_rename_map(code, dict(zip(names, pool)))


def perturb_single_chars(code, seed=0, language="Python"):
    rng = random.Random(seed)
    names = _renameable(code, language)
    pool = list(string.ascii_lowercase)
    suffix = 0
    while len(pool) < len(names):
        pool += [c + str(suffix) for c in string.ascii_lowercase]
        suffix += 1
    pool = pool[:len(names)]
    rng.shuffle(pool)
    return apply_rename_map(code, dict(zip(names, pool)))


def perturb_all_same(code, seed=0, language="Python"):
    return apply_rename_map(code, {n: "x" for n in _renameable(code, language)})


def perturb_numeric_vars(code, seed=0, language="Python"):
    rng = random.Random(seed)
    names = _renameable(code, language)
    indices = list(range(1, len(names) + 1))
    rng.shuffle(indices)
    return apply_rename_map(code, {n: f"v{idx}" for n, idx in zip(names, indices)})


def perturb_misleading(code, role, seed=0, language="Python"):
    rng = random.Random(seed)
    all_names = _renameable(code, language)
    role_names = extract_roles(code, language)[role]

    role_vars = sorted(n for n in all_names if n in role_names)
    other_vars = sorted(n for n in all_names if n not in role_names)

    counter_pool = _cycle_pool(MISLEADING_COUNTER[role], len(role_vars), rng)
    role_pool = _cycle_pool(ROLE_LOOKING[role], len(other_vars), rng)

    rmap = dict(zip(role_vars, counter_pool))
    rmap.update(zip(other_vars, role_pool))
    return apply_rename_map(code, rmap)


def perturb(code, strategy, seed=0, language="Python"):
    """Apply one strategy; returns perturbed code (baseline returns input).

    Raises UnsupportedRenameLanguage outside RENAME_SUPPORTED: emitting
    corrupted source would yield plausible-looking probe numbers measured on
    code the model can never have seen, which is worse than no number.
    """
    if strategy == "baseline":
        return code
    if language not in RENAME_SUPPORTED:
        raise UnsupportedRenameLanguage(
            f"renaming is validated for {sorted(RENAME_SUPPORTED)}; {language!r} "
            "would be corrupted by whole-word substitution (see RENAME_SUPPORTED "
            "notes). Run baseline-only for this language, or use a parser-based "
            "renamer."
        )
    if strategy.startswith("misleading_"):
        return perturb_misleading(code, strategy.removeprefix("misleading_"), seed, language)
    fn = {
        "random_nouns": perturb_random_nouns,
        "single_chars": perturb_single_chars,
        "all_same": perturb_all_same,
        "numeric_vars": perturb_numeric_vars,
    }[strategy]
    return fn(code, seed, language)
