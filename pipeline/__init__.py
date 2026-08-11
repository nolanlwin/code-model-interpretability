"""Common pipeline for variable-role probing experiments on XLCoST.

One dataset, five roles (index_key, accumulator, iterator, boolean,
class_struct), six naming strategies, seven languages — runnable from a
single set of scripts instead of per-role notebooks.
"""

ROLES = ["index_key", "accumulator", "iterator", "boolean", "class_struct"]

LANGUAGES = ["Python", "Java", "C++", "C", "C#", "Javascript", "PHP"]

# Role-agnostic strategies exactly as in the probing notebooks; misleading is
# role-specific (role vars get counter-role names and vice versa), so it is
# materialized once per role as misleading_<role>.
BASE_STRATEGIES = ["baseline", "random_nouns", "single_chars", "all_same", "numeric_vars"]
STRATEGIES = BASE_STRATEGIES + [f"misleading_{r}" for r in ROLES]
