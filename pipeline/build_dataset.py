"""Build the common role-labeled dataset from XLCoST.

Two configs:
  python_perturbations — Python programs under all 10 naming strategies with
                         role labels re-extracted from the transformed code
  multilingual_baseline — original programs in all 7 languages with role labels

Rows store code + role-name sets (model-agnostic); token-level labels are
computed at experiment time for whichever tokenizer is probed.

Usage:
  python -m pipeline.build_dataset --out dataset [--max-programs 500] [--splits train test]
"""

import argparse
import json
import os
from collections import Counter

from . import LANGUAGES, ROLES, STRATEGIES
from .perturb import perturb
from .roles import extract_roles
from .xlcost import load_programs


def _row(idx, language, split, strategy, code):
    roles = extract_roles(code, language)
    if not any(roles.values()):
        return None
    return {
        "id": f"{idx}:{language}:{strategy}",
        "problem_id": idx,
        "language": language,
        "split": split,
        "strategy": strategy,
        "code": code,
        "roles": {role: sorted(names) for role, names in roles.items()},
    }


def build_python_perturbations(out_dir, splits, max_programs):
    stats = Counter()
    for split in splits:
        path = os.path.join(out_dir, "python_perturbations", f"{split}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for idx, code in load_programs("Python", split, max_programs):
                for strategy in STRATEGIES:
                    row = _row(idx, "Python", split, strategy, perturb(code, strategy, seed=idx))
                    if row is None:
                        stats[f"{split}:skipped"] += 1
                        continue
                    f.write(json.dumps(row) + "\n")
                    stats[f"{split}:{strategy}"] += 1
        print(f"python_perturbations/{split}: done")
    return stats


def build_multilingual_baseline(out_dir, splits, max_programs):
    stats = Counter()
    for split in splits:
        path = os.path.join(out_dir, "multilingual_baseline", f"{split}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for language in LANGUAGES:
                try:
                    programs = load_programs(language, split, max_programs)
                    for idx, code in programs:
                        row = _row(idx, language, split, "baseline", code)
                        if row is None:
                            stats[f"{split}:{language}:skipped"] += 1
                            continue
                        f.write(json.dumps(row) + "\n")
                        stats[f"{split}:{language}"] += 1
                except FileNotFoundError as e:
                    print(f"  skip {language}/{split}: {e}")
        print(f"multilingual_baseline/{split}: done")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--max-programs", type=int, default=None,
                    help="cap programs per language/split (default: all)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = ap.parse_args()

    stats = {
        "roles": ROLES,
        "strategies": STRATEGIES,
        "max_programs": args.max_programs,
        "python_perturbations": dict(build_python_perturbations(args.out, args.splits, args.max_programs)),
        "multilingual_baseline": dict(build_multilingual_baseline(args.out, args.splits, args.max_programs)),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"\nWrote {args.out}/stats.json")


if __name__ == "__main__":
    main()
