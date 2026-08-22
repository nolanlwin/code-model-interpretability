"""Restrict every language to the problems all three share.

Reviewer point 2: pairwise matching removes subject matter as an explanation
within a cell, but each cell is matched on its OWN problem set (php-python on
1145, javascript-php on 1529), so the six cells do not see the same programs.
A boundary effect could in principle be composition rather than language.

Restricting all three languages to the 869-problem three-way intersection makes
every cell see identical problem ids, at the cost of a smaller sample.

    uv run python scripts/build_intersection.py
"""
from __future__ import annotations

import json
import pathlib

LANGS = ("python", "javascript", "php")
SPLIT = "train"
D = pathlib.Path("data/xlcost")
O = pathlib.Path("outputs/role_occ")


def main() -> int:
    ids = {}
    for s in LANGS:
        ids[s] = {json.loads(l)["problem_id"]
                  for l in (D / f"{s}_{SPLIT}.jsonl").read_text().splitlines() if l.strip()}
    shared = set.intersection(*ids.values())
    print(f"  three-way intersection: {len(shared)} problems")

    for s in LANGS:
        src = D / f"{s}_{SPLIT}.jsonl"
        dst = D / f"{s}_{SPLIT}_isect.jsonl"
        kept = [l for l in src.read_text().splitlines()
                if l.strip() and json.loads(l)["problem_id"] in shared]
        dst.write_text("\n".join(kept) + "\n")

        occ_src = O / f"all_{s}_{SPLIT}.jsonl"
        occ_dst = O / f"isect_{s}_{SPLIT}.jsonl"
        n = 0
        with occ_dst.open("w") as fh:
            for line in occ_src.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("problem_id") in shared:
                    fh.write(line + "\n")
                    n += 1
        # The completion marker other scripts guard on.
        (occ_dst.with_suffix(".jsonl.stats.json")).write_text(json.dumps(
            {"input": str(occ_src), "problems": len(shared), "occurrences_written": n,
             "restricted_to": "three-way intersection"}, indent=1))
        print(f"  {s:<11} {len(kept):>5} programs, {n:>6} occurrences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
