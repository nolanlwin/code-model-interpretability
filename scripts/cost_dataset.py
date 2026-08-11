"""
CoST (MuST-CoST) dataset loader: read the local wide CSV of GeeksforGeeks problems
aligned across languages and emit canonical JSONL for probing.

The repository ships a single wide CSV (``consolidated_data.csv``) with one row per
problem and one column per language:

  Problem ID, Problem Title, C++, Java, Python, C#, Javascript, PHP, C

This is the same corpus as MuST-CoST's ``raw_data`` (the per-problem aligned
snippets) merged into one file, so no zip download is required. For structure
probing we keep only problems that have BOTH a Java and a Python program, and emit
two canonical rows per problem so the languages pair on ``problem_id``.

Canonical record:
  {"problem_id": 1, "language": "python", "code": "...", "title": "..."}

CLI:
  build   read CSV, keep Java+Python aligned rows, write canonical JSONL
  verify  parse the CSV header + a few rows and assert the expected shape
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "consolidated_data.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "cost" / "java_python.jsonl"

# Languages we probe. Maps the CSV column header to the canonical language tag.
LANGUAGE_COLUMNS: dict[str, str] = {"Java": "java", "Python": "python"}
ID_COLUMN = "Problem ID"
TITLE_COLUMN = "Problem Title"

# CoST cells can be whole programs; raise the field-size limit so the CSV reader
# does not choke on long multi-line quoted code.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def iter_csv_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        missing = [
            c
            for c in (ID_COLUMN, *LANGUAGE_COLUMNS)
            if c not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing expected columns {missing}; "
                f"found {reader.fieldnames}"
            )
        yield from reader


def canonical_rows_from_csv(
    csv_path: Path,
    *,
    require_both: bool = True,
    max_problems: int | None = None,
) -> Iterator[dict[str, object]]:
    """
    Yield ``{"problem_id", "language", "code", "title"}`` for Java and Python.

    With ``require_both`` (default) a problem only contributes rows when it has a
    non-empty program in *every* probed language, so the languages stay aligned
    on ``problem_id``.
    """
    kept = 0
    for row in iter_csv_rows(csv_path):
        raw_id = (row.get(ID_COLUMN) or "").strip()
        if not raw_id:
            continue
        try:
            problem_id: int | str = int(raw_id)
        except ValueError:
            problem_id = raw_id

        title = (row.get(TITLE_COLUMN) or "").strip()
        codes = {
            tag: (row.get(col) or "").strip()
            for col, tag in LANGUAGE_COLUMNS.items()
        }
        if require_both and not all(codes.values()):
            continue

        emitted_any = False
        for tag, code in codes.items():
            if not code:
                continue
            yield {
                "problem_id": problem_id,
                "language": tag,
                "code": code,
                "title": title,
            }
            emitted_any = True

        if emitted_any:
            kept += 1
            if max_problems is not None and kept >= max_problems:
                break


def iter_canonical_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cmd_build(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"no such file: {csv_path}", file=sys.stderr)
        print(
            "Provide --csv pointing at the CoST consolidated CSV "
            "(columns: Problem ID, Java, Python, ...).",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_lang: dict[str, int] = {tag: 0 for tag in LANGUAGE_COLUMNS.values()}
    problem_ids: set[object] = set()
    n_out = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in canonical_rows_from_csv(
            csv_path,
            require_both=not args.allow_single,
            max_problems=args.max_problems,
        ):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            per_lang[str(rec["language"])] += 1
            problem_ids.add(rec["problem_id"])

    counts = " ".join(f"{tag}={n}" for tag, n in per_lang.items())
    print(
        f"problems={len(problem_ids)} rows={n_out} ({counts}) -> {out_path}"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"no such file: {csv_path}", file=sys.stderr)
        return 1

    both = 0
    per_lang: dict[str, int] = {tag: 0 for tag in LANGUAGE_COLUMNS.values()}
    total = 0
    for row in iter_csv_rows(csv_path):
        total += 1
        codes = {
            tag: (row.get(col) or "").strip()
            for col, tag in LANGUAGE_COLUMNS.items()
        }
        for tag, code in codes.items():
            if code:
                per_lang[tag] += 1
        if all(codes.values()):
            both += 1

    counts = " ".join(f"{tag}={n}" for tag, n in per_lang.items())
    print(
        f"cost_dataset verify: ok (rows={total} {counts} both_java_python={both})"
    )
    if both < 1:
        print("verify: no rows had both Java and Python", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CoST Java+Python loader: wide CSV -> canonical JSONL."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Write canonical {problem_id, language, code} JSONL.")
    b.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"CoST consolidated CSV (default: {DEFAULT_CSV.name}).",
    )
    b.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"Output JSONL (default: {DEFAULT_OUT}).",
    )
    b.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Stop after N kept problems (smoke tests).",
    )
    b.add_argument(
        "--allow-single",
        action="store_true",
        help="Keep a problem even if only one of Java/Python is present (breaks alignment).",
    )
    b.set_defaults(func=_cmd_build)

    v = sub.add_parser("verify", help="Check the CSV header and language coverage.")
    v.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    v.set_defaults(func=_cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
