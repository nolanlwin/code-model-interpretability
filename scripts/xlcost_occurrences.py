"""Boolean-flag occurrences over the XLCoST canonical corpus (PROTOCOL.md §1.1).

Bridges scripts/xlcost_data.py output (``problem_id``/``language``/``code``)
to the per-language occurrence extractors, and stamps every row with the
protocol's stable occurrence identity:

    occurrence_id = <problem_id>:<language>:f<function_index>:b<binding_index>:o<occurrence_index>

- ``function_index``: order of the enclosing function within the program
- ``binding_index``: order of first appearance of the variable within it
- ``occurrence_index``: order of this occurrence within (function, variable)

Renaming conditions must carry these ids through unchanged; every paired
delta joins on them (never on character spans, which renaming shifts).

Span integrity gate: each row must satisfy ``code[s:e] == variable``. Rows
that fail (the known tree-sitter byte-vs-char hazard on non-ASCII source)
are DROPPED AND COUNTED, never silently kept.

    uv run python scripts/xlcost_occurrences.py extract \
        --input data/xlcost/python_valid.jsonl --output outputs/xlcost_occ/python_valid.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from variable_occurrences import occurrence_rows_from_code  # noqa: E402

# XLCoST language name -> how to extract. python/java/go go through the
# dispatcher; javascript/php/c++ through their own library modules.
_DISPATCHER_LANGS = {"Python": "python", "Java": "java"}
# "C" deliberately reuses the C++ extractor: C is very nearly a subset,
# and tree-sitter-cpp parses the XLCoST C corpus BETTER than
# tree-sitter-c does (381/421 vs 363/421). A separate C trio would be
# 700 lines to parse strictly less.
_MODULE_LANGS = {
    "Javascript": "javascript",
    "PHP": "php",
    "C++": "cpp",
    "C": "cpp",
}


def extract_rows(language: str, code: str, tokenizer=None, max_length: int = 2048):
    if language in _DISPATCHER_LANGS:
        return occurrence_rows_from_code(
            code,
            language=_DISPATCHER_LANGS[language],
            tokenizer=tokenizer,
            max_length=max_length,
        )
    if language in _MODULE_LANGS:
        slug = _MODULE_LANGS[language]
        mod = __import__(f"{slug}_variable_occurrences")
        fn = getattr(mod, f"occurrence_rows_from_{slug}_code")
        return fn(code, tokenizer=tokenizer, max_length=max_length)
    raise SystemExit(
        f"unsupported language {language!r}: no extractor "
        f"(supported: {sorted(_DISPATCHER_LANGS | _MODULE_LANGS)})"
    )


def assign_occurrence_ids(rows: list[dict], problem_id: str, language: str) -> list[dict]:
    """Stamp protocol occurrence ids; rows arrive sorted by (line, col)."""
    fn_order: dict = {}
    for r in rows:
        key = (r.get("function_lineno", 0), r.get("function", "?"))
        fn_order.setdefault(key, len(fn_order))
    binding_order: dict = {}
    occ_counter: Counter = Counter()
    for r in rows:
        fkey = (r.get("function_lineno", 0), r.get("function", "?"))
        f_idx = fn_order[fkey]
        bkey = (fkey, r.get("variable"))
        if bkey not in binding_order:
            binding_order[bkey] = sum(1 for k in binding_order if k[0] == fkey)
        b_idx = binding_order[bkey]
        o_idx = occ_counter[bkey]
        occ_counter[bkey] += 1
        r["problem_id"] = problem_id
        r["language"] = language
        r["occurrence_id"] = f"{problem_id}:{language}:f{f_idx}:b{b_idx}:o{o_idx}"
    return rows


def span_gate(rows: list[dict], code: str) -> tuple[list[dict], int]:
    """Drop rows whose span does not slice back to the variable name.

    PHP spans cover the whole ``variable_name`` node including the ``$``
    sigil, while ``variable`` holds the bare name — accept that form too.
    """
    kept, dropped = [], 0
    for r in rows:
        span = r.get("source_span") or [0, 0]
        s, e = int(span[0]), int(span[1])
        var = r.get("variable")
        if 0 <= s < e <= len(code) and code[s:e] in (var, f"${var}"):
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def program_occurrence_rows(
    language: str, code: str, problem_id: str, tokenizer=None, max_length: int = 2048
) -> tuple[list[dict], dict]:
    """The full per-program pipeline: extract -> span gate -> conflict dedupe ->
    sort -> stable occurrence ids. Single source of truth — the renamer's
    id-preservation check calls this on renamed code, so both sides always
    apply identical rules."""
    rows, err = extract_rows(language, code, tokenizer, max_length)
    if err is not None:
        return [], {"parse_error": True, "span_dropped": 0}
    rows, dropped = span_gate(rows, code)
    by_span: dict = {}
    for r in rows:
        by_span.setdefault(
            (r.get("function"), r.get("variable"), tuple(r["source_span"])), []
        ).append(r)
    clean = [g[0] for g in by_span.values() if len({x["occurrence_type"] for x in g}) == 1]
    rows = sorted(clean, key=lambda r: (r["line"], r.get("col_offset") or 0))
    rows = assign_occurrence_ids(rows, problem_id, language)
    return rows, {"parse_error": False, "span_dropped": dropped}


def cmd_extract(args: argparse.Namespace) -> int:
    tokenizer = None
    if args.model_id:
        from token_alignment import load_alignment_tokenizer

        tokenizer = load_alignment_tokenizer(args.model_id)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Remove any previous completion marker BEFORE truncating the output.
    # Callers guard re-extraction on this file, so a marker left over from an
    # earlier complete run would vouch for a partial rewrite if this one is
    # interrupted.
    Path(str(out) + ".stats.json").unlink(missing_ok=True)
    n_prog = n_parse_err = n_rows = n_span_dropped = 0
    dedup: set = set()
    with out.open("w", encoding="utf-8") as f:
        for ln in Path(args.input).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            code, language = rec["code"], rec["language"]
            problem_id = rec["problem_id"]
            n_prog += 1
            rows, pstats = program_occurrence_rows(
                language, code, problem_id, tokenizer, args.max_length
            )
            if pstats["parse_error"]:
                n_parse_err += 1
                continue
            n_span_dropped += pstats["span_dropped"]
            for r in rows:
                if r["occurrence_id"] in dedup:
                    continue
                dedup.add(r["occurrence_id"])
                r["split"] = rec.get("split")
                f.write(json.dumps(r) + "\n")
                n_rows += 1

    stats = {
        "input": args.input,
        "programs": n_prog,
        "parse_errors": n_parse_err,
        "span_gate_dropped": n_span_dropped,
        "occurrences_written": n_rows,
        "output": str(out),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    code = (
        "def demo(flag):\n"
        "    ok = True\n"
        "    if flag and ok:\n"
        "        return flag\n"
        "    while flag:\n"
        "        pass\n"
        "    return ok\n"
    )
    rows, err = extract_rows("Python", code)
    rows, dropped = span_gate(rows, code)
    rows = assign_occurrence_ids(rows, "deadbeef", "Python")
    ids = [r["occurrence_id"] for r in rows]
    checks = [
        ("no parse error", err is None),
        ("no span drops on ASCII", dropped == 0),
        ("rows found", len(rows) >= 5),
        ("ids unique", len(ids) == len(set(ids))),
        ("id format", all(i.startswith("deadbeef:Python:f0:") for i in ids)),
        ("two bindings", any(":b1:" in i for i in ids)),
    ]
    ok = True
    for name, passed in checks:
        print(f"  {'OK ' if passed else 'FAIL'} {name}")
        ok &= passed
    # Non-ASCII gate check: a corrupted span must be dropped, not kept.
    bad = [{"variable": "flag", "source_span": [0, 4], "occurrence_type": "x"}]
    kept, d = span_gate(bad, "xöxxflag")
    print(f"  {'OK ' if (not kept and d == 1) else 'FAIL'} span gate drops misaligned row")
    ok &= not kept and d == 1
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="extract occurrences from canonical XLCoST JSONL")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--model-id", help="optional: fill token_positions via this tokenizer")
    e.add_argument("--max-length", type=int, default=2048)
    sub.add_parser("verify", help="self-check id assignment and the span gate")
    args = ap.parse_args(argv)
    return cmd_verify(args) if args.cmd == "verify" else cmd_extract(args)


if __name__ == "__main__":
    sys.exit(main())
