"""
Map source-level character spans to tokenizer positions.

Uses Hugging Face ``offset_mapping`` from a fast tokenizer so spans align with the
same tokenization contract as ``scripts/qwen_inference.py`` (truncation, special
tokens, max length).

CLI: ``align`` (JSON in/out), ``verify`` (fixture invariants + optional cases file).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# transformers is imported lazily inside the CLI commands that build a
# tokenizer. The alignment helpers above them (char_span_to_token_indices,
# span_chars_covered) are pure and are imported by tooling that runs without
# torch installed -- a module-level import here would make those unusable.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEFAULT_CASES = PROJECT_ROOT / "fixtures" / "token_alignment_cases.json"
DEFAULT_MINIMAL_SPANS = PROJECT_ROOT / "fixtures" / "minimal_forward_align_spans.json"


def _overlap_half_open(a0: int, a1: int, b0: int, b1: int) -> bool:
    """True iff [a0, a1) and [b0, b1) intersect and both are non-empty."""
    return a0 < b1 and a1 > b0 and a0 < a1 and b0 < b1


def char_span_to_token_indices(
    offset_mapping: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """
    Half-open character span ``[char_start, char_end)`` in the original string.

    Returns every token index whose offset range intersects that span (ordered
    left-to-right). Skips special tokens with empty ``(0, 0)``-style ranges.
    """
    if char_start < 0 or char_end < char_start:
        raise ValueError(f"invalid span: [{char_start}, {char_end})")
    out: list[int] = []
    for i, (ts, te) in enumerate(offset_mapping):
        if te <= ts:
            continue
        if _overlap_half_open(ts, te, char_start, char_end):
            out.append(i)
    return out


def span_chars_covered(
    code: str,
    char_start: int,
    char_end: int,
    token_indices: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
) -> bool:
    """Each codepoint index in ``[char_start, char_end)`` lies inside some selected token span."""
    n = len(code)
    for pos in range(char_start, min(char_end, n)):
        ok = False
        for i in token_indices:
            ts, te = offset_mapping[i]
            if ts <= pos < te:
                ok = True
                break
        if not ok:
            return False
    return True


def tokenize_for_alignment(
    tokenizer,
    code: str,
    *,
    max_length: int = 2048,
) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    """Match ``qwen_inference.forward_code`` tokenization defaults."""
    enc = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=tokenizer.is_fast,
    )
    row = enc["offset_mapping"][0]
    pairs = [(int(a), int(b)) for a, b in row.tolist()]
    ids = enc["input_ids"][0].tolist()
    pieces = tokenizer.convert_ids_to_tokens(ids)
    return ids, pairs, pieces


def align_one(
    *,
    code: str,
    variable: str,
    source_span: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    decoded_tokens: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    s, e = int(source_span[0]), int(source_span[1])
    if code[s:e] != variable:
        raise ValueError(
            f"code[{s}:{e}]={code[s:e]!r} does not equal variable={variable!r}"
        )
    idx = char_span_to_token_indices(offset_mapping, s, e)
    out: dict[str, Any] = {
        "variable": variable,
        "source_span": [s, e],
        "token_positions": idx,
        "decoded_tokens": [decoded_tokens[i] for i in idx],
    }
    if extra:
        for k, v in extra.items():
            if k not in out:
                out[k] = v
    return out


def align_batch(
    tokenizer,
    code: str,
    occurrences: list[dict[str, Any]],
    *,
    max_length: int = 2048,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Each occurrence dict: ``variable``, ``source_span`` [start, end); optional ``id``.
    """
    notes: list[str] = []
    if not tokenizer.is_fast:
        raise RuntimeError(
            "Tokenizer is slow; offset_mapping is unavailable. "
            "Use a fast tokenizer (e.g. Qwen2.5) for alignment."
        )
    _, offset_mapping, pieces = tokenize_for_alignment(
        tokenizer, code, max_length=max_length
    )
    results: list[dict[str, Any]] = []
    for occ in occurrences:
        extra = {k: v for k, v in occ.items() if k not in ("variable", "source_span")}
        row = align_one(
            code=code,
            variable=str(occ["variable"]),
            source_span=occ["source_span"],
            offset_mapping=offset_mapping,
            decoded_tokens=pieces,
            extra=extra or None,
        )
        if not row["token_positions"] and int(occ["source_span"][0]) < int(occ["source_span"][1]):
            oid = occ.get("id", occ.get("variable"))
            notes.append(
                f"occurrence {oid!r}: no token overlaps span; check truncation or tokenizer."
            )
        results.append(row)
    return results, notes


def verify_case(
    tokenizer,
    case: dict[str, Any],
    *,
    max_length: int,
) -> list[str]:
    """Return list of error strings (empty if ok)."""
    errs: list[str] = []
    cid = case.get("id", "?")
    code = case["code"]
    var = case["variable"]
    span = case["source_span"]
    s, e = int(span[0]), int(span[1])
    if code[s:e] != var:
        errs.append(f"{cid}: code[{s}:{e}]={code[s:e]!r} != variable {var!r}")
        return errs
    if not tokenizer.is_fast:
        errs.append(f"{cid}: tokenizer is not fast")
        return errs
    try:
        _, offset_mapping, _pieces = tokenize_for_alignment(
            tokenizer, code, max_length=max_length
        )
    except Exception as ex:  # noqa: BLE001
        errs.append(f"{cid}: tokenize failed: {ex}")
        return errs

    idx = char_span_to_token_indices(offset_mapping, s, e)
    if not idx:
        errs.append(f"{cid}: empty token_positions for non-empty span")
    if not span_chars_covered(code, s, e, idx, offset_mapping):
        errs.append(f"{cid}: span not fully covered by overlapping token offsets")
    for i in idx:
        ts, te = offset_mapping[i]
        if not _overlap_half_open(ts, te, s, e):
            errs.append(f"{cid}: token {i} offsets {(ts, te)} do not overlap span")
    return errs


def _default_spans_path(code_file: str) -> Path:
    """``dir/foo.py`` → ``dir/foo_align_spans.json`` (optional convention for ``align``)."""
    p = Path(code_file)
    return p.parent / f"{p.stem}_align_spans.json"


def cmd_align(args: argparse.Namespace) -> int:
    cf = args.code_file
    spans_arg: str | None = args.spans
    if spans_arg is None:
        if cf == "-":
            print(
                "--spans is required when code_file is '-' (stdin); "
                "or use align-bundle with one JSON body.",
                file=sys.stderr,
            )
            return 1
        default_spans = _default_spans_path(cf)
        if not default_spans.is_file():
            print(
                f"no default spans file {default_spans}; pass --spans PATH, "
                "or '-' to read spans from stdin.",
                file=sys.stderr,
            )
            return 1
        spans_arg = str(default_spans)

    if cf == "-" and spans_arg == "-":
        print(
            "cannot read both code and spans from stdin; use a file for one of them, "
            "or use the align-bundle subcommand.",
            file=sys.stderr,
        )
        return 1
    if cf == "-":
        code = sys.stdin.read()
    else:
        try:
            code = Path(cf).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"no such file: {cf}", file=sys.stderr)
            return 1

    if spans_arg == "-":
        raw = sys.stdin.read()
    else:
        try:
            raw = Path(spans_arg).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(
                f"no such file: {spans_arg}",
                file=sys.stderr,
            )
            print(
                "create that JSON list or use e.g. "
                f"{DEFAULT_MINIMAL_SPANS.relative_to(PROJECT_ROOT)} "
                "with fixtures/minimal_forward.py",
                file=sys.stderr,
            )
            return 1
    occurrences = json.loads(raw)
    if not isinstance(occurrences, list):
        print("spans JSON must be a list of objects", file=sys.stderr)
        return 1
    for i, occ in enumerate(occurrences):
        if not isinstance(occ, dict) or "variable" not in occ or "source_span" not in occ:
            print(f"occurrences[{i}] must be an object with 'variable' and 'source_span'", file=sys.stderr)
            return 1
        sp = occ["source_span"]
        if not isinstance(sp, (list, tuple)) or len(sp) != 2:
            print(f"occurrences[{i}].source_span must be [start, end)", file=sys.stderr)
            return 1

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        results, notes = align_batch(tok, code, occurrences, max_length=args.max_length)
    except (ValueError, RuntimeError) as ex:
        print(str(ex), file=sys.stderr)
        return 1

    payload: dict[str, Any] = {"alignments": results, "notes": notes}
    out_path = Path(args.output)
    if str(args.output) == "-":
        print(json.dumps(payload, indent=2))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


def cmd_align_bundle(args: argparse.Namespace) -> int:
    bp = args.bundle
    if bp == "-":
        raw = sys.stdin.read()
    else:
        try:
            raw = Path(bp).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"no such file: {bp}", file=sys.stderr)
            return 1
    bundle = json.loads(raw)
    code = bundle.get("code")
    occurrences = bundle.get("occurrences")
    if not isinstance(code, str) or not isinstance(occurrences, list):
        print("bundle JSON must have string 'code' and list 'occurrences'", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        results, notes = align_batch(tok, code, occurrences, max_length=args.max_length)
    except (ValueError, RuntimeError) as ex:
        print(str(ex), file=sys.stderr)
        return 1
    payload: dict[str, Any] = {"alignments": results, "notes": notes}
    out_path = Path(args.output)
    if str(args.output) == "-":
        print(json.dumps(payload, indent=2))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if not tok.is_fast:
        print("verify requires a fast tokenizer", file=sys.stderr)
        return 1
    cases_path = Path(args.cases)
    try:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"no such file: {args.cases}", file=sys.stderr)
        return 1
    if not isinstance(cases, list):
        print("cases file must be a JSON list", file=sys.stderr)
        return 1
    all_errs: list[str] = []
    for case in cases:
        all_errs.extend(verify_case(tok, case, max_length=args.max_length))
    if all_errs:
        for line in all_errs:
            print(line, file=sys.stderr)
        return 1
    print(f"token_alignment verify: {len(cases)} cases ok ({cases_path})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map source character spans to tokenizer indices (offset_mapping)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "align",
        help="Align JSON occurrences against a UTF-8 code file (stdin '-' for code or spans).",
    )
    a.add_argument("code_file", type=str, help="Path to UTF-8 source, or '-' for stdin (code only).")
    a.add_argument(
        "--spans",
        type=str,
        default=None,
        help=(
            "JSON file of [{variable, source_span, ...}], or '-' for stdin. "
            "If omitted and code_file is a path, default is "
            "<code_dir>/<code_stem>_align_spans.json when that file exists."
        ),
    )
    a.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    a.add_argument("--max-length", type=int, default=2048)
    a.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Write JSON here, or '-' for stdout (default: stdout).",
    )
    a.set_defaults(func=cmd_align)

    ab = sub.add_parser(
        "align-bundle",
        help="Read {\"code\": \"...\", \"occurrences\": [...]} from a file or stdin ('-').",
    )
    ab.add_argument(
        "bundle",
        type=str,
        help="Path to bundle JSON, or '-' for stdin.",
    )
    ab.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    ab.add_argument("--max-length", type=int, default=2048)
    ab.add_argument("--output", "-o", type=str, default="-")
    ab.set_defaults(func=cmd_align_bundle)

    v = sub.add_parser(
        "verify",
        help=f"Run invariant checks on fixture cases (default: {DEFAULT_CASES.name}).",
    )
    v.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    v.add_argument("--max-length", type=int, default=2048)
    v.add_argument("--cases", type=str, default=str(DEFAULT_CASES))
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
