"""
Structural (syntactic-class) labels for code tokens, shared across Java and Python.

Parses a snippet with tree-sitter, walks every **leaf** node, and assigns each leaf
a coarse ``structural_class`` from one shared label space so Java and Python probes
use the same targets:

  identifier | keyword | operator | string | number | punctuation | bool_null | comment

Each leaf record carries a half-open **character** ``source_span`` ``[start, end)`` in
the original string (tree-sitter reports byte offsets; we convert to char offsets so
spans line up with the tokenizer ``offset_mapping`` used in ``token_alignment.py``).

CLI:
  verify   parse a fixture in both languages and assert the expected classes appear
  extract  canonical JSONL (problem_id, language, code) or --code-file -> leaf JSONL
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "structure_labels" / "leaves.jsonl"

SUPPORTED_LANGUAGES = ("python", "java")

# Shared coarse label space (order is the canonical class index ordering).
STRUCTURAL_CLASSES = (
    "identifier",
    "keyword",
    "operator",
    "string",
    "number",
    "punctuation",
    "bool_null",
    "comment",
)

# Named leaf node types that map directly, regardless of language.
_IDENTIFIER_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "field_identifier",
        "scoped_identifier",
        "scoped_type_identifier",
        "dotted_name",
    }
)
_BOOL_NULL_TYPES = frozenset({"true", "false", "none", "null", "null_literal"})

# Anonymous (literal) tokens that are punctuation rather than operators.
_PUNCTUATION = frozenset(set("()[]{},;.") | {"->", "::", "..."})

# tree-sitter is unavailable until deps are installed; import lazily so the CLI
# ``--help`` and unit imports do not hard-fail in a bare environment.
_LANG_CACHE: dict[str, Any] = {}


@lru_cache(maxsize=None)
def _imports():  # pragma: no cover - thin wrapper
    from tree_sitter import Language, Parser  # noqa: PLC0415

    return Language, Parser


def get_parser(language: str):
    """Return a cached tree-sitter ``Parser`` for ``python`` or ``java``."""
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language {language!r}; expected {SUPPORTED_LANGUAGES}")
    if language in _LANG_CACHE:
        return _LANG_CACHE[language]

    Language, Parser = _imports()
    if language == "python":
        import tree_sitter_python as ts_lang  # noqa: PLC0415
    else:
        import tree_sitter_java as ts_lang  # noqa: PLC0415

    lang = Language(ts_lang.language())
    parser = Parser(lang)
    _LANG_CACHE[language] = parser
    return parser


def classify_leaf(node_type: str, text: str, is_named: bool) -> str:
    """Map a tree-sitter leaf to one shared structural class."""
    t = node_type.lower()

    if t in _BOOL_NULL_TYPES or text in ("True", "False", "None", "true", "false", "null"):
        return "bool_null"
    if t in _IDENTIFIER_TYPES:
        return "identifier"
    if "comment" in t:
        return "comment"
    if "string" in t or "char" in t or t == "escape_sequence":
        return "string"
    if any(k in t for k in ("integer", "float", "decimal", "hex", "number", "literal")) and (
        text[:1].isdigit() or text[:1] in "+-."
    ):
        return "number"

    stripped = text.strip()
    if stripped and (stripped.isidentifier() or stripped.replace("_", "a").isalpha()):
        # Anonymous alphabetic tokens are language keywords (def/if/class, and also
        # word operators like and/or/not/in/is/instanceof -> treated as keywords).
        return "keyword"
    if stripped in _PUNCTUATION or (len(stripped) == 1 and stripped in "()[]{},;.:"):
        return "punctuation"
    if stripped:
        return "operator"
    return "operator"


def _byte_to_char_map(code: str) -> tuple[bytes, list[int]]:
    """Return UTF-8 bytes and a ``byte_index -> char_index`` lookup of length len+1."""
    data = code.encode("utf-8")
    mapping = [0] * (len(data) + 1)
    char_idx = 0
    byte_idx = 0
    for ch in code:
        n = len(ch.encode("utf-8"))
        for b in range(byte_idx, byte_idx + n):
            mapping[b] = char_idx
        byte_idx += n
        char_idx += 1
    mapping[len(data)] = char_idx
    return data, mapping


def structural_leaves_from_code(
    code: str,
    language: str,
    *,
    include_comments: bool = True,
) -> list[dict[str, Any]]:
    """
    One record per leaf token: ``{source_span:[start,end], node_type, structural_class, text}``.

    ``source_span`` is a half-open character range in ``code``.
    """
    parser = get_parser(language)
    data, b2c = _byte_to_char_map(code)
    tree = parser.parse(data)

    rows: list[dict[str, Any]] = []
    stack = [tree.root_node]
    leaves: list[Any] = []
    while stack:
        node = stack.pop()
        if node.child_count == 0:
            leaves.append(node)
        else:
            stack.extend(reversed(node.children))

    for node in leaves:
        sb, eb = node.start_byte, node.end_byte
        if eb <= sb:
            continue  # zero-width (e.g. error/EOF) leaves
        text = data[sb:eb].decode("utf-8", errors="replace")
        if node.type in ("ERROR",) or node.is_error:
            continue
        cls = classify_leaf(node.type, text, node.is_named)
        if cls == "comment" and not include_comments:
            continue
        s_char, e_char = b2c[sb], b2c[eb]
        rows.append(
            {
                "source_span": [s_char, e_char],
                "node_type": node.type,
                "structural_class": cls,
                "text": text,
            }
        )

    rows.sort(key=lambda r: (r["source_span"][0], r["source_span"][1]))
    return rows


def iter_canonical_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cmd_extract(args: argparse.Namespace) -> int:
    include_comments = not args.no_comments

    if args.code_file:
        code = Path(args.code_file).read_text(encoding="utf-8")
        rows = structural_leaves_from_code(
            code, args.language, include_comments=include_comments
        )
        out_dest = None if args.output == "-" else Path(args.output)
        if out_dest is None:
            for rec in rows:
                sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            out_dest.parent.mkdir(parents=True, exist_ok=True)
            with out_dest.open("w", encoding="utf-8") as fout:
                for rec in rows:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"wrote {len(rows)} leaves -> {out_dest}")
        return 0

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"no such file: {in_path}", file=sys.stderr)
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    max_rows = args.max_rows
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(desc="structure leaves", unit="row", total=max_rows)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            pbar.update(1)
            row = json.loads(line)
            code = row.get("code") or ""
            language = (row.get("language") or args.language or "").lower()
            if language not in SUPPORTED_LANGUAGES:
                if max_rows is not None and n_in >= max_rows:
                    break
                continue
            problem_id = row.get("problem_id")
            for rec in structural_leaves_from_code(
                code, language, include_comments=include_comments
            ):
                rec["problem_id"] = problem_id
                rec["language"] = language
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
            if max_rows is not None and n_in >= max_rows:
                break
        pbar.close()

    print(f"read_rows={n_in} written_leaves={n_out} -> {out_path}")
    return 0


_VERIFY_PY = "def f(x):\n    ok = True\n    if ok and x > 0:\n        return 'hi'  # note\n    return 1\n"
_VERIFY_JAVA = (
    "class A {\n    int f(int x) {\n        boolean ok = true;\n"
    '        if (ok) { return 1; }\n        return 0; // c\n    }\n}\n'
)


def _cmd_verify(args: argparse.Namespace) -> int:
    expected = {"identifier", "keyword", "operator", "number", "punctuation", "bool_null"}
    for lang, code in (("python", _VERIFY_PY), ("java", _VERIFY_JAVA)):
        rows = structural_leaves_from_code(code, lang)
        classes = {r["structural_class"] for r in rows}
        # Every leaf span must slice back to its own text.
        for r in rows:
            s, e = r["source_span"]
            if code[s:e] != r["text"]:
                print(
                    f"verify[{lang}]: span {r['source_span']} -> {code[s:e]!r} != {r['text']!r}",
                    file=sys.stderr,
                )
                return 1
        missing = expected - classes
        if missing:
            print(
                f"verify[{lang}]: missing classes {sorted(missing)} (have {sorted(classes)})",
                file=sys.stderr,
            )
            return 1
        print(f"structure_labels verify[{lang}]: ok ({len(rows)} leaves, classes {sorted(classes)})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structural (syntactic-class) leaf labels for Java/Python via tree-sitter."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser(
        "extract",
        help="Canonical JSONL (problem_id, language, code) or --code-file -> per-leaf JSONL.",
    )
    src = ex.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Canonical JSONL path.")
    src.add_argument("--code-file", type=str, help="Single UTF-8 source file.")
    ex.add_argument(
        "--language",
        type=str,
        default=None,
        choices=[*SUPPORTED_LANGUAGES],
        help="Language for --code-file, or fallback when a JSONL row omits 'language'.",
    )
    ex.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"JSONL path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)}), or '-' for stdout.",
    )
    ex.add_argument("--max-rows", type=int, default=None)
    ex.add_argument("--no-comments", action="store_true", help="Drop comment leaves.")
    ex.set_defaults(func=_cmd_extract)

    v = sub.add_parser("verify", help="Parse a fixture in both languages and check classes.")
    v.set_defaults(func=_cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
