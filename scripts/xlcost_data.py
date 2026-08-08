"""XLCoST corpus acquisition -> canonical JSONL (PROTOCOL.md §1.2).

The OFFICIAL XLCoST release is TransCoder-tokenized (``NEW_LINE`` / ``INDENT``
/ ``DEDENT`` sentinels, spaces around all punctuation): 0/300 official Python
programs parse. Do not use it raw. This script pulls the usable HuggingFace
mirrors instead:

- Python, C++ : ``giulio98/xlcost-formatted`` (properly detokenized; Python
  parse rate measured 92%)
- others      : ``codeparrot/xlcost-text-to-code`` (tokenized; brace languages
  parse as-is at ~89%, and this script detokenizes the spacing anyway so BPE
  sees natural code — ``System.out.println`` not ``System . out . println``)

Output: ``data/xlcost/<lang>_{train,valid,test}.jsonl`` with one program per
line: ``{"problem_id", "language", "split", "code", "text"}``. ``problem_id``
is a hash of the problem DESCRIPTION, so it is stable across languages — the
grouped-split and cross-language joins key on it.

    uv run python scripts/xlcost_data.py build --language Python --split valid
    uv run python scripts/xlcost_data.py build --language Java --split all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FORMATTED_REPO = "giulio98/xlcost-formatted"  # Python, C++ (detokenized)
TOKENIZED_REPO = "codeparrot/xlcost-text-to-code"  # all 7 (tokenized)
FORMATTED_LANGS = {"Python", "C++"}
ALL_LANGS = ["Python", "Java", "C++", "C#", "Javascript", "PHP", "C"]
# The codeparrot mirror names the C# directory "Csharp".
_MIRROR_DIRNAME = {"C#": "Csharp"}
SPLITS = ["train", "valid", "test"]


def detok_brace(s: str, language: str = "") -> str:
    """Undo TransCoder spacing for brace languages (whitespace not semantic)."""
    s = s.replace(" NEW_LINE ", "\n").replace("NEW_LINE", "\n")
    # Member-access / float chains first, with lookarounds so `a . b . c` joins
    # fully in one pass (a capturing group would consume the shared letter).
    s = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", s)
    # Multi-char operators the tokenizer split. None of the joined forms can
    # arise from validly-spaced code (`a - > b` / `a = > b` parse nowhere).
    s = re.sub(r"-\s+>", "->", s)  # member deref (C/C++/PHP)
    s = re.sub(r"=\s+>", "=>", s)  # JS arrow fns, PHP array arrows
    if language == "PHP":
        s = re.sub(r"<\s*\?\s*php", "<?php", s)
        s = re.sub(r"\?\s*>", "?>", s)
        s = re.sub(r"\$\s+(?=\w)", "$", s)  # `$ arr` -> `$arr`
        s = re.sub(r"::\s+", "::", s)
    s = re.sub(r"\s+([;,)\]}])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"(\w)\s*\(", r"\1(", s)  # call spacing
    s = re.sub(r"\s*(\+\+|--)", r"\1", s)
    return s


def problem_id(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def fetch_split(language: str, split: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    repo = FORMATTED_REPO if language in FORMATTED_LANGS else TOKENIZED_REPO
    dirname = _MIRROR_DIRNAME.get(language, language)
    fname = f"data/{dirname}-program-level/{split}.json"
    local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
    raw = Path(local).read_text(encoding="utf-8")
    try:
        recs = json.loads(raw)
        if not isinstance(recs, list):
            recs = [recs]
    except json.JSONDecodeError:
        recs = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    return recs


def validate(language: str, code: str) -> bool:
    if language == "Python":
        import ast

        try:
            ast.parse(code)
            return True
        except Exception:
            return False
    if language == "Java":
        try:
            import tree_sitter
            import tree_sitter_java

            parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_java.language()))
            return not parser.parse(bytes(code, "utf8")).root_node.has_error
        except Exception:
            pass
    # Other brace languages: structural balance heuristic.
    return code.count("{") == code.count("}") and code.count("(") == code.count(")")


def build_split(language: str, split: str, out_dir: Path) -> dict:
    recs = fetch_split(language, split)
    needs_detok = language not in FORMATTED_LANGS
    kept, failed = [], 0
    for r in recs:
        code, text = r.get("code", ""), r.get("text", "")
        if needs_detok:
            code = detok_brace(code, language)
        if not code.strip() or not validate(language, code):
            failed += 1
            continue
        kept.append(
            {
                "problem_id": problem_id(text),
                "language": language,
                "split": split,
                "code": code,
                "text": text,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_slug = language.lower().replace("++", "pp").replace("#", "sharp")
    out = out_dir / f"{lang_slug}_{split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    n_problems = len({k["problem_id"] for k in kept})
    stats = {
        "language": language,
        "split": split,
        "source_repo": FORMATTED_REPO if language in FORMATTED_LANGS else TOKENIZED_REPO,
        "records_in": len(recs),
        "kept": len(kept),
        "validation_failed": failed,
        "unique_problems": n_problems,
        "output": str(out),
    }
    print(json.dumps(stats))
    return stats


def cmd_build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    splits = SPLITS if args.split == "all" else [args.split]
    langs = ALL_LANGS if args.language == "all" else [args.language]
    manifest, failures = [], 0
    for lang in langs:
        for sp in splits:
            try:
                manifest.append(build_split(lang, sp, out_dir))
            except Exception as e:  # keep building; record the failure
                failures += 1
                entry = {"language": lang, "split": sp, "error": f"{type(e).__name__}: {e}"}
                print(json.dumps(entry), file=sys.stderr)
                manifest.append(entry)
    (out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return 1 if failures else 0


def cmd_verify(_args: argparse.Namespace) -> int:
    ok = True
    # Detokenizer must produce natural member access and preserve strings' shape.
    s = "import java . io . * ; class A { void f ( ) { System . out . println ( x ) ; } }"
    d = detok_brace(s)
    checks = [
        ("member access joined", "java.io" in d and "System.out.println" in d),
        ("no NEW_LINE left", "NEW_LINE" not in detok_brace("a ; NEW_LINE b ;")),
        ("problem_id stable", problem_id(" A  b ") == problem_id("a B")),
        ("python validate", validate("Python", "def f():\n    return 1\n")),
        ("python invalid rejected", not validate("Python", "def f(:")),
    ]
    for name, passed in checks:
        print(f"  {'OK ' if passed else 'FAIL'} {name}")
        ok &= passed
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download a mirror split and emit canonical JSONL")
    b.add_argument("--language", default="Python", choices=ALL_LANGS + ["all"])
    b.add_argument("--split", default="all", choices=SPLITS + ["all"])
    b.add_argument("--out-dir", default="data/xlcost")
    sub.add_parser("verify", help="self-check the detokenizer and validators")
    args = ap.parse_args(argv)
    return cmd_verify(args) if args.cmd == "verify" else cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
