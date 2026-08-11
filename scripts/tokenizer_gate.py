"""Tokenizer gate — no model may be extracted from until this passes (PROTOCOL.md §3.2).

A roundtrip check is NOT sufficient (Yi-Coder round-trips yet has 8/17 wrong
offsets). The gate asserts, per token, that ``source[start:end]`` equals the
decoded token, on fixtures containing non-ASCII text, tabs, CRLF, and
digit-bearing identifiers.

Known hazard baked in: DeepSeek-Coder declares ``LlamaTokenizerFast`` over a
ByteLevel tokenizer.json; under transformers 5.8.0 AutoTokenizer silently
deletes whitespace and corrupts every offset (11/11 wrong). The registry loads
it via ``PreTrainedTokenizerFast`` instead. Fixed upstream in >= 5.14.0 — if
the pin ever moves, re-run this gate.

    uv run python scripts/tokenizer_gate.py run            # all six models
    uv run python scripts/tokenizer_gate.py run --models Qwen/Qwen2.5-Coder-1.5B
"""

from __future__ import annotations

import argparse
import json
import sys

PINNED_TRANSFORMERS = "5.8.0"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen3-4B-Base",
    "bigcode/starcoder2-7b",
    "ibm-granite/granite-3b-code-base-2k",
    "deepseek-ai/deepseek-coder-1.3b-base",
]

# Models whose AutoTokenizer path is known-broken under the pinned transformers.
FAST_OVERRIDE = {"deepseek-ai/deepseek-coder-1.3b-base"}

FIXTURES = [
    "if is_valid:\n    total += count\n    return found",
    "// café résumé é\nint x1_y2 = arr[idx];\n",
    "def f(a, b):\n\tflag = True\n\treturn flag  # tab-indented\n",
    "line1\r\nline2\r\n",
    "s = \"string with  double  spaces\"\nname_2 = s\n",
]


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if model_id in FAST_OVERRIDE:
        return PreTrainedTokenizerFast.from_pretrained(model_id), "PreTrainedTokenizerFast"
    return AutoTokenizer.from_pretrained(model_id), "AutoTokenizer"


def check_model(model_id: str) -> dict:
    tok, loader = load_tokenizer(model_id)
    total_bad = total_tokens = 0
    roundtrip_ok = True
    failures = []
    for fi, src in enumerate(FIXTURES):
        enc = tok(src, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if tok.decode(ids) != src:
            roundtrip_ok = False
        for i, (s, e) in enumerate(offs):
            total_tokens += 1
            piece = tok.decode([ids[i]])
            if src[s:e] != piece:
                total_bad += 1
                if len(failures) < 3:
                    failures.append(
                        {"fixture": fi, "token": piece, "span_text": src[s:e]}
                    )
    return {
        "model": model_id,
        "loader": loader,
        "tokenizer_class": type(tok).__name__,
        "roundtrip_ok": roundtrip_ok,
        "bad_offsets": total_bad,
        "total_tokens": total_tokens,
        "passed": roundtrip_ok and total_bad == 0,
        "sample_failures": failures,
    }


def cmd_run(args: argparse.Namespace) -> int:
    import transformers

    version_ok = transformers.__version__ == PINNED_TRANSFORMERS
    if not version_ok:
        msg = (
            f"transformers {transformers.__version__} != pinned {PINNED_TRANSFORMERS}: "
            "tokenizer behavior (incl. the DeepSeek workaround) is only validated on the pin."
        )
        if args.strict_version:
            raise SystemExit(f"FAIL: {msg}")
        print(f"WARNING: {msg}", file=sys.stderr)

    results, all_ok = [], True
    print(f"{'model':<42}{'loader':<26}{'roundtrip':<11}{'bad_offsets':<13}gate")
    for mid in args.models:
        try:
            res = check_model(mid)
        except Exception as e:  # download/auth failures fail the gate loudly
            res = {"model": mid, "passed": False, "error": f"{type(e).__name__}: {e}"}
            print(f"{mid:<42}LOAD FAILED: {res['error'][:60]}")
            results.append(res)
            all_ok = False
            continue
        flag = "PASS" if res["passed"] else "FAIL"
        print(
            f"{mid:<42}{res['loader']:<26}{str(res['roundtrip_ok']):<11}"
            f"{res['bad_offsets']}/{res['total_tokens']:<10}{flag}"
        )
        results.append(res)
        all_ok &= res["passed"]

    if args.output:
        from pathlib import Path

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"transformers_version": transformers.__version__, "results": results},
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out}")
    print("GATE:", "PASS — extraction may proceed" if all_ok else "FAIL — do NOT extract")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    r.add_argument("--strict-version", action="store_true",
                   help="fail (not warn) when transformers != pinned version")
    r.add_argument("--output", help="write JSON report")
    args = ap.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
