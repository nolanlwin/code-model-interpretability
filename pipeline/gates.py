"""Tokenizer offset gate and safe loading (ported from scripts/tokenizer_gate.py).

A roundtrip check is NOT sufficient — one candidate tokenizer round-tripped
correctly while corrupting half its offsets. Every extraction must load its
tokenizer through here: labels and spans are derived from offset mappings,
so corrupted offsets silently label and pool the wrong tokens (DeepSeek-Coder
under transformers 5.3-5.13 deletes whitespace outright via AutoTokenizer).
"""

from __future__ import annotations

FIXTURES = [
    "if is_valid:\n    total += count\n    return found",
    "// café résumé é\nint x1_y2 = arr[idx];\n",
    "def f(a, b):\n\tflag = True\n\treturn flag  # tab-indented\n",
    "line1\r\nline2\r\n",
    's = "string with  double  spaces"\nname_2 = s\n',
]


def offsets_ok(tok) -> tuple[bool, int, int]:
    bad = total = 0
    ok_roundtrip = True
    for src in FIXTURES:
        try:
            enc = tok(src, return_offsets_mapping=True, add_special_tokens=False)
        except Exception:
            return False, -1, -1  # no fast offsets at all
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if tok.decode(ids) != src:
            ok_roundtrip = False
        for i, (s, e) in enumerate(offs):
            total += 1
            if src[s:e] != tok.decode([ids[i]]):
                bad += 1
    return ok_roundtrip and bad == 0, bad, total


def load_tokenizer_gated(model_name: str, trust_remote_code: bool = False):
    """AutoTokenizer if it passes the offset gate; PreTrainedTokenizerFast
    fallback if that passes instead; hard failure otherwise."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    ok, bad, total = offsets_ok(tok)
    if ok:
        return tok
    fallback = PreTrainedTokenizerFast.from_pretrained(model_name)
    ok2, bad2, total2 = offsets_ok(fallback)
    if ok2:
        print(f"[gates] {model_name}: AutoTokenizer failed the offset gate "
              f"({bad}/{total} bad); using PreTrainedTokenizerFast fallback")
        return fallback
    raise RuntimeError(
        f"{model_name}: no tokenizer passes the offset gate (AutoTokenizer "
        f"{bad}/{total} bad, fast fallback {bad2}/{total2} bad). Extraction "
        "with corrupted offsets would label and pool the wrong tokens."
    )
