"""XLCoST loading: token-list reconstruction and program-level access."""

import json
import os

XLCOST_ROOT = os.environ.get(
    "XLCOST_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "XLCoST_data"),
)
NL2CODE_PROG = os.path.join(XLCOST_ROOT, "retrieval", "nl2code_search", "program_level")

SPLITS = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}


def reconstruct_code(tokens):
    """Reconstruct source from XLCoST token list (NEW_LINE/INDENT/DEDENT markers)."""
    indent_level, lines, current_line = 0, [], []
    for tok in tokens:
        if tok == "NEW_LINE":
            lines.append("    " * indent_level + " ".join(current_line))
            current_line = []
        elif tok == "INDENT":
            indent_level += 1
        elif tok == "DEDENT":
            indent_level = max(0, indent_level - 1)
        else:
            current_line.append(tok)
    if current_line:
        lines.append("    " * indent_level + " ".join(current_line))
    return "\n".join(lines).strip()


def load_programs(language="Python", split="train", max_programs=None):
    """Yield (idx, code) for programs in one XLCoST language/split."""
    path = os.path.join(NL2CODE_PROG, language, SPLITS[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    count = 0
    with open(path) as f:
        for line in f:
            if max_programs is not None and count >= max_programs:
                break
            rec = json.loads(line.strip())
            code = reconstruct_code(rec["code_tokens"])
            if code:
                yield rec["idx"], code
                count += 1
