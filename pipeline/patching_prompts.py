"""Frozen Python class_struct patching prompts (v1).

Generates matched class/def pairs that differ at exactly one keyword token.
Character spans are recorded for both sides; token indices are derived later
with add_special_tokens=False.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = "class_struct_activation_patching_v1"
TRUE_COMPLETION = " True"
FALSE_COMPLETION = " False"

# These are hashes of the exact canonical UTF-8 JSONL payloads, including the
# final newline.  Changing the prompt protocol requires an explicit version and
# hash update; validation must never derive its expectation from the file under
# test.
EVAL_SHA256 = "6077e73c158616cf5f9175e4cf49daa2e4b2016ebc8afd12edc708572e79bd7b"
SMOKE_SHA256 = "522f9b2be880a861af22b6d7948f8837a67fb00897e73e02c7de094676a27425"
EXPECTED_EVAL_SHA256 = EVAL_SHA256
EXPECTED_SMOKE_SHA256 = SMOKE_SHA256

EVAL_NAMES = [
    "Node", "Item", "Point", "Entry", "Record", "Token",
    "Buffer", "Parser", "Scanner", "Packet", "Frame", "Block",
    "Tree", "Graph", "State", "Value", "Stack", "Table",
    "Matrix", "Vector", "Widget", "Element", "Vertex", "Edge",
]
SMOKE_NAMES = [
    "Cell", "Column", "Key", "Index", "Result", "Context", "Config", "Model",
]
SMOKE_CLUSTERS = (0, 7, 13, 18, 25, 31, 38, 47)

PREFIXES = (
    "",
    "LIMIT = 4\nmode = \"safe\"\n\n",
    "def helper(value):\n    return value + 1\n\n",
    "class Helper():\n    marker = 1\n\n",
)
BODIES = (
    "    pass\n",
    "    marker = 7\n    pass\n",
    "    label = \"ready\"\n    marker = len(label)\n    pass\n",
)
GAPS = (
    "",
    "sentinel = 3\n\n",
    "left = 2\nright = left + 5\n\n",
    "values = [1, 2, 3]\ntotal = sum(values)\nstatus = total > 0\n\n",
)

N_PREFIX = len(PREFIXES)
N_BODY = len(BODIES)
N_GAP = len(GAPS)
N_CLUSTERS = N_PREFIX * N_BODY * N_GAP  # 48
N_LEXICAL = 6
N_EVAL = N_CLUSTERS * N_LEXICAL  # 288

ROW_FIELDS = frozenset({
    "schema_version",
    "pair_id",
    "cluster_id",
    "prefix_id",
    "body_id",
    "gap_id",
    "lexical_variant",
    "name",
    "clean_prompt",
    "corrupt_prompt",
    "clean_expected",
    "corrupt_expected",
    "keyword_char_span",
    "declaration_name_char_span",
    "query_name_char_span",
    "placebo_char_span",
})


def cluster_parts(cluster_id: int) -> tuple[int, int, int]:
    if not 0 <= cluster_id < N_CLUSTERS:
        raise ValueError(f"cluster_id {cluster_id} out of range")
    prefix_id = cluster_id // (N_BODY * N_GAP)
    rest = cluster_id % (N_BODY * N_GAP)
    body_id = rest // N_GAP
    gap_id = rest % N_GAP
    return prefix_id, body_id, gap_id


def eval_name(cluster_id: int, lexical_variant: int) -> str:
    if not 0 <= lexical_variant < N_LEXICAL:
        raise ValueError(f"lexical_variant {lexical_variant} out of range")
    prefix_id, body_id, gap_id = cluster_parts(cluster_id)
    group = (prefix_id + body_id + gap_id) % 4
    return EVAL_NAMES[6 * group + lexical_variant]


def _span(haystack: str, needle: str, start: int) -> list[int]:
    i = haystack.index(needle, start)
    return [i, i + len(needle)]


def _build_one(prefix: str, body: str, gap: str, name: str, keyword: str) -> tuple[str, dict]:
    kw_start = len(prefix)
    head = f"{prefix}{keyword} {name}():\n"
    decl = _span(head, name, kw_start + len(keyword))
    text = head + body + "\n" + gap + f"assert isinstance({name}, type) is"
    if not body.endswith("    pass\n"):
        raise ValueError("body must end with a unique four-space pass line")
    pass_rel = body.rfind("    pass\n")
    placebo = [len(head) + pass_rel + 4, len(head) + pass_rel + 8]
    if text[placebo[0]:placebo[1]] != "pass":
        raise RuntimeError("placebo span is not 'pass'")
    query_start = text.rfind(f"isinstance({name}, type)")
    query = _span(text, name, query_start)
    kw = [kw_start, kw_start + len(keyword)]
    if text[kw[0]:kw[1]] != keyword:
        raise RuntimeError("keyword span mismatch")
    if text[decl[0]:decl[1]] != name or text[query[0]:query[1]] != name:
        raise RuntimeError("name span mismatch")
    if query == decl:
        raise RuntimeError("declaration and query spans collided")
    return text, {
        "keyword_char_span": kw,
        "declaration_name_char_span": decl,
        "query_name_char_span": query,
        "placebo_char_span": placebo,
    }


def build_pair_prompts(cluster_id: int, name: str) -> dict:
    prefix_id, body_id, gap_id = cluster_parts(cluster_id)
    prefix, body, gap = PREFIXES[prefix_id], BODIES[body_id], GAPS[gap_id]
    clean, clean_spans = _build_one(prefix, body, gap, name, "class")
    corrupt, corrupt_spans = _build_one(prefix, body, gap, name, "def")
    return {
        "clean_prompt": clean,
        "corrupt_prompt": corrupt,
        "keyword_char_span": {"clean": clean_spans["keyword_char_span"],
                              "corrupt": corrupt_spans["keyword_char_span"]},
        "declaration_name_char_span": {
            "clean": clean_spans["declaration_name_char_span"],
            "corrupt": corrupt_spans["declaration_name_char_span"],
        },
        "query_name_char_span": {
            "clean": clean_spans["query_name_char_span"],
            "corrupt": corrupt_spans["query_name_char_span"],
        },
        "placebo_char_span": {
            "clean": clean_spans["placebo_char_span"],
            "corrupt": corrupt_spans["placebo_char_span"],
        },
        "prefix_id": prefix_id,
        "body_id": body_id,
        "gap_id": gap_id,
    }


def _row(pair_id: str, cluster_id: int, lexical_variant: int, name: str) -> dict:
    built = build_pair_prompts(cluster_id, name)
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "cluster_id": cluster_id,
        "prefix_id": built["prefix_id"],
        "body_id": built["body_id"],
        "gap_id": built["gap_id"],
        "lexical_variant": lexical_variant,
        "name": name,
        "clean_prompt": built["clean_prompt"],
        "corrupt_prompt": built["corrupt_prompt"],
        "clean_expected": True,
        "corrupt_expected": False,
        "keyword_char_span": built["keyword_char_span"],
        "declaration_name_char_span": built["declaration_name_char_span"],
        "query_name_char_span": built["query_name_char_span"],
        "placebo_char_span": built["placebo_char_span"],
    }


def generate_eval_pairs() -> list[dict]:
    rows = []
    for c in range(N_CLUSTERS):
        for j in range(N_LEXICAL):
            name = eval_name(c, j)
            rows.append(_row(f"eval-c{c:02d}-j{j}", c, j, name))
    return rows


def generate_smoke_pairs() -> list[dict]:
    rows = []
    for name, c in zip(SMOKE_NAMES, SMOKE_CLUSTERS):
        rows.append(_row(f"smoke-c{c:02d}-{name}", c, 0, name))
    return rows


def canonicalize_row(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_payload(rows: list[dict]) -> bytes:
    return "".join(canonicalize_row(row) + "\n" for row in rows).encode("utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_payload(rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def complete_program(prompt: str, completion: str) -> str:
    return prompt + completion


def exec_completed(source: str) -> None:
    tree = ast.parse(source)
    compiled = compile(tree, "<patching-prompt>", "exec")
    ns: dict = {}
    exec(compiled, ns, ns)


def assertion_outcome(prompt: str, expected_true: bool) -> tuple[bool, bool]:
    """Return (correct_completion_ok, opposite_raises_assertion)."""
    good = TRUE_COMPLETION if expected_true else FALSE_COMPLETION
    bad = FALSE_COMPLETION if expected_true else TRUE_COMPLETION
    try:
        exec_completed(complete_program(prompt, good))
        good_ok = True
    except Exception:
        good_ok = False
    opposite_assert = False
    try:
        exec_completed(complete_program(prompt, bad))
    except AssertionError:
        opposite_assert = True
    except Exception:
        opposite_assert = False
    return good_ok, opposite_assert


def validate_python_semantics(row: dict) -> None:
    ok, opp = assertion_outcome(row["clean_prompt"], True)
    if not ok or not opp:
        raise ValueError(f"{row['pair_id']} clean completion semantics failed")
    ok, opp = assertion_outcome(row["corrupt_prompt"], False)
    if not ok or not opp:
        raise ValueError(f"{row['pair_id']} corrupt completion semantics failed")


def validate_row(
    row: dict,
    dataset: Literal["eval", "smoke"],
    *,
    position: int | None = None,
) -> None:
    """Validate one row against the frozen protocol, including exact spans.

    ``dataset`` is required so an otherwise well-formed smoke row cannot be
    substituted into the evaluation set (or vice versa).  When ``position`` is
    provided, canonical row ordering is also enforced.
    """
    if not isinstance(row, dict):
        raise ValueError("prompt row must be a JSON object")
    if set(row) != ROW_FIELDS:
        missing = sorted(ROW_FIELDS - set(row))
        extra = sorted(set(row) - ROW_FIELDS)
        raise ValueError(f"prompt row schema mismatch: missing={missing}, extra={extra}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version {row['schema_version']!r}")
    for field in ("cluster_id", "prefix_id", "body_id", "gap_id", "lexical_variant"):
        if type(row[field]) is not int:  # bool is deliberately rejected
            raise ValueError(f"{field} must be an integer")
    for field in ("pair_id", "name", "clean_prompt", "corrupt_prompt"):
        if type(row[field]) is not str:
            raise ValueError(f"{field} must be a string")
    for field in (
        "keyword_char_span",
        "declaration_name_char_span",
        "query_name_char_span",
        "placebo_char_span",
    ):
        spans = row[field]
        if not isinstance(spans, dict) or set(spans) != {"clean", "corrupt"}:
            raise ValueError(f"{field} must contain exactly clean/corrupt spans")
        for side, span in spans.items():
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(type(value) is not int for value in span)
            ):
                raise ValueError(f"{field}.{side} must be a two-integer list")
            start, end = span
            prompt = row[f"{side}_prompt"]
            if not 0 <= start < end <= len(prompt):
                raise ValueError(f"{field}.{side} is out of bounds")
    if row["clean_expected"] is not True or row["corrupt_expected"] is not False:
        raise ValueError("expected labels must be the booleans True and False")

    cluster_id = row["cluster_id"]
    lexical_variant = row["lexical_variant"]
    prefix_id, body_id, gap_id = cluster_parts(cluster_id)
    if (row["prefix_id"], row["body_id"], row["gap_id"]) != (
        prefix_id,
        body_id,
        gap_id,
    ):
        raise ValueError(f"{row['pair_id']} factor IDs do not match cluster_id")

    if dataset == "eval":
        if not 0 <= lexical_variant < N_LEXICAL:
            raise ValueError(f"lexical_variant {lexical_variant} out of range")
        name = eval_name(cluster_id, lexical_variant)
        pair_id = f"eval-c{cluster_id:02d}-j{lexical_variant}"
        if position is not None:
            expected_position = cluster_id * N_LEXICAL + lexical_variant
            if position != expected_position:
                raise ValueError(
                    f"{row['pair_id']} is at row {position}, expected {expected_position}"
                )
    elif dataset == "smoke":
        if lexical_variant != 0:
            raise ValueError("smoke lexical_variant must be zero")
        try:
            smoke_index = SMOKE_CLUSTERS.index(cluster_id)
        except ValueError as exc:
            raise ValueError(f"unexpected smoke cluster {cluster_id}") from exc
        name = SMOKE_NAMES[smoke_index]
        pair_id = f"smoke-c{cluster_id:02d}-{name}"
        if position is not None and position != smoke_index:
            raise ValueError(
                f"{row['pair_id']} is at row {position}, expected {smoke_index}"
            )
    else:
        raise ValueError(f"unknown prompt dataset {dataset!r}")

    expected = _row(pair_id, cluster_id, lexical_variant, name)
    if row != expected:
        mismatches = sorted(key for key in ROW_FIELDS if row[key] != expected[key])
        raise ValueError(f"{row['pair_id']} differs from frozen protocol: {mismatches}")
    validate_python_semantics(row)


def validate_frozen_rows(rows: list[dict], dataset: Literal["eval", "smoke"]) -> None:
    expected_count = N_EVAL if dataset == "eval" else len(SMOKE_NAMES)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} {dataset} rows, got {len(rows)}")
    for position, row in enumerate(rows):
        validate_row(row, dataset, position=position)

    pair_ids = [row["pair_id"] for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"duplicate {dataset} pair_id")

    if dataset == "eval":
        names = Counter(row["name"] for row in rows)
        if names != Counter({name: 12 for name in EVAL_NAMES}):
            raise ValueError(f"unbalanced overall name distribution: {dict(names)}")
        for factor, size, expected_per_name in (
            ("prefix_id", N_PREFIX, 3),
            ("body_id", N_BODY, 4),
            ("gap_id", N_GAP, 3),
        ):
            for level in range(size):
                counts = Counter(
                    row["name"] for row in rows if row[factor] == level
                )
                expected = Counter({name: expected_per_name for name in EVAL_NAMES})
                if counts != expected:
                    raise ValueError(
                        f"name distribution is not balanced for {factor}={level}"
                    )
        for cluster_id in range(N_CLUSTERS):
            cluster_names = {
                row["name"] for row in rows if row["cluster_id"] == cluster_id
            }
            if len(cluster_names) != N_LEXICAL:
                raise ValueError(f"cluster {cluster_id} does not have six unique names")


def _load_canonical_file(
    path: Path,
    dataset: Literal["eval", "smoke"],
    expected_sha256: str,
) -> tuple[list[dict], str]:
    path = Path(path)
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{dataset} prompt SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{dataset} prompt file must use canonical LF lines")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{dataset} prompt file is not UTF-8") from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"{dataset} prompt file contains a blank line")
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError(f"{dataset} prompt file contains invalid JSON") from exc
    if any(canonicalize_row(row) != line for row, line in zip(rows, lines)):
        raise ValueError(f"{dataset} prompt file is not canonical JSONL")
    validate_frozen_rows(rows, dataset)
    generated = generate_eval_pairs() if dataset == "eval" else generate_smoke_pairs()
    if payload != canonical_payload(generated):
        raise ValueError(f"{dataset} prompt bytes differ from the frozen generator")
    return rows, actual_sha256


def default_eval_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "patching" / "class_struct_python_v1.jsonl"


def default_smoke_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "patching" / "class_struct_python_smoke_v1.jsonl"


def validate_frozen_files(
    eval_path: Path | None = None,
    smoke_path: Path | None = None,
) -> dict:
    """Strictly validate the two immutable canonical prompt fixtures."""
    eval_path = Path(eval_path) if eval_path else default_eval_path()
    smoke_path = Path(smoke_path) if smoke_path else default_smoke_path()
    eval_rows, eval_sha = _load_canonical_file(
        eval_path, "eval", EXPECTED_EVAL_SHA256
    )
    smoke_rows, smoke_sha = _load_canonical_file(
        smoke_path, "smoke", EXPECTED_SMOKE_SHA256
    )
    if {row["pair_id"] for row in eval_rows} & {
        row["pair_id"] for row in smoke_rows
    }:
        raise ValueError("evaluation and smoke pair IDs overlap")
    if {row["name"] for row in eval_rows} & {row["name"] for row in smoke_rows}:
        raise ValueError("evaluation and smoke names overlap")
    return {
        "eval_path": str(eval_path),
        "smoke_path": str(smoke_path),
        "prompt_sha256": eval_sha,
        "smoke_prompt_sha256": smoke_sha,
        "n_eval": len(eval_rows),
        "n_smoke": len(smoke_rows),
    }


def generate_frozen_files(eval_path: Path | None = None, smoke_path: Path | None = None) -> dict:
    eval_path = Path(eval_path) if eval_path else default_eval_path()
    smoke_path = Path(smoke_path) if smoke_path else default_smoke_path()
    eval_rows = generate_eval_pairs()
    smoke_rows = generate_smoke_pairs()
    validate_frozen_rows(eval_rows, "eval")
    validate_frozen_rows(smoke_rows, "smoke")
    eval_generated_sha = hashlib.sha256(canonical_payload(eval_rows)).hexdigest()
    smoke_generated_sha = hashlib.sha256(canonical_payload(smoke_rows)).hexdigest()
    if eval_generated_sha != EXPECTED_EVAL_SHA256:
        raise RuntimeError(
            "generated evaluation prompts drifted from the pinned SHA-256: "
            f"expected {EXPECTED_EVAL_SHA256}, got {eval_generated_sha}"
        )
    if smoke_generated_sha != EXPECTED_SMOKE_SHA256:
        raise RuntimeError(
            "generated smoke prompts drifted from the pinned SHA-256: "
            f"expected {EXPECTED_SMOKE_SHA256}, got {smoke_generated_sha}"
        )
    write_jsonl(eval_path, eval_rows)
    write_jsonl(smoke_path, smoke_rows)
    return validate_frozen_files(eval_path, smoke_path)


if __name__ == "__main__":
    info = generate_frozen_files()
    print(json.dumps(info, indent=2))
