"""Strict, dependency-free tests for the frozen class_struct prompt protocol."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json

import pytest

from pipeline import patching_prompts as prompts


EXPECTED_FIELDS = {
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
}


def test_exact_protocol_constants():
    assert prompts.SCHEMA_VERSION == "class_struct_activation_patching_v1"
    assert prompts.PREFIXES == (
        "",
        'LIMIT = 4\nmode = "safe"\n\n',
        "def helper(value):\n    return value + 1\n\n",
        "class Helper():\n    marker = 1\n\n",
    )
    assert prompts.BODIES == (
        "    pass\n",
        "    marker = 7\n    pass\n",
        '    label = "ready"\n    marker = len(label)\n    pass\n',
    )
    assert prompts.GAPS == (
        "",
        "sentinel = 3\n\n",
        "left = 2\nright = left + 5\n\n",
        "values = [1, 2, 3]\ntotal = sum(values)\nstatus = total > 0\n\n",
    )
    assert prompts.TRUE_COMPLETION == " True"
    assert prompts.FALSE_COMPLETION == " False"


def test_generator_exactly_matches_frozen_files_and_hashes():
    eval_rows = prompts.generate_eval_pairs()
    smoke_rows = prompts.generate_smoke_pairs()
    eval_payload = prompts.canonical_payload(eval_rows)
    smoke_payload = prompts.canonical_payload(smoke_rows)

    assert eval_payload == prompts.default_eval_path().read_bytes()
    assert smoke_payload == prompts.default_smoke_path().read_bytes()
    assert hashlib.sha256(eval_payload).hexdigest() == prompts.EVAL_SHA256
    assert hashlib.sha256(smoke_payload).hexdigest() == prompts.SMOKE_SHA256
    assert prompts.EVAL_SHA256 == prompts.EXPECTED_EVAL_SHA256
    assert prompts.SMOKE_SHA256 == prompts.EXPECTED_SMOKE_SHA256

    info = prompts.validate_frozen_files()
    assert info["n_eval"] == 288
    assert info["n_smoke"] == 8
    assert info["prompt_sha256"] == prompts.EVAL_SHA256
    assert info["smoke_prompt_sha256"] == prompts.SMOKE_SHA256


def test_regenerator_writes_the_exact_frozen_bytes(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    smoke_path = tmp_path / "smoke.jsonl"
    info = prompts.generate_frozen_files(eval_path, smoke_path)
    assert eval_path.read_bytes() == prompts.default_eval_path().read_bytes()
    assert smoke_path.read_bytes() == prompts.default_smoke_path().read_bytes()
    assert info["prompt_sha256"] == prompts.EVAL_SHA256
    assert info["smoke_prompt_sha256"] == prompts.SMOKE_SHA256


def test_all_rows_have_exact_schema_ids_values_and_spans():
    datasets = (
        ("eval", prompts.generate_eval_pairs()),
        ("smoke", prompts.generate_smoke_pairs()),
    )
    seen_ids: set[str] = set()
    for dataset, rows in datasets:
        for position, row in enumerate(rows):
            assert set(row) == EXPECTED_FIELDS == prompts.ROW_FIELDS
            assert row["schema_version"] == prompts.SCHEMA_VERSION
            assert row["clean_expected"] is True
            assert row["corrupt_expected"] is False
            assert row["pair_id"] not in seen_ids
            seen_ids.add(row["pair_id"])
            prompts.validate_row(row, dataset, position=position)

            for side, keyword in (("clean", "class"), ("corrupt", "def")):
                text = row[f"{side}_prompt"]
                expected_values = {
                    "keyword_char_span": keyword,
                    "declaration_name_char_span": row["name"],
                    "query_name_char_span": row["name"],
                    "placebo_char_span": "pass",
                }
                for field, expected in expected_values.items():
                    start, end = row[field][side]
                    assert type(start) is int and type(end) is int
                    assert 0 <= start < end <= len(text)
                    assert text[start:end] == expected
                assert (
                    row["declaration_name_char_span"][side]
                    != row["query_name_char_span"][side]
                )


def test_row_validator_rejects_within_token_corrupted_spans():
    original = prompts.generate_eval_pairs()[0]
    for field in (
        "keyword_char_span",
        "declaration_name_char_span",
        "query_name_char_span",
        "placebo_char_span",
    ):
        bad = copy.deepcopy(original)
        start, end = bad[field]["clean"]
        # This remains in bounds and inside the same lexical token; checking
        # only bounds or token overlap would incorrectly accept it.
        bad[field]["clean"] = [start + 1, end]
        with pytest.raises(ValueError, match="differs from frozen protocol"):
            prompts.validate_row(bad, "eval")


def test_eval_name_assignment_is_balanced_for_every_factor_level():
    rows = prompts.generate_eval_pairs()
    assert len(rows) == prompts.N_EVAL == 288
    assert Counter(row["name"] for row in rows) == Counter(
        {name: 12 for name in prompts.EVAL_NAMES}
    )

    for factor, levels, expected_per_name in (
        ("prefix_id", prompts.N_PREFIX, 3),
        ("body_id", prompts.N_BODY, 4),
        ("gap_id", prompts.N_GAP, 3),
    ):
        for level in range(levels):
            counts = Counter(row["name"] for row in rows if row[factor] == level)
            assert counts == Counter(
                {name: expected_per_name for name in prompts.EVAL_NAMES}
            )

    for cluster_id in range(prompts.N_CLUSTERS):
        cluster = [row for row in rows if row["cluster_id"] == cluster_id]
        prefix_id, body_id, gap_id = prompts.cluster_parts(cluster_id)
        group = (prefix_id + body_id + gap_id) % 4
        assert [row["name"] for row in cluster] == prompts.EVAL_NAMES[
            6 * group : 6 * (group + 1)
        ]
        assert [row["lexical_variant"] for row in cluster] == list(range(6))


def test_every_prompt_has_the_frozen_python_semantics():
    for row in prompts.generate_eval_pairs() + prompts.generate_smoke_pairs():
        assert prompts.assertion_outcome(row["clean_prompt"], True) == (True, True)
        assert prompts.assertion_outcome(row["corrupt_prompt"], False) == (True, True)


def test_strict_file_validation_hard_fails_byte_and_schema_drift(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    smoke_path = tmp_path / "smoke.jsonl"
    eval_path.write_bytes(prompts.default_eval_path().read_bytes())
    smoke_path.write_bytes(prompts.default_smoke_path().read_bytes())

    payload = eval_path.read_bytes()
    eval_path.write_bytes(payload.replace(b'"Node"', b'"Mode"', 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prompts.validate_frozen_files(eval_path, smoke_path)

    eval_path.write_bytes(prompts.default_eval_path().read_bytes())
    first, *rest = eval_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(first)
    row["unexpected"] = 1
    mutated = prompts.canonicalize_row(row) + "\n" + "\n".join(rest) + "\n"
    eval_path.write_text(mutated, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prompts.validate_frozen_files(eval_path, smoke_path)


def test_smoke_rows_are_exact_and_disjoint():
    eval_rows = prompts.generate_eval_pairs()
    smoke_rows = prompts.generate_smoke_pairs()
    assert [row["cluster_id"] for row in smoke_rows] == list(prompts.SMOKE_CLUSTERS)
    assert [row["name"] for row in smoke_rows] == prompts.SMOKE_NAMES
    assert {row["pair_id"] for row in eval_rows}.isdisjoint(
        row["pair_id"] for row in smoke_rows
    )
    assert {row["name"] for row in eval_rows}.isdisjoint(
        row["name"] for row in smoke_rows
    )
