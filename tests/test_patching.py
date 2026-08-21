"""Unit tests for the class_struct patching pipeline (no GPU, no HF downloads)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import patching as P
from pipeline.patching_prompts import (
    EVAL_NAMES,
    N_BODY,
    N_CLUSTERS,
    N_EVAL,
    N_GAP,
    N_LEXICAL,
    SMOKE_CLUSTERS,
    SMOKE_NAMES,
    assertion_outcome,
    canonicalize_row,
    default_eval_path,
    default_smoke_path,
    eval_name,
    generate_eval_pairs,
    generate_frozen_files,
    generate_smoke_pairs,
    load_jsonl,
    sha256_file,
    validate_python_semantics,
)

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Fake tokenizer: whitespace/word pieces; " True"/" False" forced to one token.
# ---------------------------------------------------------------------------

class FakeTokenizer:
    def __init__(self, true_id=3007, false_id=3557):
        self.true_id = true_id
        self.false_id = false_id
        self._vocab = {"class": 1, "def": 2, "pass": 3, " True": true_id, " False": false_id}

    def encode(self, text, add_special_tokens=False):
        ids, _ = self._enc(text)
        return ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False,
                 padding=False, return_tensors=None, **kwargs):
        if isinstance(text, list):
            raise AssertionError("batch call not used in these tests")
        ids, offs = self._enc(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def _enc(self, text: str):
        ids, offs = [], []
        i = 0
        while i < len(text):
            if text.startswith(" True", i):
                ids.append(self.true_id)
                offs.append((i, i + 5))
                i += 5
                continue
            if text.startswith(" False", i):
                ids.append(self.false_id)
                offs.append((i, i + 6))
                i += 6
                continue
            if text[i].isspace():
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                ids.append(100 + (ord(text[i]) % 50))
                offs.append((i, j))
                i = j
                continue
            if text[i].isalnum() or text[i] == "_":
                j = i + 1
                while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                piece = text[i:j]
                ids.append(self._vocab.get(piece, 1000 + (hash(piece) % 10000)))
                offs.append((i, j))
                i = j
                continue
            ids.append(200 + ord(text[i]) % 80)
            offs.append((i, i + 1))
            i += 1
        return ids, offs


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_eval_counts_and_clusters():
    rows = generate_eval_pairs()
    assert len(rows) == N_EVAL == 288
    clusters = [r["cluster_id"] for r in rows]
    assert sorted(set(clusters)) == list(range(48))
    for c in range(N_CLUSTERS):
        names = [r["name"] for r in rows if r["cluster_id"] == c]
        assert len(names) == 6
        assert len(set(names)) == 6
    uses = {n: 0 for n in EVAL_NAMES}
    for r in rows:
        uses[r["name"]] += 1
    assert set(uses.values()) == {12}


def test_name_index_formula():
    for c in range(N_CLUSTERS):
        for j in range(N_LEXICAL):
            prefix = c // (N_BODY * N_GAP)
            rest = c % (N_BODY * N_GAP)
            body = rest // N_GAP
            gap = rest % N_GAP
            group = (prefix + body + gap) % 4
            assert eval_name(c, j) == EVAL_NAMES[6 * group + j]


def test_smoke_disjoint_and_clusters():
    eval_ids = {r["pair_id"] for r in generate_eval_pairs()}
    smoke = generate_smoke_pairs()
    assert len(smoke) == 8
    assert [r["cluster_id"] for r in smoke] == list(SMOKE_CLUSTERS)
    assert [r["name"] for r in smoke] == list(SMOKE_NAMES)
    assert not any(r["pair_id"] in eval_ids for r in smoke)


def test_python_semantics_all_pairs():
    for row in generate_eval_pairs() + generate_smoke_pairs():
        validate_python_semantics(row)
        ok, opp = assertion_outcome(row["clean_prompt"], True)
        assert ok and opp
        ok, opp = assertion_outcome(row["corrupt_prompt"], False)
        assert ok and opp


def test_character_spans():
    row = generate_eval_pairs()[0]
    for side, kw in (("clean", "class"), ("corrupt", "def")):
        prompt = row["clean_prompt"] if side == "clean" else row["corrupt_prompt"]
        s, e = row["keyword_char_span"][side]
        assert prompt[s:e] == kw
        s, e = row["declaration_name_char_span"][side]
        assert prompt[s:e] == row["name"]
        s, e = row["query_name_char_span"][side]
        assert prompt[s:e] == row["name"]
        s, e = row["placebo_char_span"][side]
        assert prompt[s:e] == "pass"
        assert row["declaration_name_char_span"][side] != row["query_name_char_span"][side]


def test_frozen_files_stable_hash():
    generate_frozen_files()
    a = sha256_file(default_eval_path())
    generate_frozen_files()
    b = sha256_file(default_eval_path())
    assert a == b
    assert a == P.EVAL_PROMPT_SHA256
    assert sha256_file(default_smoke_path()) == P.SMOKE_PROMPT_SHA256
    rows = load_jsonl(default_eval_path())
    assert canonicalize_row(rows[0]) == json.dumps(rows[0], sort_keys=True, separators=(",", ":"))


def test_fake_tokenizer_alignment():
    tok = FakeTokenizer()
    for row in generate_eval_pairs()[:12] + generate_smoke_pairs():
        info = P.validate_pair_tokenizer(tok, row)
        assert info["n_tokens"] > 4
        assert info["declaration_index"] != info["query_index"]


def test_shifted_span_fails():
    tok = FakeTokenizer()
    row = generate_eval_pairs()[0]
    bad = dict(row)
    bad["query_name_char_span"] = {
        "clean": [row["query_name_char_span"]["clean"][0] + 1,
                  row["query_name_char_span"]["clean"][1] + 1],
        "corrupt": [row["query_name_char_span"]["corrupt"][0] + 1,
                    row["query_name_char_span"]["corrupt"][1] + 1],
    }
    with pytest.raises(P.PatchingError):
        P.validate_pair_tokenizer(tok, bad)


def test_left_padding_index():
    assert P.left_pad_index(3, 5) == 8
    with pytest.raises(P.PatchingError):
        P.left_pad_index(-1, 0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_logit_diff_sign():
    logits = np.zeros(4000)
    logits[3007] = 2.0
    logits[3557] = -1.0
    assert P.logit_diff(logits, 3007, 3557) == 3.0
    logits[3007], logits[3557] = -2.0, 1.0
    assert P.logit_diff(logits, 3007, 3557) == -3.0


def test_direction_signs():
    assert P.denoise_effect(-1.0, 0.5) == 1.5
    assert P.noise_effect(2.0, 0.5) == 1.5
    assert P.signed_effect("denoise", 2.0, -1.0, 0.5) == 1.5
    assert P.signed_effect("noise", 2.0, -1.0, 0.5) == 1.5


def test_ratio_of_means_not_mean_of_ratios():
    effects = [1.0, 1.0, 1.0]
    gaps = [1.0, 2.0, 4.0]
    rom = P.ratio_of_means(effects, gaps)
    mor = P.mean_of_ratios(effects, gaps)
    assert rom == pytest.approx(1.0 / (7.0 / 3.0))
    assert mor != pytest.approx(rom)


def test_clustered_bootstrap_resamples_whole_clusters():
    # 48 clusters × 6 identical values: CI should be tight around the mean
    values = np.repeat(np.arange(48, dtype=float), 6)
    clusters = np.repeat(np.arange(48), 6)
    ci = P.clustered_mean_ci(values, clusters, n_boot=500, seed=20260818)
    assert ci["n_clusters"] == 48
    assert ci["point"] == pytest.approx(values.mean())
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


# ---------------------------------------------------------------------------
# Random control
# ---------------------------------------------------------------------------

def test_random_vector_orthogonal_norm_deterministic():
    rng = np.random.default_rng(0)
    src = rng.standard_normal(16).astype(np.float32)
    dst = rng.standard_normal(16).astype(np.float32)
    key = P.cell_rng_key("p", 3, "query_name", "denoise", "random")
    n1 = P.random_control_noise(src, dst, key)
    n2 = P.random_control_noise(src, dst, key)
    assert np.allclose(n1, n2)
    delta = src - dst
    assert abs(float(np.dot(n1.reshape(-1), delta.reshape(-1)))) < 1e-4
    assert np.linalg.norm(n1) == pytest.approx(np.linalg.norm(delta), rel=1e-5)
    n3 = P.random_control_noise(src, dst, key + "x")
    assert not np.allclose(n1, n3)


def test_zero_delta_random_is_zero():
    v = np.ones(8, dtype=np.float32)
    noise = P.random_control_noise(v, v, "k")
    assert np.allclose(noise, 0)


# ---------------------------------------------------------------------------
# Probe conversion
# ---------------------------------------------------------------------------

def test_raw_margin_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    y = (X[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit(scaler.transform(X), y)
    w_raw, b_raw = P.raw_residual_params(scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_)
    for row in X:
        sk = float(clf.decision_function(scaler.transform(row.reshape(1, -1)))[0])
        ours = P.probe_margin(row, w_raw, b_raw)
        assert ours == pytest.approx(sk, rel=1e-5, abs=1e-5)


# ---------------------------------------------------------------------------
# Hooks / tiny models
# ---------------------------------------------------------------------------

def test_hook_replacement_and_cleanup():
    torch = pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    adapter = P.ArchitectureAdapter(model)
    assert adapter.n_blocks == 2
    assert adapter.n_hidden == 3
    ids = torch.randint(0, 128, (2, 6))
    mask = torch.ones_like(ids)
    diffs = P.compare_hook_to_hidden_states(model, adapter, ids, mask, atol=1e-4)
    assert set(diffs) == {0, 1, 2}
    kept_logits, kept_hs = P.capture_hidden_states(model, ids, mask, keep_layers=[0])
    assert kept_logits.shape[0] == 2
    assert kept_hs[0] is not None
    assert kept_hs[1] is None
    assert kept_hs[2] is None

    logits_a, hs_a = P.capture_hidden_states(model, ids, mask)
    src = hs_a[1][0, 2].detach().clone()
    # same-source patch is a no-op
    logits_b = P.patched_forward(
        model, adapter, ids, mask, 1, [(0, 2, src)],
    )
    assert torch.allclose(logits_a[0], logits_b[0], atol=1e-5)

    changed = src + 3.0
    logits_c = P.patched_forward(model, adapter, ids, mask, 1, [(0, 2, changed)])
    assert not torch.allclose(logits_a[0], logits_c[0])

    # hooks removed: a later unpatched forward matches original
    logits_d, _ = P.capture_hidden_states(model, ids, mask)
    assert torch.allclose(logits_a, logits_d, atol=1e-5)

    # exception path still removes hooks
    class Boom(Exception):
        pass

    handles = adapter.install(0, [(0, 0, src)])
    try:
        try:
            raise Boom()
        finally:
            P.remove_hooks(handles)
        raise AssertionError("unreachable")
    except Boom:
        logits_e, _ = P.capture_hidden_states(model, ids, mask)
        assert torch.allclose(logits_a, logits_e, atol=1e-5)


def test_starcoder2_tiny_adapter():
    torch = pytest.importorskip("torch")
    try:
        from transformers import Starcoder2Config, Starcoder2ForCausalLM
    except ImportError:
        pytest.skip("Starcoder2 not in this transformers build")
    cfg = Starcoder2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=64,
    )
    model = Starcoder2ForCausalLM(cfg).eval()
    adapter = P.ArchitectureAdapter(model)
    ids = torch.randint(0, 128, (1, 5))
    mask = torch.ones_like(ids)
    P.compare_hook_to_hidden_states(model, adapter, ids, mask, atol=1e-3)


# ---------------------------------------------------------------------------
# Checkpoints / lease / estimate
# ---------------------------------------------------------------------------

def test_chunk_checksum_and_duplicates(tmp_path):
    store = P.ChunkStore(tmp_path)
    row = P.make_result_row(
        prompt_sha256="a", configuration_sha256="b", model_id="m",
        model_revision="r", dtype="float16", pair_id="p0", layer=1,
        span="query_name", direction="denoise", control="target",
        signed_effect=0.1, source_D=1.0, destination_D=0.0, patched_D=0.2,
        cluster_id=0, name="Node", run_id="run",
    )
    store.write_block("core", "m", "float16", "b", 1, 0, [row])
    loaded = store.load_valid_rows()
    assert len(loaded) == 1
    with pytest.raises(P.PatchingError):
        P.finalize_chunk(tmp_path / "chunks" / "dup.jsonl", [row, row])


def test_manifest_mismatch_and_resume(tmp_path):
    store = P.ChunkStore(tmp_path)
    row = P.make_result_row(
        prompt_sha256="aaa", configuration_sha256="bbb", model_id="m",
        model_revision="r", dtype="float16", pair_id="p0", layer=0,
        span="query_name", direction="denoise", control="target",
        signed_effect=0.0, source_D=0.0, destination_D=0.0, patched_D=0.0,
        cluster_id=0, name="Node", run_id="run",
    )
    rec = store.write_block("core", "m", "float16", "bbb", 0, 0, [row])
    # corrupt checksum
    with pytest.raises(P.PatchingError):
        P.read_chunk(tmp_path / rec["path"], expected_sha="0" * 64)
    # resume skips valid
    present = {P.primary_key(r) for r in store.load_valid_rows()}
    assert P.primary_key(row) in present


def test_lease_heartbeat_release_and_duplicate_holders(tmp_path):
    path = tmp_path / "lease.json"
    P.acquire_lease(path, "a", now=1000.0)
    P.heartbeat_lease(path, "a", now=1001.0, stale_after_s=600)
    P.acquire_lease(path, "a", now=1002.0, allow_same=True)
    with pytest.raises(P.PatchingError, match="lease held"):
        P.acquire_lease(path, "b", now=1003.0)
    with pytest.raises(P.PatchingError, match="lease changed"):
        P.heartbeat_lease(path, "b", now=1004.0)
    P.release_lease(path, "a", now=1005.0)
    with pytest.raises(P.PatchingError, match="released"):
        P.heartbeat_lease(path, "a", now=1006.0)
    P.acquire_lease(path, "b", now=1006.0)
    assert P.load_lease(path).holder == "b"
    assert P.load_lease(path).released is False


def test_lease_and_stale_heartbeat(tmp_path, monkeypatch):
    path = tmp_path / "lease.json"
    P.acquire_lease(path, "a", now=1000.0)
    with pytest.raises(P.PatchingError):
        P.acquire_lease(path, "b", now=1001.0, allow_same=False)
    P.acquire_lease(path, "b", now=1000.0 + 601, allow_same=False)


def test_estimate_under_ceilings():
    est = P.estimate_all()
    expected = {
        "Qwen/Qwen2.5-1.5B": {
            "n_intervention_cells": 294,
            "item_forwards": 85_824,
            "staged_item_forwards": 87_552,
            "staged_batched_forwards": 2_736,
            "staged_new_cells": [10, 54, 230],
            "fp32_top3_item_forwards": 6_336,
            "ceiling": 90_000,
        },
        "Qwen/Qwen2.5-Coder-1.5B": {
            "n_intervention_cells": 70,
            "item_forwards": 21_312,
            "staged_item_forwards": 23_040,
            "staged_batched_forwards": 720,
            "staged_new_cells": [10, 54, 6],
            "fp32_top3_item_forwards": 6_336,
            "ceiling": 26_000,
        },
        "bigcode/starcoder2-7b": {
            "n_intervention_cells": 78,
            "item_forwards": 23_616,
            "staged_item_forwards": 25_344,
            "staged_batched_forwards": 3_168,
            "staged_new_cells": [10, 62, 6],
            "fp32_top3_item_forwards": 6_336,
            "ceiling": 26_000,
        },
    }
    for model_id, want in expected.items():
        got = est[model_id]
        for key, value in want.items():
            assert got[key] == value, (model_id, key, got[key], value)
        assert got["staged_item_forwards"] <= got["ceiling"]
        assert got["item_forwards"] <= got["ceiling"]


def test_forward_count_refusal(monkeypatch):
    monkeypatch.setitem(P.MODELS["Qwen/Qwen2.5-1.5B"], "item_forward_ceiling", 10)
    with pytest.raises(P.PatchingError):
        P.estimate_all()


def test_smoke_gate_and_causal_gate_orientation():
    rows = []
    for i in range(8):
        pid = f"s{i}"
        d_c, d_f = 1.0, -1.0
        rows.append(P.make_result_row(
            prompt_sha256="p", configuration_sha256="c", model_id="m",
            model_revision="r", dtype="float16", pair_id=pid, layer=-1,
            span="baseline_class", direction="none", control="unpatched",
            source_D=d_c, destination_D=d_c, patched_D=d_c, signed_effect=0.0,
            baseline_drift=0.0, cluster_id=i, name="Node", run_id="r",
        ))
        rows.append(P.make_result_row(
            prompt_sha256="p", configuration_sha256="c", model_id="m",
            model_revision="r", dtype="float16", pair_id=pid, layer=-1,
            span="baseline_function", direction="none", control="unpatched",
            source_D=d_f, destination_D=d_f, patched_D=d_f, signed_effect=0.0,
            baseline_drift=0.0, cluster_id=i, name="Node", run_id="r",
        ))
        for direction, effect, span, control, layer in (
            ("denoise", 0.8, "query_name", "target", 18),
            ("noise", 0.7, "query_name", "target", 18),
            ("denoise", 0.1, "placebo", "target", 18),
            ("noise", 0.1, "placebo", "target", 18),
            ("denoise", 0.05, "query_name", "random", 18),
            ("noise", 0.05, "query_name", "random", 18),
            ("denoise", 0.0, "query_name", "same_source", 18),
            ("noise", 0.0, "query_name", "same_source", 18),
            ("denoise", 0.0, "query_name", "target", 0),
            ("noise", 0.0, "query_name", "target", 0),
        ):
            rows.append(P.make_result_row(
                prompt_sha256="p", configuration_sha256="c", model_id="m",
                model_revision="r", dtype="float16", pair_id=pid, layer=layer,
                span=span, direction=direction, control=control,
                source_D=d_c if direction == "denoise" else d_f,
                destination_D=d_f if direction == "denoise" else d_c,
                patched_D=d_f + effect if direction == "denoise" else d_c - effect,
                signed_effect=effect, baseline_drift=0.0,
                cluster_id=i, name="Node", run_id="r",
            ))
    gate = P.smoke_gate(rows)
    assert gate["pass"], gate["checks"]
    assert gate["n_target_denoise"] == 8
    assert gate["n_target_noise"] == 8
    # Layer-0 identity rows must not be mixed into the probe-layer target mean.
    polluted = []
    for row in rows:
        clone = dict(row)
        if clone["layer"] == 0:
            clone["signed_effect"] = 99.0
        polluted.append(clone)
    polluted_gate = P.smoke_gate(polluted, probe_index=18)
    assert not polluted_gate["pass"]
    assert not polluted_gate["checks"]["layer0_within_tau"]
    assert polluted_gate["checks"]["mean_denoise_positive"]
    assert polluted_gate["checks"]["mean_noise_positive"]
    assert polluted_gate["checks"]["target_gt_placebo"]
    assert polluted_gate["checks"]["target_gt_random"]
    missing = [
        row for row in rows
        if not (
            row["pair_id"] == "s7" and row["layer"] == 18
            and row["span"] == "query_name" and row["direction"] == "noise"
            and row["control"] == "random"
        )
    ]
    incomplete = P.smoke_gate(missing, expected_pair_ids=[f"s{i}" for i in range(8)])
    assert not incomplete["pass"]
    assert not incomplete["checks"]["exact_row_cube"]

    den = [0.8] * 48
    noi = [0.7] * 48
    clusters = list(range(48))
    causal = P.causal_gate(den, noi, [0.1] * 48, [0.1] * 48,
                           [0.05] * 48, [0.05] * 48, [2.0] * 48, clusters, tau=1e-4)
    assert causal["pass"], causal["checks"]


def test_identity_tau_does_not_raise_causal_bar():
    assert P.drift_tau(0.0) == pytest.approx(P.DRIFT_TAU_FLOOR)
    assert P.identity_tau(0.0, "float16") == pytest.approx(P.FP16_IDENTITY_TAU)
    assert P.identity_tau(0.0, "float32") == pytest.approx(P.DRIFT_TAU_FLOOR)
    assert P.identity_tau(0.02, "float16") == pytest.approx(0.2)
    causal = P.causal_gate(
        [0.11] * 48, [0.11] * 48, [0.0] * 48, [0.0] * 48,
        [0.0] * 48, [0.0] * 48, [1.0] * 48, list(range(48)),
        tau=P.drift_tau(0.0),
    )
    assert causal["threshold"] == pytest.approx(0.10)
    assert causal["pass"], causal["checks"]


def _smoke_rows(*, d_class, d_fn, denoise, noise, same=0.0, layer0=0.0,
                placebo=0.01, random=0.005, drift=0.0):
    rows = []
    for i in range(8):
        pid = f"s{i}"
        dc, df = d_class[i], d_fn[i]
        rows.append(P.make_result_row(
            prompt_sha256="p", configuration_sha256="c", model_id="m",
            model_revision="r", dtype="float16", pair_id=pid, layer=-1,
            span="baseline_class", direction="none", control="unpatched",
            source_D=dc, destination_D=dc, patched_D=dc, signed_effect=0.0,
            baseline_drift=drift, cluster_id=i, name="Node", run_id="r",
        ))
        rows.append(P.make_result_row(
            prompt_sha256="p", configuration_sha256="c", model_id="m",
            model_revision="r", dtype="float16", pair_id=pid, layer=-1,
            span="baseline_function", direction="none", control="unpatched",
            source_D=df, destination_D=df, patched_D=df, signed_effect=0.0,
            baseline_drift=drift, cluster_id=i, name="Node", run_id="r",
        ))
        effects = {
            ("denoise", "query_name", "target", 18): denoise[i],
            ("noise", "query_name", "target", 18): noise[i],
            ("denoise", "placebo", "target", 18): placebo,
            ("noise", "placebo", "target", 18): placebo,
            ("denoise", "query_name", "random", 18): random,
            ("noise", "query_name", "random", 18): random,
            ("denoise", "query_name", "same_source", 18): same,
            ("noise", "query_name", "same_source", 18): same,
            ("denoise", "query_name", "target", 0): layer0,
            ("noise", "query_name", "target", 0): layer0,
        }
        for (direction, span, control, layer), effect in effects.items():
            rows.append(P.make_result_row(
                prompt_sha256="p", configuration_sha256="c", model_id="m",
                model_revision="r", dtype="float16", pair_id=pid, layer=layer,
                span=span, direction=direction, control=control,
                source_D=dc if direction == "denoise" else df,
                destination_D=df if direction == "denoise" else dc,
                patched_D=df + effect if direction == "denoise" else dc - effect,
                signed_effect=effect, baseline_drift=drift,
                cluster_id=i, name="Node", run_id="r",
            ))
    return rows


def test_smoke_gate_accepts_true_biased_gap_and_fp16_identity():
    # Observed Qwen 1.5B smoke shape: function D stays positive, class D is
    # larger, same-source wiggles by one fp16 logit ULP, denoise signs are 4/8.
    d_class = [3.95, 2.61, 2.41, 1.59, 3.59, 2.45, 3.05, 3.45]
    d_fn = [2.70, 1.67, 1.33, -0.28, 1.50, 2.09, 1.56, 2.55]
    denoise = [0.094, 0.031, -0.031, 0.0, 0.094, 0.047, -0.031, -0.031]
    noise = [0.078, 0.047, -0.016, 0.016, 0.094, 0.063, -0.031, -0.031]
    rows = _smoke_rows(
        d_class=d_class, d_fn=d_fn, denoise=denoise, noise=noise,
        same=0.03125, layer0=0.0, placebo=0.002, random=0.006,
    )
    gate = P.smoke_gate(rows)
    assert gate["pass"], gate
    assert gate["tau"] == pytest.approx(P.FP16_IDENTITY_TAU)
    assert gate["checks"]["function_below_class_6_of_8"]
    assert not gate["diagnostics"]["function_D_negative_6_of_8"]
    assert not gate["diagnostics"]["denoise_sign_5_of_8"]


def test_smoke_gate_rejects_inverted_class_function_gap():
    rows = _smoke_rows(
        d_class=[1.0] * 8, d_fn=[2.0] * 8,
        denoise=[0.2] * 8, noise=[0.2] * 8,
    )
    gate = P.smoke_gate(rows)
    assert not gate["pass"]
    assert not gate["checks"]["function_below_class_6_of_8"]
    assert not gate["checks"]["mean_gap_positive"]


def test_behavior_gate_uses_pair_gap_not_function_sign():
    d_class = [2.5] * 48
    d_function = [1.2] * 48
    clusters = list(range(48))
    gate = P.behavior_gate(d_class, d_function, clusters)
    assert gate["pass"], gate["checks"]
    assert gate["checks"]["pair_gap_acc_ge_0.60"]
    assert not gate["diagnostics"]["function_acc_ge_0.60"]


def test_adaptive_batch_halving_retries_same_work():
    from pipeline.run_patching import _halve_until_ok

    torch = pytest.importorskip("torch")
    seen = []

    def fn(size):
        seen.append(size)
        if size > 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return {"batch_size": size}

    assert _halve_until_ok(fn, 32) == {"batch_size": 4}
    assert seen == [32, 16, 8, 4]


def test_projected_spend():
    assert P.projected_spend_usd({"spent_usd": 10}, 45) == 55
    assert P.projected_spend_usd({"spent_usd": 10}, 5) == 15


def test_source_cache_not_mutated():
    src = np.arange(8, dtype=np.float32)
    dst = np.ones(8, dtype=np.float32)
    copy = src.copy()
    P.inject_random(src, dst, "k")
    assert np.array_equal(src, copy)


def _result_row(*, pair_id="p0", model_id="m", config="cfg", dtype="float16",
                layer=1, span="query_name", direction="denoise", control="target",
                effect=0.1, cluster_id=0, name="Node"):
    return P.make_result_row(
        prompt_sha256="prompt", configuration_sha256=config, model_id=model_id,
        model_revision="rev", dtype=dtype, pair_id=pair_id, layer=layer,
        span=span, direction=direction, control=control,
        signed_effect=effect, source_D=1.0, destination_D=-1.0,
        patched_D=-1.0 + effect, class_function_gap=2.0,
        cluster_id=cluster_id, name=name, run_id="run", baseline_drift=0.0,
    )


def test_schedule_phases_are_non_overlapping_and_complete():
    for model_id in P.MODELS:
        primary = set(P.primary_cells(model_id))
        core_delta = set(P.core_cells(model_id)) - primary
        expanded = set(P.expanded_cells(model_id))
        assert not (primary & expanded)
        assert not (core_delta & expanded)
        assert primary | core_delta | expanded == set(P.intervention_cells(model_id))
        estimate = P.estimate_forwards(model_id)
        assert estimate["staged_item_forwards"] <= estimate["ceiling"]
        assert estimate["fp32_top3_item_forwards"] == 6336


def test_namespaced_chunks_do_not_collide_and_are_portable(tmp_path):
    store = P.ChunkStore(tmp_path / "remote-run")
    a = _result_row(model_id="org/model-a", config="cfg-a")
    b = _result_row(model_id="org/model-b", config="cfg-a")
    c = _result_row(model_id="org/model-a", config="cfg-b", dtype="float32")
    rec_a = store.write_block("core", "org/model-a", "float16", "cfg-a", 1, 0, [a])
    rec_b = store.write_block("core", "org/model-b", "float16", "cfg-a", 1, 0, [b])
    rec_c = store.write_block("fp32", "org/model-a", "float32", "cfg-b", 1, 0, [c])
    paths = {rec_a["path"], rec_b["path"], rec_c["path"]}
    assert len(paths) == 3
    assert all(not Path(path).is_absolute() for path in paths)

    pulled = tmp_path / "pulled-elsewhere"
    shutil.copytree(tmp_path / "remote-run", pulled)
    rows = P.ChunkStore(pulled).load_valid_rows(strict=True)
    assert {row["model_id"] for row in rows} == {"org/model-a", "org/model-b"}
    assert {row["dtype"] for row in rows} == {"float16", "float32"}


def test_chunk_merge_resume_and_completeness_duplicates(tmp_path):
    store = P.ChunkStore(tmp_path)
    a = _result_row(pair_id="a")
    b = _result_row(pair_id="b")
    store.write_block("core", "m", "float16", "cfg", 1, 0, [a])
    store.write_block("core", "m", "float16", "cfg", 1, 0, [b])
    rows = store.load_valid_rows(strict=True)
    assert {row["pair_id"] for row in rows} == {"a", "b"}
    expected = [P.primary_key(a), P.primary_key(b)]
    assert P.completeness_report(expected, rows)["complete"]
    duplicate = P.completeness_report(expected, rows + [a])
    assert duplicate["n_duplicates"] == 1
    assert not duplicate["complete"]


def test_corrupt_chunk_is_recomputed_on_resume(tmp_path):
    store = P.ChunkStore(tmp_path)
    row = _result_row(pair_id="a")
    rec = store.write_block("core", "m", "float16", "cfg", 1, 0, [row])
    chunk = tmp_path / rec["path"]
    chunk.write_text("corrupt\n")
    assert store.load_valid_rows(strict=False) == []
    repaired = store.write_block("core", "m", "float16", "cfg", 1, 0, [row])
    assert P.read_chunk(tmp_path / repaired["path"], repaired["sha256"])[0]["pair_id"] == "a"

    # A syntactically valid but checksum-invalid file must not be trusted or
    # merged into the resumed result either.
    altered = dict(row, signed_effect=999.0)
    chunk.write_bytes(P.chunk_payload([altered]))
    assert store.load_valid_rows(strict=False) == []
    repaired = store.write_block("core", "m", "float16", "cfg", 1, 0, [row])
    loaded = P.read_chunk(tmp_path / repaired["path"], repaired["sha256"])
    assert loaded[0]["signed_effect"] == row["signed_effect"]


def test_manifest_is_immutable_and_config_hash_excludes_git(tmp_path):
    cfg = P.configuration_dict("prompt", "smoke", "code")
    assert "git_commit" not in cfg
    manifest = P.validate_or_create_manifest(
        tmp_path, run_id="run", config=cfg, metadata={"base_commit": "abc"},
    )
    assert manifest["metadata"]["base_commit"] == "abc"
    P.validate_or_create_manifest(tmp_path, run_id="run", config=cfg)
    changed = dict(cfg, code_sha256="different")
    with pytest.raises(P.PatchingError, match="manifest mismatch"):
        P.validate_or_create_manifest(tmp_path, run_id="run", config=changed)


def test_exact_pair_join_rejects_missing_or_duplicates():
    a = _result_row(pair_id="a")
    b = _result_row(pair_id="b")
    with pytest.raises(P.PatchingError, match="pair join mismatch"):
        P.exact_pair_join({"left": [a, b], "right": [a]}, expected_pair_ids=["a", "b"])
    with pytest.raises(P.PatchingError, match="duplicate"):
        P.exact_pair_join({"left": [a, a]}, expected_pair_ids=["a"])


def test_fp32_layer_selection_and_dtype_correct_random():
    model_id = "Qwen/Qwen2.5-1.5B"
    rows = []
    for layer, effect in ((1, 0.2), (2, 0.8), (3, -1.2), (4, 0.4)):
        for pair_id in ("a", "b"):
            for direction in P.DIRECTIONS:
                row = _result_row(
                    pair_id=pair_id, model_id=model_id, layer=layer,
                    direction=direction, effect=effect,
                )
                row["model_revision"] = P.MODELS[model_id]["revision"]
                rows.append(row)
    assert P.select_fp32_layers(rows, model_id) == [18, 3, 2]
    src = np.arange(16, dtype=np.float32)
    dst = np.linspace(1, 2, 16, dtype=np.float32)
    fp16 = P.patch_vector("random", src, dst, "p", 1, "query_name", "denoise", "float16")
    fp32 = P.patch_vector("random", src, dst, "p", 1, "query_name", "denoise", "float32")
    assert fp16.dtype == np.float16
    assert fp32.dtype == np.float32


def test_probe_artifact_metadata_validation(tmp_path):
    model_id = "Qwen/Qwen2.5-1.5B"
    width = 8
    path = tmp_path / "probe_seed0.npz"
    digest = P.save_probe_npz(
        path, scaler_mean=np.zeros(width), scaler_scale=np.ones(width),
        coef=np.ones((1, width)), intercept=np.zeros(1), classes=np.array([0, 1]),
    )
    P.write_json_atomic(tmp_path / "probe_meta.json", {
        "model_id": model_id,
        "model_revision": P.MODELS[model_id]["revision"],
        "layer": P.MODELS[model_id]["probe_index"],
        "dataset_revision": P.DATASET_REVISION,
        "prompt_sha256": P.EVAL_PROMPT_SHA256,
        "smoke_prompt_sha256": P.SMOKE_PROMPT_SHA256,
        "configuration_sha256": "config",
        "code_sha256": "code",
        "model_dtype": "float16",
        "hidden_size": width,
        "artifact_sha256": digest,
    })
    loaded = P.load_probe_artifact(
        path, model_id, expected_configuration_sha256="config",
        expected_code_sha256="code",
    )
    assert loaded["w_raw"].shape == (width,)
    meta = json.loads((tmp_path / "probe_meta.json").read_text())
    meta["model_revision"] = "wrong"
    P.write_json_atomic(tmp_path / "probe_meta.json", meta)
    with pytest.raises(P.PatchingError, match="model_revision"):
        P.load_probe_artifact(
            path, model_id, expected_configuration_sha256="config",
            expected_code_sha256="code",
        )
    meta["model_revision"] = P.MODELS[model_id]["revision"]
    P.write_json_atomic(tmp_path / "probe_meta.json", meta)
    with pytest.raises(P.PatchingError, match="configuration_sha256"):
        P.load_probe_artifact(
            path, model_id, expected_configuration_sha256="wrong",
            expected_code_sha256="code",
        )
    with pytest.raises(P.PatchingError, match="code_sha256"):
        P.load_probe_artifact(
            path, model_id, expected_configuration_sha256="config",
            expected_code_sha256="wrong",
        )


def test_offline_config_tokenizer_and_model_use_exact_snapshot(
    tmp_path, monkeypatch,
):
    """Transformers 5.8 must not resolve offline model IDs through the Hub."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model_id = "Qwen/Qwen2.5-1.5B"
    revision = P.MODELS[model_id]["revision"]
    cache = tmp_path / "hub"
    snapshot = (
        cache / "models--Qwen--Qwen2.5-1.5B" / "snapshots" / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"present")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))

    calls = {}

    class FakeTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "right"

    class FakeConfig:
        num_hidden_layers = 28
        hidden_size = 8

    class FakeInner:
        layers = []
        norm = object()

    class FakeModel:
        model = FakeInner()
        config = FakeConfig()

        def eval(self):
            return self

        def to(self, _device):
            return self

    def fake_tokenizer(source, **kwargs):
        calls["tokenizer"] = (source, kwargs)
        return FakeTokenizer()

    def fake_config(source, **kwargs):
        calls["config"] = (source, kwargs)
        return FakeConfig()

    def fake_model(source, **kwargs):
        calls["model"] = (source, kwargs)
        return FakeModel()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_tokenizer,
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_model,
    )

    P.validate_local_model_snapshot(model_id)
    P.load_tokenizer_pinned(model_id, local_files_only=True)
    _, _, adapter = P.load_causal_lm(
        model_id, torch.device("cpu"), local_files_only=True,
    )
    assert adapter.hidden_size == 8
    for key in ("config", "tokenizer", "model"):
        source, kwargs = calls[key]
        assert Path(source) == snapshot
        assert kwargs["local_files_only"] is True
        assert "revision" not in kwargs


def test_offline_sharded_snapshot_requires_every_weight_file(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    model_id = "Qwen/Qwen2.5-1.5B"
    revision = P.MODELS[model_id]["revision"]
    snapshot = (
        tmp_path / "hub" / "models--Qwen--Qwen2.5-1.5B"
        / "snapshots" / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    P.write_json_atomic(snapshot / "model.safetensors.index.json", {
        "weight_map": {"a": "model-00001-of-00002.safetensors",
                       "b": "model-00002-of-00002.safetensors"},
    })
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"present")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: object(),
    )
    with pytest.raises(P.PatchingError, match="missing/empty safetensors shards"):
        P.validate_local_model_snapshot(model_id)


def test_offline_snapshot_missing_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    with pytest.raises(P.PatchingError, match="missing exact local snapshot"):
        P.require_local_snapshot("Qwen/Qwen2.5-1.5B")


def test_probe_margins_only_make_sense_at_selected_layer():
    # The persisted schema can represent the protocol invariant explicitly:
    # off-layer interventions remain null while the probe layer is populated.
    off = _result_row(model_id="Qwen/Qwen2.5-1.5B", layer=17)
    on = _result_row(model_id="Qwen/Qwen2.5-1.5B", layer=18)
    on["source_probe_margin"] = 1.0
    on["destination_probe_margin"] = -1.0
    on["patched_probe_margin"] = 0.5
    assert off["source_probe_margin"] is None
    assert all(on[key] is not None for key in (
        "source_probe_margin", "destination_probe_margin", "patched_probe_margin",
    ))


def test_summaries_are_isolated_by_config_and_dtype(tmp_path):
    from pipeline.run_patching import cmd_summarize

    store = P.ChunkStore(tmp_path)
    model_id = "Qwen/Qwen2.5-1.5B"
    revision = P.MODELS[model_id]["revision"]
    for dtype, config in (("float16", "cfg16"), ("float32", "cfg32")):
        baseline = []
        interventions = []
        for i in range(2):
            pair_id = f"p{i}"
            for span, value in (("baseline_class", 1.0), ("baseline_function", -1.0)):
                row = _result_row(
                    pair_id=pair_id, model_id=model_id, config=config, dtype=dtype,
                    layer=-1, span=span, direction="none", control="unpatched",
                    effect=0.0, cluster_id=i,
                )
                row.update({
                    "model_revision": revision, "source_D": value,
                    "destination_D": value, "patched_D": value,
                    "source_probe_margin": value,
                    "source_probe_declaration_margin": value,
                })
                baseline.append(row)
            for direction in P.DIRECTIONS:
                for span, control, effect in (
                    ("query_name", "target", 0.8),
                    ("placebo", "target", 0.1),
                    ("query_name", "random", 0.05),
                ):
                    row = _result_row(
                        pair_id=pair_id, model_id=model_id, config=config,
                        dtype=dtype, layer=18, span=span, direction=direction,
                        control=control, effect=effect, cluster_id=i,
                    )
                    row["model_revision"] = revision
                    if span == "query_name" and control == "target":
                        row.update({
                            "source_probe_margin": 1.0,
                            "destination_probe_margin": -1.0,
                            "patched_probe_margin": 0.5,
                        })
                    interventions.append(row)
        store.write_block("behavior", model_id, dtype, config, -1, 0, baseline)
        store.write_block("primary", model_id, dtype, config, 18, 0, interventions)
    cmd_summarize(SimpleNamespace(run_dir=str(tmp_path), model=model_id, n_boot=50))
    index = json.loads((tmp_path / "summaries" / "index.json").read_text())
    assert len(index["summaries"]) == 2
    assert {entry["dtype"] for entry in index["summaries"]} == {"float16", "float32"}
    assert {entry["configuration_sha256"] for entry in index["summaries"]} == {"cfg16", "cfg32"}
    assert all((tmp_path / entry["path"]).is_file() for entry in index["summaries"])
    first_dir = (tmp_path / index["summaries"][0]["path"]).parent
    probe_header = (first_dir / "probe_link.csv").read_text().splitlines()[0]
    assert "denoise_patched_probe_margin" in probe_header
    assert "noise_patched_probe_margin" in probe_header
    assert "symmetric_probe_movement" in probe_header
    probe_summary = json.loads((first_dir / "probe_link_summary.json").read_text())
    assert "baseline_probe_gap_vs_behavior_spearman" in probe_summary
    assert "patched_probe_movement_vs_behavior_spearman" in probe_summary


def test_model_gate_exact_primary_cube_and_cross_phase_drift():
    model_id = "Qwen/Qwen2.5-1.5B"
    revision = P.MODELS[model_id]["revision"]
    rows = []
    pair_ids = [f"p{i}" for i in range(8)]
    for i, pair_id in enumerate(pair_ids):
        for span, value in (("baseline_class", 1.0), ("baseline_function", -1.0)):
            row = _result_row(
                pair_id=pair_id, model_id=model_id, layer=-1, span=span,
                direction="none", control="unpatched", effect=0.0, cluster_id=i,
            )
            row.update({
                "model_revision": revision, "source_D": value,
                "destination_D": value, "patched_D": value,
                "source_probe_margin": value,
                "source_probe_declaration_margin": value,
            })
            rows.append(row)
        for direction in P.DIRECTIONS:
            for layer, span, control, effect in (
                (18, "query_name", "target", 0.8),
                (18, "placebo", "target", 0.1),
                (18, "query_name", "random", 0.05),
                (18, "query_name", "same_source", 0.0),
                (0, "query_name", "target", 0.0),
            ):
                row = _result_row(
                    pair_id=pair_id, model_id=model_id, layer=layer, span=span,
                    direction=direction, control=control, effect=effect, cluster_id=i,
                )
                row.update({"model_revision": revision, "baseline_drift": 0.0005})
                rows.append(row)
    report = P.evaluate_gates(rows, model_id, expected_pair_ids=pair_ids)
    assert report["behavior"]["pass"]
    assert report["probe_ood"]["pass"]
    assert report["causal"]["pass"], report["causal"]
    assert report["max_cross_phase_drift"] == pytest.approx(0.0005)

    primary_row = next(row for row in rows if row["layer"] == 18)
    primary_row["baseline_drift"] = 0.02
    drifted = P.evaluate_gates(rows, model_id, expected_pair_ids=pair_ids)
    assert not drifted["causal"]["pass"]
    assert not drifted["causal"]["diagnostic_checks"]["max_cross_phase_drift_le_0_01"]
