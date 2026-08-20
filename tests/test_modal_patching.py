"""Controller, mount, resume, and cost tests that do not contact Modal."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pipeline import patching as P

REPO = Path(__file__).resolve().parents[1]
MODAL_SCRIPT = REPO / "scripts" / "modal_patching.py"


class _FakeImage:
    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def pip_install(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self

    def add_local_dir(self, *_args, **_kwargs):
        return self


class _FakeVolume:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.read_only = False

    @classmethod
    def from_name(cls, name, **kwargs):
        return cls(name, **kwargs)

    def with_mount_options(self, **kwargs):
        self.read_only = bool(kwargs.get("read_only", False))
        self.mount_options = kwargs
        return self

    def commit(self):
        return None

    def reload(self):
        return None


class _FakeApp:
    def __init__(self, *_args, **_kwargs):
        pass

    def function(self, **kwargs):
        def decorator(fn):
            fn.modal_kwargs = kwargs
            fn.remote = fn
            return fn
        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def _make_fake_modal() -> ModuleType:
    modal = ModuleType("modal")
    modal.Image = _FakeImage
    modal.Volume = _FakeVolume
    modal.App = _FakeApp
    modal.current_function_call_id = lambda: "test-function-call"
    modal.current_input_id = lambda: "test-input"
    return modal


def load_modal_patching():
    cached = sys.modules.get("modal_patching_under_test")
    if cached is not None:
        return cached
    previous = sys.modules.get("modal")
    sys.modules["modal"] = _make_fake_modal()
    spec = importlib.util.spec_from_file_location(
        "modal_patching_under_test", MODAL_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_patching_under_test"] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is not None:
            sys.modules["modal"] = previous
        else:
            sys.modules.pop("modal", None)
    return module


@pytest.fixture
def mp():
    return load_modal_patching()


def test_volume_plan_is_single_cross_env_data_mount(mp):
    assert mp.VOLUME_PLAN["data"]["name"] == "class-struct-data"
    assert mp.VOLUME_PLAN["data"]["environment_name"] == "main"
    assert mp.VOLUME_PLAN["data"]["mount"] == "/data"
    assert mp.VOLUME_PLAN["data"]["read_only"] is True
    assert mp.VOLUME_PLAN["results"]["mount"] == "/results"
    assert mp.VOLUME_PLAN["results"]["environment_name"] is None
    assert set(mp.COMMON_VOLUMES) == {"/results", "/data"}
    assert mp.COMMON_VOLUMES["/data"] is not mp.COMMON_VOLUMES["/results"]
    assert mp.COMMON_VOLUMES["/data"].name == "class-struct-data"
    assert mp.COMMON_VOLUMES["/data"].kwargs.get("environment_name") == "main"
    assert mp.COMMON_VOLUMES["/data"].read_only is True
    assert mp.COMMON_VOLUMES["/results"].name == "class-struct-patching-results"


def test_scientific_pins_exclude_injected_modal(mp):
    assert "modal" in mp.SOFTWARE_PINS
    assert "modal" not in mp.SCIENTIFIC_PINS
    req = (REPO / "pipeline" / "requirements.txt").read_text()
    for package, version in mp.SCIENTIFIC_PINS.items():
        assert f"{package}=={version}" in req


def test_torch_cuda_local_label_matches_pin(mp):
    assert mp._pin_matches("2.6.0", "2.6.0+cu124")
    assert mp._pin_matches("2.6.0", "2.6.0")
    assert not mp._pin_matches("2.6.0", "2.5.0+cu124")
    assert not mp._pin_matches("2.6.0", None)


def test_gpu_functions_do_not_share_one_chunk_namespace(tmp_path, mp):
    kwargs = mp.sweep_1p5b.modal_kwargs
    assert kwargs["max_containers"] == 1
    assert kwargs["retries"] == 0
    store = P.ChunkStore(tmp_path)
    qwen = store.chunk_relative_path("core", mp.QWEN, "float16", "cfg", 0, 0)
    coder = store.chunk_relative_path("core", mp.CODER, "float16", "cfg", 0, 0)
    star_fp32 = store.chunk_relative_path("fp32", mp.STAR, "float32", "cfg", 0, 0)
    assert qwen != coder
    assert coder != star_fp32
    assert "Qwen2.5-1.5B" in qwen.as_posix()
    assert "float16" in qwen.as_posix()
    assert "float32" in star_fp32.as_posix()
    assert not qwen.is_absolute()


def test_elapsed_cost_is_recorded_and_can_stop_the_run(tmp_path, monkeypatch, mp):
    monkeypatch.setattr(mp, "_commit", lambda: None)
    monkeypatch.setattr(mp, "_identity", lambda: {"function_call_id": "c1"})
    ledger = mp._record_elapsed(
        tmp_path, "qwen_smoke", "l4-2-16g", time.time() - 120, outcome="ok",
    )
    assert ledger["estimated_spent_usd"] > 0
    assert ledger["entries"][0]["elapsed_seconds"] >= 119
    huge = {
        "schema_version": 2,
        "entries": [{"estimated_usd": 48.0, "entry_id": "prior"}],
    }
    P.write_json_atomic(tmp_path / "cost_ledger.json", huge)
    with pytest.raises(P.PatchingError, match="refusing"):
        mp._refuse_if_over_budget(tmp_path, "core", 5.0)


def test_assert_manifest_rejects_changed_payload(tmp_path, monkeypatch, mp):
    monkeypatch.setattr(mp, "_manifest_payload", lambda run_id: {"run_id": run_id, "x": 1})
    with pytest.raises(P.PatchingError, match="preflight"):
        mp._assert_manifest(tmp_path, "run")
    P.write_json_atomic(tmp_path / "manifest.json", {"run_id": "run", "x": 1})
    assert mp._assert_manifest(tmp_path, "run")["x"] == 1
    P.write_json_atomic(tmp_path / "manifest.json", {"run_id": "run", "x": 2})
    with pytest.raises(P.PatchingError, match="immutable manifest"):
        mp._assert_manifest(tmp_path, "run")


def test_completeness_failure_is_raised(tmp_path, monkeypatch, mp):
    monkeypatch.setattr(mp, "_commit", lambda: None)
    P.write_json_atomic(tmp_path / "manifest.json", {
        "prompt_sha256": P.EVAL_PROMPT_SHA256,
        "configuration_sha256": "cfg",
    })
    with pytest.raises(P.PatchingError, match="completeness failed"):
        mp._write_completeness(
            tmp_path, mp.QWEN, "float16", ["behavior"], label="final",
        )


def test_model_core_runs_gates_before_the_layer_curve(monkeypatch, tmp_path, mp):
    phases = []
    monkeypatch.setattr(
        mp, "_ensure_probe",
        lambda *a, **k: {"prior": {"pass": True}},
    )
    monkeypatch.setattr(
        mp, "_run_sweep_phase",
        lambda *args, **kwargs: phases.append(args[3]) or {},
    )
    monkeypatch.setattr(mp, "_write_completeness", lambda *a, **k: {"complete": True})
    monkeypatch.setattr(mp, "_record_model_gate", lambda *a, **k: None)
    reports = [
        {"behavior": {"pass": True}, "probe_ood": {"pass": True}, "causal": {"skipped": True}},
        {"behavior": {"pass": True}, "probe_ood": {"pass": True}, "causal": {"pass": False}},
    ]
    monkeypatch.setattr(mp, "_evaluate_gates", lambda *a, **k: reports.pop(0))
    result = mp._run_model_core(
        tmp_path, mp.QWEN, SimpleNamespace(), SimpleNamespace(), "holder",
        resume=True,
    )
    assert phases == ["behavior", "primary"]
    assert result["status"] == "causal_null"


def test_qwen_gate_failure_does_not_start_replicas(monkeypatch, tmp_path, mp):
    started = []

    def fake_core(run_dir, model_id, *args, **kwargs):
        started.append(model_id)
        return {"model_id": model_id, "status": "causal_null"}

    monkeypatch.setattr(mp, "_run_model_core", fake_core)
    monkeypatch.setattr(mp, "_call_summarize", lambda *a, **k: started.append(("summarize", a[1])))
    monkeypatch.setattr(mp, "_status", lambda *a, **k: None)
    monkeypatch.setattr(mp, "_write_overnight_report", lambda *a, **k: None)
    monkeypatch.setattr(mp, "_clear_resolved_model_failure", lambda *a, **k: None)
    monkeypatch.setattr(mp, "_run_sweep_phase", lambda *a, **k: started.append("sweep"))
    out = mp._run_gated_controller(tmp_path, "holder", True)
    assert out["stopped"] == "qwen_gate"
    assert started[0] == mp.QWEN
    assert mp.CODER not in started
    assert mp.STAR not in started
    assert "sweep" not in started


def test_core_complete_models_get_expanded_and_fp32_deltas(monkeypatch, tmp_path, mp):
    started = []

    def fake_core(run_dir, model_id, *args, **kwargs):
        started.append(("core", model_id))
        return {"model_id": model_id, "status": "core_complete"}

    def fake_sweep(run_dir, model_id, sweep_fn, schedule, dtype, holder, **kwargs):
        started.append((schedule, model_id, dtype))
        return {}

    def fake_fp32(run_dir, model_id, sweep_fn, holder):
        started.append(("fp32", model_id))
        return {"layers": [18, 2, 3], "completeness": {"complete": True}}

    monkeypatch.setattr(mp, "_run_model_core", fake_core)
    monkeypatch.setattr(mp, "_run_sweep_phase", fake_sweep)
    monkeypatch.setattr(mp, "_run_fp32", fake_fp32)
    monkeypatch.setattr(mp, "_call_summarize", lambda *a, **k: None)
    monkeypatch.setattr(mp, "_status", lambda *a, **k: None)
    monkeypatch.setattr(mp, "_write_overnight_report", lambda *a, **k: None)
    monkeypatch.setattr(mp, "_clear_resolved_model_failure", lambda *a, **k: None)
    monkeypatch.setattr(
        mp, "_write_completeness",
        lambda *a, **k: {"complete": True, "label": k.get("label")},
    )
    out = mp._run_gated_controller(tmp_path, "holder", True)
    assert out["state"] == "complete"
    assert ("core", mp.QWEN) in started
    assert ("core", mp.CODER) in started
    assert ("core", mp.STAR) in started
    assert ("expanded", mp.QWEN, "float16") in started
    assert ("fp32", mp.QWEN) in started


def test_helpers_used_by_smoke_fit_and_summarize_are_module_level(mp):
    import inspect

    assert callable(mp._record_elapsed)
    assert callable(mp._refresh_lease)
    assert mp.qwen_smoke.__name__ == "qwen_smoke"
    assert mp.fit_probe.__name__ == "fit_probe"
    assert mp.summarize.__name__ == "summarize"
    assert "cmd_smoke" in inspect.getsource(mp.qwen_smoke)
    assert "cmd_fit_probe" in inspect.getsource(mp.fit_probe)
    assert "cmd_summarize" in inspect.getsource(mp.summarize)
