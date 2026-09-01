"""Modal runner for the Python class_struct activation-patching experiment.

The smoke is synchronous.  The overnight controller is CPU-only and calls one
GPU function at a time.  Scientific outputs use a dedicated results Volume;
the populated data/model Volume in ``main`` is mounted once, read-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import modal

REPO = Path(__file__).resolve().parents[1]
APP_NAME = "class-struct-patching"
RESULTS_VOLUME = "class-struct-patching-results"
DATA_VOLUME = "class-struct-data"
RUN_ID_DEFAULT = "class-struct-python-v1-20260819"
QWEN = "Qwen/Qwen2.5-1.5B"
CODER = "Qwen/Qwen2.5-Coder-1.5B"
STAR = "bigcode/starcoder2-7b"
MODELS = (QWEN, CODER, STAR)
SPEND_WARN_USD = 25.0
SPEND_HARD_USD = 50.0
LEASE_STALE_SECONDS = 10 * 60
HEARTBEAT_SECONDS = 60
EXPECTED_PROBE_SOURCE_SHA256 = (
    "1729bef9187b6f92a9d162c265c771a9294fffab8d406b1b696cd6234506a4f3"
)
BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
SOFTWARE_PINS = {
    "torch": "2.6.0",
    "transformers": "5.8.0",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "numpy": "2.2.6",
    "tqdm": "4.70.0",
    "datasets": "4.8.4",
    "huggingface_hub": "1.27.0",
    "modal": "1.5.4",
}
# Modal's client is injected by the runtime and is not pip-installed into the
# image. Preflight still records it, but only the scientific stack is a
# hard mismatch.
SCIENTIFIC_PINS = {
    package: version for package, version in SOFTWARE_PINS.items()
    if package != "modal"
}


def _discover_base_commit() -> str:
    inherited = os.environ.get("PATCHING_BASE_GIT_COMMIT")
    if inherited:
        return inherited
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


BASE_GIT_COMMIT = _discover_base_commit()

# Published component rates, with a deliberately conservative L40S estimate.
RESOURCE_USD_PER_HOUR = {
    "cpu2-4g": 2 * 0.0473 + 4 * 0.008,
    "cpu2-8g": 2 * 0.0473 + 8 * 0.008,
    "cpu4-8g": 4 * 0.0473 + 8 * 0.008,
    "cpu8-16g": 8 * 0.0473 + 16 * 0.008,
    "l4-2-16g": 0.80 + 2 * 0.0473 + 16 * 0.008,
    "l4-4-32g": 0.80 + 4 * 0.0473 + 32 * 0.008,
    "l40s-4-48g": 2.50 + 4 * 0.0473 + 48 * 0.008,
}
PROJECTED_PHASE_USD = {
    "extract_1p5b": 2.0,
    "extract_7b": 3.0,
    "fit_probe": 0.5,
    "behavior_1p5b": 0.5,
    "behavior_7b": 0.75,
    "primary_1p5b": 1.0,
    "primary_7b": 2.0,
    "core_1p5b": 4.0,
    "core_7b": 8.0,
    "expanded_1p5b": 9.0,
    "expanded_7b": 1.0,
    "fp32_1p5b": 3.0,
    "fp32_7b": 5.0,
}

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .pip_install(
        "transformers==5.8.0",
        "scipy==1.15.3",
        "scikit-learn==1.7.2",
        "numpy==2.2.6",
        "tqdm==4.70.0",
        "datasets==4.8.4",
        "huggingface_hub==1.27.0",
    )
    .env({"PATCHING_BASE_GIT_COMMIT": BASE_GIT_COMMIT})
    .add_local_dir(
        str(REPO), remote_path="/root/code-model-interpretability",
        ignore=[
            "**/.git/**", "**/.venv/**", "**/results/**", "**/notebooks/**",
            "**/XLCoST_data/**", "**/outputs/**", "**/.mypy_cache/**",
            "**/.DS_Store", "**/kaggle-amd-gpu-setup.json",
        ],
    )
)
app = modal.App(APP_NAME, image=image)
# Cross-environment mounts: results live in the current (`patching`) env;
# models/dataset live on the existing `main` volume. Modal rejects mounting
# the same Volume twice, so `/data` is the single data mount (hf/ and
# dataset/ are subdirectories, not separate mounts).
VOLUME_PLAN = {
    "results": {
        "name": RESULTS_VOLUME,
        "mount": "/results",
        "environment_name": None,
        "read_only": False,
        "create_if_missing": True,
    },
    "data": {
        "name": DATA_VOLUME,
        "mount": "/data",
        "environment_name": "main",
        "read_only": True,
        "create_if_missing": False,
    },
}
results_vol = modal.Volume.from_name(
    RESULTS_VOLUME, create_if_missing=True,
)
data_vol = modal.Volume.from_name(
    DATA_VOLUME, environment_name="main",
).with_mount_options(read_only=True)
COMMON_VOLUMES = {"/results": results_vol, "/data": data_vol}
# Release paid accelerators promptly between the gated phases.  The model is
# deliberately unloaded at the end of each worker, so a warm container would
# only retain the GPU reservation while CPU-side gates are running.
FN_KW = dict(max_containers=1, retries=0, scaledown_window=2)
_VOLUME_LOCK = threading.RLock()


def _prep() -> None:
    import sys
    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.chdir("/root/code-model-interpretability")
    if "/root/code-model-interpretability" not in sys.path:
        sys.path.insert(0, "/root/code-model-interpretability")


def _run_dir(run_id: str) -> Path:
    path = Path("/results/runs") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _commit() -> None:
    with _VOLUME_LOCK:
        results_vol.commit()


def _reload() -> None:
    with _VOLUME_LOCK:
        results_vol.reload()


def _identity() -> dict[str, str | None]:
    """Actual Modal identifiers; never invented phase labels."""
    return {
        "function_call_id": modal.current_function_call_id(),
        "input_id": modal.current_input_id(),
        "task_id": os.environ.get("MODAL_TASK_ID"),
    }


def _error_blob(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__, "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-12000:],
    }


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def _software_versions() -> dict[str, str | None]:
    import platform
    from importlib import metadata
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in (*SOFTWARE_PINS, "tokenizers", "safetensors"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _pin_matches(expected: str, installed: str | None) -> bool:
    """Accept PEP 440 local labels from CUDA wheels, e.g. 2.6.0+cu124."""
    if not installed:
        return False
    return installed.split("+", 1)[0] == expected


def _refresh_lease(run_dir: Path, holder: str | None, *, commit: bool = False) -> None:
    if not holder:
        return
    from pipeline import patching as P
    with _VOLUME_LOCK:
        path = run_dir / "lease.json"
        P.heartbeat_lease(path, holder, stale_after_s=LEASE_STALE_SECONDS)
        lease = _read_json(path, {})
        lease["last_worker"] = _identity()
        P.write_json_atomic(path, lease)
        if commit:
            _commit()


def _status(
    run_dir: Path, *, lease_holder: str | None = None, commit: bool = True,
    **updates: Any,
) -> dict:
    from pipeline import patching as P
    with _VOLUME_LOCK:
        path = run_dir / "status.json"
        cur = _read_json(path, {})
        cur.update(updates)
        cur.update({"heartbeat": time.time(), "worker": _identity()})
        P.write_json_atomic(path, cur)
        _refresh_lease(run_dir, lease_holder, commit=False)
        if commit:
            _commit()
    print(json.dumps({"status_event": updates}, default=str), flush=True)
    return cur


def _ledger_spent(ledger: dict) -> float:
    if "entries" in ledger:
        return sum(float(x.get("estimated_usd", 0.0)) for x in ledger["entries"])
    return float(ledger.get("spent_usd", 0.0))


def _record_elapsed(
    run_dir: Path, phase: str, resource: str, started_at: float, *,
    outcome: str, details: dict[str, Any] | None = None,
) -> dict:
    """Record 115% of elapsed cost, with a one-minute billing floor."""
    from pipeline import patching as P
    ended_at = time.time()
    elapsed_s = max(0.0, ended_at - started_at)
    billed_s = max(60.0, elapsed_s) * 1.15
    ident = _identity()
    call_id = ident.get("function_call_id") or ident.get("input_id")
    entry_id = f"{call_id or ('local:' + str(started_at))}:{phase}"
    entry = {
        "entry_id": entry_id, "phase": phase, "resource": resource,
        "hourly_usd": RESOURCE_USD_PER_HOUR[resource],
        "started_at": started_at, "ended_at": ended_at,
        "elapsed_seconds": elapsed_s, "conservative_billed_seconds": billed_s,
        "estimated_usd": billed_s / 3600 * RESOURCE_USD_PER_HOUR[resource],
        "outcome": outcome, "worker": ident, "details": details or {},
    }
    path = run_dir / "cost_ledger.json"
    entries = [
        x for x in _read_json(path, {}).get("entries", [])
        if x.get("entry_id") != entry_id
    ] + [entry]
    ledger = {
        "schema_version": 2, "entries": entries,
        "estimated_spent_usd": sum(x["estimated_usd"] for x in entries),
        "updated_at": ended_at,
    }
    P.write_json_atomic(path, ledger)
    _commit()
    return ledger


def _active_controller_cost(run_dir: Path) -> float:
    started = _read_json(run_dir / "status.json", {}).get("controller_started_at")
    if not started:
        return 0.0
    elapsed = max(0.0, time.time() - float(started))
    return elapsed / 3600 * RESOURCE_USD_PER_HOUR["cpu2-4g"] * 1.15


def _refuse_if_over_budget(
    run_dir: Path, phase: str, extra: float, lease_holder: str | None = None,
) -> None:
    from pipeline import patching as P
    spent = _ledger_spent(_read_json(run_dir / "cost_ledger.json", {}))
    projected = spent + _active_controller_cost(run_dir) + float(extra)
    if projected >= SPEND_HARD_USD:
        _status(
            run_dir, lease_holder=lease_holder,
            budget_stop={
                "phase": phase, "spent_estimate_usd": spent,
                "next_phase_usd": float(extra), "projected_usd": projected,
                "hard_usd": SPEND_HARD_USD,
            },
        )
        raise P.PatchingError(
            f"refusing {phase}: projected ${projected:.2f} reaches "
            f"${SPEND_HARD_USD:.2f} experiment stop"
        )
    if projected >= SPEND_WARN_USD:
        _status(
            run_dir, lease_holder=lease_holder,
            budget_warning={
                "phase": phase, "spent_estimate_usd": spent,
                "next_phase_usd": float(extra), "projected_usd": projected,
                "warning_usd": SPEND_WARN_USD,
            },
        )
        print(f"WARNING {phase}: projected spend ${projected:.2f}", flush=True)


def _heartbeat_loop(run_dir: Path, holder: str, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            _refresh_lease(run_dir, holder, commit=True)
        except BaseException as exc:
            print(f"lease heartbeat failed: {type(exc).__name__}: {exc}", flush=True)


@contextmanager
def _tracked_phase(
    run_dir: Path, phase: str, resource: str, *,
    lease_holder: str | None = None, **fields: Any,
):
    started = time.time()
    stop = threading.Event()
    thread = None
    _status(
        run_dir, lease_holder=lease_holder, state="running", phase=phase,
        phase_started_at=started, **fields,
    )
    if lease_holder:
        thread = threading.Thread(
            target=_heartbeat_loop, args=(run_dir, lease_holder, stop), daemon=True
        )
        thread.start()
    outcome, error = "ok", None
    try:
        yield
    except BaseException as exc:
        outcome, error = "failed", _error_blob(exc)
        try:
            _status(
                run_dir, lease_holder=lease_holder, state="worker_failed",
                phase=phase, error=error,
            )
        finally:
            raise
    finally:
        stop.set()
        if thread:
            thread.join(timeout=5)
        _record_elapsed(
            run_dir, phase, resource, started, outcome=outcome,
            details={**fields, **({"error": error} if error else {})},
        )


def _manifest_payload(run_id: str) -> dict:
    from pipeline import patching as P
    from pipeline.patching_prompts import default_eval_path, default_smoke_path, sha256_file
    prompt_sha = sha256_file(default_eval_path())
    smoke_sha = sha256_file(default_smoke_path())
    probe_source = Path("/data/dataset/python_perturbations/train.jsonl")
    if not probe_source.is_file():
        raise P.PatchingError(f"missing frozen probe source {probe_source}")
    probe_source_sha = P.sha256_file(probe_source)
    if probe_source_sha != EXPECTED_PROBE_SOURCE_SHA256:
        raise P.PatchingError(
            f"probe source hash {probe_source_sha} != pinned "
            f"{EXPECTED_PROBE_SOURCE_SHA256}"
        )
    code_sha = P.bundle_sha256(P.default_bundle_paths())
    cfg = P.configuration_dict(prompt_sha, smoke_sha, code_sha)
    payload = {
        **cfg, "run_id": run_id,
        "configuration_sha256": P.configuration_sha256(cfg),
        "base_git_commit": BASE_GIT_COMMIT,
        "base_image": BASE_IMAGE,
        "software_pins": SOFTWARE_PINS,
        "software_versions": _software_versions(),
        "probe_source_jsonl_sha256": probe_source_sha,
        "estimates": P.estimate_all(),
        "resources": {
            "qwen_smoke": "L4/16GiB/20min",
            "extract_1p5b": "L4/16GiB/90min",
            "extract_7b": "L4/32GiB/2h",
            "sweep_1p5b": "L4/16GiB/10h",
            "sweep_7b": "L4/32GiB/8h",
            "fp32_7b": "L40S/48GiB/90min",
            "controller": "CPU/4GiB/24h",
        },
    }
    # Match the exact JSON representation written to the Volume (notably,
    # schedule tuples become arrays). This makes equality strict and stable.
    return json.loads(json.dumps(payload, sort_keys=True))


def _assert_manifest(run_dir: Path, run_id: str) -> dict:
    from pipeline import patching as P
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise P.PatchingError("run preflight first: manifest.json is missing")
    stored, current = _read_json(path), _manifest_payload(run_id)
    if stored != current:
        raise P.PatchingError(
            "immutable manifest mismatch; prompts, code, model revisions, or "
            "configuration changed. Use a new run ID"
        )
    return stored


def _require_smoke(run_dir: Path, manifest: dict) -> dict:
    from pipeline import patching as P
    rec = _read_json(run_dir / "smoke_receipt.json")
    if not rec:
        raise P.PatchingError("run the synchronous Qwen smoke before detaching")
    if (
        rec.get("configuration_sha256") != manifest["configuration_sha256"]
        or rec.get("prompt_sha256") != manifest["smoke_prompt_sha256"]
        or rec.get("model_id") != QWEN
        or not rec.get("pass")
    ):
        raise P.PatchingError("no passing Qwen smoke for the current manifest")
    return rec


def _phase_callback(
    run_dir: Path, holder: str | None, phase: str,
) -> Callable[..., None]:
    def callback(event: Any = None, **kwargs: Any) -> None:
        payload = dict(event) if isinstance(event, dict) else {}
        if event is not None and not isinstance(event, dict):
            payload["event"] = str(event)
        payload.update(kwargs)
        # Called after fsync + atomic rename. This Volume commit makes both the
        # chunk and updated index durable and cross-container visible.
        _status(
            run_dir, lease_holder=holder, state="running", phase=phase,
            last_checkpoint=payload,
        )
    return callback


def _record_model_gate(run_dir: Path, model_id: str, stage: str, report: dict) -> None:
    from pipeline import patching as P
    path = run_dir / "gate_report.json"
    blob = _read_json(path, {"models": {}})
    blob.setdefault("models", {}).setdefault(model_id, {})[stage] = report
    P.write_json_atomic(path, blob)
    _commit()


def _model_rows(run_dir: Path, model_id: str, dtype: str = "float16") -> list[dict]:
    from pipeline import patching as P
    cfg = _read_json(run_dir / "manifest.json", {})["configuration_sha256"]
    return [
        r for r in P.ChunkStore(run_dir).load_valid_rows()
        if r.get("model_id") == model_id and r.get("dtype") == dtype
        and r.get("configuration_sha256") == cfg
    ]


def _evaluate_gates(run_dir: Path, model_id: str) -> dict:
    # Includes the 10k clustered bootstrap for the synthetic probe-OOD link;
    # this helper runs in the CPU controller, never in a GPU function.
    from pipeline.run_patching import evaluate_model_results
    return evaluate_model_results(run_dir, model_id, dtype="float16")


def _schedule_union(model_id: str, schedules: Iterable[str], layers=None) -> list[tuple]:
    from pipeline import patching as P
    cells = []
    for schedule in schedules:
        cells.extend(P.schedule_cells(
            model_id, schedule, layers=layers if schedule == "fp32" else None
        ))
    return list(dict.fromkeys(cells))


def _write_completeness(
    run_dir: Path, model_id: str, dtype: str, schedules: Iterable[str], *,
    layers: list[int] | None = None, label: str,
) -> dict:
    from pipeline import patching as P
    from pipeline.patching_prompts import default_eval_path, load_jsonl
    schedules = list(schedules)
    manifest = _read_json(run_dir / "manifest.json")
    expected = P.expected_intervention_keys(
        model_id, load_jsonl(default_eval_path()), manifest["prompt_sha256"],
        manifest["configuration_sha256"], dtype,
        cells=_schedule_union(model_id, schedules, layers),
    )
    rows = _model_rows(run_dir, model_id, dtype)
    expected_set = set(expected)
    scoped = [row for row in rows if P.primary_key(row) in expected_set]
    report = P.completeness_report(expected, scoped)
    report["n_out_of_scope"] = len(rows) - len(scoped)
    report.update({
        "model_id": model_id, "dtype": dtype, "schedules": schedules,
        "layers": layers,
    })
    out = run_dir / "summaries" / "completeness"
    out.mkdir(parents=True, exist_ok=True)
    P.write_json_atomic(out / f"{model_id.split('/')[-1]}-{label}.json", report)
    _commit()
    final_scope = label in {"final", "fp32-final", "full", "fp32"}
    if not report["complete"] or (final_scope and report["n_out_of_scope"]):
        raise P.PatchingError(
            f"completeness failed {model_id} {label}: "
            f"missing={report['n_missing']} extra={report['n_extra']} "
            f"out_of_scope={report['n_out_of_scope']} finite={report['finite']}"
        )
    return report


def _schedule_is_complete(
    run_dir: Path, model_id: str, schedule: str, dtype: str,
    layers: list[int] | None,
) -> bool:
    """Cheap CPU resume check performed before a GPU container is requested."""
    from pipeline import patching as P
    from pipeline.patching_prompts import default_eval_path, load_jsonl
    manifest = _read_json(run_dir / "manifest.json")
    expected = set(P.expected_intervention_keys(
        model_id, load_jsonl(default_eval_path()), manifest["prompt_sha256"],
        manifest["configuration_sha256"], dtype,
        cells=P.schedule_cells(model_id, schedule, layers=layers),
    ))
    present = {P.primary_key(row) for row in _model_rows(run_dir, model_id, dtype)}
    return expected.issubset(present)


@app.function(cpu=2, memory=8192, timeout=15 * 60, volumes=COMMON_VOLUMES, **FN_KW)
def preflight(run_id: str = RUN_ID_DEFAULT):
    _prep()
    _reload()
    from pipeline import patching as P
    from pipeline.patching_prompts import (
        default_eval_path, default_smoke_path, load_jsonl,
        validate_frozen_files, validate_python_semantics,
    )
    run_dir = _run_dir(run_id)
    with _tracked_phase(run_dir, "preflight", "cpu2-8g"):
        if BASE_GIT_COMMIT == "unknown":
            raise P.PatchingError(
                "base Git commit was not injected while building the Modal image"
            )
        validate_frozen_files()
        installed = _software_versions()
        mismatches = {
            package: {"expected": expected, "installed": installed.get(package)}
            for package, expected in SCIENTIFIC_PINS.items()
            if not _pin_matches(expected, installed.get(package))
        }
        if mismatches:
            raise P.PatchingError(f"software pin mismatch: {mismatches}")
        rows = load_jsonl(default_eval_path()) + load_jsonl(default_smoke_path())
        for row in rows:
            validate_python_semantics(row)
        for model_id in P.MODELS:
            model_config = P.validate_local_model_snapshot(model_id)
            got_blocks = int(getattr(model_config, "num_hidden_layers", -1))
            got_width = int(getattr(model_config, "hidden_size", -1))
            got_hidden = got_blocks + 1
            if (
                got_blocks != P.MODELS[model_id]["n_blocks"]
                or got_hidden != P.MODELS[model_id]["n_hidden"]
                or got_width <= 0
            ):
                raise P.PatchingError(
                    f"cached config mismatch for {model_id}: "
                    f"num_hidden_layers={got_blocks}, n_hidden={got_hidden}, "
                    f"hidden_size={got_width}"
                )
            tok = P.load_tokenizer_pinned(model_id, local_files_only=True)
            for row in rows:
                P.validate_pair_tokenizer(tok, row, model_id)
            P.discover_completion_ids(tok, model_id)
        manifest = _manifest_payload(run_id)
        path = run_dir / "manifest.json"
        old = _read_json(path)
        if old is not None and old != manifest:
            raise P.PatchingError("refusing to overwrite immutable manifest")
        if old is None:
            P.write_json_atomic(path, manifest)
            _commit()
        _status(
            run_dir, state="preflight_ok", phase="preflight",
            configuration_sha256=manifest["configuration_sha256"],
        )
    return {"run_id": run_id, "configuration_sha256": manifest["configuration_sha256"]}


@app.function(
    gpu="L4", cpu=2, memory=16384, timeout=20 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def qwen_smoke(run_id: str = RUN_ID_DEFAULT):
    _prep()
    _reload()
    from pipeline import patching as P
    from pipeline.run_patching import cmd_smoke
    run_dir = _run_dir(run_id)
    manifest = _assert_manifest(run_dir, run_id)
    with _tracked_phase(run_dir, "qwen_smoke", "l4-2-16g", model_id=QWEN):
        smoke_dir = run_dir / "smoke"
        args = SimpleNamespace(
            model=QWEN, out=str(smoke_dir), run_id=run_id, probe=None,
            allow_download=False, verify_hooks=True,
            checkpoint_callback=_phase_callback(run_dir, None, "qwen_smoke"),
            progress_callback=None,
        )
        smoke_error = None
        try:
            cmd_smoke(args)
        except SystemExit as exc:
            # cmd_smoke writes its gate report before raising on a scientific
            # no-go. Persist a negative receipt so the reason is inspectable.
            smoke_error = exc
        gate = _read_json(smoke_dir / "gate_report.json")
        passed = bool(gate and gate.get("smoke", {}).get("pass"))
        receipt = {
            "model_id": QWEN,
            "configuration_sha256": manifest["configuration_sha256"],
            "prompt_sha256": manifest["smoke_prompt_sha256"],
            "pass": passed, "gate": gate, "worker": _identity(),
            "completed_at": time.time(),
        }
        P.write_json_atomic(run_dir / "smoke_receipt.json", receipt)
        _status(
            run_dir, state="smoke_passed" if passed else "smoke_failed",
            phase="qwen_smoke", smoke_pass=passed,
        )
    if not passed:
        # A negative scientific gate is not an infrastructure failure, but the
        # synchronous Modal command must still exit non-zero so no one detaches
        # the overnight controller by accident.
        raise P.PatchingError("Qwen smoke gate failed") from smoke_error
    return receipt


def _probe_paths(run_dir: Path, model_id: str) -> tuple[Path, Path]:
    root = run_dir / "probes" / model_id.split("/")[-1]
    return root / "hidden", root


def _extract(model_id: str, run_id: str, holder: str | None, resource: str):
    _prep()
    _reload()
    from pipeline.run_patching import extract_probe_layer
    run_dir = _run_dir(run_id)
    _assert_manifest(run_dir, run_id)
    hidden, _ = _probe_paths(run_dir, model_id)
    phase = f"extract_probe:{model_id}"
    with _tracked_phase(
        run_dir, phase, resource, lease_holder=holder, model_id=model_id,
    ):
        extract_probe_layer(
            model_id, "/data/dataset", hidden, split="train", allow_download=False
        )
        _status(
            run_dir, lease_holder=holder, state="running", phase=phase,
            probe_dump=str(hidden),
        )
    return str(hidden)


@app.function(
    gpu="L4", cpu=2, memory=16384, timeout=90 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def extract_probe_1p5b(
    model_id: str, run_id: str = RUN_ID_DEFAULT, lease_holder: str | None = None,
):
    return _extract(model_id, run_id, lease_holder, "l4-2-16g")


@app.function(
    gpu="L4", cpu=4, memory=32768, timeout=2 * 60 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def extract_probe_7b(
    model_id: str = STAR, run_id: str = RUN_ID_DEFAULT,
    lease_holder: str | None = None,
):
    return _extract(model_id, run_id, lease_holder, "l4-4-32g")


@app.function(
    cpu=8, memory=16384, timeout=60 * 60, volumes=COMMON_VOLUMES, **FN_KW,
)
def fit_probe(
    model_id: str, run_id: str = RUN_ID_DEFAULT, lease_holder: str | None = None,
):
    _prep()
    _reload()
    from pipeline import patching as P
    from pipeline.run_patching import cmd_fit_probe
    run_dir = _run_dir(run_id)
    _assert_manifest(run_dir, run_id)
    hidden, out = _probe_paths(run_dir, model_id)
    phase = f"fit_probe:{model_id}"
    with _tracked_phase(
        run_dir, phase, "cpu8-16g", lease_holder=lease_holder, model_id=model_id,
    ):
        cmd_fit_probe(SimpleNamespace(
            model=model_id, hidden=str(hidden / "hidden.npy"),
            labels=str(hidden / "labels.npy"), programs=str(hidden / "programs.npy"),
            out=str(out), strict_prior=False,
        ))
        meta = _read_json(out / "probe_meta.json")
        if not meta:
            raise P.PatchingError(f"probe fit wrote no metadata for {model_id}")
        _status(
            run_dir, lease_holder=lease_holder, state="running", phase=phase,
            probe_prior_pass=bool(meta.get("prior", {}).get("pass")),
        )
    return meta


def _sweep(
    model_id: str, run_id: str, cells: str, dtype: str, resource: str,
    lease_holder: str | None, layers: list[int] | None = None,
    verify_hooks: bool = False,
):
    _prep()
    _reload()
    from pipeline.run_patching import cmd_sweep
    run_dir = _run_dir(run_id)
    _assert_manifest(run_dir, run_id)
    _, probe_dir = _probe_paths(run_dir, model_id)
    probe = probe_dir / "probe_seed0.npz"
    if not probe.is_file():
        raise FileNotFoundError(f"missing frozen probe {probe}")
    phase = f"sweep:{model_id}:{cells}:{dtype}"
    callback = _phase_callback(run_dir, lease_holder, phase)
    with _tracked_phase(
        run_dir, phase, resource, lease_holder=lease_holder, model_id=model_id,
        schedule=cells, dtype=dtype, layers=layers,
    ):
        cmd_sweep(SimpleNamespace(
            model=model_id, out=str(run_dir), cells=cells, dtype=dtype,
            layers=layers, run_id=run_id, probe=str(probe), max_pairs=None,
            allow_download=False, verify_hooks=verify_hooks,
            checkpoint_callback=callback, progress_callback=None,
        ))
        _status(
            run_dir, lease_holder=lease_holder, state="running", phase=phase,
            phase_complete=True,
        )
    return {
        "model_id": model_id, "schedule": cells, "dtype": dtype,
        "layers": layers, "worker": _identity(),
    }


@app.function(
    gpu="L4", cpu=2, memory=16384, timeout=10 * 60 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def sweep_1p5b(
    model_id: str, run_id: str = RUN_ID_DEFAULT, cells: str = "core",
    dtype: str = "float16", lease_holder: str | None = None,
    layers: list[int] | None = None, verify_hooks: bool = False,
):
    return _sweep(
        model_id, run_id, cells, dtype, "l4-2-16g", lease_holder, layers,
        verify_hooks,
    )


@app.function(
    gpu="L4", cpu=4, memory=32768, timeout=8 * 60 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def sweep_7b(
    model_id: str = STAR, run_id: str = RUN_ID_DEFAULT, cells: str = "core",
    dtype: str = "float16", lease_holder: str | None = None,
    layers: list[int] | None = None, verify_hooks: bool = False,
):
    return _sweep(
        model_id, run_id, cells, dtype, "l4-4-32g", lease_holder, layers,
        verify_hooks,
    )


@app.function(
    gpu="L40S", cpu=4, memory=49152, timeout=90 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def fp32_7b(
    run_id: str = RUN_ID_DEFAULT, lease_holder: str | None = None,
    layers: list[int] | None = None,
):
    if not layers:
        raise ValueError("fp32_7b requires three frozen layers")
    return _sweep(
        STAR, run_id, "fp32", "float32", "l40s-4-48g", lease_holder,
        layers, False,
    )


@app.function(
    cpu=4, memory=8192, timeout=60 * 60, volumes=COMMON_VOLUMES, **FN_KW,
)
def summarize(
    run_id: str = RUN_ID_DEFAULT, model_id: str | None = None,
    lease_holder: str | None = None,
):
    """All fitting, bootstraps, confidence intervals, and reports stay CPU-only."""
    _prep()
    _reload()
    from pipeline import patching as P
    from pipeline.run_patching import cmd_summarize
    run_dir = _run_dir(run_id)
    _assert_manifest(run_dir, run_id)
    phase = f"summarize:{model_id or 'all'}"
    with _tracked_phase(
        run_dir, phase, "cpu4-8g", lease_holder=lease_holder,
        model_id=model_id,
    ):
        cmd_summarize(SimpleNamespace(run_dir=str(run_dir), model=model_id))
        report = {
            "run_id": run_id, "generated_at": time.time(),
            "status": _read_json(run_dir / "status.json", {}),
            "gate_report": _read_json(run_dir / "gate_report.json", {}),
        }
        P.write_json_atomic(run_dir / "overnight_report.json", report)
        _status(
            run_dir, lease_holder=lease_holder, state="running", phase=phase,
            summary_complete=True,
        )
    return report


def _probe_ready(run_dir: Path, model_id: str) -> bool:
    from pipeline import patching as P
    hidden, out = _probe_paths(run_dir, model_id)
    manifest = _read_json(run_dir / "manifest.json", {})
    needed = (
        hidden / "hidden.npy", hidden / "labels.npy", hidden / "programs.npy",
        out / "probe_seed0.npz", out / "probe_meta.json",
    )
    if not all(p.is_file() for p in needed):
        return False
    try:
        P.load_probe_artifact(
            out / "probe_seed0.npz", model_id,
            expected_configuration_sha256=manifest.get("configuration_sha256"),
            expected_code_sha256=manifest.get("code_sha256"),
        )
    except (OSError, ValueError, P.PatchingError):
        return False
    return True


def _extraction_ready(run_dir: Path, model_id: str) -> bool:
    """Validate a durable GPU dump before deciding whether to re-extract it."""
    from pipeline import patching as P
    hidden, _ = _probe_paths(run_dir, model_id)
    meta = _read_json(hidden / "extraction_meta.json", {})
    expected = {
        "model_id": model_id,
        "model_revision": P.MODELS[model_id]["revision"],
        "layer": P.MODELS[model_id]["probe_index"],
        "dataset_revision": P.DATASET_REVISION,
        "n_programs": 400,
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        return False
    source_jsonl = Path("/data/dataset/python_perturbations/train.jsonl")
    if (
        not source_jsonl.is_file()
        or meta.get("source_jsonl_sha256") != P.sha256_file(source_jsonl)
    ):
        return False
    for filename, field in (
        ("hidden.npy", "hidden_sha256"),
        ("labels.npy", "labels_sha256"),
        ("programs.npy", "programs_sha256"),
    ):
        path = hidden / filename
        if not path.is_file() or meta.get(field) != P.sha256_file(path):
            return False
    return True


def _ensure_probe(
    run_dir: Path, model_id: str, extract_fn, lease_holder: str, *,
    resume: bool,
) -> dict:
    if resume and _probe_ready(run_dir, model_id):
        print(f"resume: valid probe already present for {model_id}", flush=True)
        return _read_json(_probe_paths(run_dir, model_id)[1] / "probe_meta.json")
    kind = "7b" if model_id == STAR else "1p5b"
    phase = f"extract_{kind}"
    if resume and _extraction_ready(run_dir, model_id):
        print(f"resume: valid extraction already present for {model_id}", flush=True)
    else:
        _refuse_if_over_budget(
            run_dir, phase, PROJECTED_PHASE_USD[phase], lease_holder
        )
        extract_fn.remote(model_id, run_dir.name, lease_holder)
        _reload()
    _refuse_if_over_budget(
        run_dir, "fit_probe", PROJECTED_PHASE_USD["fit_probe"], lease_holder
    )
    meta = fit_probe.remote(model_id, run_dir.name, lease_holder)
    _reload()
    return meta


def _run_sweep_phase(
    run_dir: Path, model_id: str, sweep_fn, schedule: str, dtype: str,
    lease_holder: str, *, layers: list[int] | None = None,
    verify_hooks: bool = False,
) -> dict:
    if _schedule_is_complete(run_dir, model_id, schedule, dtype, layers):
        print(
            f"resume: skip complete {model_id} {schedule} {dtype} {layers or ''}",
            flush=True,
        )
        return {
            "model_id": model_id, "schedule": schedule, "dtype": dtype,
            "layers": layers, "resumed": True,
        }
    kind = "7b" if model_id == STAR else "1p5b"
    projection = PROJECTED_PHASE_USD[f"{schedule}_{kind}"]
    # The replica-model expanded schedule is only the six probe-layer
    # declaration controls missing from the gate. Qwen's is the large
    # all-layer control delta.
    if schedule == "expanded" and model_id == CODER:
        projection = 0.5
    _refuse_if_over_budget(
        run_dir, f"{model_id}:{schedule}:{dtype}", projection, lease_holder
    )
    result = sweep_fn.remote(
        model_id, run_dir.name, schedule, dtype, lease_holder, layers, verify_hooks
    )
    _reload()
    return result


def _run_model_core(
    run_dir: Path, model_id: str, extract_fn, sweep_fn, lease_holder: str, *,
    resume: bool,
) -> dict:
    """Probe, baseline behavior gate, primary causal gate, then layer curve."""
    result: dict[str, Any] = {"model_id": model_id, "status": "running"}
    probe_meta = _ensure_probe(
        run_dir, model_id, extract_fn, lease_holder, resume=resume
    )
    result["probe_prior"] = probe_meta.get("prior", {"pass": False})
    _record_model_gate(run_dir, model_id, "probe_prior", result["probe_prior"])
    if not result["probe_prior"].get("pass", False):
        result["status"] = "probe_link_null"
        return result
    _run_sweep_phase(
        run_dir, model_id, sweep_fn, "behavior", "float16", lease_holder,
        verify_hooks=True,
    )
    behavior_completeness = _write_completeness(
        run_dir, model_id, "float16", ["behavior"], label="behavior"
    )
    gates = _evaluate_gates(run_dir, model_id)
    result["behavior"] = gates["behavior"]
    result["probe_ood"] = gates["probe_ood"]
    _record_model_gate(run_dir, model_id, "behavior", gates["behavior"])
    _record_model_gate(run_dir, model_id, "probe_ood", gates["probe_ood"])
    if not gates["probe_ood"].get("pass", False):
        result["status"] = "probe_link_null"
        result["completeness"] = behavior_completeness
        return result
    if not gates["behavior"]["pass"]:
        result["status"] = "behavioral_null"
        result["completeness"] = behavior_completeness
        return result

    _run_sweep_phase(
        run_dir, model_id, sweep_fn, "primary", "float16", lease_holder
    )
    primary_completeness = _write_completeness(
        run_dir, model_id, "float16", ["behavior", "primary"],
        label="primary",
    )
    gates = _evaluate_gates(run_dir, model_id)
    result["causal"] = gates["causal"]
    _record_model_gate(run_dir, model_id, "primary", gates)
    if not gates["causal"]["pass"]:
        result["status"] = "causal_null"
        result["completeness"] = primary_completeness
        return result

    _run_sweep_phase(
        run_dir, model_id, sweep_fn, "core", "float16", lease_holder
    )
    result["completeness"] = _write_completeness(
        run_dir, model_id, "float16", ["behavior", "primary", "core"],
        label="matched-core",
    )
    result["status"] = "core_complete"
    return result


def _freeze_fp32_layers(run_dir: Path, model_id: str) -> list[int]:
    from pipeline import patching as P
    path = run_dir / "fp32_selection.json"
    blob = _read_json(path, {"models": {}})
    selected = [
        int(x) for x in P.select_fp32_layers(_model_rows(run_dir, model_id), model_id)
    ]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise P.PatchingError(f"expected three distinct fp32 layers: {selected}")
    old = blob.setdefault("models", {}).get(model_id)
    if old is not None and old != selected:
        raise P.PatchingError(
            f"frozen fp32 layers changed for {model_id}: {old} != {selected}"
        )
    blob["models"][model_id] = selected
    P.write_json_atomic(path, blob)
    _commit()
    return selected


def _run_fp32(run_dir: Path, model_id: str, sweep_fn, holder: str) -> dict:
    layers = _freeze_fp32_layers(run_dir, model_id)
    if model_id == STAR:
        if _schedule_is_complete(run_dir, model_id, "fp32", "float32", layers):
            result = {
                "model_id": model_id, "schedule": "fp32", "dtype": "float32",
                "layers": layers, "resumed": True,
            }
        else:
            _refuse_if_over_budget(
                run_dir, f"{model_id}:fp32", PROJECTED_PHASE_USD["fp32_7b"],
                holder,
            )
            result = fp32_7b.remote(run_dir.name, holder, layers)
            _reload()
    else:
        result = _run_sweep_phase(
            run_dir, model_id, sweep_fn, "fp32", "float32", holder,
            layers=layers,
        )
    result["completeness"] = _write_completeness(
        run_dir, model_id, "float32", ["fp32"], layers=layers, label="fp32"
    )
    return result


def _record_model_failure(run_dir: Path, model_id: str, exc: BaseException) -> dict:
    from pipeline import patching as P
    record = {
        "model_id": model_id, "status": "infrastructure_failed",
        "error": _error_blob(exc), "at": time.time(),
    }
    path = run_dir / "model_failures.json"
    blob = _read_json(path, {"models": {}})
    blob.setdefault("models", {})[model_id] = record
    P.write_json_atomic(path, blob)
    _commit()
    return record


def _clear_resolved_model_failure(run_dir: Path, model_id: str) -> None:
    from pipeline import patching as P
    path = run_dir / "model_failures.json"
    blob = _read_json(path, {"models": {}, "resolved": []})
    old = blob.setdefault("models", {}).pop(model_id, None)
    if old is None:
        return
    blob.setdefault("resolved", []).append({
        **old, "resolved_at": time.time(), "resolution": "resume_succeeded",
    })
    P.write_json_atomic(path, blob)
    _commit()


def _write_overnight_report(run_dir: Path) -> None:
    """Write a small final-status handoff after the controller state is final."""
    from pipeline import patching as P
    status = _read_json(run_dir / "status.json", {})
    gates = _read_json(run_dir / "gate_report.json", {})
    failures = _read_json(run_dir / "model_failures.json", {"models": {}})
    report = {
        "run_id": run_dir.name,
        "generated_at": time.time(),
        "status": status,
        "gate_report": gates,
        "model_failures": failures,
        "cost_ledger": _read_json(run_dir / "cost_ledger.json", {}),
    }
    P.write_json_atomic(run_dir / "overnight_report.json", report)
    ledger = report["cost_ledger"]
    estimated = ledger.get("estimated_spent_usd", _ledger_spent(ledger))
    model_records = status.get("models", {})
    lines = [
        "# Overnight class_struct patching report",
        "",
        f"- Run: `{run_dir.name}`",
        f"- State: `{status.get('state', 'unknown')}`",
        f"- Phase: `{status.get('phase', 'unknown')}`",
        f"- Heartbeat: `{status.get('heartbeat', 'unknown')}`",
        f"- Infrastructure failures: "
        f"`{', '.join(status.get('infrastructure_failures', [])) or 'none'}`",
        f"- Conservative recorded cost estimate: `${float(estimated):.2f}`",
        "",
        "## Models",
        "",
    ]
    for model_id in MODELS:
        record = model_records.get(model_id, {})
        lines.append(f"- `{model_id}`: `{record.get('status', 'not attempted')}`")
    lines.extend([
        "",
        "## Outputs",
        "",
        "- Gates: `gate_report.json` and `diagnostics/`",
        "- Statistics: `summaries/`",
        "- Exact schedule checks: `summaries/completeness/`",
        "- Failures: `model_failures.json`",
        "- Cost ledger: `cost_ledger.json`",
    ])
    P.atomic_write_text(run_dir / "overnight_report.md", "\n".join(lines) + "\n")
    _commit()


def _call_summarize(run_id: str, model_id: str | None, holder: str) -> None:
    summarize.remote(run_id, model_id, holder)
    _reload()


def _run_gated_controller(run_dir: Path, holder: str, resume: bool) -> dict:
    """Sequential overnight loop. GPU work is invoked one function at a time."""
    results: dict[str, Any] = {}
    failures: list[str] = []
    try:
        qwen = _run_model_core(
            run_dir, QWEN, extract_probe_1p5b, sweep_1p5b, holder,
            resume=resume,
        )
    except BaseException as exc:
        results[QWEN] = _record_model_failure(run_dir, QWEN, exc)
        raise
    results[QWEN] = qwen
    _clear_resolved_model_failure(run_dir, QWEN)
    _call_summarize(run_dir.name, QWEN, holder)
    _status(
        run_dir, lease_holder=holder, state="running", phase="controller",
        last_model_summarized=QWEN, models=results,
    )
    if qwen["status"] != "core_complete":
        _status(
            run_dir, lease_holder=holder, state="stopped_qwen_null",
            phase="done", models=results,
        )
        _write_overnight_report(run_dir)
        return {"stopped": "qwen_gate", "models": results}

    for model_id, extract_fn, sweep_fn in (
        (CODER, extract_probe_1p5b, sweep_1p5b),
        (STAR, extract_probe_7b, sweep_7b),
    ):
        try:
            results[model_id] = _run_model_core(
                run_dir, model_id, extract_fn, sweep_fn, holder,
                resume=resume,
            )
            _clear_resolved_model_failure(run_dir, model_id)
            _call_summarize(run_dir.name, model_id, holder)
            _status(
                run_dir, lease_holder=holder, state="running",
                phase="controller", last_model_summarized=model_id,
                models=results,
            )
        except BaseException as exc:
            failures.append(model_id)
            results[model_id] = _record_model_failure(run_dir, model_id, exc)
            _status(
                run_dir, lease_holder=holder, state="running",
                phase="controller", last_model_failure=results[model_id],
            )

    # Delta only: primary/core cells are excluded. For Qwen this is the
    # large all-layer control cube; for replicas it is only the six missing
    # declaration-name controls at the preregistered layer.
    for model_id, sweep_fn in (
        (QWEN, sweep_1p5b), (CODER, sweep_1p5b), (STAR, sweep_7b)
    ):
        if results.get(model_id, {}).get("status") != "core_complete":
            continue
        try:
            _run_sweep_phase(
                run_dir, model_id, sweep_fn, "expanded", "float16", holder
            )
            results[model_id]["expanded_completeness"] = _write_completeness(
                run_dir, model_id, "float16",
                ["behavior", "primary", "core", "expanded"], label="full",
            )
        except BaseException as exc:
            if model_id == QWEN:
                results[model_id]["expanded_failure"] = _error_blob(exc)
                raise
            failures.append(model_id)
            results[model_id]["status"] = "infrastructure_failed"
            results[model_id]["expanded_failure"] = _error_blob(exc)
            _record_model_failure(run_dir, model_id, exc)

    for model_id, sweep_fn in (
        (QWEN, sweep_1p5b), (CODER, sweep_1p5b), (STAR, sweep_7b)
    ):
        if results.get(model_id, {}).get("status") != "core_complete":
            continue
        try:
            results[model_id]["fp32"] = _run_fp32(
                run_dir, model_id, sweep_fn, holder
            )
        except BaseException as exc:
            if model_id == QWEN:
                results[model_id]["fp32_failure"] = _error_blob(exc)
                raise
            failures.append(model_id)
            results[model_id]["fp32_failure"] = _error_blob(exc)
            _record_model_failure(run_dir, model_id, exc)

    _call_summarize(run_dir.name, None, holder)
    # Final exact validation after all statistics have been written.
    # Completeness failures must not be swallowed: a run is not complete
    # unless every core-complete model validates exactly.
    for model_id, record in results.items():
        if record.get("status") != "core_complete":
            continue
        _write_completeness(
            run_dir, model_id, "float16",
            ["behavior", "primary", "core", "expanded"], label="final",
        )
        if record.get("fp32"):
            _write_completeness(
                run_dir, model_id, "float32", ["fp32"],
                layers=record["fp32"]["layers"], label="fp32-final",
            )

    state = "finished_with_failures" if failures else "complete"
    _status(
        run_dir, lease_holder=holder, state=state, phase="done",
        models=results, infrastructure_failures=sorted(set(failures)),
    )
    _write_overnight_report(run_dir)
    return {"state": state, "models": results}


@app.function(
    cpu=2, memory=4096, timeout=24 * 60 * 60,
    volumes=COMMON_VOLUMES, **FN_KW,
)
def run_all_gated(run_id: str = RUN_ID_DEFAULT, resume: bool = True):
    """Detached sequential controller. It never requests a GPU itself."""
    _prep()
    _reload()
    from pipeline import patching as P
    run_dir = _run_dir(run_id)
    manifest = _assert_manifest(run_dir, run_id)
    _require_smoke(run_dir, manifest)
    ident = _identity()
    holder = ident.get("function_call_id")
    if not holder:
        raise P.PatchingError("Modal supplied no controller function-call ID")
    if not resume and any(
        p.exists() for p in (
            run_dir / "chunk_index.json", run_dir / "chunks", run_dir / "probes"
        )
    ):
        raise P.PatchingError("--no-resume requires a fresh scientific run ID")

    P.acquire_lease(
        run_dir / "lease.json", holder=holder, function_id=holder,
        # Modal may resume a preempted invocation with the same call ID.
        # A distinct concurrently launched controller has a different ID and
        # is still rejected while this lease is fresh.
        allow_same=True,
    )
    lease = _read_json(run_dir / "lease.json")
    lease.update({
        "task_id": ident.get("task_id"), "input_id": ident.get("input_id"),
        "stale_after_s": LEASE_STALE_SECONDS,
    })
    P.write_json_atomic(run_dir / "lease.json", lease)
    started = time.time()
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop, args=(run_dir, holder, stop), daemon=True,
    )
    heartbeat.start()
    _status(
        run_dir, lease_holder=holder, state="running", phase="controller",
        controller_started_at=started, controller=ident, resume=resume,
    )

    results: dict[str, Any] = {}
    outcome = "ok"
    payload: dict[str, Any] | None = None
    try:
        payload = _run_gated_controller(run_dir, holder, resume)
        results = payload.get("models") or {}
        return payload
    except BaseException as exc:
        outcome = "failed"
        status = _read_json(run_dir / "status.json", {})
        results = status.get("models") or {}
        _status(
            run_dir, lease_holder=holder, state="failed", phase="controller",
            models=results, error=_error_blob(exc),
        )
        try:
            _write_overnight_report(run_dir)
        except BaseException as report_exc:
            print(f"failed to write overnight report: {report_exc}", flush=True)
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=5)
        _record_elapsed(
            run_dir, "controller", "cpu2-4g", started, outcome=outcome,
            details={"models": list(results), "resume": resume},
        )
        try:
            if _read_json(run_dir / "status.json", {}).get("state") in {
                "complete", "finished_with_failures", "stopped_qwen_null", "failed",
            }:
                _write_overnight_report(run_dir)
        except BaseException as report_exc:
            print(f"failed to refresh final overnight report: {report_exc}", flush=True)
        try:
            with _VOLUME_LOCK:
                P.release_lease(run_dir / "lease.json", holder)
                _commit()
        except BaseException as lease_exc:
            print(f"failed to release controller lease: {lease_exc}", flush=True)


@app.local_entrypoint()
def main(stage: str = "preflight", run_id: str = RUN_ID_DEFAULT, resume: bool = True):
    if stage == "preflight":
        print(preflight.remote(run_id))
    elif stage == "smoke":
        print(qwen_smoke.remote(run_id))
    elif stage == "all":
        print(run_all_gated.remote(run_id, resume))
    else:
        raise SystemExit(f"unknown stage {stage!r}; choose preflight, smoke, or all")
