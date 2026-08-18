"""class_struct on Modal.

GPU dumps hidden states; CPU fits sklearn probes. Do not keep an L4 attached
during logistic regression — that was burning the $30 credit.

Qwen 1.5B + Coder 1.5B perturbation are already on the volume. Remaining:

  python3 scripts/modal_all_and_pull.py

That skips finished jobs, then for each remaining job:
  1. L4 extract  2. CPU probe (8 cores, no GPU)

Pull:
  modal volume get --force class-struct-data /results ./results/modal
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_NAME = "class-struct-probe"
VOLUME_NAME = "class-struct-data"
DEFAULT_MODEL = "bigcode/starcoder2-7b"
MODELS = (
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-Coder-1.5B",
    "bigcode/starcoder2-7b",
)
# Skip Java/C# on the first pass (GFG wrapper + RAM). PHP has zero labels.
CROSSLANG_LANGS = ["C++", "Javascript", "C"]

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
    )
    .pip_install(
        "transformers==5.8.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "tqdm>=4.64",
        "datasets>=2.18",
        "huggingface_hub>=0.23",
    )
    .add_local_dir(
        str(REPO),
        remote_path="/root/mech-interp",
        ignore=[
            "**/.git/**",
            "**/.venv/**",
            "**/results/**",
            "**/notebooks/**",
            "**/XLCoST_data/**",
            "**/outputs/**",
            "**/.mypy_cache/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _prep_env():
    import os

    os.environ.setdefault("HF_HOME", "/data/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
    os.environ.setdefault("HIDDEN_CACHE_DIR", "/data/hidden_cache")
    os.makedirs(os.environ["HIDDEN_CACHE_DIR"], exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


@app.function(
    volumes={"/data": vol},
    secrets=[hf_secret],
    timeout=30 * 60,
)
def download_dataset():
    from huggingface_hub import snapshot_download

    _prep_env()
    snapshot_download(
        "dhyuti-n/xlcost-variable-roles",
        repo_type="dataset",
        local_dir="/data/dataset",
    )
    vol.commit()
    print("dataset ready under /data/dataset")


def _dump_root(model: str, mode: str) -> str:
    return f"/data/hidden/{model.split('/')[-1]}/{mode}"


def _experiment_cmd(mode: str, model: str, phase: str) -> list[str]:
    import sys

    cmd = [
        sys.executable, "-m", "pipeline.run_experiment", mode,
        "--role", "class_struct",
        "--model", model,
        "--dataset", "/data/dataset",
        "--split", "train",
        "--out", "/data/results",
        "--phase", phase,
        "--dump-root", _dump_root(model, mode),
    ]
    if mode == "crosslang":
        cmd += ["--languages", *CROSSLANG_LANGS]
    return cmd


def _checkpoint_loop(stop):
    import threading

    def _ping():
        while not stop.wait(120):
            try:
                vol.commit()
                print("volume checkpoint ok", flush=True)
            except Exception as exc:
                print(f"volume checkpoint failed: {exc}", flush=True)

    threading.Thread(target=_ping, daemon=True).start()


def _run_phase(mode: str, model: str, phase: str):
    import os
    import subprocess
    import sys
    import threading

    _prep_env()
    os.chdir("/root/mech-interp")
    sys.path.insert(0, "/root/mech-interp")
    stop = threading.Event()
    _checkpoint_loop(stop)
    cmd = _experiment_cmd(mode, model, phase)
    print("running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd)
    finally:
        stop.set()
        vol.commit()
    print(f"finished {phase} {mode} {model}", flush=True)


@app.function(
    gpu="L4",
    volumes={"/data": vol},
    secrets=[hf_secret],
    timeout=6 * 60 * 60,
    memory=65536,
)
def extract_hidden(mode: str, model: str):
    _run_phase(mode, model, "extract")


@app.function(
    cpu=8,
    volumes={"/data": vol},
    timeout=12 * 60 * 60,
    memory=32768,
)
def fit_probes(mode: str, model: str):
    _run_phase(mode, model, "probe")


def _csv_data_rows(path: str) -> int:
    import os

    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


def _job_done(mode: str, model: str) -> bool:
    slug = model.split("/")[-1]
    base = f"/data/results/{slug}/class_struct/{mode}"
    if mode == "perturbation":
        return _csv_data_rows(f"{base}/summary.csv") >= 6
    return _csv_data_rows(f"{base}/crosslang.csv") >= 4


@app.function(
    volumes={"/data": vol},
    timeout=24 * 60 * 60,
)
def run_all():
    """Skip finished jobs. GPU extract then CPU probe for each remaining job."""
    import json
    import os
    import traceback

    os.makedirs("/data/results", exist_ok=True)
    status_path = "/data/results/sweep_status.json"
    status = {"completed": [], "failed": [], "skipped": [], "running": None}

    def save():
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        vol.commit()

    for mode in ("perturbation", "crosslang"):
        for model in MODELS:
            if _job_done(mode, model):
                print(f"SKIP {mode} {model} (already complete)", flush=True)
                status["skipped"].append({"mode": mode, "model": model})
                save()
                continue
            print(f"=== {mode} {model} ===", flush=True)
            status["running"] = {"mode": mode, "model": model}
            save()
            try:
                extract_hidden.remote(mode, model)
                fit_probes.remote(mode, model)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"FAILED {mode} {model}: {msg}", flush=True)
                traceback.print_exc()
                status["failed"].append({"mode": mode, "model": model, "error": msg})
                status["running"] = None
                save()
                low = msg.lower()
                if any(s in low for s in ("credit", "quota", "out of money", "insufficient")):
                    print("credits/quota exhausted; stopping sweep", flush=True)
                    break
                continue
            status["completed"].append({"mode": mode, "model": model})
            status["running"] = None
            save()
        else:
            continue
        break
    print("sweep stopped", flush=True)
    print(json.dumps(status, indent=2), flush=True)


@app.local_entrypoint()
def main(stage: str = "all", model: str = DEFAULT_MODEL):
    """stage: download | perturbation | crosslang | all"""
    if stage not in {"download", "perturbation", "crosslang", "all"}:
        raise SystemExit(f"unknown stage {stage!r}")
    if stage == "all":
        call = run_all.spawn()
        print(f"spawned run_all {getattr(call, 'object_id', call)}; laptop can disconnect", flush=True)
        return
    if stage == "download":
        download_dataset.remote()
        return
    extract_hidden.remote(stage, model)
    fit_probes.remote(stage, model)
    print(f"finished {stage} {model}", flush=True)
