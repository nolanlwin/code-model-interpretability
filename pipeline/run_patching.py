"""CLI for class_struct activation patching (local/remote-neutral).

Subcommands: validate, estimate, smoke, sweep, summarize, check-completeness,
generate-prompts, fit-probe.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import patching as P
from .patching_prompts import (
    default_eval_path,
    default_smoke_path,
    generate_frozen_files,
    load_jsonl,
    sha256_file,
    validate_frozen_files,
    validate_python_semantics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "class-struct-python-v1-20260819"


def _software_versions() -> dict:
    from importlib import metadata

    out = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "modal"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def _cfg_hashes() -> tuple[str, str, str, dict]:
    validate_frozen_files()
    eval_path, smoke_path = default_eval_path(), default_smoke_path()
    prompt_sha = sha256_file(eval_path)
    smoke_sha = sha256_file(smoke_path)
    code_sha = P.bundle_sha256(P.default_bundle_paths())
    cfg = P.configuration_dict(prompt_sha, smoke_sha, code_sha)
    return prompt_sha, smoke_sha, P.configuration_sha256(cfg), cfg


def cmd_generate(_args):
    info = generate_frozen_files()
    print(json.dumps(info, indent=2))


def cmd_validate(args):
    prompt_sha, smoke_sha, config_sha, cfg = _cfg_hashes()
    eval_rows = load_jsonl(default_eval_path())
    smoke_rows = load_jsonl(default_smoke_path())
    if len(eval_rows) != 288:
        raise SystemExit(f"eval rows {len(eval_rows)} != 288")
    if len(smoke_rows) != 8:
        raise SystemExit(f"smoke rows {len(smoke_rows)} != 8")
    if prompt_sha != P.EVAL_PROMPT_SHA256 or smoke_sha != P.SMOKE_PROMPT_SHA256:
        raise SystemExit("frozen prompt hash mismatch")
    eval_ids = {r["pair_id"] for r in eval_rows}
    if any(r["pair_id"] in eval_ids for r in smoke_rows):
        raise SystemExit("smoke/eval pair_id overlap")
    for row in eval_rows + smoke_rows:
        validate_python_semantics(row)
    tokenizers = args.tokenizers or []
    report = {
        "prompt_sha256": prompt_sha,
        "smoke_prompt_sha256": smoke_sha,
        "configuration_sha256": config_sha,
        "n_eval": len(eval_rows),
        "n_smoke": len(smoke_rows),
        "python_semantics": "ok",
        "tokenizer_models": [],
    }
    for model_id in tokenizers:
        tok = _load_tokenizer(model_id, local_files_only=not args.allow_download)
        for row in eval_rows + smoke_rows:
            P.validate_pair_tokenizer(tok, row, model_id)
        true_id, false_id = P.discover_completion_ids(tok, model_id)
        report["tokenizer_models"].append(
            {"model_id": model_id, "true_id": true_id, "false_id": false_id}
        )
    print(json.dumps(report, indent=2))
    print("validate ok")


def _load_tokenizer(model_id: str, local_files_only: bool = True):
    tok = P.load_tokenizer_pinned(model_id, local_files_only=local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def cmd_estimate(_args):
    prompt_sha, smoke_sha, config_sha, cfg = _cfg_hashes()
    estimates = P.estimate_all()
    out = {
        "experiment": P.EXPERIMENT,
        "prompt_sha256": prompt_sha,
        "configuration_sha256": config_sha,
        "models": estimates,
        "fp16_staged_gpu_only_conservative_usd": sum(
            v["fp16_staged_gpu_only_conservative_usd"] for v in estimates.values()
        ),
        "fp32_top3_gpu_only_conservative_usd": sum(
            v["fp32_top3_conservative_usd"] for v in estimates.values()
        ),
        "controller_hard_stop_usd": P.CONTROLLER_HARD_STOP_USD,
        "cost_scope_note": (
            "GPU-only modeled estimates exclude CPU/RAM, probe extraction/fitting, "
            "smoke, image startup, and preemption; the detached controller's $50 "
            "hard stop is the end-to-end experiment envelope."
        ),
    }
    print(json.dumps(out, indent=2))
    for mid, est in estimates.items():
        print(
            f"{mid}: pairs={est['n_pairs']} layers={est['n_layers']} "
            f"cells={est['n_intervention_cells']} item_forwards={est['item_forwards']} "
            f"staged_item_forwards={est['staged_item_forwards']} "
            f"staged_batched={est['staged_batched_forwards']} ceiling={est['ceiling']} "
            f"fp16_gpu_only~${est['fp16_staged_gpu_only_conservative_usd']:.2f} "
            f"fp32_top3={est['fp32_top3_item_forwards']} "
            f"fp32_gpu_only~${est['fp32_top3_conservative_usd']:.2f}"
        )


def cmd_check_completeness(args):
    run_dir = Path(args.run_dir)
    model_id = args.model
    prompt_sha, _, config_sha, _ = _cfg_hashes()
    pairs = load_jsonl(default_eval_path())
    schedule = getattr(args, "cells", "full")
    layers = getattr(args, "layers", None)
    cells = P.schedule_cells(model_id, schedule, layers=layers)
    expected = P.expected_intervention_keys(
        model_id, pairs, prompt_sha, config_sha, args.dtype, cells=cells,
    )
    rows = P.ChunkStore(run_dir).load_valid_rows(
        model_id=model_id, dtype=args.dtype,
        configuration_sha256=config_sha, strict=True,
    )
    expected_set = set(expected)
    scoped = [row for row in rows if P.primary_key(row) in expected_set]
    report = P.completeness_report(expected, scoped)
    report["n_out_of_scope"] = len(rows) - len(scoped)
    namespace = P.ChunkStore._safe_component
    out = (
        run_dir / "completeness" / namespace(model_id) / namespace(args.dtype)
        / config_sha[:16] / f"{namespace(schedule)}.json"
    )
    P.write_json_atomic(out, report)
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit("completeness failed")


def evaluate_model_results(run_dir: Path, model_id: str, *,
                           dtype: str = "float16") -> dict:
    """CPU-only behavioral, probe-OOD, and primary causal gate evaluation."""
    run_dir = Path(run_dir)
    prompt_sha, _, config_sha, _ = _cfg_hashes()
    pairs = load_jsonl(default_eval_path())
    pair_ids = [pair["pair_id"] for pair in pairs]
    rows = P.ChunkStore(run_dir).load_valid_rows(
        model_id=model_id, dtype=dtype,
        configuration_sha256=config_sha, strict=True,
    )
    rows = [row for row in rows if row.get("prompt_sha256") == prompt_sha]
    gates = P.evaluate_gates(
        rows, model_id, dtype=dtype, configuration_sha256=config_sha,
        expected_pair_ids=pair_ids,
    )
    namespace = P.ChunkStore._safe_component
    path = (
        run_dir / "diagnostics" / namespace(model_id) / namespace(dtype)
        / config_sha[:16] / "gate_report.json"
    )
    P.write_json_atomic(path, gates)
    return gates


def cmd_summarize(args):
    run_dir = Path(args.run_dir)
    rows = P.ChunkStore(run_dir).load_valid_rows(strict=True)
    if args.model:
        rows = [r for r in rows if r["model_id"] == args.model]
    n_boot = int(getattr(args, "n_boot", P.N_BOOT))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["configuration_sha256"], row["model_id"], row["model_revision"],
            row["dtype"], row["prompt_sha256"],
        )].append(row)
    index = {"summaries": []}
    for (config_sha, model_id, revision, dtype, prompt_sha), scope in sorted(grouped.items()):
        # Smoke and paper prompts must never share a statistical table.
        kind = "eval" if prompt_sha == P.EVAL_PROMPT_SHA256 else "smoke"
        namespace = P.ChunkStore._safe_component
        summary_dir = (
            run_dir / "summaries" / namespace(model_id) / namespace(dtype)
            / config_sha[:16] / kind
        )
        summary_dir.mkdir(parents=True, exist_ok=True)
        baseline_clean = [
            row for row in scope if row.get("layer") == -1
            and row.get("span") == "baseline_class" and row.get("control") == "unpatched"
        ]
        baseline_function = [
            row for row in scope if row.get("layer") == -1
            and row.get("span") == "baseline_function" and row.get("control") == "unpatched"
        ]
        pair_ids = {row["pair_id"] for row in baseline_clean + baseline_function}
        baselines = P.exact_pair_join(
            {"clean": baseline_clean, "function": baseline_function},
            expected_pair_ids=pair_ids,
        ) if pair_ids else []
        baseline_by_pair = {
            item["clean"]["pair_id"]: item for item in baselines
        }
        cell_groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in scope:
            if row.get("layer", -1) < 0 or row.get("signed_effect") is None:
                continue
            cell_groups[(row["layer"], row["span"], row["direction"], row["control"])].append(row)
        records = []
        sensitivity = {}
        for (layer, span, direction, control), cell_rows in sorted(cell_groups.items()):
            effects = np.asarray([row["signed_effect"] for row in cell_rows], dtype=np.float64)
            clusters = [row["cluster_id"] for row in cell_rows]
            ci = P.clustered_mean_ci(effects, clusters, n_boot=n_boot)
            record = {
                "configuration_sha256": config_sha,
                "prompt_sha256": prompt_sha,
                "model_id": model_id,
                "model_revision": revision,
                "dtype": dtype,
                "layer": layer,
                "span": span,
                "direction": direction,
                "control": control,
                "n": len(cell_rows),
                "n_clusters": ci["n_clusters"],
                "mean_effect": ci["point"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "recovery": None,
                "recovery_ci_low": None,
                "recovery_ci_high": None,
                "recovery_denominator_crosses_zero": None,
                "minus_placebo": None,
                "minus_placebo_ci_low": None,
                "minus_placebo_ci_high": None,
                "minus_random": None,
                "minus_random_ci_low": None,
                "minus_random_ci_high": None,
            }
            if all(row["pair_id"] in baseline_by_pair for row in cell_rows):
                gaps = [
                    baseline_by_pair[row["pair_id"]]["clean"]["source_D"]
                    - baseline_by_pair[row["pair_id"]]["function"]["source_D"]
                    for row in cell_rows
                ]
                recovery = P.clustered_ratio_ci(
                    effects, gaps, clusters, n_boot=n_boot,
                )
                record.update({
                    "recovery": recovery["point"],
                    "recovery_ci_low": recovery["ci_low"],
                    "recovery_ci_high": recovery["ci_high"],
                    "recovery_denominator_crosses_zero": recovery["denominator_crosses_zero"],
                })
            if control == "target" and span in P.NAME_SPANS:
                for comparison, selector in (
                    ("placebo", (layer, "placebo", direction, "target")),
                    ("random", (layer, span, direction, "random")),
                ):
                    other = cell_groups.get(selector, [])
                    if not other:
                        continue
                    joined = P.exact_pair_join(
                        {"target": cell_rows, comparison: other},
                        expected_pair_ids=[row["pair_id"] for row in cell_rows],
                    )
                    delta = [
                        item["target"]["signed_effect"] - item[comparison]["signed_effect"]
                        for item in joined
                    ]
                    delta_clusters = [item["target"]["cluster_id"] for item in joined]
                    delta_ci = P.clustered_mean_ci(delta, delta_clusters, n_boot=n_boot)
                    record[f"minus_{comparison}"] = delta_ci["point"]
                    record[f"minus_{comparison}_ci_low"] = delta_ci["ci_low"]
                    record[f"minus_{comparison}_ci_high"] = delta_ci["ci_high"]
            records.append(record)
            sensitivity[f"L{layer}:{span}:{direction}:{control}"] = P.leave_one_name_out(
                effects, [row["name"] for row in cell_rows],
            )
        summary_path = summary_dir / "summary.csv"
        fieldnames = list(records[0]) if records else [
            "configuration_sha256", "prompt_sha256", "model_id", "model_revision",
            "dtype", "layer", "span", "direction", "control", "n",
        ]
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        P.write_json_atomic(summary_dir / "summary.json", records)
        P.write_json_atomic(summary_dir / "leave_one_name_out.json", sensitivity)

        probe_records = []
        probe_summary = {"available": False}
        if model_id in P.MODELS:
            layer = P.MODELS[model_id]["probe_index"]
            denoise = cell_groups.get((layer, "query_name", "denoise", "target"), [])
            noise = cell_groups.get((layer, "query_name", "noise", "target"), [])
            if denoise and noise:
                joined = P.exact_pair_join(
                    {"denoise": denoise, "noise": noise},
                    expected_pair_ids=[row["pair_id"] for row in denoise],
                )
                for item in joined:
                    drow, nrow = item["denoise"], item["noise"]
                    if drow.get("source_probe_margin") is None:
                        continue
                    denoise_probe_movement = (
                        drow["patched_probe_margin"] - drow["destination_probe_margin"]
                    )
                    noise_probe_movement = (
                        nrow["destination_probe_margin"] - nrow["patched_probe_margin"]
                    )
                    probe_records.append({
                        "configuration_sha256": config_sha,
                        "model_id": model_id,
                        "model_revision": revision,
                        "dtype": dtype,
                        "layer": layer,
                        "pair_id": drow["pair_id"],
                        "cluster_id": drow["cluster_id"],
                        "name": drow["name"],
                        "probe_gap": drow["source_probe_margin"] - drow["destination_probe_margin"],
                        "denoise_source_probe_margin": drow["source_probe_margin"],
                        "denoise_destination_probe_margin": drow["destination_probe_margin"],
                        "denoise_patched_probe_margin": drow["patched_probe_margin"],
                        "denoise_probe_movement": denoise_probe_movement,
                        "noise_source_probe_margin": nrow["source_probe_margin"],
                        "noise_destination_probe_margin": nrow["destination_probe_margin"],
                        "noise_patched_probe_margin": nrow["patched_probe_margin"],
                        "noise_probe_movement": noise_probe_movement,
                        "symmetric_probe_movement": (
                            denoise_probe_movement + noise_probe_movement
                        ) / 2,
                        "symmetric_behavior": (drow["signed_effect"] + nrow["signed_effect"]) / 2,
                    })
                if probe_records:
                    probe_summary = {
                        "available": True,
                        "baseline_probe_gap_vs_behavior_spearman": P.clustered_spearman(
                            [row["probe_gap"] for row in probe_records],
                            [row["symmetric_behavior"] for row in probe_records],
                            [row["cluster_id"] for row in probe_records],
                            n_boot=n_boot,
                        ),
                        "patched_probe_movement_vs_behavior_spearman": P.clustered_spearman(
                            [row["symmetric_probe_movement"] for row in probe_records],
                            [row["symmetric_behavior"] for row in probe_records],
                            [row["cluster_id"] for row in probe_records],
                            n_boot=n_boot,
                        ),
                        "interpretation_note": (
                            "Patched-probe movement is descriptive because full-residual "
                            "replacement directly changes the probed vector."
                        ),
                    }
        probe_path = summary_dir / "probe_link.csv"
        probe_fields = [
            "configuration_sha256", "model_id", "model_revision", "dtype", "layer",
            "pair_id", "cluster_id", "name", "probe_gap",
            "denoise_source_probe_margin", "denoise_destination_probe_margin",
            "denoise_patched_probe_margin", "denoise_probe_movement",
            "noise_source_probe_margin", "noise_destination_probe_margin",
            "noise_patched_probe_margin", "noise_probe_movement",
            "symmetric_probe_movement", "symmetric_behavior",
        ]
        with probe_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=probe_fields)
            writer.writeheader()
            writer.writerows(probe_records)
        P.write_json_atomic(summary_dir / "probe_link_summary.json", probe_summary)
        index["summaries"].append({
            "model_id": model_id, "model_revision": revision, "dtype": dtype,
            "configuration_sha256": config_sha, "prompt_sha256": prompt_sha,
            "kind": kind,
            "path": summary_path.relative_to(run_dir).as_posix(),
        })
    index_path = run_dir / "summaries" / "index.json"
    P.write_json_atomic(index_path, index)
    print(f"wrote {len(index['summaries'])} isolated summary groups to {index_path}")


def cmd_evaluate(args):
    report = evaluate_model_results(Path(args.run_dir), args.model, dtype=args.dtype)
    print(json.dumps(report, indent=2))
    if not report.get("behavior", {}).get("pass", False):
        raise SystemExit("behavior gate failed")
    if not report.get("probe_ood", {}).get("pass", False):
        raise SystemExit("probe OOD gate failed")
    if not report.get("causal", {}).get("pass", False):
        raise SystemExit("causal gate failed")


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("cpu")  # protocol is CUDA fp16; CPU for local tests
    return torch.device("cpu")


def _indices_from_view(view: dict) -> dict:
    return {
        "query_name": view["indices"]["query_name_char_span"],
        "declaration_name": view["indices"]["declaration_name_char_span"],
        "placebo": view["indices"]["placebo_char_span"],
    }


def run_source_block(model, tokenizer, adapter, pairs, true_id, false_id, device,
                     *, layers: list[int] | tuple[int, ...], batch_size: int,
                     probe=None, repeat_logits: bool = True,
                     baseline_reference: dict | None = None):
    """Microbatched clean/corrupt sources retaining only requested span vectors.

    Full hidden-state tuples remain on the accelerator only long enough to copy
    three vectors for each requested layer.  The drift repeat requests no hidden
    states and transfers only the True/False logit difference.
    """
    import torch

    requested_layers = sorted(set(int(layer) for layer in layers))
    if probe is not None:
        requested_layers = sorted(set(requested_layers) | {int(probe["layer"])})
    for layer in requested_layers:
        if not 0 <= layer < adapter.n_hidden:
            raise P.PatchingError(f"source layer {layer} out of range")
    caches: list[dict] = [{} for _ in pairs]
    views = {
        side: [P.pair_token_view(tokenizer, row, side) for row in pairs]
        for side in ("clean", "corrupt")
    }

    for side in ("clean", "corrupt"):
        cursor = 0
        size = max(1, min(int(batch_size), len(pairs)))
        while cursor < len(pairs):
            stop = min(len(pairs), cursor + size)
            indices = list(range(cursor, stop))
            texts = [
                pairs[i]["clean_prompt" if side == "clean" else "corrupt_prompt"]
                for i in indices
            ]
            ids = mask = logits2 = hidden_states = None
            try:
                ids, mask, pad_lens = P.pad_batch(tokenizer, texts, device)
                if requested_layers:
                    logits2, hidden_states = P.capture_hidden_states(
                        model, ids, mask, readout_ids=(true_id, false_id),
                        keep_layers=requested_layers,
                    )
                    d_current = (
                        logits2[:, 0] - logits2[:, 1]
                    ).detach().float().cpu().numpy()
                else:
                    hidden_states = ()
                    d_current = P.forward_logit_diffs(
                        model, ids, mask, true_id, false_id,
                    )
                if repeat_logits:
                    d_repeat = P.forward_logit_diffs(
                        model, ids, mask, true_id, false_id,
                    )
                else:
                    d_repeat = d_current.copy()
                for local_i, global_i in enumerate(indices):
                    idx = _indices_from_view(views[side][global_i])
                    vectors = {}
                    for layer in requested_layers:
                        tensor = hidden_states[layer]
                        if tensor is None:
                            raise P.PatchingError(
                                f"source capture dropped required layer {layer}"
                            )
                        vectors[layer] = {
                            span: tensor[
                                local_i, P.left_pad_index(raw_index, pad_lens[local_i])
                            ].detach().cpu().numpy()
                            for span, raw_index in idx.items()
                        }
                    reference = None
                    if baseline_reference is not None:
                        reference = baseline_reference.get((pairs[global_i]["pair_id"], side))
                    primary_d = float(reference["D"] if reference else d_current[local_i])
                    repeat_d = float(d_repeat[local_i])
                    drift = abs(repeat_d - primary_d)
                    caches[global_i][side] = {
                        "D": primary_d,
                        "D_repeat": repeat_d,
                        "drift": drift,
                        "vectors": vectors,
                        "indices": idx,
                    }
                    if probe is not None:
                        w, b = probe["w_raw"], probe["b_raw"]
                        caches[global_i][side]["probe"] = {
                            span: P.probe_margin(
                                vectors[probe["layer"]][span], w, b,
                            )
                            for span in idx
                        }
                del ids, mask, logits2, hidden_states
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                cursor = stop
            except torch.cuda.OutOfMemoryError as exc:
                ids = mask = logits2 = hidden_states = None
                if not str(device).startswith("cuda") or size == 1:
                    raise P.PatchingError("source forward OOM at batch size 1") from exc
                size = max(1, size // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"source OOM; retrying at batch_size={size}", flush=True)
    return caches


def _halve_until_ok(fn, batch_size: int, min_size: int = 1):
    import torch
    size = batch_size
    while size >= min_size:
        try:
            return fn(size)
        except torch.cuda.OutOfMemoryError:
            if size <= min_size:
                gc.collect()
                torch.cuda.empty_cache()
                raise P.PatchingError("OOM at batch size 1") from None
            next_size = max(min_size, size // 2)
            print(f"OOM at batch_size={size}; retrying at {next_size}", flush=True)
            size = next_size
            gc.collect()
            torch.cuda.empty_cache()
    raise P.PatchingError("invalid adaptive batch size")


def run_intervention_block(
    model, tokenizer, adapter, caches, pairs, cells, true_id, false_id, device,
    batch_size: int, prompt_sha: str, config_sha: str, model_id: str,
    dtype: str, run_id: str, probe=None, allowed_jobs: set | None = None,
) -> list[dict]:
    meta = P.MODELS[model_id]
    rows_out = []
    # group cells by (layer, recipient_side) so one forward can pack many pairs
    by_layer = defaultdict(list)
    for cell in cells:
        by_layer[cell[0]].append(cell)

    for layer, layer_cells in by_layer.items():
        jobs = []
        for i, pair in enumerate(pairs):
            cache = caches[i]
            for _layer, span, direction, control in layer_cells:
                if allowed_jobs is not None and (
                    pair["pair_id"], (_layer, span, direction, control)
                ) not in allowed_jobs:
                    continue
                # choose_source_dest decides recipient
                clean_v = cache["clean"]["vectors"][layer][span]
                corr_v = cache["corrupt"]["vectors"][layer][span]
                source, dest, recip = P.choose_source_dest(direction, control, clean_v, corr_v)
                vec = P.patch_vector(
                    control, source, dest, pair["pair_id"], layer, span,
                    direction, dtype=dtype,
                )
                jobs.append({
                    "i": i, "span": span, "direction": direction, "control": control,
                    "recip": recip, "vec": vec,
                    "token": cache[recip]["indices"][span],
                })

        def run_jobs(bs):
            local_rows = []
            for side in ("clean", "corrupt"):
                side_jobs = [job for job in jobs if job["recip"] == side]
                for start in range(0, len(side_jobs), bs):
                    sub = side_jobs[start:start + bs]
                    texts = [
                        pairs[j["i"]]["clean_prompt" if side == "clean" else "corrupt_prompt"]
                        for j in sub
                    ]
                    input_ids, mask, pad_lens = P.pad_batch(tokenizer, texts, device)
                    specs = []
                    for b, j in enumerate(sub):
                        pos = P.left_pad_index(j["token"], pad_lens[b])
                        specs.append((b, pos, j["vec"]))
                    logits = P.patched_forward(
                        model, adapter, input_ids, mask, layer, specs,
                        readout_ids=(true_id, false_id),
                    )
                    for b, j in enumerate(sub):
                        i = j["i"]
                        pair = pairs[i]
                        d_class = caches[i]["clean"]["D"]
                        d_fn = caches[i]["corrupt"]["D"]
                        d_p = float((logits[b, 0] - logits[b, 1]).detach().float().cpu())
                        effect = P.signed_effect(j["direction"], d_class, d_fn, d_p)
                        src_m = dst_m = pat_m = None
                        if probe is not None and layer == int(probe["layer"]):
                            src_side = "clean" if j["direction"] == "denoise" else "corrupt"
                            dst_side = "corrupt" if src_side == "clean" else "clean"
                            if j["control"] == "same_source":
                                src_side = dst_side = j["recip"]
                            src_m = caches[i][src_side]["probe"][j["span"]]
                            dst_m = caches[i][dst_side]["probe"][j["span"]]
                            # patched margin on replaced residual
                            pat_m = P.probe_margin(j["vec"], probe["w_raw"], probe["b_raw"])
                        if j["control"] == "same_source":
                            source_d = destination_d = caches[i][j["recip"]]["D"]
                        else:
                            source_d = d_class if j["direction"] == "denoise" else d_fn
                            destination_d = d_fn if j["direction"] == "denoise" else d_class
                        local_rows.append(P.make_result_row(
                            run_id=run_id, cluster_id=pair["cluster_id"], name=pair["name"],
                            source_D=source_d, destination_D=destination_d,
                            patched_D=d_p, signed_effect=effect,
                            class_function_gap=P.class_function_gap(d_class, d_fn),
                            source_probe_margin=src_m, destination_probe_margin=dst_m,
                            patched_probe_margin=pat_m,
                            baseline_drift=max(caches[i]["clean"]["drift"], caches[i]["corrupt"]["drift"]),
                            batch_size=bs, prompt_sha256=prompt_sha,
                            configuration_sha256=config_sha, model_id=model_id,
                            model_revision=meta["revision"], dtype=dtype,
                            pair_id=pair["pair_id"], layer=layer, span=j["span"],
                            direction=j["direction"], control=j["control"],
                        ))
                    del logits, input_ids, mask
            return local_rows

        rows_out.extend(_halve_until_ok(run_jobs, batch_size))
    return rows_out


def baseline_rows(caches, pairs, prompt_sha, config_sha, model_id, dtype, run_id):
    meta = P.MODELS[model_id]
    out = []
    for i, pair in enumerate(pairs):
        for side, span in (("clean", "baseline_class"), ("corrupt", "baseline_function")):
            D = caches[i][side]["D"]
            for control, dval in (("unpatched", D), ("drift", caches[i][side]["D_repeat"])):
                out.append(P.make_result_row(
                    run_id=run_id, cluster_id=pair["cluster_id"], name=pair["name"],
                    source_D=dval, destination_D=dval, patched_D=dval,
                    signed_effect=0.0,
                    class_function_gap=P.class_function_gap(
                        caches[i]["clean"]["D"], caches[i]["corrupt"]["D"]),
                    source_probe_margin=(caches[i][side].get("probe") or {}).get("query_name"),
                    source_probe_declaration_margin=(
                        caches[i][side].get("probe") or {}
                    ).get("declaration_name"),
                    destination_probe_margin=(caches[i][side].get("probe") or {}).get("query_name"),
                    patched_probe_margin=(caches[i][side].get("probe") or {}).get("query_name"),
                    baseline_drift=caches[i][side]["drift"],
                    batch_size=len(pairs), prompt_sha256=prompt_sha,
                    configuration_sha256=config_sha, model_id=model_id,
                    model_revision=meta["revision"], dtype=dtype,
                    pair_id=pair["pair_id"], layer=-1, span=span,
                    direction="none", control=control,
                ))
    return out


def run_pairs_phase(
    model_id: str, pairs: list[dict], cells, run_dir: Path, phase: str,
    dtype: str = "float16", run_id: str = DEFAULT_RUN_ID,
    local_files_only: bool = True, probe_path: Path | None = None,
    allow_download: bool = False, prompt_sha: str | None = None,
    checkpoint_callback=None, progress_callback=None,
    verify_hooks: bool = False,
):
    import torch

    eval_sha, smoke_sha, config_sha, cfg = _cfg_hashes()
    prompt_sha = prompt_sha or eval_sha
    P.validate_or_create_manifest(
        run_dir, run_id=run_id, config=cfg,
        metadata={"base_commit": os.environ.get("PATCHING_BASE_COMMIT")},
    )
    device = _device()
    tok, model, adapter = P.load_causal_lm(
        model_id, device, dtype=dtype, local_files_only=local_files_only and not allow_download,
    )
    adapter.assert_expected(model_id)
    true_id, false_id = P.discover_completion_ids(tok, model_id)
    probe = None
    if probe_path and Path(probe_path).is_file():
        probe = P.load_probe_artifact(
            probe_path, model_id,
            expected_configuration_sha256=config_sha,
            expected_code_sha256=cfg["code_sha256"],
        )
        if int(np.asarray(probe["w_raw"]).size) != adapter.hidden_size:
            raise P.PatchingError(
                f"probe width {np.asarray(probe['w_raw']).size} != model hidden size {adapter.hidden_size}"
            )
    if verify_hooks and pairs:
        ids, mask, _ = P.pad_batch(tokenizer=tok, texts=[pairs[0]["clean_prompt"]], device=device)
        diffs = P.compare_hook_to_hidden_states(model, adapter, ids, mask)
        print(f"verified hook mapping {model_id}: max_diff={max(diffs.values()):.6g}", flush=True)
        del ids, mask
    store = P.ChunkStore(run_dir)
    mb = 2 if dtype == "float32" and "7b" in model_id else P.MODELS[model_id]["microbatch"]
    present_rows = store.load_valid_rows(
        model_id=model_id, dtype=dtype, configuration_sha256=config_sha,
    )
    present = {P.primary_key(r) for r in present_rows}
    baseline_reference = {}
    for row in present_rows:
        if row.get("layer") != -1 or row.get("control") != "unpatched":
            continue
        side = "clean" if row.get("span") == "baseline_class" else "corrupt"
        baseline_reference[(row["pair_id"], side)] = {"D": row["source_D"]}
    n_blocks = int(np.ceil(len(pairs) / P.BLOCK_SIZE))
    for b in range(n_blocks):
        block = pairs[b * P.BLOCK_SIZE:(b + 1) * P.BLOCK_SIZE]
        missing_baseline = any(
            (prompt_sha, config_sha, model_id, P.MODELS[model_id]["revision"], dtype,
             pair["pair_id"], -1, span, "none", control, P.RANDOM_SEED) not in present
            for pair in block
            for span, control in (
                ("baseline_class", "unpatched"), ("baseline_function", "unpatched"),
                ("baseline_class", "drift"), ("baseline_function", "drift"),
            )
        )
        missing_by_layer: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
        for pair_i, pair in enumerate(block):
            for cell in cells:
                key = (
                    prompt_sha, config_sha, model_id, P.MODELS[model_id]["revision"],
                    dtype, pair["pair_id"], cell[0], cell[1], cell[2], cell[3],
                    P.RANDOM_SEED,
                )
                if key not in present:
                    missing_by_layer[cell[0]].append((pair_i, cell))
        if not missing_baseline and not missing_by_layer:
            print(f"skip {phase} block {b} (complete)", flush=True)
            continue
        needed_layers = sorted(missing_by_layer)
        caches = run_source_block(
            model, tok, adapter, block, true_id, false_id, device,
            layers=needed_layers, batch_size=mb, probe=probe,
            repeat_logits=missing_baseline,
            baseline_reference=baseline_reference or None,
        )
        if missing_baseline:
            base = baseline_rows(
                caches, block, prompt_sha, config_sha, model_id, dtype, run_id,
            )
            base = [row for row in base if P.primary_key(row) not in present]
            rec = store.write_block(
                "behavior", model_id, dtype, config_sha, -1, b, base,
            )
            present.update(P.primary_key(row) for row in base)
            if checkpoint_callback:
                checkpoint_callback(rec)
            if progress_callback:
                progress_callback(rec)
        for layer in needed_layers:
            jobs = missing_by_layer[layer]
            pair_indices = sorted({pair_i for pair_i, _ in jobs})
            subset_pairs = [block[i] for i in pair_indices]
            subset_caches = [caches[i] for i in pair_indices]
            allowed = {(pair_i, cell) for pair_i, cell in jobs}
            cells_for_subset = sorted({cell for _, cell in jobs})
            inter = run_intervention_block(
                model, tok, adapter, subset_caches, subset_pairs,
                cells_for_subset, true_id, false_id, device, mb,
                prompt_sha, config_sha, model_id, dtype, run_id, probe,
                allowed_jobs={
                    (block[pair_i]["pair_id"], cell) for pair_i, cell in allowed
                },
            )
            wanted = {
                (
                    block[pair_i]["pair_id"], cell[0], cell[1], cell[2], cell[3]
                )
                for pair_i, cell in allowed
            }
            inter = [
                row for row in inter
                if (row["pair_id"], row["layer"], row["span"],
                    row["direction"], row["control"]) in wanted
                and P.primary_key(row) not in present
            ]
            rec = store.write_block(
                phase, model_id, dtype, config_sha, layer, b, inter,
            )
            present.update(P.primary_key(row) for row in inter)
            if checkpoint_callback:
                checkpoint_callback(rec)
            if progress_callback:
                progress_callback(rec)
            print(
                f"wrote {phase} {model_id} {dtype} layer={layer} block={b} n={len(inter)}",
                flush=True,
            )
        del caches
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return store


def cmd_smoke(args):
    pairs = load_jsonl(default_smoke_path())
    model_id = args.model
    probe_i = P.MODELS[model_id]["probe_index"]
    cells = P.smoke_cells(probe_i)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    _, smoke_sha, config_sha, _ = _cfg_hashes()
    run_pairs_phase(
        model_id, pairs, cells, run_dir, phase="smoke", dtype="float16",
        run_id=args.run_id, allow_download=args.allow_download,
        probe_path=args.probe,
        prompt_sha=smoke_sha,
        checkpoint_callback=getattr(args, "checkpoint_callback", None),
        progress_callback=getattr(args, "progress_callback", None),
        verify_hooks=getattr(args, "verify_hooks", False),
    )
    rows = P.ChunkStore(run_dir).load_valid_rows(
        model_id=model_id, dtype="float16", configuration_sha256=config_sha,
    )
    rows = [row for row in rows if row.get("prompt_sha256") == smoke_sha]
    gate = P.smoke_gate(
        rows, probe_index=probe_i,
        expected_pair_ids=[pair["pair_id"] for pair in pairs],
    )
    P.write_json_atomic(run_dir / "gate_report.json", {"smoke": gate})
    print(json.dumps(gate, indent=2))
    if not gate["pass"]:
        raise SystemExit("smoke gate failed")


def cmd_sweep(args):
    pairs = load_jsonl(default_eval_path())
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    model_id = args.model
    layers = getattr(args, "layers", None)
    cells = P.schedule_cells(model_id, args.cells, layers=layers)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_pairs_phase(
        model_id, pairs, cells, run_dir, phase=args.cells, dtype=args.dtype,
        run_id=args.run_id, allow_download=args.allow_download, probe_path=args.probe,
        checkpoint_callback=getattr(args, "checkpoint_callback", None),
        progress_callback=getattr(args, "progress_callback", None),
        verify_hooks=getattr(args, "verify_hooks", False),
    )


def extract_probe_layer(model_id: str, dataset_dir: str, out_dir: Path,
                        split: str = "train", allow_download: bool = False):
    """Extract only the preregistered hidden index for Python baseline class_struct."""
    import json
    import torch

    from .probing import MAX_SEQ_LEN, build_token_dataset

    meta = P.MODELS[model_id]
    layer = meta["probe_index"]
    device = _device()
    tok, model, adapter = P.load_causal_lm(
        model_id, device, dtype="float16",
        local_files_only=not allow_download,
    )
    adapter.assert_expected(model_id)
    path = os.path.join(dataset_dir, "python_perturbations", f"{split}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("language") != "Python" or row.get("strategy") != "baseline":
                continue
            if not (row.get("roles") or {}).get("class_struct"):
                continue
            rows.append(row)
    if len(rows) != 400:
        raise P.PatchingError(
            f"probe protocol requires exactly 400 Python baseline class_struct programs; got {len(rows)}"
        )
    source_program_ids = [
        str(next((row[key] for key in ("program_id", "problem_id", "idx") if row.get(key) is not None), ""))
        for row in rows
    ]
    if any(not value for value in source_program_ids) or len(set(source_program_ids)) != 400:
        raise P.PatchingError("probe source must contain 400 unique non-empty program IDs")
    data, skipped = build_token_dataset(rows, "class_struct", tok)
    if skipped or len(data) != 400:
        raise P.PatchingError(
            f"probe tokenization changed the frozen cohort: {len(data)} kept, {skipped} skipped"
        )
    print(f"probe extract {model_id}: {len(data)} programs ({skipped} skipped)", flush=True)
    probe_ids = tok("x", return_tensors="pt")["input_ids"][0].tolist()
    leading = len(probe_ids) - len(tok.tokenize("x"))
    hidden, labels, programs = [], [], []
    with torch.inference_mode():
        for sample in data:
            enc = tok(sample["code"], return_tensors="pt", truncation=True,
                      max_length=MAX_SEQ_LEN, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            try:
                out = model(**enc, use_cache=False, output_hidden_states=True, logits_to_keep=1)
            except TypeError:
                out = model(**enc, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states[layer][0]
            n_content = enc["input_ids"].shape[1] - leading
            lab = sample["labels"][:n_content]
            n = len(lab)
            if n == 0:
                continue
            vec = hs[leading:leading + n].half().cpu().numpy()
            hidden.append(vec)
            labels.extend(lab)
            programs.extend([sample["program_id"]] * n)
            del out
    X = np.concatenate(hidden, axis=0)
    y = np.asarray(labels)
    g = np.asarray(programs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "hidden.npy", X)
    np.save(out_dir / "labels.npy", y)
    np.save(out_dir / "programs.npy", g)
    extraction_meta = {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": meta["revision"],
        "layer": layer,
        "hidden_size": int(X.shape[1]),
        "dataset_revision": P.DATASET_REVISION,
        "dataset_split": split,
        "source_jsonl_sha256": sha256_file(Path(path)),
        "n_programs": int(len(set(g.tolist()))),
        "n_tokens": int(len(y)),
        "source_program_ids_sha256": P.sha256_bytes(
            "\n".join(source_program_ids).encode()
        ),
        "token_program_ids_sha256": P.sha256_bytes(
            "\n".join(map(str, g.tolist())).encode()
        ),
        "hidden_sha256": sha256_file(out_dir / "hidden.npy"),
        "labels_sha256": sha256_file(out_dir / "labels.npy"),
        "programs_sha256": sha256_file(out_dir / "programs.npy"),
        "created_at": time.time(),
    }
    P.write_json_atomic(out_dir / "extraction_meta.json", extraction_meta)
    print(f"saved {X.shape} tokens to {out_dir}", flush=True)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out_dir


def cmd_extract_probe(args):
    extract_probe_layer(args.model, args.dataset, Path(args.out),
                        split=args.split, allow_download=args.allow_download)


def cmd_fit_probe(args):
    """CPU fit from a selected-layer activation dump (npy + labels + programs)."""
    hidden = np.load(args.hidden)
    labels = np.load(args.labels)
    programs = np.load(args.programs, allow_pickle=False)
    hidden_path, labels_path, programs_path = map(
        Path, (args.hidden, args.labels, args.programs),
    )
    extraction_meta_path = hidden_path.parent / "extraction_meta.json"
    if not extraction_meta_path.is_file():
        raise P.PatchingError(f"missing extraction metadata {extraction_meta_path}")
    extraction = json.loads(extraction_meta_path.read_text())
    expected_extract = {
        "model_id": args.model,
        "model_revision": P.MODELS[args.model]["revision"],
        "layer": P.MODELS[args.model]["probe_index"],
        "dataset_revision": P.DATASET_REVISION,
        "n_programs": 400,
    }
    for key, expected in expected_extract.items():
        if extraction.get(key) != expected:
            raise P.PatchingError(
                f"probe extraction metadata {key}={extraction.get(key)!r} != {expected!r}"
            )
    checksums = {
        "hidden_sha256": sha256_file(hidden_path),
        "labels_sha256": sha256_file(labels_path),
        "programs_sha256": sha256_file(programs_path),
    }
    for key, digest in checksums.items():
        if extraction.get(key) != digest:
            raise P.PatchingError(f"probe extraction checksum mismatch for {key}")
    unique_programs = sorted(set(map(str, programs.tolist())))
    if len(unique_programs) != 400:
        raise P.PatchingError(f"probe fit requires 400 programs, got {len(unique_programs)}")
    fit = P.fit_probe_link(hidden, labels, programs)
    prompt_sha, smoke_sha, config_sha, cfg = _cfg_hashes()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    primary = fit["primary"]
    sha = P.save_probe_npz(
        out / "probe_seed0.npz",
        scaler_mean=primary["scaler_mean"], scaler_scale=primary["scaler_scale"],
        coef=primary["coef"], intercept=primary["intercept"], classes=primary["classes"],
    )
    artifact_hashes = {}
    for f in fit["fits"]:
        artifact_hashes[str(f["seed"])] = P.save_probe_npz(
            out / f"probe_seed{f['seed']}.npz",
            scaler_mean=f["scaler_mean"], scaler_scale=f["scaler_scale"],
            coef=f["coef"], intercept=f["intercept"], classes=f["classes"],
        )
    meta = {
        "schema_version": 1,
        "model_id": args.model,
        "model_revision": P.MODELS[args.model]["revision"],
        "layer": P.MODELS[args.model]["probe_index"],
        "hidden_size": int(hidden.shape[1]),
        "dataset_revision": P.DATASET_REVISION,
        "prompt_sha256": prompt_sha,
        "smoke_prompt_sha256": smoke_sha,
        "configuration_sha256": config_sha,
        "code_sha256": cfg["code_sha256"],
        "model_dtype": "float16",
        "base_git_commit": os.environ.get("PATCHING_BASE_GIT_COMMIT", "unknown"),
        "n_programs": len(unique_programs),
        "test_f1_mean": fit["test_f1_mean"],
        "test_acc_mean": fit["test_acc_mean"],
        "per_seed_metrics": [
            {
                "seed": int(item["seed"]),
                "test_f1": float(item["test_f1"]),
                "test_accuracy": float(item["test_acc"]),
            }
            for item in fit["fits"]
        ],
        "artifact_sha256": sha,
        "artifact_sha256_by_seed": artifact_hashes,
        "n_tokens": int(len(labels)),
        "program_ids_sha256": P.sha256_bytes("\n".join(unique_programs).encode()),
        "split_hashes": {
            str(seed): {
                name: P.sha256_bytes(
                    "\n".join(sorted(unique_programs[i] for i in indices)).encode()
                )
                for name, indices in zip(
                    ("train", "validation", "test"),
                    P.hash_split(np.asarray(unique_programs), seed),
                )
            }
            for seed in range(5)
        },
        "extraction": extraction,
        "software": _software_versions(),
        "created_at": time.time(),
        "prior": P.probe_matches_prior(fit, args.model),
    }
    P.write_json_atomic(out / "probe_meta.json", meta)
    print(json.dumps(meta, indent=2))
    if args.strict_prior and not meta["prior"]["pass"]:
        raise SystemExit("probe does not match prior CSV within 0.002")


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m pipeline.run_patching")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate-prompts")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate")
    p.add_argument("--tokenizers", nargs="*", default=[])
    p.add_argument("--allow-download", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("estimate")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("smoke")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--out", default="results/patching/smoke")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--probe", default=None)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--verify-hooks", action="store_true")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--cells", choices=["behavior", "primary", "core", "expanded", "full", "fp32"],
        default="full",
    )
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--probe", default=None)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--verify-hooks", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("summarize")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--n-boot", type=int, default=P.N_BOOT)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("check-completeness")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="float16")
    p.add_argument(
        "--cells", choices=["behavior", "primary", "core", "expanded", "full", "fp32"],
        default="full",
    )
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.set_defaults(func=cmd_check_completeness)

    p = sub.add_parser("extract-probe")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--allow-download", action="store_true")
    p.set_defaults(func=cmd_extract_probe)

    p = sub.add_parser("fit-probe")
    p.add_argument("--model", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--programs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strict-prior", action="store_true")
    p.set_defaults(func=cmd_fit_probe)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
