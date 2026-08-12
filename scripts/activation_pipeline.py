"""
Days 14–15: extract residual hidden states per boolean-flag occurrence and write
a manifest JSONL (``activation_path``, metadata) plus compressed NumPy ``.npz``
files (``first``, ``last``, ``mean`` pooling; shape ``[num_layers+1, hidden_size]``).

Uses one model load for the whole ``extract`` run; each canonical row gets one
forward pass. Storage is ``numpy.savez_compressed`` (no raw pickle blobs).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import signal
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from qwen_inference import (  # noqa: E402
    forward_hidden_cached,
    load_causal_lm_tokenizer,
    resolve_device,
    resolve_dtype,
)
from variable_occurrences import occurrence_rows_from_code, SUPPORTED_LANGUAGES  # noqa: E402
from go_csn_parse import iter_top_level_functions as iter_top_level_go_functions, parse_go  # noqa: E402
from java_csn_parse import iter_top_level_methods, parse_java  # noqa: E402
from javascript_csn_parse import (  # noqa: E402
    iter_top_level_functions as iter_top_level_js_functions,
    parse_javascript,
)
from php_csn_parse import iter_top_level_functions as iter_top_level_php_functions, parse_php_module  # noqa: E402
from ruby_csn_parse import iter_top_level_methods as iter_top_level_ruby_methods, parse_ruby  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "activations_v0" / "manifest.jsonl"
DEFAULT_TENSOR_DIR = PROJECT_ROOT / "outputs" / "activations_v0" / "npz"


def _function_source_len(code: str, function_name: str, *, language: str = "python") -> int | None:
    if language == "java":
        tree = parse_java(code)
        for method in iter_top_level_methods(tree.root_node):
            if method.name == function_name:
                return method.end_byte - method.start_byte
        return None
    if language == "go":
        tree = parse_go(code)
        for fn in iter_top_level_go_functions(tree.root_node):
            if fn.name == function_name:
                return fn.end_byte - fn.start_byte
        return None
    if language == "javascript":
        tree = parse_javascript(code)
        for fn in iter_top_level_js_functions(tree.root_node):
            if fn.name == function_name:
                return fn.end_byte - fn.start_byte
        return None
    if language == "php":
        module = parse_php_module(code)
        for fn in iter_top_level_php_functions(module):
            if fn.name == function_name:
                return fn.end_byte - fn.start_byte
        return None
    if language == "ruby":
        tree = parse_ruby(code)
        for method in iter_top_level_ruby_methods(tree.root_node):
            if method.name == function_name:
                return method.end_byte - method.start_byte
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name:
            seg = ast.get_source_segment(code, n)
            return len(seg) if seg else None
    return None


def _rel_to_manifest(manifest_file: Path, npz_file: Path) -> str:
    root = manifest_file.resolve().parent
    try:
        return str(npz_file.resolve().relative_to(root))
    except ValueError:
        return npz_file.name


def _hidden_stack_numpy(hs: tuple[torch.Tensor, ...]) -> np.ndarray:
    """``(num_layers+1, seq, hidden)`` float32 CPU."""
    arrs = [hs[i][0].detach().float().cpu().numpy() for i in range(len(hs))]
    return np.stack(arrs, axis=0).astype(np.float32, copy=False)


def _pool_tokens(
    stack: np.ndarray,
    token_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``stack`` is ``[L+1, seq, H]``. Returns ``first``, ``last``, ``mean`` each ``[L+1, H]``.
    """
    if not token_indices:
        h = stack.shape[2]
        z = np.zeros((stack.shape[0], h), dtype=np.float32)
        return z, z, z
    idxs = sorted(set(int(i) for i in token_indices if i >= 0))
    if not idxs or idxs[-1] >= stack.shape[1]:
        raise IndexError("token index out of range for hidden sequence")
    sl = stack[:, idxs, :]
    first = sl[:, 0, :].copy()
    last = sl[:, -1, :].copy()
    mean = sl.mean(axis=1).astype(np.float32, copy=False)
    return first, last, mean


def _occurrence_frequencies(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r["variable"]) for r in rows if r.get("variable")))


def _iter_manifest_records(manifest_path: Path):
    if not manifest_path.is_file():
        return
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


_NPZ_ROW_RE = re.compile(r"^row_(\d+)_occ_\d+\.npz$")


def _source_rows_from_npz_dir(tensor_dir: Path) -> set[int]:
    if not tensor_dir.is_dir():
        return set()
    rows: set[int] = set()
    for p in tensor_dir.glob("row_*_occ_*.npz"):
        m = _NPZ_ROW_RE.match(p.name)
        if m:
            rows.add(int(m.group(1)))
    return rows


def completed_source_rows(
    manifest_path: Path,
    tensor_dir: Path | None = None,
) -> set[int]:
    """Rows considered done for resume (new markers + legacy manifest/npz)."""
    done: set[int] = set()
    legacy_npz_lines: set[int] = set()

    for rec in _iter_manifest_records(manifest_path):
        sr = rec.get("source_row")
        if sr is None:
            continue
        row = int(sr)
        if rec.get("row_status") == "complete":
            done.add(row)
        elif rec.get("parse_error") or rec.get("occurrence_count") == 0:
            done.add(row)
        elif rec.get("activation_path"):
            # Pre-row_status manifests: any saved activation implies the row was processed.
            legacy_npz_lines.add(row)

    done |= legacy_npz_lines
    if tensor_dir is not None:
        done |= _source_rows_from_npz_dir(tensor_dir)
    return done


def _load_completed_source_rows(
    manifest_path: Path,
    tensor_dir: Path | None = None,
) -> set[int]:
    return completed_source_rows(manifest_path, tensor_dir)


def _max_completed_source_row(
    manifest_path: Path,
    tensor_dir: Path | None = None,
) -> int:
    return max(completed_source_rows(manifest_path, tensor_dir), default=0)


def _manifest_write(fout, rec: dict[str, Any]) -> None:
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.flush()


def _mark_row_complete(
    fout,
    *,
    source_row: int,
    repo: str | None,
    path: str | None,
    activation_count: int,
) -> None:
    _manifest_write(
        fout,
        {
            "source_row": source_row,
            "repo": repo,
            "path": path,
            "row_status": "complete",
            "activation_count": activation_count,
        },
    )


def cmd_extract(args: argparse.Namespace) -> int:
    if getattr(args, "resume", False):
        args.skip_processed = True
        args.append_manifest = True

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"no such file: {in_path}", file=sys.stderr)
        return 1

    manifest_path = Path(args.manifest)
    tensor_dir = Path(args.tensor_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    model, tok = load_causal_lm_tokenizer(
        args.model_id, device, dtype, eager_attn=False
    )

    n_in = 0
    n_written = 0
    n_skip = 0
    n_err = 0
    n_skip_processed = 0
    n_new_rows = 0
    max_rows = args.max_rows
    start_row = max(0, int(args.start_row))
    if args.resume and start_row == 0 and manifest_path.is_file():
        start_row = _max_completed_source_row(manifest_path, tensor_dir)
    completed_rows = (
        _load_completed_source_rows(manifest_path, tensor_dir)
        if args.skip_processed
        else set()
    )
    if args.skip_processed and completed_rows:
        n_done = len(completed_rows)
        max_done = max(completed_rows)
        print(
            f"resume: {n_done} completed source rows recognized "
            f"(max source_row={max_done})",
            flush=True,
        )
    manifest_mode = "a" if args.append_manifest and manifest_path.is_file() else "w"
    if start_row:
        print(f"resume: skipping first {start_row} input lines", flush=True)
    if max_rows is not None:
        print(f"extract: up to {max_rows} new input rows this run", flush=True)

    stop_path = Path(args.stop_file) if getattr(args, "stop_file", None) else None
    if stop_path is not None:
        print(f"stop file: {stop_path} (create this file to exit early)", flush=True)

    stop_requested = False

    def _on_stop_signal(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("\nstop signal received — finishing current row then exiting", flush=True)

    signal.signal(signal.SIGINT, _on_stop_signal)
    signal.signal(signal.SIGTERM, _on_stop_signal)

    def _should_stop() -> bool:
        if stop_requested:
            return True
        return stop_path is not None and stop_path.is_file()

    try:
        with in_path.open(encoding="utf-8") as fin, manifest_path.open(
            manifest_mode, encoding="utf-8"
        ) as fout:
            pbar = tqdm(
                desc="activations",
                unit="row",
                total=max_rows,
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                if n_in <= start_row:
                    continue
                row = json.loads(line)
                code = row.get("code") or ""
                repo = row.get("repo")
                path = row.get("path")
                source_row = row.get("source_row", n_in)
                if args.skip_processed and int(source_row) in completed_rows:
                    n_skip_processed += 1
                    continue
                if _should_stop():
                    print("stopped before row (stop file or signal)", flush=True)
                    break
                pbar.update(1)

                occ, err = occurrence_rows_from_code(
                    code,
                    language=args.language,
                    repo=repo,
                    path=path,
                    source_row=int(source_row) if source_row is not None else n_in,
                    tokenizer=tok,
                    max_length=args.max_length,
                )
                if err is not None:
                    n_err += 1
                    _manifest_write(
                        fout,
                        {
                            "parse_error": err,
                            "source_row": source_row,
                            "repo": repo,
                            "path": path,
                            "activation_path": None,
                        },
                    )
                    _mark_row_complete(
                        fout,
                        source_row=int(source_row),
                        repo=repo,
                        path=path,
                        activation_count=0,
                    )
                    n_new_rows += 1
                    pbar.set_postfix(src=int(source_row), npz=n_written, err=n_err, refresh=False)
                    if max_rows is not None and n_new_rows >= max_rows:
                        break
                    continue

                if not occ:
                    _manifest_write(
                        fout,
                        {
                            "source_row": source_row,
                            "repo": repo,
                            "path": path,
                            "occurrence_count": 0,
                            "activation_path": None,
                            "note": "no boolean_flag occurrences",
                        },
                    )
                    _mark_row_complete(
                        fout,
                        source_row=int(source_row),
                        repo=repo,
                        path=path,
                        activation_count=0,
                    )
                    n_new_rows += 1
                    pbar.set_postfix(src=int(source_row), npz=n_written, err=n_err, refresh=False)
                    if max_rows is not None and n_new_rows >= max_rows:
                        break
                    continue

                if _should_stop():
                    print("stopped before forward pass (stop file or signal)", flush=True)
                    break

                hs, meta = forward_hidden_cached(
                    model, tok, code, max_length=args.max_length, output_attentions=False
                )
                stack = _hidden_stack_numpy(hs)
                del hs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                seq_len = int(meta["seq_len"])
                n_layers = int(meta["num_hidden_layers"])
                hidden_size = int(meta["hidden_size"])
                token_len = seq_len
                func_len = _function_source_len(
                    code, str(occ[0].get("function", "")), language=args.language
                )
                freq = _occurrence_frequencies(occ)
                row_npz = 0

                for j, rec in enumerate(occ):
                    idxs = rec.get("token_positions")
                    if not idxs:
                        n_skip += 1
                        _manifest_write(
                            fout,
                            {
                                **{k: rec.get(k) for k in (
                                    "repo", "path", "source_row", "variable",
                                    "role", "occurrence_type", "line", "function",
                                    "detection_pattern",
                                )},
                                "token_positions": idxs,
                                "activation_path": None,
                                "skip_reason": "missing_token_positions",
                                "model_id": args.model_id,
                                "token_len": token_len,
                                "function_len_chars": func_len,
                                "occurrence_frequency": freq.get(str(rec.get("variable")), 0),
                            },
                        )
                        continue
                    idxs = [int(i) for i in idxs if 0 <= int(i) < seq_len]
                    if not idxs:
                        n_skip += 1
                        _manifest_write(
                            fout,
                            {
                                **{k: rec.get(k) for k in (
                                    "repo", "path", "source_row", "variable",
                                    "role", "occurrence_type", "line", "function",
                                )},
                                "token_positions": rec.get("token_positions"),
                                "activation_path": None,
                                "skip_reason": "token_index_out_of_range",
                                "model_id": args.model_id,
                                "seq_len": seq_len,
                            },
                        )
                        continue

                    try:
                        first, last, mean = _pool_tokens(stack, idxs)
                    except IndexError:
                        n_skip += 1
                        _manifest_write(
                            fout,
                            {
                                "variable": rec.get("variable"),
                                "source_row": rec.get("source_row"),
                                "activation_path": None,
                                "skip_reason": "pool_index_error",
                            },
                        )
                        continue

                    fname = f"row_{int(rec.get('source_row', source_row)):08d}_occ_{j:03d}.npz"
                    npz_path = tensor_dir / fname
                    np.savez_compressed(
                        npz_path,
                        first=first,
                        last=last,
                        mean=mean,
                        token_positions=np.asarray(idxs, dtype=np.int32),
                    )
                    rel = _rel_to_manifest(manifest_path, npz_path)
                    out_rec: dict[str, Any] = {
                        "repo": rec.get("repo"),
                        "path": rec.get("path"),
                        "source_row": rec.get("source_row"),
                        "function": rec.get("function"),
                        "variable": rec.get("variable"),
                        "role": rec.get("role"),
                        "occurrence_type": rec.get("occurrence_type"),
                        "line": rec.get("line"),
                        "token_positions": idxs,
                        "layer": None,
                        "activation_path": rel,
                        "activation_format": "npz_compressed",
                        "pooling": ["first", "last", "mean"],
                        "tensor_shape": [n_layers + 1, hidden_size],
                        "model_id": args.model_id,
                        "token_len": token_len,
                        "function_len_chars": func_len,
                        "occurrence_frequency": freq.get(str(rec.get("variable")), 0),
                    }
                    _manifest_write(fout, out_rec)
                    n_written += 1
                    row_npz += 1

                _mark_row_complete(
                    fout,
                    source_row=int(source_row),
                    repo=repo,
                    path=path,
                    activation_count=row_npz,
                )
                del stack
                n_new_rows += 1
                pbar.set_postfix(
                    src=int(source_row),
                    npz=n_written,
                    err=n_err,
                    refresh=False,
                )
                if max_rows is not None and n_new_rows >= max_rows:
                    break
            pbar.close()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"read_rows={n_in} new_rows={n_new_rows} activations_npz={n_written} "
        f"skipped_occurrences={n_skip} skipped_completed={n_skip_processed} "
        f"parse_errors={n_err} -> {manifest_path}"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    _FIXTURES = {
        "java": "boolean_occurrence_sample.java",
        "go": "boolean_occurrence_sample.go",
        "javascript": "boolean_occurrence_sample.js",
        "php": "boolean_occurrence_sample.php",
        "ruby": "boolean_occurrence_sample.rb",
        "python": "boolean_occurrence_sample.py",
    }
    sample_path = PROJECT_ROOT / "fixtures" / _FIXTURES[args.language]
    sample = sample_path.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        one = td_path / "one.jsonl"
        one.write_text(
            json.dumps(
                {
                    "code": sample,
                    "repo": "fixture",
                    "path": sample_path.name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = td_path / "manifest.jsonl"
        tens = td_path / "npz"
        ns = argparse.Namespace(
            input=str(one),
            manifest=str(manifest),
            tensor_dir=str(tens),
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
            max_rows=1,
            start_row=0,
            append_manifest=False,
            skip_processed=False,
            resume=False,
            language=args.language,
        )
        rc = cmd_extract(ns)
        if rc != 0:
            return rc
        lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        saved: list[dict[str, Any]] = []
        for ln in lines:
            d = json.loads(ln)
            if d.get("activation_path"):
                saved.append(d)
        if not saved:
            print("verify: no activation_path in manifest", file=sys.stderr)
            return 1
        p = saved[0]["activation_path"]
        npz_path = manifest.parent / p
        if not npz_path.is_file():
            print(f"verify: missing npz {npz_path}", file=sys.stderr)
            return 1
        with np.load(npz_path) as data:
            for k in ("first", "last", "mean"):
                if k not in data:
                    print(f"verify: npz missing {k}", file=sys.stderr)
                    return 1
                a = data[k]
                if a.ndim != 2:
                    print(f"verify: bad ndim for {k}: {a.shape}", file=sys.stderr)
                    return 1
            shape = tuple(int(x) for x in data["first"].shape)
        print(
            f"activation_pipeline verify ({args.language}): ok "
            f"({len(saved)} npz rows; tensor {shape})"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Days 14–15: hidden activations + manifest JSONL.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser(
        "extract",
        help="Read canonical JSONL; forward Qwen once per row; save .npz + manifest lines.",
    )
    ex.add_argument("--input", type=str, required=True, help="Canonical JSONL (repo, path, code).")
    ex.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    ex.add_argument("--tensor-dir", type=str, default=str(DEFAULT_TENSOR_DIR))
    ex.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    ex.add_argument("--device", type=str, default=None)
    ex.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ex.add_argument("--max-length", type=int, default=2048)
    ex.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many newly processed input rows (not counting skipped resume rows).",
    )
    ex.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Skip the first N non-empty input lines (0-based count before filtering).",
    )
    ex.add_argument(
        "--append-manifest",
        action="store_true",
        help="Append to an existing manifest instead of overwriting.",
    )
    ex.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip input rows already marked complete in the manifest.",
    )
    ex.add_argument(
        "--resume",
        action="store_true",
        help="Resume: append manifest, skip completed rows, jump --start-row to last complete row.",
    )
    ex.add_argument(
        "--stop-file",
        type=str,
        default=None,
        help="If this path exists, exit cleanly after the current input row.",
    )
    ex.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="python",
        help="Source language for occurrence extraction (default: python).",
    )
    ex.set_defaults(func=cmd_extract)

    v = sub.add_parser("verify", help="Run extract on one fixture row into a temp dir.")
    v.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    v.add_argument("--device", type=str, default=None)
    v.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    v.add_argument("--max-length", type=int, default=2048)
    v.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="python",
        help="Fixture language to verify (default: python).",
    )
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
