"""
Headless per-token activation extractor for structural (syntactic-class) probing.

Generalizes ``activation_pipeline.py`` (boolean occurrences) to the multi-class
structural labels from ``structure_labels.py``, over CoST canonical JSONL
(``problem_id``, ``language``, ``code``). One model load for the whole run; one
forward per program. For every labeled token we store its residual vector at all
layers.

Output:
- ``--tensor-dir`` / ``row_<problem>_<language>.npz`` per program with:
    ``activations``    float32 ``[n_tokens, num_layers+1, hidden_size]``
    ``class_idx``      int32 ``[n_tokens]`` (index into ``CLASS_LIST``)
    ``token_index``    int32 ``[n_tokens]``
- ``--manifest`` JSONL: one row per program (``activation_path``, ``problem_id``,
  ``language``, ``token_count``, per-class counts, ``tensor_shape``, ``model_id``).

``verify`` runs one fixture program (Python + Java) in a temp dir.
"""

from __future__ import annotations

import argparse
import json
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
from structure_labels import (  # noqa: E402
    STRUCTURAL_CLASSES,
    structural_leaves_from_code,
)
import token_alignment as _tokalign  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "structure_activations" / "manifest.jsonl"
DEFAULT_TENSOR_DIR = PROJECT_ROOT / "outputs" / "structure_activations" / "npz"

# Fixed class ordering shared by both languages (drop comment by default).
CLASS_LIST = [c for c in STRUCTURAL_CLASSES if c != "comment"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}


def token_class_map(
    code: str,
    language: str,
    tokenizer,
    *,
    max_length: int = 2048,
    include_comments: bool = False,
) -> tuple[dict[int, int], int]:
    """
    Return ``{token_index: class_idx}`` (first labeled char wins) and the seq length.

    Uses the same fast-tokenizer ``offset_mapping`` contract as ``token_alignment``.
    """
    _, offset_mapping, _ = _tokalign.tokenize_for_alignment(
        tokenizer, code, max_length=max_length
    )
    seq_len = len(offset_mapping)
    leaves = structural_leaves_from_code(code, language, include_comments=include_comments)

    # Per-character class index (-1 = unlabeled).
    char_cls = np.full(len(code), -1, dtype=np.int64)
    for leaf in leaves:
        cls = leaf["structural_class"]
        if cls not in CLASS_TO_IDX:
            continue
        s, e = leaf["source_span"]
        char_cls[s:e] = CLASS_TO_IDX[cls]

    out: dict[int, int] = {}
    for ti, (cs, ce) in enumerate(offset_mapping):
        if ce <= cs:
            continue
        for pos in range(cs, min(ce, len(code))):
            if char_cls[pos] >= 0:
                out[ti] = int(char_cls[pos])
                break
    return out, seq_len


def _hidden_stack_numpy(hs: tuple[torch.Tensor, ...]) -> np.ndarray:
    """``[num_layers+1, seq, hidden]`` float32 CPU."""
    arrs = [hs[i][0].detach().float().cpu().numpy() for i in range(len(hs))]
    return np.stack(arrs, axis=0).astype(np.float32, copy=False)


def cmd_extract(args: argparse.Namespace) -> int:
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
    model, tok = load_causal_lm_tokenizer(args.model_id, device, dtype, eager_attn=False)
    if not tok.is_fast:
        print("tokenizer is not fast; offset_mapping unavailable", file=sys.stderr)
        return 1

    n_in = 0
    n_written = 0
    n_tokens = 0
    n_skip = 0
    max_rows = args.max_rows

    try:
        with in_path.open(encoding="utf-8") as fin, manifest_path.open(
            "w", encoding="utf-8"
        ) as fout:
            pbar = tqdm(desc="structure activations", unit="row", total=max_rows)
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                pbar.update(1)
                row = json.loads(line)
                code = row.get("code") or ""
                language = (row.get("language") or args.language or "").lower()
                problem_id = row.get("problem_id")
                if language not in ("python", "java"):
                    n_skip += 1
                    if max_rows is not None and n_in >= max_rows:
                        break
                    continue

                try:
                    tcm, _seq = token_class_map(
                        code, language, tok, max_length=args.max_length
                    )
                except Exception as ex:  # noqa: BLE001
                    n_skip += 1
                    fout.write(
                        json.dumps(
                            {
                                "problem_id": problem_id,
                                "language": language,
                                "activation_path": None,
                                "skip_reason": f"label_error: {ex}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if max_rows is not None and n_in >= max_rows:
                        break
                    continue

                if not tcm:
                    fout.write(
                        json.dumps(
                            {
                                "problem_id": problem_id,
                                "language": language,
                                "token_count": 0,
                                "activation_path": None,
                                "note": "no labeled tokens",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if max_rows is not None and n_in >= max_rows:
                        break
                    continue

                hs, meta = forward_hidden_cached(
                    model, tok, code, max_length=args.max_length, output_attentions=False
                )
                stack = _hidden_stack_numpy(hs)  # [L+1, seq, H]
                del hs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                seq_len = int(meta["seq_len"])
                n_layers = int(meta["num_hidden_layers"])
                hidden_size = int(meta["hidden_size"])

                token_indices = sorted(ti for ti in tcm if ti < seq_len)
                if not token_indices:
                    del stack
                    n_skip += 1
                    continue
                sel = stack[:, token_indices, :]  # [L+1, n_tok, H]
                activations = np.transpose(sel, (1, 0, 2)).astype(
                    np.float32, copy=False
                )  # [n_tok, L+1, H]
                class_idx = np.asarray([tcm[ti] for ti in token_indices], dtype=np.int32)
                tok_idx = np.asarray(token_indices, dtype=np.int32)
                del stack

                fname = f"row_{problem_id}_{language}.npz"
                npz_path = tensor_dir / fname
                np.savez_compressed(
                    npz_path,
                    activations=activations,
                    class_idx=class_idx,
                    token_index=tok_idx,
                )
                try:
                    rel = str(npz_path.resolve().relative_to(manifest_path.resolve().parent))
                except ValueError:
                    rel = npz_path.name

                counts = Counter(CLASS_LIST[c] for c in class_idx.tolist())
                fout.write(
                    json.dumps(
                        {
                            "problem_id": problem_id,
                            "language": language,
                            "activation_path": rel,
                            "activation_format": "npz_compressed",
                            "token_count": int(class_idx.shape[0]),
                            "class_counts": dict(counts),
                            "class_list": CLASS_LIST,
                            "tensor_shape": [int(class_idx.shape[0]), n_layers + 1, hidden_size],
                            "model_id": args.model_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1
                n_tokens += int(class_idx.shape[0])

                if max_rows is not None and n_in >= max_rows:
                    break
            pbar.close()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"read_rows={n_in} npz_written={n_written} tokens={n_tokens} "
        f"skipped={n_skip} -> {manifest_path}"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    samples = [
        ("python", "def f(x):\n    ok = True\n    if ok and x > 0:\n        return 1\n    return 0\n"),
        ("java", "class A {\n    int f(int x) {\n        boolean ok = true;\n        if (ok) return 1;\n        return 0;\n    }\n}\n"),
    ]
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        rows_path = td_path / "rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as f:
            for i, (lang, code) in enumerate(samples):
                f.write(
                    json.dumps({"problem_id": i, "language": lang, "code": code}) + "\n"
                )
        manifest = td_path / "manifest.jsonl"
        tens = td_path / "npz"
        ns = argparse.Namespace(
            input=str(rows_path),
            manifest=str(manifest),
            tensor_dir=str(tens),
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
            max_rows=None,
            language=None,
        )
        rc = cmd_extract(ns)
        if rc != 0:
            return rc
        saved = [
            json.loads(ln)
            for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        saved = [d for d in saved if d.get("activation_path")]
        if len(saved) < 2:
            print("verify: expected 2 npz rows (python + java)", file=sys.stderr)
            return 1
        npz = manifest.parent / saved[0]["activation_path"]
        with np.load(npz) as data:
            a = data["activations"]
            if a.ndim != 3:
                print(f"verify: bad activations ndim {a.shape}", file=sys.stderr)
                return 1
        print(f"structure_activations verify: ok ({len(saved)} programs, tensor {a.shape})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Headless per-token structural-class activation extractor."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="CoST canonical JSONL -> per-program npz + manifest.")
    ex.add_argument("--input", type=str, required=True, help="JSONL (problem_id, language, code).")
    ex.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    ex.add_argument("--tensor-dir", type=str, default=str(DEFAULT_TENSOR_DIR))
    ex.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    ex.add_argument("--device", type=str, default=None)
    ex.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ex.add_argument("--max-length", type=int, default=2048)
    ex.add_argument("--language", type=str, default=None, choices=["python", "java"],
                    help="Fallback language when a JSONL row omits it.")
    ex.add_argument("--max-rows", type=int, default=None)
    ex.set_defaults(func=cmd_extract)

    v = sub.add_parser("verify", help="Run extract on a Python + Java fixture in a temp dir.")
    v.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    v.add_argument("--device", type=str, default=None)
    v.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    v.add_argument("--max-length", type=int, default=2048)
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
