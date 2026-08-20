"""Residual-stream extraction over XLCoST occurrences -> memmap store.

Protocol-native replacement for the npz-per-occurrence format (which spent
491,762 bytes per occurrence, 92% of it redundant). One forward per program,
mean-pooled residuals per occurrence, written to:

    <out-dir>/shard.npy    float16 memmap [N, num_layers+1, hidden]
    <out-dir>/index.jsonl  one row per occurrence, same order as the shard
    <out-dir>/meta.json    model id, dtype, shapes, source files

Resumable: rows are written in occurrence order; on restart the completed
count is read from index.jsonl and extraction continues from there. Layer 0
of the stack is the embedding layer (pre-transformer) — label axes
"embed, 1..L", never "0..L".

DeepSeek-Coder loads via PreTrainedTokenizerFast (the AutoTokenizer path
under transformers 5.8.0 deletes whitespace and corrupts every offset).

    uv run python scripts/extract_activations.py run \
        --canonical data/xlcost/python_train.jsonl \
        --occurrences outputs/xlcost_occ/python_train.jsonl \
        --model-id Qwen/Qwen2.5-Coder-1.5B \
        --out-dir outputs/activations_xlcost/python_qwen25coder15b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from token_alignment import char_span_to_token_indices, tokenize_for_alignment  # noqa: E402
from tokenizer_gate import FAST_OVERRIDE  # noqa: E402


def load_model_and_tokenizer(model_id: str, device: str, dtype_flag: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    if model_id in FAST_OVERRIDE:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    else:
        tok = AutoTokenizer.from_pretrained(model_id)
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if dtype_flag == "auto":
        dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_flag]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, tok, device


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            yield json.loads(ln)


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    canonical = {r["problem_id"]: r["code"] for r in read_jsonl(Path(args.canonical))}
    occurrences = list(read_jsonl(Path(args.occurrences)))
    if args.max_occurrences:
        occurrences = occurrences[: args.max_occurrences]
    n_total = len(occurrences)
    if not n_total:
        raise SystemExit("no occurrences")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, shard_path, meta_path = (
        out_dir / "index.jsonl", out_dir / "shard.npy", out_dir / "meta.json",
    )

    model, tok, device = load_model_and_tokenizer(args.model_id, args.device, args.dtype)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    canon_sha = file_sha256(Path(args.canonical))
    occ_sha = file_sha256(Path(args.occurrences))
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        expected = {
            "model_id": args.model_id,
            "n_occurrences": n_total,
            "canonical_sha256": canon_sha,
            "occurrences_sha256": occ_sha,
            "max_length": args.max_length,
            # Resuming with a different --label-field would append records
            # labelled from one field to records labelled from another, while
            # meta.json kept advertising the original. The probe would then
            # train on two incompatible class schemes, or die with a
            # single-class error, with nothing in the store to explain why.
            # Stores written before this flag existed have no label_field key
            # and were necessarily occurrence_type, so that is the default on
            # both sides of the comparison.
            "label_field": args.label_field,
        }
        meta.setdefault("label_field", "occurrence_type")
        legacy = [k for k in ("canonical_sha256", "occurrences_sha256") if k not in meta]
        mismatch = {
            k: (meta.get(k), v)
            for k, v in expected.items()
            if k in meta and meta.get(k) != v
        }
        if mismatch:
            raise SystemExit(
                f"store at {out_dir} was built from different inputs — refusing to "
                f"resume into a stale shard. Mismatches: {mismatch}. Use a new "
                "--out-dir (or delete the store) if the inputs legitimately changed."
            )
        if legacy:
            print(
                f"WARNING: legacy store (no input hashes recorded: {legacy}) — "
                "input identity cannot be verified; resuming on model+count only.",
                flush=True,
            )
    else:
        meta = {
            "model_id": args.model_id,
            "canonical": str(args.canonical),
            "occurrences": str(args.occurrences),
            "canonical_sha256": canon_sha,
            "occurrences_sha256": occ_sha,
            "n_occurrences": n_total,
            "shape": [n_total, n_layers + 1, hidden],
            "dtype": "float16",
            "label_field": args.label_field,
            "pooling": "mean",
            "max_length": args.max_length,
            "layer_indexing": "0 = embedding (pre-transformer)",
        }
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    mode = "r+" if shard_path.is_file() else "w+"
    shard = np.lib.format.open_memmap(
        shard_path, mode=mode, dtype=np.float16, shape=(n_total, n_layers + 1, hidden)
    )

    done = sum(1 for _ in read_jsonl(index_path)) if index_path.is_file() else 0
    if done:
        print(f"resuming at occurrence {done}/{n_total}", flush=True)

    # Cache one forward per program: consecutive occurrences share programs.
    cache_key, cache_stack, cache_offsets, cache_len = None, None, None, 0
    n_written, n_skipped = 0, 0
    with index_path.open("a", encoding="utf-8") as idx:
        for i in range(done, n_total):
            occ = occurrences[i]
            pid = occ["problem_id"]
            code = canonical.get(pid)
            row_meta = {
                "row": i,
                "occurrence_id": occ["occurrence_id"],
                "problem_id": pid,
                "language": occ.get("language"),
                # The store has ONE label slot, always called occurrence_type.
                # --label-field says which occurrence field fills it, because
                # the two producers disagree: the boolean workstream writes
                # `occurrence_type`, role_occurrences.py writes `role`. Reading
                # the wrong one stores nulls, and probe.py then drops every
                # record -- after the GPU time is spent.
                "occurrence_type": occ.get(args.label_field),
                "variable": occ.get("variable"),
                "function": occ.get("function"),
                "detection_pattern": occ.get("detection_pattern"),
                "split": occ.get("split"),
            }
            if code is None:
                row_meta["skip"] = "missing_program"
                n_skipped += 1
                idx.write(json.dumps(row_meta) + "\n")
                continue
            if pid != cache_key:
                ids, offsets, _ = tokenize_for_alignment(tok, code, max_length=args.max_length)
                enc = torch.tensor([ids], device=device)
                with torch.no_grad():
                    out = model(enc, output_hidden_states=True, use_cache=False)
                # [L+1, seq, H] on CPU float32 for pooling precision.
                cache_stack = torch.stack(out.hidden_states, dim=0)[:, 0].float().cpu().numpy()
                cache_offsets, cache_len, cache_key = offsets, len(ids), pid
                del out
            s, e = occ["source_span"]
            positions = char_span_to_token_indices(cache_offsets, int(s), int(e))
            positions = [p for p in positions if 0 <= p < cache_len]
            if not positions:
                row_meta["skip"] = "no_token_positions"  # e.g. truncated past max_length
                n_skipped += 1
                idx.write(json.dumps(row_meta) + "\n")
                continue
            pooled = cache_stack[:, positions, :].mean(axis=1)  # [L+1, H]
            shard[i] = pooled.astype(np.float16)
            row_meta["token_len"] = cache_len
            row_meta["n_span_tokens"] = len(positions)
            idx.write(json.dumps(row_meta) + "\n")
            n_written += 1
            if (i + 1) % args.log_every == 0:
                print(f"{i + 1}/{n_total}  written={n_written} skipped={n_skipped}", flush=True)

    shard.flush()
    print(
        f"done: {n_written} written, {n_skipped} skipped, "
        f"store={out_dir} ({shard_path.stat().st_size / 1e9:.2f} GB)"
    )
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Pooling math check without any model: fake stack, known span means."""
    stack = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)  # [L+1=2, seq=4, H=3]
    pooled = stack[:, [1, 2], :].mean(axis=1)
    expected = (stack[:, 1] + stack[:, 2]) / 2
    ok = np.allclose(pooled, expected)
    offsets = [(0, 3), (3, 6), (7, 10)]
    ok &= char_span_to_token_indices(offsets, 3, 6) == [1]
    ok &= char_span_to_token_indices(offsets, 2, 8) == [0, 1, 2]
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--canonical", required=True)
    r.add_argument("--occurrences", required=True)
    r.add_argument("--model-id", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    r.add_argument("--max-length", type=int, default=2048)
    r.add_argument("--max-occurrences", type=int, default=0, help="0 = all")
    r.add_argument("--label-field", default="occurrence_type",
                   choices=["occurrence_type", "role"],
                   help="which occurrence field becomes the store's label: "
                        "'occurrence_type' for the boolean workstream, 'role' "
                        "for role_occurrences.py output (cross-lingual work)")
    r.add_argument("--log-every", type=int, default=200)
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    return cmd_verify(args) if args.cmd == "verify" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
