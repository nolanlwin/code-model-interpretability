"""Run the probing experiments from the common dataset.

Modes:
  perturbation — one role, all naming strategies on Python; per-layer CSV,
                 best-layer summary with deltas vs baseline, probe cosines
  crosslang    — train the Python baseline probe for one role, evaluate on
                 every other language (in-domain probes + transfer accuracy)

Usage:
  python -m pipeline.run_experiment perturbation --role accumulator \
      --model Qwen/Qwen2.5-1.5B --dataset dataset --split train --max-programs 500
  python -m pipeline.run_experiment crosslang --role index_key \
      --model Qwen/Qwen2.5-1.5B --dataset dataset --max-programs 300
"""

import argparse
import csv
import json
import os

import torch

from . import LANGUAGES, ROLES, STRATEGIES
from .probing import (best_layer, build_token_dataset, cross_evaluate,
                      extract_hidden_states, load_model, probe_cosine,
                      train_probes)


def load_rows(dataset_dir, config, split, language=None, strategy=None, max_programs=None):
    path = os.path.join(dataset_dir, config, f"{split}.jsonl")
    rows, seen = [], set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if language and row["language"] != language:
                continue
            if strategy and row["strategy"] != strategy:
                continue
            if max_programs is not None:
                if row["problem_id"] not in seen and len(seen) >= max_programs:
                    continue
                seen.add(row["problem_id"])
            rows.append(row)
    return rows


def _relevant_strategies(role):
    # Role-specific misleading only; other roles' misleading variants are noise here.
    return [s for s in STRATEGIES
            if not s.startswith("misleading_") or s == f"misleading_{role}"]


def run_perturbation(args, tokenizer, model, leading_special, device, out_dir):
    all_results = {}
    for strategy in _relevant_strategies(args.role):
        rows = load_rows(args.dataset, "python_perturbations", args.split,
                         strategy=strategy, max_programs=args.max_programs)
        data, skipped = build_token_dataset(rows, args.role, tokenizer)
        if not data:
            print(f"[{strategy}] no usable programs, skipping")
            continue
        print(f"[{strategy}] {len(data)} programs ({skipped} skipped)")
        hidden, labels = extract_hidden_states(data, tokenizer, model, leading_special, device)
        all_results[strategy] = train_probes(hidden, labels)
        bl = best_layer(all_results[strategy])
        print(f"  best layer {bl}: acc={all_results[strategy][bl]['test_acc']:.3f} "
              f"F1={all_results[strategy][bl]['test_f1']:.3f}")

    layers = sorted(next(iter(all_results.values())))
    with open(os.path.join(out_dir, "per_layer.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "layer", "train_acc", "test_acc", "train_f1", "test_f1"])
        for s, res in all_results.items():
            for li in layers:
                r = res[li]
                w.writerow([s, li, f"{r['train_acc']:.4f}", f"{r['test_acc']:.4f}",
                            f"{r['train_f1']:.4f}", f"{r['test_f1']:.4f}"])

    base_f1 = max(all_results["baseline"][li]["test_f1"] for li in layers) \
        if "baseline" in all_results else None
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "best_layer", "test_acc", "test_f1", "delta_f1_vs_baseline"])
        for s, res in all_results.items():
            bl = best_layer(res)
            f1 = res[bl]["test_f1"]
            delta = f"{f1 - base_f1:+.4f}" if base_f1 is not None and s != "baseline" else ""
            w.writerow([s, bl, f"{res[bl]['test_acc']:.4f}", f"{f1:.4f}", delta])

    if "baseline" in all_results:
        with open(os.path.join(out_dir, "cosine_vs_baseline.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strategy"] + [f"layer_{li}" for li in layers])
            for s, res in all_results.items():
                if s == "baseline":
                    continue
                sims = probe_cosine(all_results["baseline"], res)
                w.writerow([s] + [f"{sims[li]:.4f}" for li in layers])


def run_crosslang(args, tokenizer, model, leading_special, device, out_dir):
    py_rows = load_rows(args.dataset, "multilingual_baseline", args.split,
                        language="Python", max_programs=args.max_programs)
    py_data, _ = build_token_dataset(py_rows, args.role, tokenizer)
    print(f"[Python] {len(py_data)} programs")
    py_hidden, py_labels = extract_hidden_states(py_data, tokenizer, model, leading_special, device)
    py_results = train_probes(py_hidden, py_labels)
    py_best = best_layer(py_results)
    print(f"  in-domain best layer {py_best}: F1={py_results[py_best]['test_f1']:.3f}")

    with open(os.path.join(out_dir, "crosslang.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "programs", "indomain_best_layer", "indomain_test_f1",
                    "transfer_acc_at_py_best", "transfer_f1_at_py_best"])
        w.writerow(["Python", len(py_data), py_best,
                    f"{py_results[py_best]['test_f1']:.4f}", "", ""])
        for language in LANGUAGES:
            if language == "Python":
                continue
            rows = load_rows(args.dataset, "multilingual_baseline", args.split,
                             language=language, max_programs=args.max_programs)
            data, skipped = build_token_dataset(rows, args.role, tokenizer)
            if not data:
                print(f"[{language}] no usable programs, skipping")
                continue
            print(f"[{language}] {len(data)} programs ({skipped} skipped)")
            hidden, labels = extract_hidden_states(data, tokenizer, model, leading_special, device)
            in_results = train_probes(hidden, labels)
            in_best = best_layer(in_results)
            xfer = cross_evaluate(py_results, hidden, labels)
            w.writerow([language, len(data), in_best,
                        f"{in_results[in_best]['test_f1']:.4f}",
                        f"{xfer[py_best]['acc']:.4f}", f"{xfer[py_best]['f1']:.4f}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["perturbation", "crosslang"])
    ap.add_argument("--role", required=True, choices=ROLES)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-programs", type=int, default=None)
    ap.add_argument("--out", default="results/unified")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  model={args.model}")
    tokenizer, model, leading_special = load_model(args.model, device)

    out_dir = os.path.join(args.out, args.model.split("/")[-1], args.role, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "perturbation":
        run_perturbation(args, tokenizer, model, leading_special, device, out_dir)
    else:
        run_crosslang(args, tokenizer, model, leading_special, device, out_dir)
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
