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
  python -m pipeline.run_experiment crosslang --role class_struct \
      --model Qwen/Qwen2.5-1.5B --dataset dataset --split train \
      --languages C++ Javascript C

Hidden-state dumps go to disk. `--phase both` (default) is still one GPU
extract plus sklearn. `--phase extract|probe` and `--dump-root` split that
across jobs; other roles can ignore those flags.
"""

import argparse
import csv
import json
import os

import torch

from . import LANGUAGES, ROLES, STRATEGIES
from .probing import (META, best_layer, build_token_dataset,
                      control_selectivity, cross_evaluate, dump_exists,
                      extract_hidden_states, layer_keys, load_hidden_dump,
                      load_model, probe_cosine, train_probes)
from .stats import git_commit


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


def _write_run_meta(out_dir, args):
    import json
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump({"git_commit": git_commit(), "model": args.model,
                   "role": args.role, "split": args.split,
                   "max_programs": args.max_programs,
                   "protocol": "hash_split_70_10_20 | val-selected layer | 5 seeds | BCa CIs | control task"},
                  f, indent=1)


def run_perturbation(args, tokenizer, model, leading_special, device, out_dir):
    _write_run_meta(out_dir, args)
    all_results = {}
    for strategy in _relevant_strategies(args.role):
        dump_dir = os.path.join(args.dump_root, strategy) if args.dump_root else None
        hidden = labels = programs = None
        if args.phase in ("both", "extract"):
            if dump_dir and dump_exists(dump_dir):
                print(f"[{strategy}] dump exists, skip extract")
            else:
                rows = load_rows(args.dataset, "python_perturbations", args.split,
                                 strategy=strategy, max_programs=args.max_programs)
                data, skipped = build_token_dataset(rows, args.role, tokenizer)
                if not data:
                    print(f"[{strategy}] no usable programs, skipping")
                    continue
                print(f"[{strategy}] {len(data)} programs ({skipped} skipped)")
                hidden, labels, programs = extract_hidden_states(
                    data, tokenizer, model, leading_special, device, out_dir=dump_dir)
                if args.phase == "extract":
                    if hasattr(hidden, "close"):
                        hidden.close(delete=False)
                    print(f"  dumped {strategy} -> {dump_dir}")
                    continue
        if args.phase in ("both", "probe"):
            if hidden is None:
                if not dump_dir or not dump_exists(dump_dir):
                    print(f"[{strategy}] no dump, skip probe")
                    continue
                print(f"[{strategy}] probing from {dump_dir}")
                hidden, labels, programs = load_hidden_dump(dump_dir)
            try:
                res = train_probes(hidden, labels, programs)
                control_selectivity(hidden, labels, programs, res)
            finally:
                if hasattr(hidden, "close"):
                    hidden.close(delete=True)
            all_results[strategy] = res
            m = res[META]
            print(f"  selected layer {m['selected_layer']} (val): "
                  f"F1={m['test_f1_mean']:.3f}±{m['test_f1_std']:.3f} "
                  f"CI[{m['ci_low']:.3f},{m['ci_high']:.3f}] "
                  f"selectivity={m['selectivity']:+.3f}")
            _write_perturbation_csvs(out_dir, all_results)


def _write_perturbation_csvs(out_dir, all_results):
    """Rewrite CSVs after each strategy so a kill still leaves finished rows."""
    if not all_results:
        return
    layers = layer_keys(next(iter(all_results.values())))
    per_layer = os.path.join(out_dir, "per_layer.csv")
    with open(per_layer, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "layer", "train_acc", "test_acc", "train_f1", "test_f1"])
        for s, res in all_results.items():
            for li in layers:
                r = res[li]
                w.writerow([s, li, f"{r['train_acc']:.4f}", f"{r['test_acc']:.4f}",
                            f"{r['train_f1']:.4f}", f"{r['test_f1']:.4f}"])
        f.flush()
        os.fsync(f.fileno())

    base_f1 = all_results["baseline"][META]["test_f1_mean"] \
        if "baseline" in all_results else None
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "selected_layer", "test_f1_mean", "test_f1_std",
                    "ci_low", "ci_high", "majority_f1", "control_f1",
                    "selectivity", "n_programs", "delta_f1_vs_baseline"])
        for s, res in all_results.items():
            m = res[META]
            f1 = m["test_f1_mean"]
            delta = f"{f1 - base_f1:+.4f}" if base_f1 is not None and s != "baseline" else ""
            w.writerow([s, m["selected_layer"], f"{f1:.4f}", f"{m['test_f1_std']:.4f}",
                        f"{m['ci_low']:.4f}", f"{m['ci_high']:.4f}",
                        f"{m['majority_f1']:.4f}", f"{m['control_f1']:.4f}",
                        f"{m['selectivity']:+.4f}", m["n_programs"], delta])
        f.flush()
        os.fsync(f.fileno())

    others = [s for s in all_results if s != "baseline"]
    if "baseline" in all_results and others:
        with open(os.path.join(out_dir, "cosine_vs_baseline.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strategy"] + [f"layer_{li}" for li in layers])
            for s in others:
                sims = probe_cosine(all_results["baseline"], all_results[s])
                w.writerow([s] + [f"{sims[li]:.4f}" for li in layers])
            f.flush()
            os.fsync(f.fileno())


def run_crosslang(args, tokenizer, model, leading_special, device, out_dir):
    _write_run_meta(out_dir, args)

    def _dump(name):
        return os.path.join(args.dump_root, name) if args.dump_root else None

    def _extract_lang(language, rows_language):
        dest = _dump(language)
        if dest and dump_exists(dest):
            print(f"[{language}] dump exists, skip extract")
            return
        rows = load_rows(args.dataset, "multilingual_baseline", args.split,
                         language=rows_language, max_programs=args.max_programs)
        data, skipped = build_token_dataset(rows, args.role, tokenizer)
        if not data:
            print(f"[{language}] no usable programs, skipping")
            return
        print(f"[{language}] {len(data)} programs ({skipped} skipped)")
        hidden, labels, programs = extract_hidden_states(
            data, tokenizer, model, leading_special, device, out_dir=dest)
        if hasattr(hidden, "close"):
            hidden.close(delete=False)

    if args.phase == "extract" or (args.phase == "both" and args.dump_root):
        _extract_lang("Python", "Python")
        if args.phase == "extract":
            targets = args.languages or [lang for lang in LANGUAGES if lang != "Python"]
            for language in targets:
                if language != "Python":
                    _extract_lang(language, language)
            return

    py_dump = _dump("Python")
    if py_dump and dump_exists(py_dump):
        py_hidden, py_labels, py_programs = load_hidden_dump(py_dump)
        n_py = int(len(set(py_programs.tolist())))
        print(f"[Python] probing from dump ({n_py} programs)")
    else:
        py_rows = load_rows(args.dataset, "multilingual_baseline", args.split,
                            language="Python", max_programs=args.max_programs)
        py_data, _ = build_token_dataset(py_rows, args.role, tokenizer)
        print(f"[Python] {len(py_data)} programs")
        py_hidden, py_labels, py_programs = extract_hidden_states(
            py_data, tokenizer, model, leading_special, device, out_dir=py_dump)
        n_py = len(py_data)
    try:
        py_results = train_probes(py_hidden, py_labels, py_programs)
        py_best = best_layer(py_results)
        print(f"  in-domain selected layer {py_best} (val): "
              f"F1={py_results[META]['test_f1_mean']:.3f}")

        with open(os.path.join(out_dir, "crosslang.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["language", "programs", "indomain_selected_layer",
                        "indomain_test_f1_mean", "indomain_ci_low", "indomain_ci_high",
                        "transfer_acc_at_py_best", "transfer_f1_at_py_best"])
            pm = py_results[META]
            w.writerow(["Python", n_py, py_best, f"{pm['test_f1_mean']:.4f}",
                        f"{pm['ci_low']:.4f}", f"{pm['ci_high']:.4f}", "", ""])
            f.flush()
            os.fsync(f.fileno())
            targets = args.languages or [lang for lang in LANGUAGES if lang != "Python"]
            for language in targets:
                if language == "Python":
                    continue
                dest = _dump(language)
                hidden = labels = programs = None
                n_data = None
                if dest and dump_exists(dest):
                    print(f"[{language}] probing from {dest}")
                    hidden, labels, programs = load_hidden_dump(dest)
                    n_data = int(len(set(programs.tolist())))
                else:
                    rows = load_rows(args.dataset, "multilingual_baseline", args.split,
                                     language=language, max_programs=args.max_programs)
                    data, skipped = build_token_dataset(rows, args.role, tokenizer)
                    if not data:
                        print(f"[{language}] no usable programs, skipping")
                        continue
                    print(f"[{language}] {len(data)} programs ({skipped} skipped)")
                    hidden, labels, programs = extract_hidden_states(
                        data, tokenizer, model, leading_special, device, out_dir=dest)
                    n_data = len(data)
                try:
                    in_results = train_probes(hidden, labels, programs)
                    in_best = best_layer(in_results)
                    im = in_results[META]
                    xfer = cross_evaluate(py_results, hidden, labels)
                    w.writerow([language, n_data, in_best, f"{im['test_f1_mean']:.4f}",
                                f"{im['ci_low']:.4f}", f"{im['ci_high']:.4f}",
                                f"{xfer[py_best]['acc']:.4f}", f"{xfer[py_best]['f1']:.4f}"])
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if hasattr(hidden, "close"):
                        hidden.close(delete=True)
    finally:
        if hasattr(py_hidden, "close"):
            py_hidden.close(delete=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["perturbation", "crosslang"])
    ap.add_argument("--role", required=True, choices=ROLES)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-programs", type=int, default=None)
    ap.add_argument(
        "--languages", nargs="+", default=None, choices=LANGUAGES,
        help="crosslang target languages (default: all except Python). "
             "Python is always the source probe. For class_struct skip Java/C# "
             "on the first pass — nearly every program is a GFG wrapper and the "
             "hidden-state matrices will OOM.",
    )
    ap.add_argument("--out", default="results/unified")
    ap.add_argument(
        "--phase", choices=["both", "extract", "probe"], default="both",
        help="both = GPU dump + sklearn (default). extract = hidden states only. "
             "probe = sklearn only from --dump-root (no GPU).",
    )
    ap.add_argument(
        "--dump-root", default=None,
        help="Directory for per-strategy / per-language hidden-state dumps. "
             "Required for --phase extract|probe.",
    )
    args = ap.parse_args()
    if args.phase != "both" and not args.dump_root:
        raise SystemExit("--dump-root is required for --phase extract|probe")

    out_dir = os.path.join(args.out, args.model.split("/")[-1], args.role, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = model = leading_special = None
    if args.phase == "probe":
        device = "cpu"
        print(f"device=cpu  model={args.model}  phase=probe")
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"device={device}  model={args.model}  phase={args.phase}")
        tokenizer, model, leading_special = load_model(args.model, device)

    if args.mode == "perturbation":
        run_perturbation(args, tokenizer, model, leading_special, device, out_dir)
    else:
        run_crosslang(args, tokenizer, model, leading_special, device, out_dir)
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
