"""python_probe"""

# # Python Iterator Probe (Clean)
# 
# This notebook isolates the Python probe workflow from `label4.ipynb` for cleaner experiments and fair Python-vs-Java comparison.

# Run this cell to mount Google Drive if running in Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running on Google Colab. Skipping Drive mount.")

# Run this cell to upload your CSV dataset to Colab if running on Google Colab
try:
    from google.colab import files
    import os
    os.makedirs('data', exist_ok=True)
    print("Running on Google Colab. Please upload the required CSV file:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        dest = os.path.join('data', filename)
        os.rename(filename, dest)
        print(f"Moved {filename} to {dest}")
except ImportError:
    print("Not running on Google Colab. Skipping file upload helper.")

import sys
import subprocess

# Programmatically install dependencies if missing
def install_dependencies():
    required = {'numpy', 'pandas', 'torch', 'transformers', 'scikit-learn', 'matplotlib'}
    try:
        installed = {pkg.split('==')[0].lower() for pkg in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split()}
    except Exception:
        installed = set()
    missing = required - installed
    if missing:
        print(f"Installing missing dependencies: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        print("All dependencies installed successfully.")

install_dependencies()

import os
import ast
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Model + tokenizer
model_name = "Qwen/Qwen2.5-Coder-1.5B"
local_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B/snapshots/df3ce67c0e24480f20468b6ef2894622d69eb73b")
model_ref = local_model_path if os.path.exists(local_model_path) else model_name
use_local_only = os.path.exists(local_model_path)

print("model ref:", model_ref)
print("local only:", use_local_only)

try:
    print("Attempting to load model locally...")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=use_local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=use_local_only,
        dtype=torch.float32
    ).to(device)
except Exception as e:
    print(f"Local load failed: {e}")
    print("Falling back to downloading from Hugging Face hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        dtype=torch.float32
    ).to(device)

model.eval()
print("device:", device)

# Token/span alignment helpers
def line_col_to_char_offset(code, line, col):
    lines = code.splitlines(keepends=True)
    return sum(len(lines[i]) for i in range(line - 1)) + col


def overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def align_spans_to_tokens(spans, offset_mapping):
    labels = []
    for start, end in offset_mapping:
        label = 0
        for s, e in spans:
            if overlaps(start, end, s, e):
                label = 1
                break
        labels.append(label)
    return labels

# AST helpers for iterator binder + body use extraction
def extract_target_names(target):
    names = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(extract_target_names(elt))
    return names


def node_to_span(code, node):
    start = line_col_to_char_offset(code, node.lineno, node.col_offset)
    end = line_col_to_char_offset(code, node.end_lineno, node.end_col_offset)
    return (start, end)


def find_iterator_and_body_use_spans(code):
    tree = ast.parse(code)
    spans = []

    def visit_for_node(for_node):
        iterator_names = extract_target_names(for_node.target)

        def add_target_spans(target):
            if isinstance(target, ast.Name):
                spans.append(node_to_span(code, target))
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    add_target_spans(elt)

        add_target_spans(for_node.target)

        def collect_uses(node, blocked_names=None):
            if blocked_names is None:
                blocked_names = set()

            if isinstance(node, ast.For):
                inner_bound = extract_target_names(node.target)
                new_blocked = blocked_names | (inner_bound & iterator_names)
                for child in node.body:
                    collect_uses(child, new_blocked)
                for child in node.orelse:
                    collect_uses(child, new_blocked)
                return

            if isinstance(node, ast.Name):
                if node.id in iterator_names and node.id not in blocked_names and isinstance(node.ctx, ast.Load):
                    spans.append(node_to_span(code, node))

            for child in ast.iter_child_nodes(node):
                collect_uses(child, blocked_names)

        for stmt in for_node.body:
            collect_uses(stmt)
        for stmt in for_node.orelse:
            collect_uses(stmt)

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            visit_for_node(node)

    return spans

def build_token_level_data_for_snippet(code, layer_idx):
    spans = find_iterator_and_body_use_spans(code)

    tok = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    labels = align_spans_to_tokens(spans, tok["offset_mapping"])

    inputs = tokenizer(code, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx][0].detach().cpu()
    y = torch.tensor(labels, dtype=torch.long)
    tokens = [tokenizer.decode([tok_id]) for tok_id in tok["input_ids"]]
    offsets = tok["offset_mapping"]

    assert hidden.shape[0] == len(y), f"Mismatch: hidden {hidden.shape[0]} vs labels {len(y)}"
    return hidden, y, tokens, offsets

def collect_dataset(df, snippet_indices, layer_idx, print_every=10, max_snippets=None):
    X_list, y_list = [], []
    skipped, kept = 0, 0

    if max_snippets is not None:
        snippet_indices = snippet_indices[:max_snippets]

    total = len(snippet_indices)

    for j, idx in enumerate(snippet_indices):
        if j % print_every == 0:
            print(f"[PY layer {layer_idx}] processing {j}/{total}")

        code = df.loc[idx, "code"]

        try:
            X_snip, y_snip, _, _ = build_token_level_data_for_snippet(code, layer_idx=layer_idx)

            if len(y_snip) == 0 or y_snip.sum().item() == 0:
                skipped += 1
                continue

            X_list.append(X_snip.numpy())
            y_list.append(y_snip.numpy())
            kept += 1

        except Exception as e:
            skipped += 1
            print(f"Skipping snippet {idx} because of error: {e}")

    if len(X_list) == 0:
        raise ValueError("No valid snippets were collected.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print("Kept snippets   :", kept)
    print("Skipped snippets:", skipped)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Positive labels:", y.sum())
    print("Total labels   :", len(y))
    print("Positive rate  :", y.mean())

    return X, y

# Data prep (full dataset, capped by max lines per snippet)
import os
if os.path.exists('/content/drive'):
    py_csv_path = "/content/drive/MyDrive/algoverse/data/python_iterator_snippets.csv"
else:
    py_csv_path = "./data/python_iterator_snippets.csv"
df_raw = pd.read_csv(py_csv_path)

max_lines_per_snippet = 50
df_raw["line_count"] = df_raw["code"].apply(lambda x: len(str(x).splitlines()))
df = df_raw[df_raw["line_count"] <= max_lines_per_snippet].reset_index(drop=True)

snippet_indices = list(range(len(df)))
train_idx, test_idx = train_test_split(snippet_indices, test_size=0.2, random_state=42)

print("raw python shape:", df_raw.shape)
print("filtered python shape (<=50 lines):", df.shape)
print("train python snippets:", len(train_idx))
print("test python snippets :", len(test_idx))

# Probe training at one layer (debug run)
layer_idx_py = 4
max_train_snippets = None
max_test_snippets = None

X_train_py, y_train_py = collect_dataset(
    df,
    train_idx,
    layer_idx=layer_idx_py,
    print_every=5,
    max_snippets=max_train_snippets
)

X_test_py, y_test_py = collect_dataset(
    df,
    test_idx,
    layer_idx=layer_idx_py,
    print_every=5,
    max_snippets=max_test_snippets
)

probe_py = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
probe_py.fit(X_train_py, y_train_py)
y_pred_py = probe_py.predict(X_test_py)

print("accuracy :", accuracy_score(y_test_py, y_pred_py))
print("precision:", precision_score(y_test_py, y_pred_py, zero_division=0))
print("recall   :", recall_score(y_test_py, y_pred_py, zero_division=0))
print("f1       :", f1_score(y_test_py, y_pred_py, zero_division=0))
print("\nClassification report:\n")
print(classification_report(y_test_py, y_pred_py, zero_division=0))

baseline_pred_py = np.zeros_like(y_test_py)
print("baseline accuracy :", accuracy_score(y_test_py, baseline_pred_py))
print("baseline precision:", precision_score(y_test_py, baseline_pred_py, zero_division=0))
print("baseline recall   :", recall_score(y_test_py, baseline_pred_py, zero_division=0))
print("baseline f1       :", f1_score(y_test_py, baseline_pred_py, zero_division=0))

# Layer-wise sweep on full filtered split
sample_inputs = tokenizer("for i in range(3):\n    print(i)", return_tensors="pt", add_special_tokens=False).to(device)
with torch.no_grad():
    sample_outputs = model(**sample_inputs, output_hidden_states=True)

num_layers = len(sample_outputs.hidden_states)
print("Number of hidden-state layers:", num_layers)

layer_results_py = []

for layer_idx in range(num_layers):
    print(f"\n=== Python Layer {layer_idx} ===")

    X_train_py, y_train_py = collect_dataset(
        df, train_idx, layer_idx=layer_idx, print_every=5, max_snippets=max_train_snippets
    )
    X_test_py, y_test_py = collect_dataset(
        df, test_idx, layer_idx=layer_idx, print_every=5, max_snippets=max_test_snippets
    )

    probe_py = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    probe_py.fit(X_train_py, y_train_py)
    y_pred_py = probe_py.predict(X_test_py)

    prec = precision_score(y_test_py, y_pred_py, zero_division=0)
    rec = recall_score(y_test_py, y_pred_py, zero_division=0)
    f1 = f1_score(y_test_py, y_pred_py, zero_division=0)

    print(f"precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    layer_results_py.append({
        "layer": layer_idx,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

results_py_df = pd.DataFrame(layer_results_py)
results_py_df

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(results_py_df["layer"], results_py_df["f1"], marker="o")
plt.xlabel("Layer")
plt.ylabel("F1 score")
plt.title("Layer-wise probe F1 on Python iterator tracking")
plt.grid(True)
plt.show()
