"""java_ifelse_probe"""

"""java_ifelse_probe"""

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

"""java_ifelse_probe"""

"""java_ifelse_probe"""

"""java_ifelse_probe"""

# # Java If-Else Probe
# 
# This notebook trains a probe to identify tokens inside if-else conditional blocks in Java.

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
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Model + tokenizer

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

# Java conditional block matching

JAVA_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

def java_find_matching_brace(code, open_idx):
    depth = 0
    for i in range(open_idx, len(code)):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1

def find_if_else_spans_java(code):
    spans = []
    
    # Match: if (...) {
    if_pat = re.compile(r"\bif\s*\([^)]*\)\s*\{")
    for m in if_pat.finditer(code):
        brace_open = code.find("{", m.start())
        if brace_open != -1:
            brace_close = java_find_matching_brace(code, brace_open)
            if brace_close != -1:
                spans.append((brace_open + 1, brace_close))
                
    # Match: else {
    else_pat = re.compile(r"\belse\s*\{")
    for m in else_pat.finditer(code):
        brace_open = code.find("{", m.start())
        if brace_open != -1:
            brace_close = java_find_matching_brace(code, brace_open)
            if brace_close != -1:
                spans.append((brace_open + 1, brace_close))
                
    return spans

def find_all_name_spans_java(code):
    return [(m.start(), m.end()) for m in JAVA_IDENT_RE.finditer(code)]

def build_token_level_data_for_snippet_java(code, layer_idx, only_identifiers=False):
    spans = find_if_else_spans_java(code)

    tok = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    labels = align_spans_to_tokens(spans, tok["offset_mapping"])

    inputs = tokenizer(code, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx][0].detach().cpu()
    y = torch.tensor(labels, dtype=torch.long)

    tokens = [tokenizer.decode([tok_id]) for tok_id in tok["input_ids"]]
    offsets = tok["offset_mapping"]

    if only_identifiers:
        name_spans = find_all_name_spans_java(code)
        name_mask = align_spans_to_tokens(name_spans, offsets)
        keep = [i for i, m in enumerate(name_mask) if m == 1]

        hidden = hidden[keep]
        y = y[keep]
        tokens = [tokens[i] for i in keep]
        offsets = [offsets[i] for i in keep]

    assert hidden.shape[0] == len(y), f"Mismatch: hidden {hidden.shape[0]} vs labels {len(y)}"
    return hidden, y, tokens, offsets

def collect_dataset_java(df, snippet_indices, layer_idx, print_every=10, max_snippets=None, only_identifiers=False):
    X_list, y_list = [], []
    skipped, kept = 0, 0

    if max_snippets is not None:
        snippet_indices = snippet_indices[:max_snippets]

    total = len(snippet_indices)

    for j, idx in enumerate(snippet_indices):
        if j % print_every == 0:
            print(f"[JAVA layer {layer_idx}] processing {j}/{total}")

        code = df.loc[idx, "code"]

        try:
            X_snip, y_snip, _, _ = build_token_level_data_for_snippet_java(
                code,
                layer_idx=layer_idx,
                only_identifiers=only_identifiers
            )

            if len(y_snip) == 0 or y_snip.sum().item() == 0:
                skipped += 1
                continue

            X_list.append(X_snip.numpy())
            y_list.append(y_snip.numpy())
            kept += 1

        except Exception as e:
            skipped += 1
            print(f"Skipping snippet {idx} due to error: {e}")

    if len(X_list) == 0:
        raise ValueError("No valid Java snippets collected.")

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

# Load and validate Java snippets

import os
if os.path.exists('/content/drive'):
    java_csv_path = "/content/drive/MyDrive/algoverse/data/java_ifelse_snippets.csv"
else:
    java_csv_path = "./data/java_ifelse_snippets.csv"
df_java = pd.read_csv(java_csv_path)

print("raw java shape:", df_java.shape)

def is_valid_java_like(code):
    if not isinstance(code, str) or len(code.strip()) == 0:
        return False
    if code.count("{") != code.count("}"):
        return False
    if "if (" not in code:
        return False
    if "System.out.println" not in code:
        return False
    return True


def has_if_else_java(code):
    return bool(re.search(r"\bif\s*\(", code))


df_java["is_valid_java_like"] = df_java["code"].apply(is_valid_java_like)
df_java["has_if_else"] = df_java["code"].apply(has_if_else_java)

print(df_java[["is_valid_java_like", "has_if_else"]].value_counts(dropna=False))

df_java = df_java[(df_java["is_valid_java_like"]) & (df_java["has_if_else"])].reset_index(drop=True)
max_lines_per_snippet = 50
df_java["line_count"] = df_java["code"].apply(lambda x: len(str(x).splitlines()))
df_java = df_java[df_java["line_count"] <= max_lines_per_snippet].reset_index(drop=True)
print("filtered java shape (<=50 lines):", df_java.shape)

# Train/test split (full filtered dataset)

java_snippet_indices = list(range(len(df_java)))
train_idx_java, test_idx_java = train_test_split(
    java_snippet_indices,
    test_size=0.2,
    random_state=42
)

print("train java snippets:", len(train_idx_java))
print("test java snippets :", len(test_idx_java))

# Probe training at one layer (debug run)

layer_idx_java = 4

X_train_java, y_train_java = collect_dataset_java(
    df_java,
    train_idx_java,
    layer_idx=layer_idx_java,
    print_every=5,
    max_snippets=None,
    only_identifiers=False
)

X_test_java, y_test_java = collect_dataset_java(
    df_java,
    test_idx_java,
    layer_idx=layer_idx_java,
    print_every=5,
    max_snippets=None,
    only_identifiers=False
)

probe_java = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
probe_java.fit(X_train_java, y_train_java)
y_pred_java = probe_java.predict(X_test_java)

print("accuracy :", accuracy_score(y_test_java, y_pred_java))
print("precision:", precision_score(y_test_java, y_pred_java, zero_division=0))
print("recall   :", recall_score(y_test_java, y_pred_java, zero_division=0))
print("f1       :", f1_score(y_test_java, y_pred_java, zero_division=0))
print("\nClassification report:\n")
print(classification_report(y_test_java, y_pred_java, zero_division=0))

baseline_pred_java = np.zeros_like(y_test_java)

print("baseline accuracy :", accuracy_score(y_test_java, baseline_pred_java))
print("baseline precision:", precision_score(y_test_java, baseline_pred_java, zero_division=0))
print("baseline recall   :", recall_score(y_test_java, baseline_pred_java, zero_division=0))
print("baseline f1       :", f1_score(y_test_java, baseline_pred_java, zero_division=0))

# Layer-wise sweep

sample_inputs = tokenizer(
    "if (x > 3) { System.out.println(x); }",
    return_tensors="pt",
    add_special_tokens=False
).to(device)

with torch.no_grad():
    sample_outputs = model(**sample_inputs, output_hidden_states=True)

num_layers = len(sample_outputs.hidden_states)
print("Number of hidden-state layers:", num_layers)

layer_results_java = []

for layer_idx in range(num_layers):
    print(f"\n=== Java Layer {layer_idx} ===")

    X_train_java, y_train_java = collect_dataset_java(
        df_java, train_idx_java, layer_idx=layer_idx, print_every=5, max_snippets=None, only_identifiers=False
    )
    X_test_java, y_test_java = collect_dataset_java(
        df_java, test_idx_java, layer_idx=layer_idx, print_every=5, max_snippets=None, only_identifiers=False
    )

    probe_java = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    probe_java.fit(X_train_java, y_train_java)
    y_pred_java = probe_java.predict(X_test_java)

    prec = precision_score(y_test_java, y_pred_java, zero_division=0)
    rec = recall_score(y_test_java, y_pred_java, zero_division=0)
    f1 = f1_score(y_test_java, y_pred_java, zero_division=0)

    print(f"precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    layer_results_java.append({
        "layer": layer_idx,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

results_java_df = pd.DataFrame(layer_results_java)
print(results_java_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(results_java_df["layer"], results_java_df["f1"], marker="o")
plt.xlabel("Layer")
plt.ylabel("F1 score")
plt.title("Layer-wise probe F1 on Java conditional (if-else) tracking")
plt.grid(True)
plt.show()
