"""csharp_class_struct_probe"""

"""csharp_class_struct_probe"""

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

"""csharp_class_struct_probe"""

"""csharp_class_struct_probe"""

# # C# Class/Struct Probe
# 
# This notebook trains a probe to identify tokens inside C# class and struct definitions.

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

# C# class/struct span detector

def find_matching_brace(code, open_idx):
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

def find_class_struct_spans_csharp(code):
    spans = []
    pattern = re.compile(r'\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b')
    for m in pattern.finditer(code):
        start_pos = m.start()
        search_start = m.end()
        brace_pos = -1
        invalid = False
        for idx in range(search_start, len(code)):
            char = code[idx]
            if char == '{':
                brace_pos = idx
                break
            if char in (';', '(', ')', '=', ','):
                invalid = True
                break
        
        if invalid or brace_pos == -1:
            continue
            
        matching_brace = find_matching_brace(code, brace_pos)
        if matching_brace == -1:
            continue
            
        end_pos = matching_brace + 1
        for idx in range(matching_brace + 1, len(code)):
            if code[idx].isspace():
                continue
            if code[idx] == ';':
                end_pos = idx + 1
                break
            break
            
        spans.append((start_pos, end_pos))
       return spans

def build_token_level_data_for_snippet(code, layer_idx):
    spans = find_class_struct_spans_csharp(code)

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
            print(f"[CS layer {layer_idx}] processing {j}/{total}")

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

# Data prep

import os
if os.path.exists('/content/drive'):
    csharp_csv_path = "/content/drive/MyDrive/algoverse/data/csharp_class_struct_snippets.csv"
else:
    csharp_csv_path = "./data/csharp_class_struct_snippets.csv"
df_raw = pd.read_csv(csharp_csv_path)

max_lines_per_snippet = None
df_raw["line_count"] = df_raw["code"].apply(lambda x: len(str(x).splitlines()))
if max_lines_per_snippet is not None:
    df = df_raw[df_raw["line_count"] <= max_lines_per_snippet].reset_index(drop=True)
else:
    df = df_raw.copy().reset_index(drop=True)

snippet_indices = list(range(len(df)))
train_idx, test_idx = train_test_split(snippet_indices, test_size=0.2, random_state=42)

print("raw csharp shape:", df_raw.shape)
print("filtered csharp shape (<=50 lines):", df.shape)
print("train csharp snippets:", len(train_idx))
print("test csharp snippets :", len(test_idx))

# Probe training at one layer (debug run)

layer_idx_csharp = 4
max_train_snippets = None
max_test_snippets = None

X_train_csharp, y_train_csharp = collect_dataset(
    df,
    train_idx,
    layer_idx=layer_idx_csharp,
    print_every=20,
    max_snippets=max_train_snippets
)

X_test_csharp, y_test_csharp = collect_dataset(
    df,
    test_idx,
    layer_idx=layer_idx_csharp,
    print_every=20,
    max_snippets=max_test_snippets
)

probe_csharp = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
probe_csharp.fit(X_train_csharp, y_train_csharp)
y_pred_csharp = probe_csharp.predict(X_test_csharp)

print("accuracy :", accuracy_score(y_test_csharp, y_pred_csharp))
print("precision:", precision_score(y_test_csharp, y_pred_csharp, zero_division=0))
print("recall   :", recall_score(y_test_csharp, y_pred_csharp, zero_division=0))
print("f1       :", f1_score(y_test_csharp, y_pred_csharp, zero_division=0))
print("\nClassification report:\n")
print(classification_report(y_test_csharp, y_pred_csharp, zero_division=0))

baseline_pred_csharp = np.zeros_like(y_test_csharp)
print("baseline accuracy :", accuracy_score(y_test_csharp, baseline_pred_csharp))
print("baseline precision:", precision_score(y_test_csharp, baseline_pred_csharp, zero_division=0))
print("baseline recall   :", recall_score(y_test_csharp, baseline_pred_csharp, zero_division=0))
print("baseline f1       :", f1_score(y_test_csharp, baseline_pred_csharp, zero_division=0))

# Layer-wise sweep on full filtered split

sample_inputs = tokenizer("class MyClass { int x; }", return_tensors="pt", add_special_tokens=False).to(device)
with torch.no_grad():
    sample_outputs = model(**sample_inputs, output_hidden_states=True)

num_layers = len(sample_outputs.hidden_states)
print("Number of hidden-state layers:", num_layers)

layer_results_csharp = []

for layer_idx in range(num_layers):
    print(f"\n=== C# Layer {layer_idx} ===")

    X_train_csharp, y_train_csharp = collect_dataset(
        df, train_idx, layer_idx=layer_idx, print_every=20, max_snippets=max_train_snippets
    )
    X_test_csharp, y_test_csharp = collect_dataset(
        df, test_idx, layer_idx=layer_idx, print_every=20, max_snippets=max_test_snippets
    )

    probe_csharp = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    probe_csharp.fit(X_train_csharp, y_train_csharp)
    y_pred_csharp = probe_csharp.predict(X_test_csharp)

    prec = precision_score(y_test_csharp, y_pred_csharp, zero_division=0)
    rec = recall_score(y_test_csharp, y_pred_csharp, zero_division=0)
    f1 = f1_score(y_test_csharp, y_pred_csharp, zero_division=0)

    print(f"precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    layer_results_csharp.append({
        "layer": layer_idx,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

results_csharp_df = pd.DataFrame(layer_results_csharp)
results_csharp_df

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(results_csharp_df["layer"], results_csharp_df["f1"], marker="o")
plt.xlabel("Layer")
plt.ylabel("F1 score")
plt.title("Layer-wise probe F1 on C# class/struct tracking")
plt.grid(True)
plt.show()
