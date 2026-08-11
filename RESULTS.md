# Probing Results — Index/Key Variable Detection in Code LLMs

**Model:** `Qwen/Qwen2.5-1.5B` (28 layers, hidden size 1536)  
**Task:** Binary token classification — detect variables used as array indices or dictionary keys  
**Probe:** Logistic Regression (class-balanced, C=1.0) trained on per-layer hidden states  
**Dataset:** XLCoST (872K parallel code-text pairs, 7 languages)

---

## 1. Baseline — Python (Original Variable Names)

**Data:** 500 XLCoST Python programs → 255 with labeled index/key vars  
**Tokens:** 63,851 total | 3,737 positive (5.9%) | imbalance 16.1:1  
**Naive baseline (all-zero classifier):** 94.1%

### Per-Layer Probe Accuracy & Macro F1

| Layer | Train Acc | Test Acc | Train F1 | Test F1 |
|------:|----------:|---------:|---------:|--------:|
| 0 | 0.951 | 0.950 | 0.838 | 0.836 |
| 1 | 0.982 | 0.977 | 0.928 | 0.912 |
| 2 | 0.988 | 0.983 | 0.950 | 0.930 |
| 3 | 0.992 | 0.987 | 0.966 | 0.944 |
| 4 | 0.994 | 0.988 | 0.974 | 0.948 |
| 5 | 0.996 | 0.988 | 0.982 | 0.951 |
| 6 | 0.996 | 0.989 | 0.983 | 0.952 |
| 7 | 0.997 | 0.990 | 0.987 | 0.955 |
| **8** | **0.998** | **0.991** | **0.990** | **0.959** |
| 9 | 0.998 | 0.990 | 0.990 | 0.955 |
| 10 | 0.998 | 0.988 | 0.992 | 0.949 |
| 11 | 0.999 | 0.989 | 0.995 | 0.951 |
| 12 | 0.999 | 0.988 | 0.995 | 0.946 |
| 13 | 0.999 | 0.990 | 0.996 | 0.954 |
| 14 | 0.999 | 0.989 | 0.997 | 0.952 |
| 15 | 0.999 | 0.989 | 0.998 | 0.950 |
| 16 | 1.000 | 0.989 | 0.998 | 0.953 |
| 17 | 0.999 | 0.988 | 0.998 | 0.947 |
| 18 | 0.999 | 0.988 | 0.998 | 0.947 |
| 19 | 1.000 | 0.989 | 0.998 | 0.951 |
| 20 | 1.000 | 0.988 | 0.998 | 0.947 |
| 21 | 0.999 | 0.988 | 0.998 | 0.948 |
| 22 | 1.000 | 0.988 | 0.998 | 0.947 |
| 23 | 1.000 | 0.988 | 0.999 | 0.946 |
| 24 | 1.000 | 0.987 | 1.000 | 0.941 |
| 25 | 1.000 | 0.987 | 0.999 | 0.941 |
| 26 | 1.000 | 0.986 | 1.000 | 0.938 |
| 27 | 1.000 | 0.987 | 1.000 | 0.942 |
| 28 | 0.999 | 0.986 | 0.997 | 0.938 |

### Best Layer (Layer 8) — Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| non-index | 1.00 | 0.99 | 1.00 | 12,024 |
| index_key | 0.89 | 0.96 | 0.92 | 747 |
| **macro avg** | **0.94** | **0.98** | **0.96** | 12,771 |

**Confusion Matrix (Layer 8):**
- TN: 11,934 | FP: 90 | FN: 29 | TP: 718
- TPR (recall): 0.961 | FPR: 0.007 | Precision: 0.889

---

## 2. Cross-Language Generalization (Baseline)

Python-trained probe (layer-wise) applied directly to other languages — no retraining.

| Language | Best Cross-Lang Acc | Best Layer | Cosine Similarity @ Layer 8 |
|----------|--------------------:|-----------:|----------------------------:|
| C++ | 0.984 | 7 | 0.2960 |
| C | 0.973 | 6 | 0.1513 |
| C# | 0.985 | 14 | 0.2237 |
| Javascript | 0.988 | 9 | 0.2849 |
| Java | 0.985 | 14 | 0.1914 |

> Cosine similarity measures alignment of Python vs. target-language probe weight vectors.
> Low cosine with high accuracy suggests the concept is encoded in parallel but not identical directions.

---

## 3. Renamed Variables — Does the Model Rely on Surface Names?

Variables in all programs were replaced with random everyday nouns (e.g., `i` → `drum`, `key` → `leaf`, `arr` → `mesh`) from a pool of 120 words. The probe is then retrained and evaluated.

**Data:** 500 renamed Python programs → 252 with labels  
**Tokens:** 61,357 total | 3,521 positive (5.7%) | imbalance 16.4:1  
**Naive baseline:** 94.3%

### Per-Layer Probe Accuracy & Macro F1 (Renamed)

| Layer | Train Acc | Test Acc | Train F1 | Test F1 |
|------:|----------:|---------:|---------:|--------:|
| 0 | 0.876 | 0.878 | 0.705 | 0.708 |
| 1 | 0.966 | 0.955 | 0.878 | 0.839 |
| 2 | 0.977 | 0.965 | 0.909 | 0.866 |
| 3 | 0.990 | 0.978 | 0.956 | 0.909 |
| 4 | 0.990 | 0.976 | 0.957 | 0.898 |
| 5 | 0.991 | 0.978 | 0.962 | 0.910 |
| 6 | 0.993 | 0.980 | 0.968 | 0.914 |
| 7 | 0.995 | 0.982 | 0.978 | 0.921 |
| 8 | 0.995 | 0.981 | 0.977 | 0.919 |
| 9 | 0.995 | 0.981 | 0.978 | 0.919 |
| 10 | 0.995 | 0.981 | 0.979 | 0.917 |
| 11 | 0.995 | 0.980 | 0.978 | 0.914 |
| 12 | 0.996 | 0.983 | 0.982 | 0.925 |
| **13** | **0.997** | **0.983** | **0.985** | **0.926** |
| 14 | 0.997 | 0.982 | 0.988 | 0.918 |
| 15 | 0.997 | 0.982 | 0.988 | 0.919 |
| 16 | 0.997 | 0.982 | 0.987 | 0.921 |
| 17 | 0.998 | 0.982 | 0.989 | 0.921 |
| 18 | 0.998 | 0.982 | 0.989 | 0.919 |
| 19 | 0.998 | 0.982 | 0.991 | 0.919 |
| 20 | 0.997 | 0.982 | 0.988 | 0.920 |
| 21 | 0.998 | 0.979 | 0.989 | 0.909 |
| 22 | 0.997 | 0.980 | 0.989 | 0.912 |
| 23 | 0.999 | 0.980 | 0.994 | 0.912 |
| 24 | 0.999 | 0.979 | 0.996 | 0.908 |
| 25 | 0.999 | 0.981 | 0.997 | 0.913 |
| 26 | 1.000 | 0.980 | 0.998 | 0.910 |
| 27 | 1.000 | 0.978 | 0.999 | 0.899 |
| 28 | 0.995 | 0.976 | 0.979 | 0.895 |

### Best Layer (Layer 13) — Classification Report (Renamed)

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| non-index | 0.99 | 0.99 | 0.99 | 11,568 |
| index_key | 0.83 | 0.89 | 0.86 | 704 |
| **macro avg** | **0.91** | **0.94** | **0.93** | 12,272 |

**Confusion Matrix (Layer 13):**
- TN: 11,439 | FP: 129 | FN: 74 | TP: 630
- TPR (recall): 0.895 | FPR: 0.011 | Precision: 0.830

### Baseline vs. Renamed — Accuracy Delta by Layer

| Layer | Original | Renamed | Δ (orig − renamed) |
|------:|---------:|--------:|--------------------:|
| 0 | 0.950 | 0.878 | +0.072 |
| 1 | 0.977 | 0.955 | +0.022 |
| 2 | 0.983 | 0.965 | +0.018 |
| 3 | 0.987 | 0.978 | +0.009 |
| 4 | 0.988 | 0.976 | +0.012 |
| 5 | 0.988 | 0.978 | +0.010 |
| 6 | 0.989 | 0.980 | +0.009 |
| 7 | 0.990 | 0.982 | +0.008 |
| 8 | 0.991 | 0.981 | +0.010 |
| 9 | 0.990 | 0.981 | +0.009 |
| 10 | 0.988 | 0.981 | +0.007 |
| 11 | 0.989 | 0.980 | +0.009 |
| 12 | 0.988 | 0.983 | +0.005 |
| 13 | 0.990 | 0.983 | +0.007 |
| 14 | 0.989 | 0.982 | +0.007 |
| 15 | 0.989 | 0.982 | +0.007 |
| 16 | 0.989 | 0.982 | +0.007 |
| 17 | 0.988 | 0.982 | +0.006 |
| 18 | 0.988 | 0.982 | +0.006 |
| 19 | 0.989 | 0.982 | +0.007 |
| 20 | 0.988 | 0.982 | +0.006 |
| 21 | 0.988 | 0.979 | +0.009 |
| 22 | 0.988 | 0.980 | +0.008 |
| 23 | 0.988 | 0.980 | +0.008 |
| 24 | 0.987 | 0.979 | +0.008 |
| 25 | 0.987 | 0.981 | +0.006 |
| 26 | 0.986 | 0.980 | +0.006 |
| 27 | 0.987 | 0.978 | +0.009 |
| 28 | 0.986 | 0.976 | +0.010 |

> **Interpretation:** The accuracy drop from renaming is very small (< 1pp at mid-to-late layers).
> This strongly suggests the model encodes **syntactic role** (position inside a subscript expression)
> rather than surface variable names like `i`, `j`, or `key`.

### Cross-Language Generalization (Renamed Variables)

Python-renamed probe applied to other languages (also with renamed variables):

| Language | Best Layer | Best Test Acc |
|----------|----------:|--------------:|
| C++ | 9 | 0.987 |
| C | 12 | 0.983 |
| C# | 15 | 0.989 |
| Javascript | 13 | 0.994 |

---

## 4. Cross-Dataset Generalization — CodeComplex

Tests whether probes trained on XLCoST generalize to **CodeComplex** — a different distribution
of real-world Java programs (4,517 Codeforces competitive programming submissions).

**Dataset:** `codeparrot/codecomplex`  
**Complexity classes:** constant, linear, logn, nlogn, quadratic, cubic, np  
**Programs used:** 300 (205 with valid index/key labels)  
**Tokens:** 85,406 total | 5,308 positive (6.2%) | imbalance 15.1:1  
**Naive baseline:** 93.8%

### CodeComplex-Trained Probe (per layer)

| Layer | Train Acc | Test Acc | Train F1 | Test F1 |
|------:|----------:|---------:|---------:|--------:|
| 0 | 0.962 | 0.959 | 0.871 | 0.863 |
| 1 | 0.987 | 0.983 | 0.951 | 0.932 |
| 2 | 0.990 | 0.985 | 0.959 | 0.940 |
| 3 | 0.992 | 0.987 | 0.967 | 0.947 |
| 4 | 0.994 | 0.988 | 0.974 | 0.951 |
| 5 | 0.995 | 0.989 | 0.978 | 0.956 |
| 6 | 0.995 | 0.989 | 0.980 | 0.953 |
| **7** | **0.996** | **0.990** | **0.983** | **0.959** |
| 8 | 0.996 | 0.988 | 0.982 | 0.952 |
| 9 | 0.996 | 0.989 | 0.982 | 0.955 |
| 10 | 0.996 | 0.989 | 0.984 | 0.953 |
| 11 | 0.996 | 0.988 | 0.984 | 0.951 |
| 12 | 0.996 | 0.988 | 0.982 | 0.952 |
| 13 | 0.996 | 0.987 | 0.985 | 0.948 |
| 14 | 0.996 | 0.987 | 0.984 | 0.946 |
| 15 | 0.996 | 0.986 | 0.984 | 0.943 |
| 16 | 0.996 | 0.987 | 0.984 | 0.945 |
| 17 | 0.996 | 0.987 | 0.985 | 0.947 |
| 18 | 0.997 | 0.985 | 0.987 | 0.939 |
| 19 | 0.997 | 0.985 | 0.987 | 0.939 |
| 20 | 0.996 | 0.984 | 0.984 | 0.934 |
| 21 | 0.997 | 0.984 | 0.987 | 0.934 |
| 22 | 0.997 | 0.985 | 0.989 | 0.937 |
| 23 | 0.997 | 0.984 | 0.988 | 0.934 |
| 24 | 0.997 | 0.984 | 0.987 | 0.933 |
| 25 | 0.998 | 0.982 | 0.991 | 0.927 |
| 26 | 0.998 | 0.984 | 0.992 | 0.933 |
| 27 | 0.999 | 0.983 | 0.994 | 0.931 |
| 28 | 0.995 | 0.983 | 0.981 | 0.931 |

### Best Layer (Layer 7) — Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| non-index | 1.00 | 0.99 | 0.99 | 16,020 |
| index_key | 0.89 | 0.96 | 0.92 | 1,062 |
| **macro avg** | **0.94** | **0.98** | **0.96** | 17,082 |

**Confusion Matrix (Layer 7):**
- TN: 15,893 | FP: 127 | FN: 42 | TP: 1,020
- TPR (recall): 0.960 | FPR: 0.008 | Precision: 0.889

### XLCoST → CodeComplex (Zero-Shot Transfer)

XLCoST Java-trained probe applied directly to CodeComplex (no retraining):

| Layer | CC Probe Test F1 | XLCoST→CC F1 | XLCoST→CC Acc |
|------:|-----------------:|-------------:|--------------:|
| 0 | 0.863 | 0.859 | 0.969 |
| 1 | 0.932 | 0.818 | 0.959 |
| 5 | 0.956 | 0.862 | 0.970 |
| 6 | 0.953 | 0.890 | 0.973 |
| 7 | 0.959 | 0.905 | 0.979 |
| 8 | 0.952 | 0.903 | 0.977 |
| **13** | **0.948** | **0.911** | **0.980** |
| 10 | 0.953 | 0.908 | 0.978 |
| 11 | 0.951 | 0.907 | 0.978 |
| 12 | 0.952 | 0.907 | 0.978 |

**Best zero-shot transfer: F1 = 0.911, Acc = 0.980 (layer 13)**

### CodeComplex → XLCoST Java (Reverse Transfer)

CodeComplex-trained probe applied to XLCoST Java data:

| Layer | XLCoST Probe Test F1 | CC→XLCoST F1 |
|------:|---------------------:|-------------:|
| 7 | 0.976 | 0.861 |
| 8 | 0.983 | 0.869 |
| **20** | **0.972** | **0.893** |
| 6 | 0.967 | 0.890 |
| 5 | 0.966 | 0.889 |

**Best reverse transfer: F1 = 0.893 (layer 20)**

---

## 5. Summary

| Experiment | Best Layer | Test Acc | Test F1 (macro) |
|---|---:|---:|---:|
| Python baseline (original names) | 8 | 0.991 | 0.959 |
| Python renamed variables | 13 | 0.983 | 0.926 |
| CodeComplex Java (in-domain) | 7 | 0.990 | 0.959 |
| XLCoST Java → CodeComplex (zero-shot) | 13 | 0.980 | 0.911 |
| CodeComplex → XLCoST Java (zero-shot) | 20 | — | 0.893 |

### Key Findings

1. **High probe accuracy across all conditions** — Qwen2.5-1.5B reliably encodes index/key variable roles. Even the embedding layer (layer 0) already achieves F1 ~0.84, suggesting some role information is in the token embeddings themselves.

2. **Variable renaming has minimal impact** — Accuracy drops less than 1pp at mid/late layers when all variable names are replaced with random nouns. The model detects syntactic position (inside `[...]`), not surface-form heuristics like `i` or `key`.

3. **Strong cross-language generalization** — A Python-trained probe transfers directly to Java, C++, C, C#, and JavaScript with 97–98.8% accuracy, despite different syntax. The concept is encoded in a language-agnostic way.

4. **Cross-dataset transfer holds** — Probes trained on XLCoST (algorithmic snippets) generalize to CodeComplex (real competitive programming, more complex, messy code) with F1 = 0.911 zero-shot.

5. **Best representation in early-to-middle layers** — Peak performance is typically around layers 7–13 (out of 28). Later layers show slight degradation, suggesting role information is sharpest in the middle of the network.

---

## 6. Visualizations

All plots are saved in `results/`:

| File | Description |
|------|-------------|
| `results/baseline/probe_accuracy_index_key.png` | Per-layer probe accuracy (test/train) — baseline |
| `results/baseline/cross_language_all.png` | Cross-language accuracy for all 5 languages |
| `results/baseline/cross_language_all.pdf` | PDF version for paper |
| `results/baseline/cosine_similarity_all.png` | Probe weight cosine similarity vs Python |
| `results/baseline/cosine_similarity_all.pdf` | PDF version |
| `results/renamed/probe_accuracy_renamed.png` | Per-layer accuracy on renamed variables |
| `results/renamed/cross_language_all_renamed.png` | Cross-language accuracy (renamed) |
| `results/renamed/cosine_similarity_all_renamed.png` | Cosine similarity (renamed) |
| `results/renamed/comparison_orig_vs_renamed.png` | Side-by-side: original vs renamed accuracy |
| `results/codecomplex/probe_accuracy_f1_codecomplex.png` | CodeComplex probe accuracy/F1 |
| `results/codecomplex/cosine_similarity_xlcost_vs_codecomplex.png` | XLCoST vs CodeComplex probe alignment |
