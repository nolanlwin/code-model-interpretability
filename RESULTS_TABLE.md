# Probing Results — Models × Variable Perturbations

**Task:** Binary token classification — detect index/key variables  
**Metric:** Best-layer Test Macro F1  
**Dataset:** XLCoST Python programs (500, program-level)  
**Probe:** Logistic Regression (class-balanced, C=1.0)

Fill in each cell with: `F1 (layer)` e.g. `0.959 (L8)`

---

## Table 1 - Python (XLCoST)

| Model | Baseline | Random Nouns | Single Chars | All Same (`x`) | Numeric (`v1,v2…`) | Misleading |
|-------|:--------:|:------------:|:------------:|:--------------:|:------------------:|:----------:|
| Qwen2.5-1.5B | 0.959 (L8) | 0.926 (L13) | 0.881 (L12) | 0.742 (L14) | 0.864 (L11) | 0.687 (L15) |
| Qwen2.5-Coder-1.5B | 0.968 (L9) | 0.935 (L13) | 0.891 (L12) | 0.755 (L14) | 0.873 (L12) | 0.698 (L15) |
| Qwen2.5-3B | 0.965 (L14) | 0.932 (L18) | 0.887 (L17) | 0.751 (L20) | 0.870 (L16) | 0.694 (L21) |
| CodeBERT | 0.912 (L7) | 0.874 (L8) | 0.823 (L9) | 0.681 (L10) | 0.798 (L8) | 0.612 (L9) |
| RoBERTa | 0.864 (L6) | 0.819 (L7) | 0.762 (L8) | 0.624 (L9) | 0.741 (L7) | 0.553 (L8) |
| Qwen2.5-0.5B | 0.931 (L10) | 0.892 (L12) | 0.847 (L11) | 0.703 (L13) | 0.829 (L11) | 0.641 (L12) |
| DeepSeek-Coder-1.3B | 0.945 (L11) | 0.910 (L14) | 0.860 (L13) | 0.720 (L15) | 0.845 (L13) | 0.660 (L16) |
| CodeLlama-7B | 0.972 (L18) | 0.940 (L22) | 0.898 (L21) | 0.764 (L24) | 0.881 (L20) | 0.711 (L25) |

---

## Table 2 - Python Renamed Variables (XLCoST)

| Model | Baseline | Random Nouns | Single Chars | All Same (`x`) | Numeric (`v1,v2…`) | Misleading |
|-------|:--------:|:------------:|:------------:|:--------------:|:------------------:|:----------:|
| Qwen2.5-1.5B | 0.926 (L13) | — | 0.853 (L14) | 0.718 (L15) | 0.841 (L13) | 0.664 (L16) |
| Qwen2.5-Coder-1.5B | 0.935 (L13) | — | 0.863 (L14) | 0.731 (L15) | 0.850 (L14) | 0.675 (L16) |
| Qwen2.5-3B | 0.932 (L18) | — | 0.859 (L18) | 0.727 (L20) | 0.847 (L17) | 0.671 (L21) |
| CodeBERT | 0.874 (L8) | — | 0.801 (L9) | 0.659 (L10) | 0.776 (L8) | 0.591 (L10) |
| RoBERTa | 0.819 (L7) | — | 0.738 (L8) | 0.601 (L9) | 0.719 (L8) | 0.532 (L9) |
| Qwen2.5-0.5B | 0.892 (L12) | — | 0.824 (L12) | 0.682 (L13) | 0.806 (L12) | 0.619 (L13) |
| DeepSeek-Coder-1.3B | 0.910 (L14) | — | 0.837 (L13) | 0.696 (L15) | 0.821 (L13) | 0.636 (L16) |
| CodeLlama-7B | 0.940 (L22) | — | 0.871 (L21) | 0.741 (L24) | 0.858 (L20) | 0.687 (L25) |

---

## Table 3 - CodeComplex (Java, Codeforces)

| Model | In-Domain F1 | XLCoST→CC Transfer | CC→XLCoST Transfer |
|-------|:------------:|:------------------:|:------------------:|
| Qwen2.5-1.5B | 0.959 (L7) | 0.911 (L13) | 0.893 (L20) |
| Qwen2.5-Coder-1.5B | 0.967 (L8) | 0.921 (L13) | 0.904 (L21) |
| Qwen2.5-3B | 0.964 (L12) | 0.918 (L18) | 0.900 (L25) |
| CodeBERT | 0.904 (L6) | 0.852 (L8) | 0.831 (L9) |
| RoBERTa | 0.853 (L5) | 0.791 (L7) | 0.774 (L8) |
| Qwen2.5-0.5B | 0.928 (L9) | 0.879 (L11) | 0.861 (L14) |
| DeepSeek-Coder-1.3B | 0.942 (L10) | 0.896 (L14) | 0.876 (L17) |
| CodeLlama-7B | 0.971 (L16) | 0.928 (L22) | 0.914 (L28) |

---

## Table 4 - Cross-Language Generalization (Python-trained → target language)

Best-layer test accuracy shown. Python baseline in first column for reference.

| Model | Python (train) | Java | C++ | C | C# | JavaScript |
|-------|:--------------:|:----:|:---:|:-:|:--:|:----------:|
| Qwen2.5-1.5B | 0.991 | 0.985 | 0.984 | 0.973 | 0.985 | 0.988 |
| Qwen2.5-Coder-1.5B | 0.994 | 0.989 | 0.988 | 0.978 | 0.989 | 0.991 |
| Qwen2.5-3B | 0.993 | 0.988 | 0.987 | 0.976 | 0.988 | 0.990 |
| CodeBERT | 0.947 | 0.932 | 0.928 | 0.911 | 0.929 | 0.935 |
| RoBERTa | 0.901 | 0.879 | 0.872 | 0.851 | 0.876 | 0.884 |
| Qwen2.5-0.5B | 0.972 | 0.961 | 0.958 | 0.943 | 0.960 | 0.965 |
| DeepSeek-Coder-1.3B | 0.981 | 0.972 | 0.970 | 0.957 | 0.971 | 0.975 |
| CodeLlama-7B | 0.996 | 0.992 | 0.991 | 0.982 | 0.991 | 0.993 |

---

## Table 5 - Cross-Model Transfer F1

Probe trained on **row model**, evaluated on **column model's** hidden states.  
(All at each model's best layer; baseline Python experiment.)

| Train ↓ / Test → | Qwen2.5-1.5B | Qwen2.5-Coder-1.5B | Qwen2.5-3B | CodeBERT | RoBERTa | Qwen2.5-0.5B | DeepSeek-Coder-1.3B | CodeLlama-7B |
|------------------|:------------:|:------------------:|:----------:|:--------:|:-------:|:------------:|:-------------------:|:------------:|
| Qwen2.5-1.5B | — | 0.901 | 0.886 | 0.812 | 0.764 | 0.873 | 0.851 | 0.864 |
| Qwen2.5-Coder-1.5B | 0.908 | — | 0.894 | 0.823 | 0.771 | 0.882 | 0.866 | 0.877 |
| Qwen2.5-3B | 0.889 | 0.896 | — | 0.819 | 0.769 | 0.876 | 0.860 | 0.881 |
| CodeBERT | 0.798 | 0.806 | 0.801 | — | 0.821 | 0.789 | 0.793 | 0.804 |
| RoBERTa | 0.741 | 0.748 | 0.745 | 0.806 | — | 0.732 | 0.738 | 0.749 |
| Qwen2.5-0.5B | 0.864 | 0.871 | 0.860 | 0.793 | 0.749 | — | 0.838 | 0.842 |
| DeepSeek-Coder-1.3B | 0.847 | 0.859 | 0.852 | 0.798 | 0.744 | 0.836 | — | 0.854 |
| CodeLlama-7B | 0.871 | 0.884 | 0.879 | 0.807 | 0.756 | 0.847 | 0.858 | — |

---

## Table 6 - Cross-Perturbation Transfer F1 (Qwen2.5-1.5B)

Probe trained on **row strategy**, evaluated on **column strategy's** hidden states.

| Train ↓ / Test → | Baseline | Rand Nouns | Single Chars | All Same | Numeric | Misleading |
|------------------|:--------:|:----------:|:------------:|:--------:|:-------:|:----------:|
| Baseline | — | 0.891 | 0.847 | 0.703 | 0.829 | 0.642 |
| Random Nouns | 0.912 | — | 0.853 | 0.718 | 0.836 | 0.658 |
| Single Chars | 0.864 | 0.847 | — | 0.741 | 0.812 | 0.671 |
| All Same | 0.701 | 0.694 | 0.726 | — | 0.683 | 0.612 |
| Numeric | 0.839 | 0.824 | 0.806 | 0.689 | — | 0.634 |
| Misleading | 0.647 | 0.652 | 0.668 | 0.601 | 0.625 | — |

---

## Summary: Δ F1 vs Baseline (best layer, Qwen2.5-1.5B)

| Perturbation | Python F1 | Δ vs Baseline | Interpretation |
|---|:---:|:---:|---|
| Baseline | 0.959 | — | Original names |
| Random Nouns | 0.926 | −0.033 | Neutral replacement |
| Single Chars | 0.881 | −0.078 | Minimal lexical signal |
| All Same (`x`) | 0.742 | −0.217 | Identity stripped; role inferred from context |
| Numeric (`v1…`) | 0.864 | −0.095 | Structured but uninformative names |
| Misleading | 0.687 | −0.272 | Adversarial names |

---

## Notes

- **Baseline** = original variable names from XLCoST (e.g. `i`, `key`, `arr`)
- **Misleading** = index/key vars renamed to accumulator-sounding names (`total`, `count`, `result`); non-index vars renamed to index-sounding names (`i`, `j`, `k`, `idx`)
- **All Same** = every variable renamed to `x` — tests if identity alone carries role signal
- **Cross-model transfer** uses source model's best layer index; if target model has fewer layers, nearest available layer is used
- Results sourced from notebooks: `probing_variable_roles.ipynb`, `probing_renamed_variables.ipynb`, `probing_codecomplex.ipynb`, `probing_more_perturbations.ipynb`
