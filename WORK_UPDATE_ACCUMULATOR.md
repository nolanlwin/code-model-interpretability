# Work Update — Accumulator Variable Role Probing

**Date:** June 5, 2026  
**Dataset:** XLCoST — Python (500 programs), Java/C++/C/C# (300 programs each)  
**Probe:** Logistic Regression (class-balanced, C=1.0), 80/20 train/test split  
**Metric:** Best-layer Test Macro F1  
**Variable role:** Accumulator — variables that collect running values inside loops via `+=`, `-=`, `.append()`, `.extend()`, etc.

---

## Model: Qwen2.5-1.5B

| Property | Value |
|---|---|
| Architecture | Qwen2 (decoder-only transformer) |
| Parameters | ~1.5B |
| Hidden size | 1536 |
| Transformer layers | 28 (all full attention) |
| Attention heads | 12 (2 key-value heads, GQA) |
| Intermediate size (FFN) | 8960 |
| Activation | SiLU |
| Max context length | 131,072 tokens |
| Precision | BFloat16 |
| Vocab size | 151,936 |
| Training | General-purpose LLM; not code-specific |

Hidden states are extracted from all 29 positions (embedding layer + 28 transformer layers). Probes are trained independently per layer; the best-performing layer is reported. No fine-tuning is applied — the model is used frozen.

---

## 1. Perturbation Experiments (Python, XLCoST)

Six variable-naming strategies applied to the same 500 Python programs. Accumulator variable labels are derived from AST structure (not variable names), so the probe must rely on hidden-state context to detect the role.

| Strategy | Best Layer | Test Acc | Test F1 | Δ vs Baseline |
|---|:---:|:---:|:---:|:---:|
| Baseline (original names) | L9 | 0.984 | 0.893 | — |
| Random nouns | L10 | 0.979 | 0.870 | −0.023 |
| Single chars (a, b, c…) | L9 | 0.971 | 0.840 | −0.053 |
| All same (`x`) | L5 | 0.986 | 0.969 | **+0.076** |
| Numeric (`v1`, `v2`…) | L3 | 0.999 | 0.986 | **+0.093** |
| Misleading (accum→idx names, others→accum names) | L9 | 0.996 | 0.972 | **+0.079** |

**Key observations:**

- Baseline F1 (0.893) is notably lower than the index/key role (0.959), suggesting accumulators are harder to detect from the baseline representation — likely because accumulator names (`total`, `count`, `result`) are lexically diverse and overlap with non-accumulator variables.
- Counterintuitively, removing or adversarially replacing variable names **improves** probe accuracy for accumulators. When all variables are renamed to `x` (+0.076 F1) or given misleading index-like names (+0.079 F1), the structural signal — being the target of `+=` or `.append()` inside a loop — becomes the only distinguishing feature, and the probe exploits it more cleanly.
- This is the **opposite** of the index/key result, where misleading names caused the largest performance drop (−0.272 F1). For index/key, the model leans on lexical cues (`i`, `j`, `idx`); for accumulators, those cues are less reliable and structural context dominates.
- The numeric (`v1`, `v2`…) strategy peaks at layer 3 — much earlier than other strategies (all at L5–L10) — suggesting structured but opaque naming forces the model to encode the role earlier in the network.

---

## 2. Cross-Language Generalization

Python-trained probe (baseline, L9) applied to programs in other languages. Language-specific accumulator extraction uses regex patterns for `+=`, `++`, `.add()`, `.push()`, `.append()` etc.

| Language | Programs used | In-Domain F1 (best layer) | Cross-Lang Accuracy (Python probe) |
|---|:---:|:---:|:---:|
| Python | 181 / 500 | 0.893 (L9) | — |
| Java | 208 / 300 | 0.952 (L9) | 0.948 |
| C++ | 204 / 300 | 0.939 (L7) | 0.948 |
| C | 165 / 300 | 0.970 (L11) | 0.943 |
| C# | 207 / 300 | 0.950 (L9) | 0.945 |
| JavaScript | in progress | — | — |
| PHP | pending | — | — |

**Key observations:**

- Cross-language transfer is strong (0.943–0.948 accuracy) despite the Python probe never seeing any Java, C++, C, or C# programs. The model encodes accumulator semantics in a language-agnostic direction in hidden space.
- In-domain F1 for Java/C++/C/C# (0.939–0.970) is consistently **higher** than Python (0.893). This is likely because accumulator patterns in compiled languages are syntactically more regular — explicit `+=` in a loop body is less ambiguous than Python's mix of `+=` and `.append()` idioms.
- The best probe layer aligns closely across languages (mostly L7–L11), consistent with the finding for index/key variables that mid-to-late layers encode semantic role information.

---

## 3. Status

| Experiment | Status |
|---|:---:|
| Perturbation strategies × 6 | ✅ Complete |
| Perturbation plots (acc/F1, Δ F1, cosine sim, bar chart) | ✅ Saved |
| Cross-language — Java | ✅ Complete |
| Cross-language — C++ | ✅ Complete |
| Cross-language — C | ✅ Complete |
| Cross-language — C# | ✅ Complete |
| Cross-language — JavaScript | 🔄 In progress |
| Cross-language — PHP | ⏳ Pending |
| Multi-model comparison (CodeBERT, RoBERTa, Qwen2.5-0.5B) | ⏳ Pending |
| Cross-model transfer matrix | ⏳ Pending |
| CSV saves | ⏳ Pending |

---

## 4. Comparison: Accumulator vs Index/Key Role

| | Index/Key (best) | Accumulator (best) |
|---|:---:|:---:|
| Baseline F1 | 0.959 (L8) | 0.893 (L9) |
| Misleading Δ F1 | −0.272 | **+0.079** |
| All-same Δ F1 | −0.217 | **+0.076** |
| Cross-lang acc range | 0.973–0.988 | 0.943–0.948 |
| Best probe layer (Python) | L8 | L9 |

The contrasting behavior under misleading perturbation is the most interesting finding so far: the model detects index/key roles partly from lexical cues (variable names), but detects accumulator roles from structural/positional context. This supports the hypothesis that the model has learned syntactic-role representations that go beyond surface naming.
