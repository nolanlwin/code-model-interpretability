# Running the structure-probing notebook on Google Colab

The notebook [`notebooks/structure_probing_cost.ipynb`](../notebooks/structure_probing_cost.ipynb)
is built to run on a Colab GPU. This guide covers billing, GPU choice, and the exact steps.

## GPU choice: a T4 is enough (A100 optional)

This job runs **forward passes only** — there is **no GPU training** (the probes are CPU
`scikit-learn` `LogisticRegression`). So:

- `Qwen/Qwen2.5-1.5B` in fp16 is ~3 GB of weights vs the T4's 16 GB — fits easily.
- Full run over the ~1,417 aligned programs is **a few minutes** on a T4 (the existing
  boolean notebook did 500 programs in ~53 min on **CPU**).
- The real constraint is **host RAM** for the stacked probe features, handled by the
  notebook's program cap + token subsampling — this is independent of the GPU.

An A100 works and won't hurt, but it does **not** meaningfully speed this up (the bottleneck
is the per-program Python loop + CPU probe fits) and burns compute units far faster.

## Billing: Pay As You Go is the best fit

For a short, occasional job, **Pay As You Go** beats a subscription:

- ~$9.99 for 100 compute units, **no subscription**, units valid 90 days.
- One run costs roughly **1–3 units on an A100** or **<1 on a T4** — one purchase covers many runs.
- Pay As You Go still grants the **same premium GPUs (T4 / L4 / A100) and high-RAM machines**,
  subject to availability; it only lacks Pro's *priority* and *background execution*.

Choose **Colab Pro** (~$10/month) only if you run often or keep losing the A100 to demand.
Students/faculty can also check for a free "No cost to students and educators" (SheerID)
button on the signup page, though Google has paused new education sign-ups at times.

## Step by step

### A. Get compute
1. Sign into Google -> https://colab.research.google.com/signup
2. Under **Pay As You Go**, buy **100 compute units** (~$9.99). Keep the receipt for reimbursement.
   (Or subscribe to **Colab Pro** if you prefer the monthly plan.)

### B. Open the notebook and pick the GPU
3. https://colab.research.google.com -> **File > Upload notebook** -> upload
   `notebooks/structure_probing_cost.ipynb`. (Or **File > Open notebook > GitHub** if the repo is pushed.)
4. **Runtime > Change runtime type**:
   - Hardware accelerator: **T4 GPU** (recommended) or **A100 GPU**.
   - Runtime shape: **High-RAM** (the High-RAM slider also selects the **A100 80GB** variant).
   - **Save**.
5. Confirm the GPU — run a cell with `!nvidia-smi` (expect `Tesla T4` or `A100-SXM4-...`).
   If you asked for A100 but see a T4, none was free — retry later or just proceed on the T4.

### C. Get the data into the session
6. Left **Files** panel (folder icon) -> **Upload** -> `consolidated_data.csv` (~12 MB).
   The notebook auto-detects it at `consolidated_data.csv` or `/content/...`.
7. To persist data/results across sessions, mount Drive and point `CSV_PATH` at it:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```
   then use a path under `/content/drive/MyDrive/`.

### D. Run
8. **Runtime > Run all.** The notebook pip-installs deps, loads Qwen on the GPU (first run
   downloads ~3 GB of weights), builds tree-sitter labels, extracts per-layer activations,
   trains the within- and cross-lingual probes, and renders the plots.
9. Results display inline and save to `output/` (`structure_probe_macro_f1.png`,
   the confusion-matrix PNG, and `structure_probe_results.json`).

### E. Stop billing when done
10. **Runtime > Disconnect and delete runtime** so idle time doesn't keep burning units
    (matters most on A100). The compute-units counter is under the resources gauge (top-right).

## Tuning

In the config cell:
- `MAX_PROGRAMS_PER_LANG` / `MAX_TOKENS_PER_LANG` — raise for tighter estimates (more RAM/time).
- `MAX_SEQ_LEN` — tokenizer truncation per program.
- `INCLUDE_COMMENTS` — set `True` to add an 8th `comment` class.
- `MODEL_NAME` — swap to `Qwen/Qwen2.5-Coder-1.5B` to compare a code-specialized model.
