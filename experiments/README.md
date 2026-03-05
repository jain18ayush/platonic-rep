# Experiments

## Setup

From the repo root:

```bash
uv sync
```

## Intrinsic Dimensionality

### Notebook (CPU, small models)

```bash
jupyter notebook experiments/intrinsic_dimensionality.ipynb
```

Runs end-to-end on CPU with 64 samples, 3 ViTs (tiny/small/base), and bloomz-560m. Produces `experiments/intrinsic_dimensionality.png`.

### GPU script (full model sweep)

Three stages — run from the repo root:

```bash
# 1. Extract per-layer features (saves to results/features/)
uv run python experiments/intrinsic_dimensionality_gpu.py extract --modality language
uv run python experiments/intrinsic_dimensionality_gpu.py extract --modality vision

# Extract a specific model subset
uv run python experiments/intrinsic_dimensionality_gpu.py extract --modality language --models "bigscience/bloomz-560m,bigscience/bloomz-1b1"

# 2. Analyze — compute participation ratio per layer + cross-modal alignment
uv run python experiments/intrinsic_dimensionality_gpu.py analyze

# 3. Plot — generate figures in results/intrinsic_dimensionality/
uv run python experiments/intrinsic_dimensionality_gpu.py plot
```

Outputs:
- `results/intrinsic_dimensionality/analysis.json`
- `results/intrinsic_dimensionality/fig_dim_gap_vs_scale.png`
- `results/intrinsic_dimensionality/fig_pr_across_layers.png`

## Cross-Model Linear Predictability (R²)

### Notebook (CPU, small models)

```bash
jupyter notebook experiments/cross_model_r2.ipynb
```

Runs end-to-end on CPU with 64 samples, 3 ViTs, and bloomz-560m. Measures cross-validated R² via ridge regression between all layer combinations. Produces `experiments/cross_model_r2.png`.

### GPU script (full model sweep)

Three stages — run from the repo root:

```bash
# 1. Extract features (same format as intrinsic dim — skips if .pt files already exist)
uv run python experiments/cross_model_r2_gpu.py extract --modality language
uv run python experiments/cross_model_r2_gpu.py extract --modality vision

# Extract a specific model
uv run python experiments/cross_model_r2_gpu.py extract --modality vision --models "vit_tiny_patch16_224.augreg_in21k"

# 2. Analyze — find best layer pairs via mutual_knn, then compute R² (fwd + rev)
uv run python experiments/cross_model_r2_gpu.py analyze

# 3. Plot — generate figures in results/cross_model_r2/
uv run python experiments/cross_model_r2_gpu.py plot
```

Outputs:
- `results/cross_model_r2/analysis.json`
- `results/cross_model_r2/fig_r2_vs_scale.png`
- `results/cross_model_r2/fig_r2_across_layers.png`
- `results/cross_model_r2/fig_r2_vs_alignment.png`

## Notes

- The `extract` stage is shared between experiments — feature `.pt` files in `results/features/` are reused if they already exist.
- GPU scripts use the `val` model set defined in `tasks.py` (12 LLMs + 16 vision models).
- Custom batch sizes: `--batch_size 8` (default: 4 for language, 32 for vision).
