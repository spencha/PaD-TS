# Metric Alignment: Diffusion-TS vs PaD-TS

## 1. Summary Table

| Metric | Diffusion-TS | PaD-TS | Compatible? | Action Needed |
|--------|:---:|:---:|---|---|
| Discriminative Score | ✓ | ✓ | **Yes** — same code lineage | None |
| Predictive Score | ✓ | ✓ | **Yes** — same code lineage | None |
| Context-FID | ✓ | ✗ | N/A | Use `compute_population_metrics.py` to apply to PaD-TS outputs |
| Cross-Correlation (cacf) | ✓ | ✗ | N/A | Use `compute_population_metrics.py` to apply to PaD-TS outputs |
| VDS | ✗ | ✓ | N/A | Use `compute_population_metrics.py` to apply to Diffusion-TS outputs |
| FDDS | ✗ | ✓ | N/A | Use `compute_population_metrics.py` to apply to Diffusion-TS outputs |

## 2. Detailed Metric-by-Metric Comparison

### Discriminative Score — COMPATIBLE

| Aspect | Diffusion-TS | PaD-TS |
|--------|-------------|--------|
| Architecture | 1-layer GRU, hidden=dim/2 | 1-layer GRU, hidden=dim/2 |
| Activation | tanh (GRU) + sigmoid (output) | tanh (GRU) + sigmoid (output) |
| Loss | Binary cross-entropy | Binary cross-entropy |
| Training iters | 2000 | 2000 |
| Batch size | 128 | 128 |
| Train/test split | 80/20 | 80/20 |
| Score formula | \|0.5 - accuracy\| | \|0.5 - accuracy\| |
| Default eval runs | 3 (visualize_results.py) | 5 (eval_run.py) |
| Sample size | min(500, N) in viz scripts | Full dataset |

**Verdict:** Same architecture and protocol. Differences in number of runs and
sample size are configurable — use `compute_population_metrics.py` with matching
`--disc_iters` for fair comparison.

### Predictive Score — COMPATIBLE

| Aspect | Diffusion-TS | PaD-TS |
|--------|-------------|--------|
| Architecture | 1-layer GRU, hidden=dim/2 | 1-layer GRU, hidden=dim/2 |
| Training strategy | Train on synthetic, test on real | Train on synthetic, test on real |
| Target | Last feature, one-step-ahead | Last feature, one-step-ahead |
| Loss | MAE | MAE |
| Training iters | 5000 | 5000 |
| Batch size | 128 | 128 |
| Default eval runs | 3 | 5 |
| Sample size | min(500, N) in viz scripts | Full dataset |

**Verdict:** Identical implementation. Same caveats about runs/sample size.

### Context-FID — PaD-TS DOES NOT HAVE

Only in Diffusion-TS (`Utils/context_fid.py`). Uses TS2Vec encoder (trained on
real data) to embed time series into 320-d space, then computes standard FID
between real and synthetic embedding distributions.

**Action:** `compute_population_metrics.py` imports this from `~/Diffusion-TS/`.

### Cross-Correlation (cacf) — DIFFERENT from PaD-TS correlation metrics

| Aspect | Diffusion-TS `CrossCorrelLoss` | PaD-TS `cross_correlation_distribution` |
|--------|------|------|
| What it computes | Lag-1 cross-autocorrelation between all feature pairs | Pearson correlation matrix (lag-0) per sample |
| Normalization | z-score per batch/time | None (raw corrcoef) |
| Includes diagonal | Yes (lower triangular with diagonal) | No (lower triangular without diagonal) |
| Output | Scalar: sum of absolute differences / 10 | Vector of correlations per sample → MMD |
| Purpose | Temporal dependency preservation | Population distribution of correlations |

**These are measuring different things.** Include both for completeness.

### VDS — Diffusion-TS DOES NOT HAVE

Only in PaD-TS (`eval_utils/MMD.py:VDS_Naive`). Per-feature MMD on flattened
value distributions (randomly samples 10,000 values per feature).
Measures marginal distribution preservation at the population level.

**Action:** `compute_population_metrics.py` applies this to Diffusion-TS outputs.

### FDDS — Diffusion-TS DOES NOT HAVE

Only in PaD-TS (`eval_utils/MMD.py:BMMD_Naive` + `cross_correlation_distribution`).
MMD on the distribution of pairwise feature correlations across samples.
Measures whether generated data preserves the population-level correlation structure.

**Action:** `compute_population_metrics.py` applies this to Diffusion-TS outputs.

## 3. Output Format Compatibility

| Aspect | Diffusion-TS | PaD-TS |
|--------|-------------|--------|
| Shape | (N, T, D) | (N, T, D) |
| Default window | 24 | 24 |
| Saved range | [0, 1] (unnormalized before save) | [-1, 1] (raw diffusion output) |
| File pattern | `ddpm_fake_{name}.npy` | `ddpm_fake_{name}_{window}.npy` |
| Save directory | `OUTPUT/{name}/` | `OUTPUT/{name}_{window}/` |

**Key difference:** PaD-TS saves in [-1, 1], Diffusion-TS saves in [0, 1].
Use `convert_outputs.py` or the `--padts_range`/`--diffts_range` flags in
the comparison scripts to handle this automatically.

## 4. Preprocessing Differences

| Aspect | Diffusion-TS | PaD-TS | Impact |
|--------|-------------|--------|--------|
| Raw normalization | MinMaxScaler → [0,1] → [-1,1] | MinMaxScaler → [0,1] → [-1,1] | Same |
| Ground truth files | `_norm_truth_` in [0,1] | `_norm_truth_` in [0,1] | Same |
| Window stride | 1 | 1 | Same |
| Train proportion | 0.8 default | 1.0 default | **Different** — PaD-TS uses all data for training |
| MinMaxScaler fit | On training split | On full data | Minor difference |

**The train proportion difference means ground truth sample counts may differ.**
Both save `_norm_truth_` files; use PaD-TS ground truth for both evaluations
to ensure identical real data baseline.

## 5. Recommended Workflow

```bash
# After both training jobs complete:

cd ~/PaD-TS/comparison

# Full comparison on stock dataset
python compute_population_metrics.py \
    --real ../OUTPUT/samples/stock_norm_truth_24_train.npy \
    --fake_diffts ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
    --fake_padts ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
    --name stock \
    --diffts_range zero1 --padts_range neg1to1 \
    --output results_stock.json

# Side-by-side visualizations
python compare_visualizations.py \
    --real ../OUTPUT/samples/stock_norm_truth_24_train.npy \
    --diffts ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
    --padts ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
    --name stock --outdir ./figures \
    --diffts_range zero1 --padts_range neg1to1
```
