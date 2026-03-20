"""
Side-by-side visualization of Diffusion-TS vs PaD-TS generated time series.

Generates publication-quality figures for presentation slides:
  1. Sample trajectories (real vs Diffusion-TS vs PaD-TS)
  2. t-SNE embedding
  3. PCA projection
  4. Marginal distributions per feature
  5. Temporal autocorrelation

All plots use consistent axis scales, colors, and labeling.

Usage:
    python compare_visualizations.py \
        --real ../OUTPUT/samples/stock_norm_truth_24_train.npy \
        --diffts ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
        --padts ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
        --name stock \
        --outdir ./figures

    # If Diffusion-TS outputs are in [-1,1] or original scale:
    python compare_visualizations.py \
        --real ../OUTPUT/samples/stock_norm_truth_24_train.npy \
        --diffts ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
        --padts ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
        --name stock --outdir ./figures \
        --diffts_range zero1 --padts_range neg1to1
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Consistent style ──────────────────────────────────────────────────────

COLORS = {
    "Real": "#2c3e50",
    "Diffusion-TS": "#e74c3c",
    "PaD-TS": "#3498db",
}
ALPHA = 0.6
DPI = 300
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (14, 10)


def unnorm_neg1to1(x):
    return (x + 1) * 0.5


def load_data(path, data_range):
    data = np.load(path)
    if data_range == "neg1to1":
        data = unnorm_neg1to1(data)
    elif data_range == "original":
        raise ValueError("Original-scale data must be converted first. Use convert_outputs.py.")
    return data


# ── Figure 1: Sample Trajectories ─────────────────────────────────────────

def plot_samples(real, diffts, padts, name, outdir, n_samples=3, n_features=4):
    n_features = min(n_features, real.shape[2])
    fig, axes = plt.subplots(n_features, 3, figsize=(15, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = axes[np.newaxis, :]

    titles = ["Real", "Diffusion-TS", "PaD-TS"]
    datasets = [real, diffts, padts]

    for col, (title, data) in enumerate(zip(titles, datasets)):
        color = COLORS[title]
        for row in range(n_features):
            ax = axes[row, col]
            for s in range(min(n_samples, data.shape[0])):
                ax.plot(data[s, :, row], color=color, alpha=0.5 + 0.15 * (n_samples - s), linewidth=1)
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Feature {row}", fontsize=10)
            ax.tick_params(labelsize=8)

    plt.suptitle(f"{name.upper()} — Sample Trajectories", fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_samples.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: t-SNE ───────────────────────────────────────────────────────

def plot_tsne(real, diffts, padts, name, outdir, max_n=1500):
    n = min(max_n, real.shape[0], diffts.shape[0], padts.shape[0])
    r = real[:n].reshape(n, -1)
    d = diffts[:n].reshape(n, -1)
    p = padts[:n].reshape(n, -1)

    combined = np.concatenate([r, d, p], axis=0)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
    emb = tsne.fit_transform(combined)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(emb[:n, 0], emb[:n, 1], s=8, alpha=ALPHA, c=COLORS["Real"], label="Real")
    ax.scatter(emb[n:2*n, 0], emb[n:2*n, 1], s=8, alpha=ALPHA, c=COLORS["Diffusion-TS"], label="Diffusion-TS")
    ax.scatter(emb[2*n:, 0], emb[2*n:, 1], s=8, alpha=ALPHA, c=COLORS["PaD-TS"], label="PaD-TS")
    ax.legend(fontsize=11, markerscale=3)
    ax.set_title(f"{name.upper()} — t-SNE Embedding", fontsize=14)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_tsne.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: PCA ─────────────────────────────────────────────────────────

def plot_pca(real, diffts, padts, name, outdir, max_n=2000):
    n = min(max_n, real.shape[0], diffts.shape[0], padts.shape[0])
    r = real[:n].reshape(n, -1)
    d = diffts[:n].reshape(n, -1)
    p = padts[:n].reshape(n, -1)

    pca = PCA(n_components=2)
    pca.fit(r)
    r_pc = pca.transform(r)
    d_pc = pca.transform(d)
    p_pc = pca.transform(p)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(r_pc[:, 0], r_pc[:, 1], s=8, alpha=ALPHA, c=COLORS["Real"], label="Real")
    ax.scatter(d_pc[:, 0], d_pc[:, 1], s=8, alpha=ALPHA, c=COLORS["Diffusion-TS"], label="Diffusion-TS")
    ax.scatter(p_pc[:, 0], p_pc[:, 1], s=8, alpha=ALPHA, c=COLORS["PaD-TS"], label="PaD-TS")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.legend(fontsize=11, markerscale=3)
    ax.set_title(f"{name.upper()} — PCA Projection", fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_pca.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Marginal Distributions ──────────────────────────────────────

def plot_marginals(real, diffts, padts, name, outdir, n_features=6):
    n_features = min(n_features, real.shape[2])
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :] if n_cols > 1 else np.array([[axes]])
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        r_vals = real[:, :, i].flatten()
        d_vals = diffts[:, :, i].flatten()
        p_vals = padts[:, :, i].flatten()

        bins = np.linspace(
            min(r_vals.min(), d_vals.min(), p_vals.min()),
            max(r_vals.max(), d_vals.max(), p_vals.max()),
            60,
        )
        ax.hist(r_vals, bins=bins, density=True, alpha=0.4, color=COLORS["Real"], label="Real")
        ax.hist(d_vals, bins=bins, density=True, alpha=0.4, color=COLORS["Diffusion-TS"],
                label="Diffusion-TS", histtype="step", linewidth=1.5)
        ax.hist(p_vals, bins=bins, density=True, alpha=0.4, color=COLORS["PaD-TS"],
                label="PaD-TS", histtype="step", linewidth=1.5)
        ax.set_title(f"Feature {i}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{name.upper()} — Marginal Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_marginals.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Temporal Autocorrelation ────────────────────────────────────

def plot_autocorrelation(real, diffts, padts, name, outdir, max_lag=12, n_features=4):
    n_features = min(n_features, real.shape[2])
    lags = list(range(1, max_lag + 1))
    max_samples = 500

    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4), sharey=True)
    if n_features == 1:
        axes = [axes]

    for feat_idx, ax in enumerate(axes):
        for label, data in [("Real", real), ("Diffusion-TS", diffts), ("PaD-TS", padts)]:
            series = data[:max_samples, :, feat_idx]
            corrs = []
            for lag in lags:
                x = series[:, :-lag]
                y = series[:, lag:]
                per_sample = [np.corrcoef(x[i], y[i])[0, 1] for i in range(len(series))
                              if len(x[i]) > 1]
                corrs.append(np.nanmean(per_sample))
            ax.plot(lags, corrs, marker="o", markersize=3, label=label,
                    color=COLORS[label], linewidth=1.5)

        ax.set_xlabel("Lag", fontsize=10)
        ax.set_title(f"Feature {feat_idx}", fontsize=10)
        if feat_idx == 0:
            ax.set_ylabel("Autocorrelation", fontsize=10)
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{name.upper()} — Temporal Autocorrelation", fontsize=14, y=1.03)
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_autocorr.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Side-by-side visualization: Diffusion-TS vs PaD-TS")
    parser.add_argument("--real", required=True, help="Ground truth .npy ([0,1] range)")
    parser.add_argument("--diffts", required=True, help="Diffusion-TS generated .npy")
    parser.add_argument("--padts", required=True, help="PaD-TS generated .npy")
    parser.add_argument("--name", required=True, help="Dataset name (for titles)")
    parser.add_argument("--outdir", default="./figures", help="Output directory for figures")
    parser.add_argument("--diffts_range", default="zero1", choices=["zero1", "neg1to1"],
                        help="Diffusion-TS output range")
    parser.add_argument("--padts_range", default="neg1to1", choices=["zero1", "neg1to1"],
                        help="PaD-TS output range")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data for {args.name}...")
    real = np.load(args.real)
    diffts = load_data(args.diffts, args.diffts_range)
    padts = load_data(args.padts, args.padts_range)

    # Match sample counts
    n = min(real.shape[0], diffts.shape[0], padts.shape[0])
    real, diffts, padts = real[:n], diffts[:n], padts[:n]

    print(f"  Real:        {real.shape}, [{real.min():.3f}, {real.max():.3f}]")
    print(f"  Diffusion-TS:{diffts.shape}, [{diffts.min():.3f}, {diffts.max():.3f}]")
    print(f"  PaD-TS:      {padts.shape}, [{padts.min():.3f}, {padts.max():.3f}]")

    print(f"\nGenerating figures in {args.outdir}/...")
    plot_samples(real, diffts, padts, args.name, args.outdir)
    plot_tsne(real, diffts, padts, args.name, args.outdir)
    plot_pca(real, diffts, padts, args.name, args.outdir)
    plot_marginals(real, diffts, padts, args.name, args.outdir)
    plot_autocorrelation(real, diffts, padts, args.name, args.outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
