"""
Quick visualization of PaD-TS results (works with single method).
Run directly on login node — no GPU needed.

Usage:
    cd ~/PaD-TS/comparison
    conda activate padts
    python quick_viz.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

COLORS = {"Real": "#2c3e50", "PaD-TS": "#3498db"}
DPI = 200
OUTDIR = "./figures"
os.makedirs(OUTDIR, exist_ok=True)

DATASETS = {
    "stock": {
        "real": "../OUTPUT/samples/stock_norm_truth_24_train.npy",
        "padts": "../OUTPUT/stock_24/ddpm_fake_stock_24.npy",
    },
    "energy": {
        "real": "../OUTPUT/samples/energy_norm_truth_24_train.npy",
        "padts": "../OUTPUT/energy_24/ddpm_fake_energy_24.npy",
    },
    "fmri": {
        "real": "../OUTPUT/samples/fmri_norm_truth_24_train.npy",
        "padts": "../OUTPUT/fmri_24/ddpm_fake_fmri_24.npy",
    },
    "mujoco": {
        "real": "../OUTPUT/samples/mujoco_norm_truth_24_train.npy",
        "padts": "../OUTPUT/mujoco_24/ddpm_fake_mujoco_24.npy",
    },
    "sine": {
        "real": "../OUTPUT/samples/sine_ground_truth_24_train.npy",
        "padts": "../OUTPUT/sine_24/ddpm_fake_sine_24.npy",
    },
}


def unnorm(x):
    return (x + 1) * 0.5


def make_plots(name, real, fake):
    n = min(real.shape[0], fake.shape[0])
    real, fake = real[:n], fake[:n]
    n_feat = real.shape[2]

    print(f"\n{'='*50}")
    print(f"  {name.upper()}: real {real.shape}, fake {fake.shape}")
    print(f"  Real  [{real.min():.3f}, {real.max():.3f}]")
    print(f"  Fake  [{fake.min():.3f}, {fake.max():.3f}]")

    # 1. Sample trajectories
    nf = min(4, n_feat)
    fig, axes = plt.subplots(nf, 2, figsize=(12, 3 * nf), sharex=True)
    if nf == 1:
        axes = axes[np.newaxis, :]
    for row in range(nf):
        for s in range(min(5, n)):
            axes[row, 0].plot(real[s, :, row], color=COLORS["Real"], alpha=0.4, linewidth=0.8)
            axes[row, 1].plot(fake[s, :, row], color=COLORS["PaD-TS"], alpha=0.4, linewidth=0.8)
        axes[row, 0].set_ylabel(f"Feat {row}", fontsize=9)
    axes[0, 0].set_title("Real", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("PaD-TS", fontsize=12, fontweight="bold")
    plt.suptitle(f"{name.upper()} — Sample Trajectories", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{name}_samples.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}_samples.png")

    # 2. t-SNE
    tsne_n = min(1000, n)
    r_flat = real[:tsne_n].reshape(tsne_n, -1)
    f_flat = fake[:tsne_n].reshape(tsne_n, -1)
    combined = np.concatenate([r_flat, f_flat])
    emb = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42).fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[:tsne_n, 0], emb[:tsne_n, 1], s=6, alpha=0.5, c=COLORS["Real"], label="Real")
    ax.scatter(emb[tsne_n:, 0], emb[tsne_n:, 1], s=6, alpha=0.5, c=COLORS["PaD-TS"], label="PaD-TS")
    ax.legend(fontsize=10, markerscale=3)
    ax.set_title(f"{name.upper()} — t-SNE", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{name}_tsne.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}_tsne.png")

    # 3. PCA
    pca = PCA(n_components=2).fit(r_flat)
    r_pc = pca.transform(r_flat)
    f_pc = pca.transform(f_flat)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r_pc[:, 0], r_pc[:, 1], s=6, alpha=0.5, c=COLORS["Real"], label="Real")
    ax.scatter(f_pc[:, 0], f_pc[:, 1], s=6, alpha=0.5, c=COLORS["PaD-TS"], label="PaD-TS")
    v = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({v[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({v[1]*100:.1f}%)")
    ax.legend(fontsize=10, markerscale=3)
    ax.set_title(f"{name.upper()} — PCA", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{name}_pca.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}_pca.png")

    # 4. Marginal distributions
    nf_plot = min(6, n_feat)
    ncols = 3
    nrows = (nf_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()
    for i in range(nf_plot):
        ax = axes[i]
        rv = real[:, :, i].flatten()
        fv = fake[:, :, i].flatten()
        bins = np.linspace(min(rv.min(), fv.min()), max(rv.max(), fv.max()), 60)
        ax.hist(rv, bins=bins, density=True, alpha=0.4, color=COLORS["Real"], label="Real")
        ax.hist(fv, bins=bins, density=True, alpha=0.6, color=COLORS["PaD-TS"],
                label="PaD-TS", histtype="step", linewidth=1.5)
        ax.set_title(f"Feature {i}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(nf_plot, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{name.upper()} — Marginal Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{name}_marginals.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}_marginals.png")

    # 5. Autocorrelation
    nf_ac = min(4, n_feat)
    fig, axes = plt.subplots(1, nf_ac, figsize=(4 * nf_ac, 4), sharey=True)
    if nf_ac == 1:
        axes = [axes]
    lags = list(range(1, 13))
    for fi, ax in enumerate(axes):
        for label, data in [("Real", real), ("PaD-TS", fake)]:
            series = data[:500, :, fi]
            corrs = []
            for lag in lags:
                c = [np.corrcoef(series[i, :-lag], series[i, lag:])[0, 1]
                     for i in range(len(series)) if series.shape[1] > lag]
                corrs.append(np.nanmean(c))
            ax.plot(lags, corrs, marker="o", markersize=3, label=label,
                    color=COLORS[label], linewidth=1.5)
        ax.set_xlabel("Lag")
        ax.set_title(f"Feature {fi}", fontsize=10)
        if fi == 0:
            ax.set_ylabel("Autocorrelation")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{name.upper()} — Autocorrelation", fontsize=14, y=1.03)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{name}_autocorr.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}_autocorr.png")


if __name__ == "__main__":
    for name, paths in DATASETS.items():
        if not os.path.exists(paths["real"]):
            print(f"SKIP {name}: {paths['real']} not found")
            continue
        if not os.path.exists(paths["padts"]):
            print(f"SKIP {name}: {paths['padts']} not found")
            continue

        real = np.load(paths["real"])
        fake = np.load(paths["padts"])
        # PaD-TS outputs are in [-1,1], unnormalize to [0,1]
        fake = unnorm(fake)
        make_plots(name, real, fake)

    print(f"\nAll figures saved to {OUTDIR}/")
