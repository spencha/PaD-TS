"""
MHEALTH heterogeneity analysis: does the generated data preserve the
multi-sensor coupling structure of wearable data?

Since subject labels are lost during preprocessing (all 10 subjects
are concatenated), we focus on sensor-level structure:

1. Cross-channel correlation matrices (real vs generated)
   - Do methods preserve accelerometer-ECG coupling?
   - Do methods preserve within-body-position correlations?

2. Per-sensor-group distributional analysis
   - Chest sensors (acc + ECG): channels 0-4
   - Left ankle sensors (acc + gyro + mag): channels 5-13
   - Right arm sensors (acc + gyro + mag): channels 14-22

3. Variability analysis
   - Per-sample standard deviation distribution
   - Inter-quartile range distribution

Usage:
    python analyze_heterogeneity.py \
        --real ../OUTPUT/samples/mhealth_norm_truth_24_train.npy \
        --padts ../OUTPUT/mhealth_24/ddpm_fake_mhealth_24.npy \
        --diffts ~/Diffusion-TS/OUTPUT/mhealth/ddpm_fake_mhealth.npy \
        --outdir results/mhealth
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── MHEALTH sensor channel groups ─────────────────────────────────────────

CHANNEL_NAMES = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_lead1", "ecg_lead2",
    "l_ankle_acc_x", "l_ankle_acc_y", "l_ankle_acc_z",
    "l_ankle_gyro_x", "l_ankle_gyro_y", "l_ankle_gyro_z",
    "l_ankle_mag_x", "l_ankle_mag_y", "l_ankle_mag_z",
    "r_arm_acc_x", "r_arm_acc_y", "r_arm_acc_z",
    "r_arm_gyro_x", "r_arm_gyro_y", "r_arm_gyro_z",
    "r_arm_mag_x", "r_arm_mag_y", "r_arm_mag_z",
]

SENSOR_GROUPS = {
    "Chest (Acc+ECG)": list(range(0, 5)),
    "L. Ankle (Acc+Gyro+Mag)": list(range(5, 14)),
    "R. Arm (Acc+Gyro+Mag)": list(range(14, 23)),
}

COLORS = {
    "Real": "#2c3e50",
    "Diffusion-TS": "#e74c3c",
    "PaD-TS": "#3498db",
}
DPI = 300


def unnorm(x):
    return (x + 1) * 0.5


def load_data(path, data_range):
    data = np.load(path)
    if data_range == "neg1to1":
        data = unnorm(data)
    return data


# ── 1. Cross-channel correlation matrices ─────────────────────────────────

def compute_correlation_matrix(data):
    """Compute mean Pearson correlation matrix across all samples."""
    n_samples, seq_len, n_features = data.shape
    # Flatten time into samples for correlation
    flat = data.reshape(-1, n_features)
    return np.corrcoef(flat.T)


def plot_correlation_matrices(real, methods, outdir):
    """Side-by-side correlation matrices."""
    n_plots = 1 + len(methods)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    corr_real = compute_correlation_matrix(real)

    all_data = [("Real", corr_real)]
    for name, data in methods.items():
        all_data.append((name, compute_correlation_matrix(data)))

    vmin, vmax = -1, 1
    for idx, (title, corr) in enumerate(all_data):
        ax = axes[idx]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, len(CHANNEL_NAMES), 3))
        ax.set_xticklabels([CHANNEL_NAMES[i][:6] for i in range(0, len(CHANNEL_NAMES), 3)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(0, len(CHANNEL_NAMES), 3))
        ax.set_yticklabels([CHANNEL_NAMES[i][:6] for i in range(0, len(CHANNEL_NAMES), 3)],
                           fontsize=7)

        # Draw sensor group boundaries
        for group_name, channels in SENSOR_GROUPS.items():
            start = channels[0] - 0.5
            end = channels[-1] + 0.5
            rect = plt.Rectangle((start, start), end - start, end - start,
                                 fill=False, edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Pearson r")
    plt.suptitle("MHEALTH — Cross-Channel Correlation Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_correlation_matrices.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return corr_real, {name: corr for name, corr in all_data[1:]}


# ── 2. Correlation difference heatmaps ────────────────────────────────────

def plot_correlation_differences(corr_real, method_corrs, outdir):
    """Heatmap of |corr_real - corr_generated| per method."""
    n_plots = len(method_corrs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for idx, (name, corr) in enumerate(method_corrs.items()):
        ax = axes[idx]
        diff = np.abs(corr_real - corr)
        im = ax.imshow(diff, cmap="Reds", vmin=0, vmax=0.5, aspect="equal")
        ax.set_title(f"|Real - {name}|", fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, len(CHANNEL_NAMES), 3))
        ax.set_xticklabels([CHANNEL_NAMES[i][:6] for i in range(0, len(CHANNEL_NAMES), 3)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(0, len(CHANNEL_NAMES), 3))
        ax.set_yticklabels([CHANNEL_NAMES[i][:6] for i in range(0, len(CHANNEL_NAMES), 3)],
                           fontsize=7)

        # Annotate mean absolute error
        mae = diff[np.tril_indices_from(diff, k=-1)].mean()
        ax.text(0.02, 0.98, f"MAE: {mae:.4f}", transform=ax.transAxes,
                va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        for group_name, channels in SENSOR_GROUPS.items():
            start = channels[0] - 0.5
            end = channels[-1] + 0.5
            rect = plt.Rectangle((start, start), end - start, end - start,
                                 fill=False, edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)

    fig.colorbar(im, ax=axes, shrink=0.8, label="|Δr|")
    plt.suptitle("MHEALTH — Correlation Difference from Real Data", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_correlation_differences.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Per-sensor-group distribution comparison ───────────────────────────

def plot_sensor_group_distributions(real, methods, outdir):
    """Compare marginal distributions per sensor group."""
    fig, axes = plt.subplots(len(SENSOR_GROUPS), 1, figsize=(12, 4 * len(SENSOR_GROUPS)))

    for idx, (group_name, channels) in enumerate(SENSOR_GROUPS.items()):
        ax = axes[idx]

        real_vals = real[:, :, channels].flatten()
        bins = np.linspace(real_vals.min(), real_vals.max(), 80)

        ax.hist(real_vals, bins=bins, density=True, alpha=0.4,
                color=COLORS["Real"], label="Real")

        for name, data in methods.items():
            vals = data[:, :, channels].flatten()
            ax.hist(vals, bins=bins, density=True, alpha=0.6,
                    color=COLORS[name], label=name, histtype="step", linewidth=1.5)

        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.suptitle("MHEALTH — Marginal Distributions by Sensor Group", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_sensor_group_distributions.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. Variability analysis ───────────────────────────────────────────────

def plot_variability(real, methods, outdir):
    """Compare per-sample variability (std) distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-sample standard deviation
    ax = axes[0]
    real_std = real.std(axis=1).mean(axis=1)  # mean std across features per sample
    bins = np.linspace(0, real_std.max() * 1.5, 60)
    ax.hist(real_std, bins=bins, density=True, alpha=0.4, color=COLORS["Real"], label="Real")
    for name, data in methods.items():
        fake_std = data.std(axis=1).mean(axis=1)
        ax.hist(fake_std, bins=bins, density=True, alpha=0.6, color=COLORS[name],
                label=name, histtype="step", linewidth=1.5)
    ax.set_title("Per-Sample Temporal Variability (Std Dev)", fontsize=11)
    ax.set_xlabel("Mean std across channels")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # Per-sample IQR
    ax = axes[1]
    real_iqr = np.percentile(real, 75, axis=1).mean(axis=1) - np.percentile(real, 25, axis=1).mean(axis=1)
    bins = np.linspace(0, real_iqr.max() * 1.5, 60)
    ax.hist(real_iqr, bins=bins, density=True, alpha=0.4, color=COLORS["Real"], label="Real")
    for name, data in methods.items():
        fake_iqr = np.percentile(data, 75, axis=1).mean(axis=1) - np.percentile(data, 25, axis=1).mean(axis=1)
        ax.hist(fake_iqr, bins=bins, density=True, alpha=0.6, color=COLORS[name],
                label=name, histtype="step", linewidth=1.5)
    ax.set_title("Per-Sample Inter-Quartile Range", fontsize=11)
    ax.set_xlabel("Mean IQR across channels")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    plt.suptitle("MHEALTH — Sample Variability Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_variability.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. Summary statistics table ───────────────────────────────────────────

def compute_summary(real, methods, outdir):
    """Compute and save per-group correlation MAE and distribution distance."""
    import json

    corr_real = compute_correlation_matrix(real)
    results = {}

    for name, data in methods.items():
        corr_fake = compute_correlation_matrix(data)
        diff = np.abs(corr_real - corr_fake)

        method_results = {
            "overall_corr_mae": float(diff[np.tril_indices_from(diff, k=-1)].mean()),
        }

        for group_name, channels in SENSOR_GROUPS.items():
            sub_real = corr_real[np.ix_(channels, channels)]
            sub_fake = corr_fake[np.ix_(channels, channels)]
            sub_diff = np.abs(sub_real - sub_fake)
            tril = np.tril_indices_from(sub_diff, k=-1)
            method_results[f"{group_name}_corr_mae"] = float(sub_diff[tril].mean())

        # Cross-group: chest-to-ankle coupling
        chest = SENSOR_GROUPS["Chest (Acc+ECG)"]
        ankle = SENSOR_GROUPS["L. Ankle (Acc+Gyro+Mag)"]
        cross_real = corr_real[np.ix_(chest, ankle)]
        cross_fake = corr_fake[np.ix_(chest, ankle)]
        method_results["chest_ankle_coupling_mae"] = float(np.abs(cross_real - cross_fake).mean())

        # Cross-group: chest-to-arm coupling
        arm = SENSOR_GROUPS["R. Arm (Acc+Gyro+Mag)"]
        cross_real = corr_real[np.ix_(chest, arm)]
        cross_fake = corr_fake[np.ix_(chest, arm)]
        method_results["chest_arm_coupling_mae"] = float(np.abs(cross_real - cross_fake).mean())

        results[name] = method_results

    # Print summary table
    print("\n" + "=" * 70)
    print("MHEALTH Heterogeneity Summary")
    print("=" * 70)
    header = f"{'Metric':<35}"
    for name in methods:
        header += f" {name:<18}"
    print(header)
    print("-" * 70)

    metric_keys = [
        ("Overall Corr MAE", "overall_corr_mae"),
        ("Chest (Acc+ECG) Corr MAE", "Chest (Acc+ECG)_corr_mae"),
        ("L. Ankle Corr MAE", "L. Ankle (Acc+Gyro+Mag)_corr_mae"),
        ("R. Arm Corr MAE", "R. Arm (Acc+Gyro+Mag)_corr_mae"),
        ("Chest↔Ankle Coupling MAE", "chest_ankle_coupling_mae"),
        ("Chest↔Arm Coupling MAE", "chest_arm_coupling_mae"),
    ]

    for label, key in metric_keys:
        row = f"{label:<35}"
        for name in methods:
            val = results[name].get(key, float("nan"))
            row += f" {val:<18.6f}"
        print(row)
    print("=" * 70)
    print("(All metrics: lower = better)")

    # Save to JSON
    path = os.path.join(outdir, "heterogeneity_summary.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MHEALTH heterogeneity analysis")
    parser.add_argument("--real", required=True, help="Ground truth .npy ([0,1] range)")
    parser.add_argument("--diffts", default=None, help="Diffusion-TS generated .npy")
    parser.add_argument("--padts", default=None, help="PaD-TS generated .npy")
    parser.add_argument("--diffts_range", default="zero1", choices=["zero1", "neg1to1"])
    parser.add_argument("--padts_range", default="neg1to1", choices=["zero1", "neg1to1"])
    parser.add_argument("--outdir", default="./results/mhealth")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    real = np.load(args.real)
    methods = {}
    if args.diffts:
        methods["Diffusion-TS"] = load_data(args.diffts, args.diffts_range)
    if args.padts:
        methods["PaD-TS"] = load_data(args.padts, args.padts_range)

    if not methods:
        print("ERROR: Provide at least one of --diffts or --padts")
        return

    # Match sample counts
    n = min(real.shape[0], *(d.shape[0] for d in methods.values()))
    real = real[:n]
    methods = {k: v[:n] for k, v in methods.items()}

    n_features = real.shape[2]
    print(f"MHEALTH heterogeneity analysis")
    print(f"  Real: {real.shape}, Features: {n_features}")
    for name, data in methods.items():
        print(f"  {name}: {data.shape}")

    if n_features != 23:
        print(f"  WARNING: Expected 23 MHEALTH channels, got {n_features}. "
              "Channel labels may be incorrect.")

    print("\n1. Cross-channel correlation matrices...")
    corr_real, method_corrs = plot_correlation_matrices(real, methods, args.outdir)

    print("\n2. Correlation difference heatmaps...")
    plot_correlation_differences(corr_real, method_corrs, args.outdir)

    print("\n3. Per-sensor-group distributions...")
    plot_sensor_group_distributions(real, methods, args.outdir)

    print("\n4. Variability analysis...")
    plot_variability(real, methods, args.outdir)

    print("\n5. Summary statistics...")
    compute_summary(real, methods, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
