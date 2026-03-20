"""
MHEALTH subject-level heterogeneity analysis.

Evaluates how well generative methods preserve subject-level structure
in multi-subject wearable sensor data. This is the core research question:
can diffusion models borrow strength across heterogeneous subjects without
collapsing individual differences?

Analyses:
  1. Per-subject discriminative scores — can a classifier distinguish
     real vs generated data within each subject's windows?
  2. Pooled vs per-subject generation quality — does the method do
     better on some subjects than others? (Reveals bias.)
  3. Inter-subject vs intra-subject distributional distance — does
     generated data preserve the between-subject variation?
  4. Cross-channel correlation preservation per subject
  5. Subject-level summary table and figures

Requires:
  - mhealth_window_subjects.npy (from preprocess_mhealth.py)
  - Ground truth .npy and generated .npy files

Usage:
    python analyze_heterogeneity.py \
        --real ../OUTPUT/samples/mhealth_norm_truth_24_train.npy \
        --subject_labels ../dataset/mhealth_window_subjects.npy \
        --padts ../OUTPUT/mhealth_24/ddpm_fake_mhealth_24.npy \
        --diffts ~/Diffusion-TS/OUTPUT/mhealth/ddpm_fake_mhealth.npy \
        --outdir results/mhealth
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add PaD-TS root for eval_utils imports
PADTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PADTS_ROOT not in sys.path:
    sys.path.insert(0, PADTS_ROOT)

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
    "L. Ankle": list(range(5, 14)),
    "R. Arm": list(range(14, 23)),
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


def get_subject_indices(window_subjects, real_n):
    """
    Map window-level subject labels to the actual indices in the training
    data. CustomDataset uses proportion=1.0 with random permutation
    (seed=123) to create training samples.

    Since PaD-TS uses proportion=1.0, all windows become training data,
    but they are randomly permuted. We need to track which permuted index
    maps to which subject.

    Returns dict: subject_id -> array of indices into the training data.
    """
    n_windows = len(window_subjects)

    # Reproduce the same permutation used by CustomDataset.divide()
    # with ratio=1.0, seed=123
    st0 = np.random.get_state()
    np.random.seed(123)
    regular_train_num = int(np.ceil(n_windows * 1.0))
    id_rdm = np.random.permutation(n_windows)
    train_ids = id_rdm[:regular_train_num]  # All of them when proportion=1.0
    np.random.set_state(st0)

    # train_ids[i] is the original window index that ended up at position i
    # in the training set. So training sample i came from original window train_ids[i],
    # which has subject label window_subjects[train_ids[i]].
    train_subjects = window_subjects[train_ids]

    # Only use indices up to real_n (the actual number of training samples)
    train_subjects = train_subjects[:real_n]

    subject_indices = {}
    for s in np.unique(train_subjects):
        if s == -1:
            continue  # Skip boundary windows
        subject_indices[int(s)] = np.where(train_subjects == s)[0]

    return subject_indices, train_subjects


# ── 1. Per-subject discriminative scores ──────────────────────────────────

def per_subject_discriminative(real, fake, subject_indices, method_name):
    """
    For each subject, train a discriminative classifier on that subject's
    real windows vs a matching number of generated windows.
    """
    from eval_utils.discriminative_metric import discriminative_score_metrics

    results = {}
    print(f"\n  Per-subject discriminative scores ({method_name}):")
    print(f"  {'Subject':<10} {'N_windows':<12} {'Disc Score':<15} {'Fake Acc':<12} {'Real Acc':<12}")
    print("  " + "-" * 60)

    for subj_id in sorted(subject_indices.keys()):
        idx = subject_indices[subj_id]
        if len(idx) < 50:
            print(f"  {subj_id:<10} {len(idx):<12} SKIPPED (too few samples)")
            continue

        real_subj = real[idx]
        # Sample same number of generated windows (generated data has no subject structure)
        n = len(idx)
        fake_idx = np.random.choice(fake.shape[0], size=min(n, fake.shape[0]), replace=False)
        fake_subj = fake[fake_idx]

        try:
            score, fake_acc, real_acc, _ = discriminative_score_metrics(real_subj, fake_subj)
            results[subj_id] = {
                "n_windows": int(n),
                "disc_score": float(score),
                "fake_acc": float(fake_acc),
                "real_acc": float(real_acc),
            }
            print(f"  {subj_id:<10} {n:<12} {score:<15.4f} {fake_acc:<12.4f} {real_acc:<12.4f}")
        except Exception as e:
            print(f"  {subj_id:<10} {n:<12} ERROR: {e}")
            results[subj_id] = {"n_windows": int(n), "error": str(e)}

    return results


# ── 2. Pooled vs per-subject quality ──────────────────────────────────────

def plot_per_subject_scores(all_results, outdir):
    """Bar chart of discriminative scores per subject, grouped by method."""
    methods = list(all_results.keys())
    subjects = sorted(set(
        s for m in methods for s in all_results[m].keys()
        if isinstance(all_results[m][s], dict) and "disc_score" in all_results[m][s]
    ))

    if not subjects:
        print("  No per-subject scores to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(subjects))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(methods))

    for i, method in enumerate(methods):
        scores = []
        for s in subjects:
            entry = all_results[method].get(s, {})
            scores.append(entry.get("disc_score", np.nan))
        bars = ax.bar(x + offsets[i], scores, width / len(methods) * 1.8,
                      label=method, color=COLORS.get(method, f"C{i}"), alpha=0.8)

    ax.set_xlabel("Subject ID", fontsize=12)
    ax.set_ylabel("Discriminative Score (|0.5 - acc|) ↓", fontsize=11)
    ax.set_title("Per-Subject Generation Quality", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects])
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_per_subject_disc.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Inter-subject vs intra-subject distance ───────────────────────────

def compute_mmd_simple(x, y):
    """Simple MMD^2 estimate with RBF kernel (bandwidth=median heuristic)."""
    from sklearn.metrics.pairwise import rbf_kernel

    # Subsample for speed
    n = min(500, x.shape[0], y.shape[0])
    x = x[np.random.choice(x.shape[0], n, replace=False)].reshape(n, -1)
    y = y[np.random.choice(y.shape[0], n, replace=False)].reshape(n, -1)

    # Median heuristic for bandwidth
    from scipy.spatial.distance import cdist
    dists = cdist(x[:100], y[:100], "sqeuclidean")
    gamma = 1.0 / np.median(dists[dists > 0])

    kxx = rbf_kernel(x, x, gamma=gamma).mean()
    kyy = rbf_kernel(y, y, gamma=gamma).mean()
    kxy = rbf_kernel(x, y, gamma=gamma).mean()
    return float(kxx + kyy - 2 * kxy)


def inter_intra_subject_analysis(real, fake, subject_indices, method_name, outdir):
    """
    Compare:
    - Intra-subject distance: MMD between real and generated within each subject
    - Inter-subject distance: MMD between different subjects in real data
    - Inter-subject in generated: do generated samples preserve subject differences?
    """
    subjects = sorted(subject_indices.keys())
    if len(subjects) < 2:
        print("  Need at least 2 subjects for inter/intra analysis.")
        return {}

    print(f"\n  Inter/intra-subject analysis ({method_name}):")

    # Intra-subject: real vs generated per subject
    intra_scores = {}
    for s in subjects:
        idx = subject_indices[s]
        if len(idx) < 30:
            continue
        real_s = real[idx]
        fake_sample = fake[np.random.choice(fake.shape[0], min(len(idx), fake.shape[0]), replace=False)]
        mmd = compute_mmd_simple(real_s, fake_sample)
        intra_scores[s] = mmd

    # Inter-subject in real data: pairwise MMD between subjects
    inter_real = []
    pairs = []
    for i, s1 in enumerate(subjects):
        for s2 in subjects[i + 1:]:
            idx1 = subject_indices[s1]
            idx2 = subject_indices[s2]
            if len(idx1) < 30 or len(idx2) < 30:
                continue
            mmd = compute_mmd_simple(real[idx1], real[idx2])
            inter_real.append(mmd)
            pairs.append((s1, s2))

    mean_intra = np.mean(list(intra_scores.values())) if intra_scores else float("nan")
    mean_inter = np.mean(inter_real) if inter_real else float("nan")

    print(f"    Mean intra-subject MMD (real vs gen): {mean_intra:.6f}")
    print(f"    Mean inter-subject MMD (real vs real): {mean_inter:.6f}")
    if mean_inter > 0:
        ratio = mean_intra / mean_inter
        print(f"    Ratio (intra/inter): {ratio:.4f}")
        print(f"    Interpretation: {'Generated data is close to real' if ratio < 0.5 else 'Generated data differs from real'}")

    results = {
        "intra_subject_mmd": {str(k): v for k, v in intra_scores.items()},
        "mean_intra_mmd": float(mean_intra),
        "mean_inter_mmd_real": float(mean_inter),
        "intra_inter_ratio": float(mean_intra / mean_inter) if mean_inter > 0 else None,
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-subject intra MMD
    ax = axes[0]
    subjs = sorted(intra_scores.keys())
    vals = [intra_scores[s] for s in subjs]
    color = COLORS.get(method_name, "#333333")
    ax.bar(range(len(subjs)), vals, color=color, alpha=0.7)
    ax.axhline(y=mean_inter, color="gray", linestyle="--", linewidth=2,
               label=f"Mean inter-subject (real): {mean_inter:.4f}")
    ax.set_xticks(range(len(subjs)))
    ax.set_xticklabels([str(s) for s in subjs])
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("MMD (real vs generated)")
    ax.set_title(f"{method_name}: Per-Subject Quality", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: inter-subject distances in real data
    ax = axes[1]
    if inter_real:
        ax.hist(inter_real, bins=15, color=COLORS["Real"], alpha=0.5, label="Inter-subject (real)")
        ax.axvline(x=mean_intra, color=color, linewidth=2, linestyle="--",
                   label=f"Mean intra (gen): {mean_intra:.4f}")
        ax.set_xlabel("MMD")
        ax.set_ylabel("Count")
        ax.set_title("Subject Separability", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle(f"MHEALTH — Subject-Level Analysis ({method_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, f"mhealth_inter_intra_{method_name.lower().replace('-', '_')}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return results


# ── 4. Per-subject correlation preservation ───────────────────────────────

def per_subject_correlation_analysis(real, fake, subject_indices, method_name, outdir):
    """Compute per-subject correlation MAE."""
    results = {}
    subjects = sorted(subject_indices.keys())

    for s in subjects:
        idx = subject_indices[s]
        if len(idx) < 30:
            continue

        real_s = real[idx]
        # Use matching count of fake samples
        n = min(len(idx), fake.shape[0])
        fake_s = fake[np.random.choice(fake.shape[0], n, replace=False)]

        corr_real = np.corrcoef(real_s.reshape(-1, real_s.shape[2]).T)
        corr_fake = np.corrcoef(fake_s.reshape(-1, fake_s.shape[2]).T)

        # Replace NaN with 0
        corr_real = np.nan_to_num(corr_real)
        corr_fake = np.nan_to_num(corr_fake)

        tril = np.tril_indices_from(corr_real, k=-1)
        mae = np.abs(corr_real[tril] - corr_fake[tril]).mean()
        results[s] = float(mae)

    return results


def plot_per_subject_correlation_mae(all_corr_results, outdir):
    """Bar chart of per-subject correlation MAE by method."""
    methods = list(all_corr_results.keys())
    subjects = sorted(set(s for m in methods for s in all_corr_results[m].keys()))

    if not subjects:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(subjects))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(methods))

    for i, method in enumerate(methods):
        vals = [all_corr_results[method].get(s, np.nan) for s in subjects]
        ax.bar(x + offsets[i], vals, width / len(methods) * 1.8,
               label=method, color=COLORS.get(method, f"C{i}"), alpha=0.8)

    ax.set_xlabel("Subject ID", fontsize=12)
    ax.set_ylabel("Correlation MAE ↓", fontsize=11)
    ax.set_title("Per-Subject Cross-Channel Correlation Preservation", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_per_subject_corr_mae.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. Cross-channel correlation matrices (kept from original) ────────────

def plot_correlation_matrices(real, methods, outdir):
    """Side-by-side full correlation matrices."""
    n_plots = 1 + len(methods)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    def corr_mat(data):
        return np.nan_to_num(np.corrcoef(data.reshape(-1, data.shape[2]).T))

    corr_real = corr_mat(real)
    all_data = [("Real", corr_real)]
    for name, data in methods.items():
        all_data.append((name, corr_mat(data)))

    for idx, (title, corr) in enumerate(all_data):
        ax = axes[idx]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, len(CHANNEL_NAMES), 3))
        ax.set_xticklabels([CHANNEL_NAMES[i][:6] for i in range(0, len(CHANNEL_NAMES), 3)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(0, len(CHANNEL_NAMES), 3))
        ax.set_yticklabels([CHANNEL_NAMES[i][:6] for i in range(0, len(CHANNEL_NAMES), 3)],
                           fontsize=7)
        for gname, channels in SENSOR_GROUPS.items():
            s, e = channels[0] - 0.5, channels[-1] + 0.5
            ax.add_patch(plt.Rectangle((s, s), e - s, e - s,
                                       fill=False, edgecolor="black", linewidth=1.5))

    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Pearson r")
    plt.suptitle("MHEALTH — Cross-Channel Correlation Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "mhealth_correlation_matrices.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MHEALTH subject-level heterogeneity analysis")
    parser.add_argument("--real", required=True, help="Ground truth .npy ([0,1] range)")
    parser.add_argument("--subject_labels", required=True,
                        help="Per-window subject IDs from preprocess_mhealth.py (mhealth_window_subjects.npy)")
    parser.add_argument("--diffts", default=None, help="Diffusion-TS generated .npy")
    parser.add_argument("--padts", default=None, help="PaD-TS generated .npy")
    parser.add_argument("--diffts_range", default="zero1", choices=["zero1", "neg1to1"])
    parser.add_argument("--padts_range", default="neg1to1", choices=["zero1", "neg1to1"])
    parser.add_argument("--outdir", default="./results/mhealth")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    real = np.load(args.real)
    window_subjects = np.load(args.subject_labels)
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

    # Map window subjects to training indices
    subject_indices, train_subjects = get_subject_indices(window_subjects, n)

    print("=" * 70)
    print("MHEALTH Subject-Level Heterogeneity Analysis")
    print("=" * 70)
    print(f"  Real data: {real.shape}")
    for name, data in methods.items():
        print(f"  {name}: {data.shape}")
    print(f"  Subject labels: {len(window_subjects)} windows")
    print(f"  Valid subjects in training data:")
    for s in sorted(subject_indices.keys()):
        print(f"    Subject {s:2d}: {len(subject_indices[s]):6d} windows")

    n_valid = sum(len(v) for v in subject_indices.values())
    n_boundary = n - n_valid
    if n_boundary > 0:
        print(f"    Boundary/invalid: {n_boundary} windows")

    all_results = {}

    # ── Analysis 1: Cross-channel correlation matrices
    print("\n" + "=" * 70)
    print("1. Cross-channel correlation matrices")
    print("=" * 70)
    plot_correlation_matrices(real, methods, args.outdir)

    # ── Analysis 2: Per-subject discriminative scores
    print("\n" + "=" * 70)
    print("2. Per-subject discriminative scores")
    print("=" * 70)
    disc_results = {}
    for name, data in methods.items():
        disc_results[name] = per_subject_discriminative(real, data, subject_indices, name)
    all_results["per_subject_discriminative"] = disc_results
    plot_per_subject_scores(disc_results, args.outdir)

    # ── Analysis 3: Inter-subject vs intra-subject distance
    print("\n" + "=" * 70)
    print("3. Inter-subject vs intra-subject distributional distance")
    print("=" * 70)
    inter_intra_results = {}
    for name, data in methods.items():
        inter_intra_results[name] = inter_intra_subject_analysis(
            real, data, subject_indices, name, args.outdir
        )
    all_results["inter_intra_subject"] = inter_intra_results

    # ── Analysis 4: Per-subject correlation preservation
    print("\n" + "=" * 70)
    print("4. Per-subject cross-channel correlation preservation")
    print("=" * 70)
    corr_results = {}
    for name, data in methods.items():
        corr_results[name] = per_subject_correlation_analysis(
            real, data, subject_indices, name, args.outdir
        )
        print(f"  {name} per-subject corr MAE:")
        for s in sorted(corr_results[name].keys()):
            print(f"    Subject {s:2d}: {corr_results[name][s]:.4f}")
    all_results["per_subject_correlation_mae"] = {
        m: {str(k): v for k, v in d.items()} for m, d in corr_results.items()
    }
    plot_per_subject_correlation_mae(corr_results, args.outdir)

    # ── Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<40}", end="")
    for name in methods:
        print(f" {name:<18}", end="")
    print()
    print("-" * (40 + 18 * len(methods)))

    # Pooled discriminative score
    for name in methods:
        scores = [v["disc_score"] for v in disc_results[name].values()
                  if isinstance(v, dict) and "disc_score" in v]
        all_results.setdefault("pooled_disc", {})[name] = {
            "mean": float(np.mean(scores)) if scores else None,
            "std": float(np.std(scores)) if scores else None,
        }

    row = f"{'Mean per-subj disc score':<40}"
    for name in methods:
        entry = all_results["pooled_disc"][name]
        if entry["mean"] is not None:
            row += f" {entry['mean']:.4f} ± {entry['std']:.4f}  "
        else:
            row += f" {'N/A':<18}"
    print(row)

    # Intra/inter ratio
    row = f"{'Intra/inter MMD ratio':<40}"
    for name in methods:
        ratio = inter_intra_results.get(name, {}).get("intra_inter_ratio")
        if ratio is not None:
            row += f" {ratio:<18.4f}"
        else:
            row += f" {'N/A':<18}"
    print(row)

    # Mean correlation MAE
    row = f"{'Mean per-subj corr MAE':<40}"
    for name in methods:
        vals = list(corr_results.get(name, {}).values())
        if vals:
            row += f" {np.mean(vals):<18.4f}"
        else:
            row += f" {'N/A':<18}"
    print(row)

    print("=" * (40 + 18 * len(methods)))
    print("(All metrics: lower = better)")

    # Save full results
    path = os.path.join(args.outdir, "heterogeneity_summary.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {path}")


if __name__ == "__main__":
    main()
