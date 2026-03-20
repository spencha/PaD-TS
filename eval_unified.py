"""
Unified evaluation script for comparing PaD-TS and Diffusion-TS outputs.

Computes 6 metrics on any .npy output file:
  1. Discriminative Score  (shared: both repos use same TimeGAN implementation)
  2. Predictive Score      (shared: same implementation)
  3. Context-FID           (from Diffusion-TS, requires ts2vec)
  4. Cross-Correlation     (from Diffusion-TS, lag-1 cacf)
  5. VDS                   (from PaD-TS, value distribution similarity)
  6. FDDS                  (from PaD-TS, feature correlation distribution)

Usage:
    # Evaluate PaD-TS stocks output
    python eval_unified.py \
        --fake ./OUTPUT/stock_24/ddpm_fake_stock_24.npy \
        --real ./OUTPUT/samples/stock_norm_truth_24_train.npy \
        --name stock --method padts

    # Evaluate Diffusion-TS stocks output
    python eval_unified.py \
        --fake ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
        --real ./OUTPUT/samples/stock_norm_truth_24_train.npy \
        --name stock --method diffts \
        --fake_range original

    # Both methods, save results to JSON
    python eval_unified.py \
        --fake output1.npy --real ground_truth.npy \
        --name stock --method padts \
        --output results/eval_stock_padts.json

Notes:
    - Ground truth (.npy) should be in [0,1] range (the _norm_truth_ files).
    - PaD-TS fake outputs are in [-1,1] range (default --fake_range neg1to1).
    - Diffusion-TS fake outputs may be in original scale (use --fake_range original
      and provide --scaler_data to refit MinMaxScaler).
    - Context-FID requires the ts2vec package from Diffusion-TS. If unavailable,
      that metric is skipped.
"""

import os
import sys
import json
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Suppress TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def unnormalize_to_zero_to_one(x):
    """Map [-1, 1] -> [0, 1]"""
    return (x + 1) * 0.5


def load_and_normalize(fake_path, real_path, fake_range, scaler_data_path=None):
    """Load fake and real data, ensure both are in [0, 1] range."""
    real = np.load(real_path)
    fake = np.load(fake_path)

    if fake_range == "neg1to1":
        fake = unnormalize_to_zero_to_one(fake)
    elif fake_range == "zero1":
        pass  # already [0, 1]
    elif fake_range == "original":
        # Re-normalize to [0, 1] using MinMaxScaler fitted on scaler_data
        from sklearn.preprocessing import MinMaxScaler

        if scaler_data_path is None:
            raise ValueError(
                "--scaler_data is required when --fake_range=original. "
                "Provide the raw CSV used for training so we can refit the scaler."
            )
        import pandas as pd

        raw = pd.read_csv(scaler_data_path, header=0).values
        scaler = MinMaxScaler().fit(raw)
        n, t, d = fake.shape
        fake = scaler.transform(fake.reshape(-1, d)).reshape(n, t, d)

    # Match sample counts
    n = min(fake.shape[0], real.shape[0])
    fake = fake[:n]
    real = real[:n]

    print(f"Loaded: real {real.shape}, fake {fake.shape}")
    print(f"  Real  range: [{real.min():.4f}, {real.max():.4f}]")
    print(f"  Fake  range: [{fake.min():.4f}, {fake.max():.4f}]")

    return real, fake


# ── Metric 1: Discriminative Score ──────────────────────────────────────────

def compute_discriminative(real, fake, iterations=5):
    """GRU classifier, score = |0.5 - accuracy|. Lower = better."""
    from eval_utils.discriminative_metric import discriminative_score_metrics

    scores = []
    for i in range(iterations):
        score, fake_acc, real_acc, _ = discriminative_score_metrics(real, fake)
        scores.append(score)
        print(f"  Disc iter {i}: {score:.4f} (fake_acc={fake_acc:.4f}, real_acc={real_acc:.4f})")
    return scores


# ── Metric 2: Predictive Score ──────────────────────────────────────────────

def compute_predictive(real, fake, iterations=5):
    """GRU predictor trained on fake, MAE on real. Lower = better."""
    from eval_utils.predictive_metric import predictive_score_metrics

    scores = []
    for i in range(iterations):
        score = predictive_score_metrics(real, fake)
        scores.append(score)
        print(f"  Pred iter {i}: {score:.4f}")
    return scores


# ── Metric 3: Context-FID ──────────────────────────────────────────────────

def compute_context_fid(real, fake, iterations=3):
    """TS2Vec embedding + FID. Lower = better. Requires Diffusion-TS ts2vec."""
    try:
        # Add Diffusion-TS to path for ts2vec import
        diffts_path = os.path.expanduser("~/Diffusion-TS")
        if diffts_path not in sys.path:
            sys.path.insert(0, diffts_path)
        from Utils.context_fid import Context_FID
    except ImportError:
        print("  SKIPPED: ts2vec not found. Install from ~/Diffusion-TS or pip install ts2vec.")
        return None

    scores = []
    n = min(1000, real.shape[0])
    for i in range(iterations):
        idx = np.random.permutation(real.shape[0])[:n]
        score = Context_FID(real[idx], fake[idx])
        scores.append(float(score))
        print(f"  C-FID iter {i}: {score:.4f}")
    return scores


# ── Metric 4: Cross-Correlation Score ──────────────────────────────────────

def compute_cross_correlation(real, fake):
    """Lag-1 cross-autocorrelation difference. Lower = better."""
    import torch

    try:
        diffts_path = os.path.expanduser("~/Diffusion-TS")
        if diffts_path not in sys.path:
            sys.path.insert(0, diffts_path)
        from Utils.cross_correlation import CrossCorrelLoss
    except ImportError:
        print("  SKIPPED: cross_correlation.py not found in ~/Diffusion-TS/Utils/.")
        return None

    real_t = torch.tensor(real).float()
    fake_t = torch.tensor(fake).float()

    loss_fn = CrossCorrelLoss(real_t, name="cross_correl")
    score = loss_fn(fake_t).item()
    print(f"  CrossCorr: {score:.6f}")
    return [score]


# ── Metric 5: VDS (Value Distribution Similarity) ─────────────────────────

def compute_vds(real, fake):
    """Per-feature MMD on value distributions. Lower = better."""
    import torch
    from eval_utils.MMD import VDS_Naive

    real_t = torch.tensor(real).float()
    fake_t = torch.tensor(fake).float()

    score = VDS_Naive(real_t, fake_t, "rbf").mean().item()
    print(f"  VDS: {score:.6f}")
    return [score]


# ── Metric 6: FDDS (Feature-correlation Distribution Distance) ────────────

def compute_fdds(real, fake):
    """MMD on cross-correlation distributions. Lower = better."""
    import torch
    from eval_utils.MMD import BMMD_Naive, cross_correlation_distribution

    real_t = torch.tensor(real).float()
    fake_t = torch.tensor(fake).float()

    real_ccd = cross_correlation_distribution(real_t).unsqueeze(-1)
    fake_ccd = cross_correlation_distribution(fake_t).unsqueeze(-1)

    score = BMMD_Naive(real_ccd, fake_ccd, "rbf").mean().item()
    print(f"  FDDS: {score:.6f}")
    return [score]


# ── Main ───────────────────────────────────────────────────────────────────

def summarize(scores):
    """Return mean ± std string."""
    if scores is None:
        return "N/A"
    arr = np.array(scores)
    return f"{arr.mean():.4f} ± {arr.std():.4f}"


def main():
    parser = argparse.ArgumentParser(description="Unified TS generation evaluation")
    parser.add_argument("--fake", required=True, help="Path to generated .npy file")
    parser.add_argument("--real", required=True, help="Path to ground truth .npy file ([0,1] range)")
    parser.add_argument("--name", required=True, help="Dataset name (for logging)")
    parser.add_argument("--method", required=True, help="Method name (padts, diffts, etc.)")
    parser.add_argument(
        "--fake_range",
        default="neg1to1",
        choices=["neg1to1", "zero1", "original"],
        help="Range of fake data: neg1to1 (PaD-TS default), zero1, original (needs --scaler_data)",
    )
    parser.add_argument("--scaler_data", default=None, help="Raw CSV for re-fitting MinMaxScaler")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--disc_iters", type=int, default=5, help="Discriminative score iterations")
    parser.add_argument("--pred_iters", type=int, default=5, help="Predictive score iterations")
    parser.add_argument("--fid_iters", type=int, default=3, help="Context-FID iterations")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["disc", "pred", "cfid", "xcorr", "vds", "fdds"],
        help="Metrics to skip",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Unified Evaluation: {args.method} on {args.name}")
    print("=" * 60)

    real, fake = load_and_normalize(args.fake, args.real, args.fake_range, args.scaler_data)

    results = {"method": args.method, "dataset": args.name}

    # 1. Discriminative Score
    if "disc" not in args.skip:
        print("\n── Discriminative Score ──")
        scores = compute_discriminative(real, fake, args.disc_iters)
        results["discriminative"] = {"scores": scores, "summary": summarize(scores)}
    else:
        print("\n── Discriminative Score: SKIPPED ──")

    # 2. Predictive Score
    if "pred" not in args.skip:
        print("\n── Predictive Score ──")
        scores = compute_predictive(real, fake, args.pred_iters)
        results["predictive"] = {"scores": scores, "summary": summarize(scores)}
    else:
        print("\n── Predictive Score: SKIPPED ──")

    # 3. Context-FID
    if "cfid" not in args.skip:
        print("\n── Context-FID ──")
        scores = compute_context_fid(real, fake, args.fid_iters)
        results["context_fid"] = {"scores": scores, "summary": summarize(scores)}
    else:
        print("\n── Context-FID: SKIPPED ──")

    # 4. Cross-Correlation
    if "xcorr" not in args.skip:
        print("\n── Cross-Correlation ──")
        scores = compute_cross_correlation(real, fake)
        results["cross_correlation"] = {"scores": scores, "summary": summarize(scores)}
    else:
        print("\n── Cross-Correlation: SKIPPED ──")

    # 5. VDS
    if "vds" not in args.skip:
        print("\n── VDS (Value Distribution Similarity) ──")
        scores = compute_vds(real, fake)
        results["vds"] = {"scores": scores, "summary": summarize(scores)}
    else:
        print("\n── VDS: SKIPPED ──")

    # 6. FDDS
    if "fdds" not in args.skip:
        print("\n── FDDS (Feature Distribution Distance) ──")
        scores = compute_fdds(real, fake)
        results["fdds"] = {"scores": scores, "summary": summarize(scores)}
    else:
        print("\n── FDDS: SKIPPED ──")

    # Summary table
    print("\n" + "=" * 60)
    print(f"RESULTS: {args.method} on {args.name}")
    print("=" * 60)
    print(f"{'Metric':<25} {'Score (↓ better)':<25}")
    print("-" * 50)
    for key in ["discriminative", "predictive", "context_fid", "cross_correlation", "vds", "fdds"]:
        if key in results:
            print(f"{key:<25} {results[key]['summary']:<25}")
    print("=" * 60)

    # Save to JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
