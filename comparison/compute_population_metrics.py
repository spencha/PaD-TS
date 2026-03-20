"""
Compute ALL metrics from both repos on any .npy output for apples-to-apples
comparison. Covers:

From PaD-TS (population-level):
  - VDS  (Value Distribution Similarity) — per-feature marginal MMD
  - FDDS (Feature Distribution Distance Score) — cross-correlation MMD

From Diffusion-TS:
  - Context-FID — TS2Vec embedding + Fréchet distance
  - Cross-Correlation Score — lag-1 cacf absolute difference

Shared (identical implementations in both repos):
  - Discriminative Score — GRU classifier, |0.5 - accuracy|
  - Predictive Score — GRU predictor, train-on-synthetic MAE

Usage:
    # Full comparison on stock dataset
    python compute_population_metrics.py \
        --real ../OUTPUT/samples/stock_norm_truth_24_train.npy \
        --fake_diffts ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
        --fake_padts ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
        --name stock \
        --diffts_range zero1 --padts_range neg1to1 \
        --output results_stock.json

    # Single method evaluation
    python compute_population_metrics.py \
        --real ../OUTPUT/samples/stock_norm_truth_24_train.npy \
        --fake_padts ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
        --name stock --padts_range neg1to1 \
        --output results_stock_padts.json
"""

import os
import sys
import json
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add PaD-TS root to path for eval_utils imports
PADTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PADTS_ROOT not in sys.path:
    sys.path.insert(0, PADTS_ROOT)

DIFFTS_ROOT = os.path.expanduser("~/Diffusion-TS")


def unnorm(x):
    """[-1,1] → [0,1]"""
    return (x + 1) * 0.5


def load_fake(path, data_range):
    data = np.load(path)
    if data_range == "neg1to1":
        data = unnorm(data)
    return data


def summarize(scores):
    if scores is None:
        return "N/A"
    a = np.array(scores, dtype=float)
    return f"{a.mean():.4f} ± {a.std():.4f}"


# ── Shared metrics (from PaD-TS eval_utils) ────────────────────────────────

def metric_discriminative(real, fake, iters=5):
    from eval_utils.discriminative_metric import discriminative_score_metrics
    scores = []
    for i in range(iters):
        s, _, _, _ = discriminative_score_metrics(real, fake)
        scores.append(float(s))
        print(f"    iter {i}: {s:.4f}")
    return scores


def metric_predictive(real, fake, iters=5):
    from eval_utils.predictive_metric import predictive_score_metrics
    scores = []
    for i in range(iters):
        s = predictive_score_metrics(real, fake)
        scores.append(float(s))
        print(f"    iter {i}: {s:.4f}")
    return scores


# ── PaD-TS population metrics ─────────────────────────────────────────────

def metric_vds(real, fake):
    import torch
    from eval_utils.MMD import VDS_Naive
    r = torch.tensor(real).float()
    f = torch.tensor(fake).float()
    score = VDS_Naive(r, f, "rbf").mean().item()
    print(f"    VDS: {score:.6f}")
    return [score]


def metric_fdds(real, fake):
    import torch
    from eval_utils.MMD import BMMD_Naive, cross_correlation_distribution
    r = torch.tensor(real).float()
    f = torch.tensor(fake).float()
    r_ccd = cross_correlation_distribution(r).unsqueeze(-1)
    f_ccd = cross_correlation_distribution(f).unsqueeze(-1)
    score = BMMD_Naive(r_ccd, f_ccd, "rbf").mean().item()
    print(f"    FDDS: {score:.6f}")
    return [score]


# ── Diffusion-TS metrics ──────────────────────────────────────────────────

def metric_context_fid(real, fake, iters=3):
    try:
        if DIFFTS_ROOT not in sys.path:
            sys.path.insert(0, DIFFTS_ROOT)
        from Utils.context_fid import Context_FID
    except ImportError:
        print("    SKIPPED (ts2vec not available)")
        return None

    scores = []
    n = min(1000, real.shape[0], fake.shape[0])
    for i in range(iters):
        idx = np.random.permutation(min(real.shape[0], fake.shape[0]))[:n]
        s = float(Context_FID(real[idx], fake[idx]))
        scores.append(s)
        print(f"    iter {i}: {s:.4f}")
    return scores


def metric_cross_corr(real, fake):
    try:
        if DIFFTS_ROOT not in sys.path:
            sys.path.insert(0, DIFFTS_ROOT)
        import torch
        from Utils.cross_correlation import CrossCorrelLoss
    except ImportError:
        print("    SKIPPED (cross_correlation.py not available)")
        return None

    r = torch.tensor(real).float()
    f = torch.tensor(fake).float()
    loss_fn = CrossCorrelLoss(r, name="cross_correl")
    score = loss_fn(f).item()
    print(f"    CrossCorr: {score:.6f}")
    return [score]


# ── Main ───────────────────────────────────────────────────────────────────

def evaluate_one(label, real, fake, skip):
    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"  Real:  {real.shape}, [{real.min():.3f}, {real.max():.3f}]")
    print(f"  Fake:  {fake.shape}, [{fake.min():.3f}, {fake.max():.3f}]")
    print(f"{'='*50}")

    n = min(real.shape[0], fake.shape[0])
    real_n, fake_n = real[:n], fake[:n]

    results = {}

    if "disc" not in skip:
        print("  Discriminative Score:")
        results["discriminative"] = metric_discriminative(real_n, fake_n)

    if "pred" not in skip:
        print("  Predictive Score:")
        results["predictive"] = metric_predictive(real_n, fake_n)

    if "cfid" not in skip:
        print("  Context-FID:")
        results["context_fid"] = metric_context_fid(real_n, fake_n)

    if "xcorr" not in skip:
        print("  Cross-Correlation:")
        results["cross_correlation"] = metric_cross_corr(real_n, fake_n)

    if "vds" not in skip:
        print("  VDS:")
        results["vds"] = metric_vds(real_n, fake_n)

    if "fdds" not in skip:
        print("  FDDS:")
        results["fdds"] = metric_fdds(real_n, fake_n)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute all metrics for method comparison")
    parser.add_argument("--real", required=True, help="Ground truth .npy ([0,1] range)")
    parser.add_argument("--fake_diffts", default=None, help="Diffusion-TS generated .npy")
    parser.add_argument("--fake_padts", default=None, help="PaD-TS generated .npy")
    parser.add_argument("--name", required=True, help="Dataset name")
    parser.add_argument("--diffts_range", default="zero1", choices=["zero1", "neg1to1"])
    parser.add_argument("--padts_range", default="neg1to1", choices=["zero1", "neg1to1"])
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["disc", "pred", "cfid", "xcorr", "vds", "fdds"])
    args = parser.parse_args()

    if args.fake_diffts is None and args.fake_padts is None:
        raise ValueError("Provide at least one of --fake_diffts or --fake_padts")

    real = np.load(args.real)
    all_results = {"dataset": args.name}

    if args.fake_diffts:
        diffts = load_fake(args.fake_diffts, args.diffts_range)
        all_results["diffusion_ts"] = evaluate_one("Diffusion-TS", real, diffts, args.skip)

    if args.fake_padts:
        padts = load_fake(args.fake_padts, args.padts_range)
        all_results["padts"] = evaluate_one("PaD-TS", real, padts, args.skip)

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"COMPARISON TABLE: {args.name}")
    print("=" * 70)
    print(f"{'Metric':<25} {'Diffusion-TS':<22} {'PaD-TS':<22}")
    print("-" * 70)

    metrics = ["discriminative", "predictive", "context_fid", "cross_correlation", "vds", "fdds"]
    for m in metrics:
        d_val = summarize(all_results.get("diffusion_ts", {}).get(m)) if args.fake_diffts else "—"
        p_val = summarize(all_results.get("padts", {}).get(m)) if args.fake_padts else "—"
        print(f"{m:<25} {d_val:<22} {p_val:<22}")
    print("=" * 70)
    print("(All metrics: lower = better)")

    if args.output:
        # Convert for JSON serialization
        for method in ["diffusion_ts", "padts"]:
            if method in all_results:
                for k, v in all_results[method].items():
                    if v is not None:
                        all_results[method][k] = {
                            "scores": [float(x) for x in v],
                            "summary": summarize(v),
                        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
