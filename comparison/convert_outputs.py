"""
Convert .npy outputs between PaD-TS and Diffusion-TS formats so either
repo's evaluation code can consume the other's generated samples.

PaD-TS saves generated samples in [-1, 1] range.
Diffusion-TS saves generated samples in [0, 1] range (after unnormalize).

Both use shape (num_samples, window, num_features).

Usage:
    # Convert PaD-TS output → Diffusion-TS format ([-1,1] → [0,1])
    python convert_outputs.py \
        --input ../OUTPUT/stock_24/ddpm_fake_stock_24.npy \
        --output ./converted/padts_stock_for_diffts.npy \
        --from_format padts --to_format diffts

    # Convert Diffusion-TS output → PaD-TS eval format ([0,1] stays [0,1])
    python convert_outputs.py \
        --input ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
        --output ./converted/diffts_stock_for_padts.npy \
        --from_format diffts --to_format padts

    # Convert Diffusion-TS output from original scale → [0,1] using scaler
    python convert_outputs.py \
        --input ~/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy \
        --output ./converted/diffts_stock_normalized.npy \
        --from_format diffts_original --to_format diffts \
        --scaler_csv ../dataset/stock_data.csv
"""

import os
import argparse
import numpy as np


def unnormalize_to_zero_to_one(x):
    """[-1, 1] → [0, 1]"""
    return (x + 1) * 0.5


def normalize_to_neg_one_to_one(x):
    """[0, 1] → [-1, 1]"""
    return x * 2 - 1


def rescale_with_scaler(data, csv_path):
    """Re-normalize original-scale data to [0, 1] using MinMaxScaler fitted on CSV."""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    raw = pd.read_csv(csv_path, header=0).values
    scaler = MinMaxScaler().fit(raw)
    n, t, d = data.shape
    return scaler.transform(data.reshape(-1, d)).reshape(n, t, d)


def main():
    parser = argparse.ArgumentParser(description="Convert .npy outputs between PaD-TS and Diffusion-TS formats")
    parser.add_argument("--input", required=True, help="Input .npy file")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument(
        "--from_format",
        required=True,
        choices=["padts", "diffts", "diffts_original"],
        help="Source format: padts ([-1,1]), diffts ([0,1]), diffts_original (raw scale)",
    )
    parser.add_argument(
        "--to_format",
        required=True,
        choices=["padts", "diffts"],
        help="Target format: padts ([-1,1] for model, [0,1] for eval), diffts ([0,1])",
    )
    parser.add_argument("--scaler_csv", default=None, help="Raw CSV for MinMaxScaler (needed for diffts_original)")
    parser.add_argument("--match_samples", default=None, help="Ground truth .npy to match sample count")
    args = parser.parse_args()

    data = np.load(args.input)
    print(f"Input:  {args.input}")
    print(f"  Shape: {data.shape}, Range: [{data.min():.4f}, {data.max():.4f}]")

    # Step 1: Convert to [0, 1] (common intermediate)
    if args.from_format == "padts":
        data_01 = unnormalize_to_zero_to_one(data)
    elif args.from_format == "diffts":
        data_01 = data
    elif args.from_format == "diffts_original":
        if args.scaler_csv is None:
            raise ValueError("--scaler_csv required for diffts_original format")
        data_01 = rescale_with_scaler(data, args.scaler_csv)

    # Step 2: Convert from [0, 1] to target format
    if args.to_format == "diffts":
        # Diffusion-TS eval expects [0, 1]
        output = data_01
    elif args.to_format == "padts":
        # PaD-TS eval also uses [0, 1] (unnormalizes before metrics)
        output = data_01

    # Optionally match sample count to ground truth
    if args.match_samples:
        gt = np.load(args.match_samples)
        n = min(output.shape[0], gt.shape[0])
        output = output[:n]
        print(f"  Matched samples to {args.match_samples}: {n}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, output.astype(np.float32))
    print(f"Output: {args.output}")
    print(f"  Shape: {output.shape}, Range: [{output.min():.4f}, {output.max():.4f}]")


if __name__ == "__main__":
    main()
