"""
Preprocess the MHEALTH dataset into the CSV format expected by PaD-TS.

The UCI MHEALTH dataset has 10 subjects, each in a separate file
(mHealth_subject{1..10}.log). Each file is a space/tab-separated table
with 24 columns:
  - Columns 0-22: 23 sensor channels
  - Column 23: activity label (1-12, 0 = null/no-activity)

This script:
  1. Reads all 10 subject files
  2. Drops the activity label column (col 23)
  3. Optionally filters out null-activity rows (label == 0)
  4. Concatenates all subjects into one long array
  5. Saves as ./dataset/mhealth_data.csv with a header row

Usage:
    # Put raw MHEALTH .log files in Data/datasets/ or specify the path
    python preprocess_mhealth.py --input_dir ./Data/datasets/MHEALTHDATASET
    python preprocess_mhealth.py --input_dir ./Data/datasets/MHEALTHDATASET --keep_null
"""

import os
import argparse
import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_lead1", "ecg_lead2",
    "left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z",
    "left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z",
    "left_ankle_mag_x", "left_ankle_mag_y", "left_ankle_mag_z",
    "right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z",
    "right_arm_gyro_x", "right_arm_gyro_y", "right_arm_gyro_z",
    "right_arm_mag_x", "right_arm_mag_y", "right_arm_mag_z",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing mHealth_subject*.log files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset/mhealth_data.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--keep_null",
        action="store_true",
        help="Keep null-activity rows (label == 0). Default: drop them.",
    )
    args = parser.parse_args()

    frames = []
    for subj_id in range(1, 11):
        fname = f"mHealth_subject{subj_id}.log"
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, skipping.")
            continue
        df = pd.read_csv(fpath, sep=r"\s+", header=None)
        if not args.keep_null:
            df = df[df.iloc[:, 23] != 0]
        # Drop activity label column
        df = df.iloc[:, :23]
        df.columns = SENSOR_COLUMNS
        frames.append(df)
        print(f"Loaded subject {subj_id}: {len(df)} rows")

    if not frames:
        raise FileNotFoundError(f"No mHealth_subject*.log files found in {args.input_dir}")

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined)} rows x {combined.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()
