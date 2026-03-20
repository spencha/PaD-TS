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
  5. Saves as ./dataset/mhealth_data.csv with a header row (23 sensor columns)
  6. Saves ./dataset/mhealth_subject_ids.npy — per-row subject ID (int 1-10)
  7. Saves ./dataset/mhealth_activity_labels.npy — per-row activity label (int 0-12)
  8. Saves ./dataset/mhealth_window_subjects.npy — per-window subject ID after
     sliding-window segmentation (windows spanning subject boundaries are dropped)

The CSV contains only the 23 sensor columns (no subject_id) so it is
directly compatible with CustomDataset. Subject metadata is stored
separately for downstream analysis.

Usage:
    python preprocess_mhealth.py --input_dir ./Data/datasets/MHEALTHDATASET
    python preprocess_mhealth.py --input_dir ./Data/datasets/MHEALTHDATASET --keep_null
    python preprocess_mhealth.py --input_dir ./Data/datasets/MHEALTHDATASET --window 48
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


def compute_window_subject_labels(subject_ids, window):
    """
    Given per-row subject IDs and a window size, compute per-window
    subject labels using the same sliding-window logic as CustomDataset.

    A window is assigned to a subject only if ALL rows in the window
    belong to the same subject. Windows that span subject boundaries
    are marked with subject_id = -1 (invalid).

    Returns:
        window_subjects: np.ndarray of shape (num_windows,) with subject
            IDs (1-10) or -1 for boundary-spanning windows.
        valid_mask: boolean array, True for windows within a single subject.
    """
    n_total = len(subject_ids)
    n_windows = max(n_total - window + 1, 0)
    window_subjects = np.full(n_windows, -1, dtype=int)

    for i in range(n_windows):
        subj_in_window = subject_ids[i : i + window]
        if np.all(subj_in_window == subj_in_window[0]):
            window_subjects[i] = subj_in_window[0]

    valid_mask = window_subjects != -1
    return window_subjects, valid_mask


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
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Window size for computing per-window subject labels (default: 24)",
    )
    args = parser.parse_args()

    frames = []
    subject_id_arrays = []
    activity_label_arrays = []

    for subj_id in range(1, 11):
        fname = f"mHealth_subject{subj_id}.log"
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, skipping.")
            continue
        df = pd.read_csv(fpath, sep=r"\s+", header=None)

        # Store activity labels before filtering
        activity_labels = df.iloc[:, 23].values.copy()

        if not args.keep_null:
            mask = df.iloc[:, 23] != 0
            df = df[mask]
            activity_labels = activity_labels[mask.values]

        n_rows = len(df)

        # Drop activity label column, keep 23 sensor columns
        df = df.iloc[:, :23]
        df.columns = SENSOR_COLUMNS
        frames.append(df)

        # Track subject ID for each row
        subject_id_arrays.append(np.full(n_rows, subj_id, dtype=int))
        activity_label_arrays.append(activity_labels.astype(int))

        print(f"Loaded subject {subj_id}: {n_rows} rows")

    if not frames:
        raise FileNotFoundError(f"No mHealth_subject*.log files found in {args.input_dir}")

    combined = pd.concat(frames, ignore_index=True)
    subject_ids = np.concatenate(subject_id_arrays)
    activity_labels = np.concatenate(activity_label_arrays)

    assert len(subject_ids) == len(combined), "Subject ID length mismatch"

    # Save training CSV (23 sensor columns only — no subject_id)
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined)} rows x {combined.shape[1]} columns to {args.output}")

    # Save per-row subject IDs
    subj_path = os.path.join(out_dir, "mhealth_subject_ids.npy")
    np.save(subj_path, subject_ids)
    print(f"Saved per-row subject IDs to {subj_path}")
    for s in range(1, 11):
        n = (subject_ids == s).sum()
        if n > 0:
            print(f"  Subject {s:2d}: {n:6d} rows")

    # Save per-row activity labels
    act_path = os.path.join(out_dir, "mhealth_activity_labels.npy")
    np.save(act_path, activity_labels)
    print(f"Saved per-row activity labels to {act_path}")

    # Compute and save per-window subject labels
    window_subjects, valid_mask = compute_window_subject_labels(subject_ids, args.window)
    win_path = os.path.join(out_dir, "mhealth_window_subjects.npy")
    np.save(win_path, window_subjects)

    n_valid = valid_mask.sum()
    n_total = len(window_subjects)
    n_boundary = n_total - n_valid
    print(f"\nPer-window subject labels (window={args.window}):")
    print(f"  Total windows:    {n_total}")
    print(f"  Valid (1 subject): {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Boundary (mixed): {n_boundary} ({100*n_boundary/n_total:.1f}%)")
    print(f"  Saved to {win_path}")

    for s in range(1, 11):
        n = (window_subjects == s).sum()
        if n > 0:
            print(f"  Subject {s:2d}: {n:6d} windows")


if __name__ == "__main__":
    main()
