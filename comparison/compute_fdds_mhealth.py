"""Quick script to compute just the FDDS score for MHEALTH (subsampled to avoid OOM)."""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_utils.MMD import BMMD_Naive, cross_correlation_distribution

ori = np.load("../OUTPUT/samples/mhealth_norm_truth_24_train.npy")
fake = np.load("../OUTPUT/mhealth_24/ddpm_fake_mhealth_24.npy")
fake = (fake + 1) * 0.5  # unnormalize [-1,1] -> [0,1]
fake = fake[:ori.shape[0]]

# Subsample to avoid OOM
n = 5000
idx = np.random.choice(ori.shape[0], n, replace=False)
ori_t = torch.tensor(ori[idx]).float()
fake_t = torch.tensor(fake[idx]).float()

print(f"Computing FDDS on {n} samples (from {ori.shape[0]} total)...")
ori_ccd = cross_correlation_distribution(ori_t).unsqueeze(-1)
fake_ccd = cross_correlation_distribution(fake_t).unsqueeze(-1)
fdds = BMMD_Naive(ori_ccd, fake_ccd, "rbf").mean()
print(f"MHEALTH FDDS Score: {fdds}")
