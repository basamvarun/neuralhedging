"""
preprocessor.py — Tensor Preparation

Wraps GBM paths and BS deltas into PyTorch tensors.

Feature tensor per step t: [S_t (raw),  BS_delta_t]  — 2 features
delta_{t-1} is the 3rd input but is added in the training loop (trainer.py).
This matches the notebook: inputs are [S_t, BS_delta_t, delta_{t-1}].
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class HedgingDataset(Dataset):
    """
    Tensors:
        S_paths   : (n_paths, n_steps+1)       raw underlying prices
        bs_deltas : (n_paths, n_steps+1)       BS benchmark deltas
        features  : (n_paths, n_steps+1, 2)    [S_t, BS_delta_t]
    """

    def __init__(self, S_paths: np.ndarray, bs_deltas: np.ndarray):
        """
        Args:
            S_paths:   (n_paths, n_steps+1)  raw underlying prices
            bs_deltas: (n_paths, n_steps+1)  BS deltas
        """
        # Feature: [S_t / S0 (normalized spot), BS_delta_t]
        # Normalization prevents network saturation and vanishing gradients due to large absolute price scales.
        S_init = S_paths[:, 0:1]
        S_norm = S_paths / S_init
        feat = np.stack([S_norm, bs_deltas], axis=-1)  # (n_paths, n_steps+1, 2)

        self.S_paths   = torch.tensor(S_paths,   dtype=torch.float32)
        self.bs_deltas = torch.tensor(bs_deltas, dtype=torch.float32)
        self.features  = torch.tensor(feat,      dtype=torch.float32)

    def __len__(self):
        return self.S_paths.shape[0]

    def __getitem__(self, idx):
        return self.S_paths[idx], self.features[idx], self.bs_deltas[idx]

    def split(self, train_ratio=0.8, seed=42):
        """Split into (train_dataset, test_dataset)."""
        n = len(self)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)
        n_train = int(n * train_ratio)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        def _subset(indices):
            ds = HedgingDataset.__new__(HedgingDataset)
            ds.S_paths   = self.S_paths[indices]
            ds.bs_deltas = self.bs_deltas[indices]
            ds.features  = self.features[indices]
            return ds

        return _subset(train_idx), _subset(test_idx)
