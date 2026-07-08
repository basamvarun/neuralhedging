"""
seed.py — Reproducibility Utilities

Seeds all random number generators for reproducible experiments:
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA (if available)
"""

import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: Integer seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic (slight performance cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
