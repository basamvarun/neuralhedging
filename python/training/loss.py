"""
loss.py — CVaR Loss Function (PyTorch)

CVaR (Conditional Value at Risk) at level alpha:
    Sort P&L ascending -> take worst alpha-fraction -> return negative mean.

Minimising this trains the model to reduce catastrophic tail losses:
    CVaR_alpha = -E[PL | PL <= VaR_alpha]
"""

import torch


def cvar_loss(pnl_values: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Compute CVaR loss from a batch of P&L values.

    Args:
        pnl_values: (batch_size,)  P&L per path (positive = profit)
        alpha:      Tail fraction, e.g. 0.05 = worst 5%

    Returns:
        Scalar CVaR loss (positive; minimise this during training)
    """
    sorted_pnl, _ = torch.sort(pnl_values)
    n = sorted_pnl.shape[0]
    n_tail = max(1, int(n * alpha))
    tail = sorted_pnl[:n_tail]
    return -tail.mean()


def mse_delta_loss(predicted_deltas: torch.Tensor,
                   bs_deltas: torch.Tensor) -> torch.Tensor:
    """
    Optional supervised loss: MSE vs Black-Scholes delta.
    Useful for pre-training or sanity checks.

    Args:
        predicted_deltas: (batch_size, n_steps)
        bs_deltas:        (batch_size, n_steps)
    """
    return torch.mean((predicted_deltas - bs_deltas) ** 2)
