"""
pnl.py — Differentiable P&L Calculation (PyTorch)

Implements the hedger P&L formula from the reference:

    PL_T = -Z(S_T) + sum_{t=0}^{n-1} [ delta_t * (S_{t+1} - S_t) ] - C_T

Where:
    Z(S_T)  = option payoff at expiry
    delta_t = hedge position held from t to t+1
    C_T     = transaction costs (optional, default 0)

All ops are differentiable so gradients flow back through delta predictions.
"""

import torch


def compute_pnl(
    S_paths: torch.Tensor,
    deltas: torch.Tensor,
    K: float,
    option_type: str = "call",
    cost_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute terminal P&L for a batch of paths.

    Args:
        S_paths:     (batch_size, n_steps+1)  underlying prices
        deltas:      (batch_size, n_steps+1)  hedge positions
        K:           Strike price
        option_type: 'call' or 'put'
        cost_rate:   Transaction cost rate (0 = no costs)

    Returns:
        pnl: (batch_size,)
    """
    # Trading P&L: sum of delta_t * (S_{t+1} - S_t)
    price_changes = S_paths[:, 1:] - S_paths[:, :-1]    # (batch, n_steps)
    trading_pnl = (deltas[:, :-1] * price_changes).sum(dim=1)  # (batch,)

    # Option payoff at expiry
    S_T = S_paths[:, -1]
    K_t = torch.tensor(K, dtype=S_paths.dtype, device=S_paths.device)
    if option_type == "call":
        payoff = torch.clamp(S_T - K_t, min=0.0)
    else:
        payoff = torch.clamp(K_t - S_T, min=0.0)

    # Transaction costs (optional)
    if cost_rate > 0.0:
        delta_changes = torch.abs(deltas[:, 1:] - deltas[:, :-1])
        costs = cost_rate * (S_paths[:, :-1] * delta_changes).sum(dim=1)
    else:
        costs = torch.zeros(S_paths.shape[0], device=S_paths.device)

    # Hedger sold the option -> payoff is a loss for the hedger
    return -payoff + trading_pnl - costs
