"""
delta_hedge.py — Black-Scholes Delta Hedging Baseline

Implements the classical BS delta hedging strategy using the same
P&L formula as the deep hedger, so results are directly comparable:

    PL_T = -Z(S_T) + sum_{t} [ bs_delta_t * (S_{t+1} - S_t) ] - C_T

This is the benchmark the neural hedger must outperform.
"""

import numpy as np
import torch
from scipy.stats import norm

from python.training.pnl import compute_pnl


def bs_delta(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str) -> float:
    """Black-Scholes delta for one option at one point in time."""
    if T <= 0 or sigma <= 1e-8:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if option_type == "call" else float(norm.cdf(d1) - 1.0)


def compute_bs_hedge_pnl(
    S_paths: np.ndarray,
    K: float,
    T_total: float,
    r: float,
    sigma: float,
    option_type: str,
    cost_rate: float = 0.0,
) -> np.ndarray:
    """
    Compute P&L for the Black-Scholes delta hedge on a batch of paths.

    Args:
        S_paths:     (n_paths, n_steps+1)
        K:           Strike price
        T_total:     Total time to expiry at step 0
        r:           Risk-free rate
        sigma:       Constant implied volatility
        option_type: 'call' or 'put'
        cost_rate:   Transaction cost rate

    Returns:
        pnl: (n_paths,)  terminal P&L per path
    """
    n_paths, n_steps_plus1 = S_paths.shape
    n_steps = n_steps_plus1 - 1
    dt = T_total / n_steps

    # --- Vectorised BS delta over all paths & timesteps ---
    t_steps = np.arange(n_steps_plus1, dtype=np.float64)
    T_rem   = np.maximum(0.0, T_total - t_steps * dt)   # (n_steps+1,)
    valid   = T_rem > 0

    S  = S_paths                          # (n_paths, n_steps+1)
    Tv = T_rem[np.newaxis, :]             # (1, n_steps+1)

    deltas = np.zeros_like(S_paths, dtype=np.float64)
    if np.any(valid):
        S_v  = S[:, valid]
        T_v  = Tv[:, valid]
        d1   = (np.log(S_v / K) + (r + 0.5 * sigma**2) * T_v) / (sigma * np.sqrt(T_v))
        from scipy.stats import norm as _norm
        if option_type == "call":
            deltas[:, valid] = _norm.cdf(d1)
        else:
            deltas[:, valid] = _norm.cdf(d1) - 1.0

    if np.any(~valid):
        S_exp = S[:, ~valid]
        if option_type == "call":
            deltas[:, ~valid] = (S_exp > K).astype(np.float64)
        else:
            deltas[:, ~valid] = -(S_exp < K).astype(np.float64)

    # Convert to tensors and reuse the same compute_pnl as the deep hedger
    S_tensor = torch.tensor(S_paths, dtype=torch.float32)
    d_tensor = torch.tensor(deltas,  dtype=torch.float32)

    with torch.no_grad():
        pnl = compute_pnl(S_tensor, d_tensor, K, option_type, cost_rate)

    return pnl.numpy(), deltas
