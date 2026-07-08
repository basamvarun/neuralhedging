"""
trainer.py — Deep Hedger Training Loop (PyTorch)

Matches the notebook training loop exactly:
    At each step t: input = [S_t, BS_delta_t, delta_{t-1}]  -> delta_t
    delta_{t-1} is the model's own previous output (fed back each step).
    Loss: CVaR at alpha=0.05.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from python.training.loss import cvar_loss
from python.training.pnl import compute_pnl


def run_hedger_on_batch(
    model: nn.Module,
    S_batch: torch.Tensor,     # (batch, n_steps+1)
    feat_batch: torch.Tensor,  # (batch, n_steps+1, input_dim)
) -> torch.Tensor:
    """
    Roll model through all timesteps.
    For DenseHedger: feeding delta_{t-1} back each step (3 features).
    For LSTM/GRU: full sequence passed in one shot (8 features).

    Returns:
        all_deltas: (batch, n_steps+1)
    """
    if hasattr(model, "rnn"):
        return model(feat_batch).squeeze(-1)  # (batch, n_steps+1)

    batch_size, n_steps_plus1, _ = feat_batch.shape
    device = S_batch.device

    delta_prev = torch.zeros(batch_size, 1, device=device)
    deltas = []

    for t in range(n_steps_plus1):
        feat_t = feat_batch[:, t, :]                     # (batch, 2) [S_t, BS_delta_t]
        x_t = torch.cat([feat_t, delta_prev], dim=-1)    # (batch, 3)
        delta_t = model(x_t)                              # (batch, 1)
        deltas.append(delta_t)
        delta_prev = delta_t.detach()

    return torch.cat(deltas, dim=-1)   # (batch, n_steps+1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    K: float,
    option_type: str,
    alpha: float = 0.05,
    cost_rate: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Run one training epoch. Returns mean CVaR loss."""
    model.train()
    total_loss = 0.0

    for S_batch, feat_batch, _ in loader:
        S_batch    = S_batch.to(device)
        feat_batch = feat_batch.to(device)

        optimiser.zero_grad()
        deltas = run_hedger_on_batch(model, S_batch, feat_batch)
        pnl    = compute_pnl(S_batch, deltas, K, option_type, cost_rate)
        loss   = cvar_loss(pnl, alpha=alpha)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    K: float,
    option_type: str,
    alpha: float = 0.05,
    cost_rate: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Evaluate model. Returns CVaR loss and P&L stats."""
    model.eval()
    all_pnl = []

    for S_batch, feat_batch, _ in loader:
        S_batch    = S_batch.to(device)
        feat_batch = feat_batch.to(device)
        deltas     = run_hedger_on_batch(model, S_batch, feat_batch)
        pnl        = compute_pnl(S_batch, deltas, K, option_type, cost_rate)
        all_pnl.append(pnl)

    all_pnl = torch.cat(all_pnl)
    return {
        "cvar_loss": cvar_loss(all_pnl, alpha).item(),
        "mean_pnl":  all_pnl.mean().item(),
        "std_pnl":   all_pnl.std().item(),
    }
