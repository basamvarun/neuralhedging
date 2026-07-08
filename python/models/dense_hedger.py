"""
dense_hedger.py — Feedforward Deep Hedger (PyTorch)

Matches the reference notebook exactly:
    Inputs:  [S_t (raw price),  BS_delta_t,  delta_{t-1}]  -> 3 features
    Hidden:  Linear(3, 32) -> Tanh       (single hidden layer, 32 units)
    Output:  Linear(32, 1) -> Tanh       (delta in (-1, 1))

The model is stateless. delta_{t-1} is fed back from the previous step
by the training loop (trainer.py).
"""

import torch
import torch.nn as nn


class DenseHedger(nn.Module):
    """
    Single-step feedforward hedger matching the notebook architecture.

    Forward:
        Input:  (batch_size, 3)   [S_t, bs_delta_t, delta_prev]
        Output: (batch_size, 1)   predicted delta in (-1, 1)
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),   # output in (-1, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3)  [S_t, bs_delta_t, delta_prev]
        Returns:
            delta: (batch_size, 1)
        """
        return self.net(x)
