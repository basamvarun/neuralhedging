"""
lstm_hedger.py — LSTM/GRU Deep Hedger (Experiment B)

Processes the entire price path as a sequence in one forward pass.
Richer input features than DenseHedger: 8 features per timestep.

Input at each step t:
    [S_t/S0, log(S_t/K), T_t, BS_delta_t, BS_gamma_t,
     BS_vega_t, BS_theta_t, delta_{t-1}]   -> 8 features

Architecture:
    LSTM(input=8, hidden=64, layers=2, dropout=0.1)
    -> Linear(64, 1) -> Tanh   -> delta_t in (-1, 1)

Unlike DenseHedger, the full sequence [t=0..T] is passed in one shot.
The delta_{t-1} is still included as a feature (shifted by one step)
to give the model awareness of its own past actions.
"""

import torch
import torch.nn as nn


class LSTMHedger(nn.Module):
    """
    Sequence-to-sequence LSTM hedger.

    Forward:
        Input:  (batch_size, n_steps+1, input_dim)
        Output: (batch_size, n_steps+1, 1)  — delta at every step

    delta_{t-1} must be provided as part of the feature tensor.
    The trainer handles building this shifted feature.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gru: bool = False,
    ):
        """
        Args:
            input_dim:  Number of input features per timestep (default 8)
            hidden_dim: LSTM/GRU hidden state size
            num_layers: Number of stacked recurrent layers
            dropout:    Dropout between recurrent layers (only if num_layers > 1)
            use_gru:    If True, use GRU instead of LSTM
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gru = use_gru

        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),  # delta in (-1, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.output_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_steps+1, input_dim)
        Returns:
            delta: (batch_size, n_steps+1, 1)
        """
        rnn_out, _ = self.rnn(x)           # (batch, seq, hidden_dim)
        delta = self.output_head(rnn_out)   # (batch, seq, 1)
        return delta


def build_lstm_features(
    S_paths: torch.Tensor,      # (batch, n_steps+1)
    bs_deltas: torch.Tensor,    # (batch, n_steps+1)
    bs_gammas: torch.Tensor,    # (batch, n_steps+1)
    bs_vegas: torch.Tensor,     # (batch, n_steps+1)
    bs_thetas: torch.Tensor,    # (batch, n_steps+1)
    T_grid: torch.Tensor,       # (n_steps+1,)   broadcast across batch
    K: float,
) -> torch.Tensor:
    """
    Build the 8-feature input tensor for LSTMHedger.

    Features per step:
        0: S_t / S0          (normalised price)
        1: log(S_t / K)      (log moneyness)
        2: T_t               (time to expiry)
        3: BS delta
        4: BS gamma
        5: BS vega
        6: BS theta
        7: delta_{t-1}       (previous hedge, shifted right by 1, 0 at t=0)

    Returns: (batch, n_steps+1, 8)
    """
    batch_size, seq_len = S_paths.shape
    device = S_paths.device

    S0 = S_paths[:, :1]                          # (batch, 1)
    S_norm = S_paths / S0                         # (batch, seq)
    log_moneyness = torch.log(S_paths / K)        # (batch, seq)
    T_tiled = T_grid.unsqueeze(0).expand(batch_size, -1)  # (batch, seq)

    # Shift deltas right by 1 to get delta_{t-1}; pad first step with 0
    delta_prev = torch.zeros(batch_size, 1, device=device)
    delta_prev = torch.cat([delta_prev, bs_deltas[:, :-1]], dim=1)  # (batch, seq)

    feat = torch.stack([
        S_norm,
        log_moneyness,
        T_tiled,
        bs_deltas,
        bs_gammas,
        bs_vegas,
        bs_thetas,
        delta_prev,
    ], dim=-1)  # (batch, seq, 8)

    return feat
