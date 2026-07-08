"""
backtester.py — Side-by-Side Evaluation: Deep Hedger vs BS Delta Hedge

Loads a trained DenseHedger checkpoint, runs both strategies on the
test paths, computes risk metrics, and plots P&L distributions.

Usage:
    cd QuantEdge
    python -m python.backtesting.backtester
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from python.data.loader import load_and_parse
from python.features.feature_engine import build_training_data
from python.data.preprocessor import HedgingDataset
from python.training.trainer import run_hedger_on_batch
from python.training.pnl import compute_pnl
from python.hedging.delta_hedge import compute_bs_hedge_pnl
from python.risk.metrics import summary, print_summary

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATHS = [
    "data/raw/20260204_option_minute_prices_non_expiry.csv",
    "data/raw/20260205_option_minute_prices_expiry.csv",
]
TARGET_SYMBOL   = "NIFTY2621025700CE"
ALPHA           = 0.05
BATCH_SIZE      = 64
R               = 0.05   # risk-free rate (must match training)
SIGMA           = 0.6    # volatility (must match training)
N_PATHS         = 1000
SEED            = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str) -> tuple:
    """Load checkpoint and reconstruct the model with matching architecture."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("architecture", "dense").lower()
    hidden_size = ckpt.get("hidden_size", 32)
    num_layers = ckpt.get("num_layers", 2)
    dropout = ckpt.get("dropout", 0.1)

    if arch in ["lstm", "gru"]:
        from python.models.lstm_hedger import LSTMHedger
        model = LSTMHedger(input_dim=8, hidden_dim=hidden_size, num_layers=num_layers,
                           dropout=dropout, use_gru=(arch == "gru")).to(DEVICE)
    else:
        from python.models.dense_hedger import DenseHedger
        model = DenseHedger(input_dim=3, hidden_dim=hidden_size).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_deep_hedger_pnl(model, loader, K, option_type, device, cost_rate: float = 0.0) -> np.ndarray:
    all_pnl = []
    for S_batch, feat_batch, _ in loader:
        S_batch    = S_batch.to(device)
        feat_batch = feat_batch.to(device)
        deltas     = run_hedger_on_batch(model, S_batch, feat_batch)
        pnl        = compute_pnl(S_batch, deltas, K, option_type, cost_rate)
        all_pnl.append(pnl.cpu().numpy())
    return np.concatenate(all_pnl)


def plot_pnl_distributions(dh_pnl: np.ndarray, bs_pnl: np.ndarray,
                            alpha: float = 0.05) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle("P&L Distribution: Deep Hedger vs Black-Scholes Delta Hedge",
                     fontsize=14)

        sns.histplot(dh_pnl, bins=50, kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title("Deep Hedger")
        axes[0].set_xlabel("P&L")
        axes[0].axvline(np.percentile(dh_pnl, alpha * 100),
                        color="red", linestyle="--", label=f"VaR {alpha:.0%}")
        axes[0].legend()

        sns.histplot(bs_pnl, bins=50, kde=True, ax=axes[1], color="coral")
        axes[1].set_title("Black-Scholes Delta Hedge")
        axes[1].set_xlabel("P&L")
        axes[1].axvline(np.percentile(bs_pnl, alpha * 100),
                        color="red", linestyle="--", label=f"VaR {alpha:.0%}")
        axes[1].legend()

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/pnl_comparison.png", dpi=150)
        plt.show()
        print("Plot saved to results/pnl_comparison.png")
    except ImportError:
        print("matplotlib/seaborn not installed — skipping plot.")


def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="QuantEdge Model Backtester")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (defaults to best available)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        if os.path.exists("checkpoints/lstm_hedger.pt"):
            checkpoint_path = "checkpoints/lstm_hedger.pt"
        else:
            checkpoint_path = "checkpoints/dense_hedger.pt"

    # 1. Load real data
    print("Loading market data...")
    df = load_and_parse(DATA_PATHS, TARGET_SYMBOL)
    S0          = float(df["S"].iloc[0])
    K           = float(df["strike"].iloc[0])
    option_type = df["option_type"].iloc[0]

    # 2. Compute T_total and dt from the real data (mirrors train.py exactly)
    t_first = df["observation_datetime"].iloc[0]
    t_last  = df["observation_datetime"].iloc[-1]
    T_total_years = (t_last - t_first).total_seconds() / (365.25 * 24 * 3600)
    num_steps     = len(df) - 1
    dt            = T_total_years / num_steps

    print(f"  S0={S0:.0f}  K={K:.0f}  num_steps={num_steps}")
    print(f"  T_total_years={T_total_years:.6f}  dt={dt:.8f}")

    # 3. Load trained model to get architecture info first
    print(f"\nLoading model from {checkpoint_path}...")
    model, ckpt = load_model(checkpoint_path)
    arch = ckpt.get("architecture", "dense").lower()
    print(f"  Loaded model architecture: {arch}")

    # 4. Generate test paths (same seed / same split as training)
    print(f"\nGenerating {N_PATHS} GBM paths ({num_steps} steps)...")
    return_all_greeks = (arch in ["lstm", "gru"])
    data = build_training_data(S0, K, T_total_years, dt, R, SIGMA, option_type,
                               num_steps, N_PATHS, seed=SEED, return_all_greeks=return_all_greeks)

    dataset = HedgingDataset(data["paths"], data["bs_deltas"])
    if return_all_greeks:
        T_grid = torch.maximum(
            torch.tensor(0.0),
            T_total_years - torch.arange(num_steps + 1, dtype=torch.float32) * dt
        )
        S_tensor = torch.tensor(data["paths"], dtype=torch.float32)
        d_tensor = torch.tensor(data["bs_deltas"], dtype=torch.float32)
        g_tensor = torch.tensor(data["bs_gammas"], dtype=torch.float32)
        v_tensor = torch.tensor(data["bs_vegas"], dtype=torch.float32)
        th_tensor = torch.tensor(data["bs_thetas"], dtype=torch.float32)

        from python.models.lstm_hedger import build_lstm_features
        lstm_features = build_lstm_features(
            S_tensor, d_tensor, g_tensor, v_tensor, th_tensor, T_grid, K
        )
        dataset.features = lstm_features

    _, test_ds = dataset.split(train_ratio=0.8, seed=SEED)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    S_test_np = test_ds.S_paths.numpy()
    print(f"  Test paths: {len(test_ds)}")

    cost_rate = ckpt.get("cost_rate", 0.0)
    print(f"  Loaded model transaction cost rate: {cost_rate}")

    # 5. Deep hedger P&L
    print(f"\nRunning Deep Hedger ({arch.upper()}) on test set...")
    dh_pnl = predict_deep_hedger_pnl(model, test_loader, K, option_type, DEVICE, cost_rate=cost_rate)

    # 6. BS baseline P&L
    print("Running Black-Scholes Delta Hedge on test set...")
    bs_pnl, _ = compute_bs_hedge_pnl(S_test_np, K, T_total_years,
                                      r=R, sigma=SIGMA,
                                      option_type=option_type,
                                      cost_rate=cost_rate)

    # 7. Print comparison table
    print()
    dh_metrics = summary(dh_pnl, ALPHA, label=f"Deep Hedger {arch.upper()} (Test)")
    bs_metrics = summary(bs_pnl, ALPHA, label="BS Delta Hedge (Test)")
    print_summary(dh_metrics)
    print_summary(bs_metrics)

    # 8. Plot
    plot_pnl_distributions(dh_pnl, bs_pnl, alpha=ALPHA)


if __name__ == "__main__":
    main()
