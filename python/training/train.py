"""
train.py — Main Training Entry Point with Advanced Optimisations

Loads Base YAML Config and Override YAML Config, runs training for DenseHedger or LSTM/GRU Hedger,
uses C++ path simulator, and applies ReduceLROnPlateau scheduler + Early Stopping.

Usage:
    cd QuantEdge
    python -m python.training.train --config configs/default.yaml --override configs/dense_experiment.yaml
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from python.utils.config import load_config_with_overrides
from python.data.loader import load_and_parse
from python.features.feature_engine import build_training_data
from python.data.preprocessor import HedgingDataset
from python.models.dense_hedger import DenseHedger
from python.models.lstm_hedger import LSTMHedger, build_lstm_features
from python.training.trainer import train_one_epoch, evaluate, run_hedger_on_batch
from python.training.pnl import compute_pnl
from python.utils.experiment import ExperimentLogger
from python.risk.metrics import summary as risk_summary


def main():
    # 1. CLI Arguments
    parser = argparse.ArgumentParser(description="QuantEdge Model Trainer")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to base YAML config file")
    parser.add_argument("--override", type=str, default="configs/dense_experiment.yaml",
                        help="Path to override YAML config file")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--use-cpp", action="store_true", default=True,
                        help="Use compiled C++ simulator for path generation")
    parser.add_argument("--architecture", type=str, default=None,
                        help="Model architecture: 'dense', 'lstm', or 'gru' (overrides config)")
    args = parser.parse_args()

    # 2. Load Configuration
    print(f"Loading config: base={args.config}, override={args.override}")
    config = load_config_with_overrides(args.config, args.override)
    
    # Resolve architecture choice
    arch = args.architecture or config.get("model", {}).get("architecture", "lstm")
    arch = arch.lower()
    print(f"Using model architecture: {arch}")

    # Seeding
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract configs
    data_dir = config.get("data", {}).get("raw_dir", "data/raw")
    expiry_file = config.get("data", {}).get("expiry_file", "20260205_option_minute_prices_expiry.csv")
    non_expiry_file = config.get("data", {}).get("non_expiry_file", "20260204_option_minute_prices_non_expiry.csv")
    
    DATA_PATHS = [
        os.path.join(data_dir, non_expiry_file),
        os.path.join(data_dir, expiry_file),
    ]
    TARGET_SYMBOL = "NIFTY2621025700CE"
    
    OPTION_TYPE = config.get("option", {}).get("option_type", "call")
    R = config.get("option", {}).get("risk_free_rate", 0.05)
    SIGMA = config.get("black_scholes", {}).get("fixed_sigma", 0.6)
    NUM_PATHS = config.get("simulation", {}).get("num_paths", 1000)
    
    EPOCHS = args.epochs
    BATCH_SIZE = config.get("training", {}).get("batch_size", 32)
    LR = config.get("training", {}).get("learning_rate", 1e-3)
    ALPHA = config.get("training", {}).get("cvar_alpha", 0.05)
    
    tc_config = config.get("transaction_costs", {})
    COST_RATE = float(tc_config.get("cost_rate", 0.0)) if tc_config.get("enabled", False) else 0.0
    print(f"Transaction costs enabled: {tc_config.get('enabled', False)} (rate: {COST_RATE})")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {DEVICE}")

    # 3. Load Real Market Data
    print("Loading real market data to calibrate S0, K, T...")
    df = load_and_parse(DATA_PATHS, TARGET_SYMBOL)

    S0 = float(df["S"].iloc[0])
    K = float(df["strike"].iloc[0])

    t_first = df["observation_datetime"].iloc[0]
    t_last = df["observation_datetime"].iloc[-1]
    time_diff_secs = (t_last - t_first).total_seconds()
    T_total_years = time_diff_secs / (365.25 * 24 * 3600)

    num_steps = len(df) - 1
    dt = T_total_years / num_steps

    print(f"  S0={S0:.0f}  K={K:.0f}")
    print(f"  num_steps={num_steps}  T_total_years={T_total_years:.6f}  dt={dt:.8f}")

    # 4. Generate GBM paths (calibrated to real data)
    sim_source = "C++ Simulator" if args.use_cpp else "NumPy Simulator"
    print(f"\nGenerating {NUM_PATHS} GBM paths ({num_steps} steps) using {sim_source}...")
    
    return_all_greeks = (arch in ["lstm", "gru"])
    data = build_training_data(S0, K, T_total_years, dt, R, SIGMA, OPTION_TYPE,
                               num_steps, NUM_PATHS, seed=seed, use_cpp=args.use_cpp,
                               return_all_greeks=return_all_greeks)

    # 5. Dataset & Dataloaders
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

        lstm_features = build_lstm_features(
            S_tensor, d_tensor, g_tensor, v_tensor, th_tensor, T_grid, K
        )
        dataset.features = lstm_features

    train_ds, test_ds = dataset.split(train_ratio=0.8, seed=seed)
    print(f"  Train paths: {len(train_ds)}   Test paths: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Model Definition
    hidden_size = config.get("model", {}).get("hidden_size", 32)
    num_layers = config.get("model", {}).get("num_layers", 2)
    dropout = config.get("model", {}).get("dropout", 0.1)

    if arch in ["lstm", "gru"]:
        model = LSTMHedger(input_dim=8, hidden_dim=hidden_size, num_layers=num_layers,
                           dropout=dropout, use_gru=(arch == "gru")).to(DEVICE)
    else:
        model = DenseHedger(input_dim=3, hidden_dim=hidden_size).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    
    # LR Scheduler (Reduce LR when test CVaR loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10, threshold=1e-4
    )

    # Initialize ExperimentLogger
    logger = ExperimentLogger(name=f"{arch}_hedger", base_dir="experiments")
    logger.log_config(config.to_dict())

    # 7. Training Loop with Early Stopping & Best Model Tracking
    print(f"\nTraining {arch.upper()} model for up to {EPOCHS} epochs...")
    
    best_test_cvar = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    early_stopping_cfg = config.get("training", {}).get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", True)
    patience = early_stopping_cfg.get("patience", 15)
    min_delta = early_stopping_cfg.get("min_delta", 1e-4)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimiser, K, OPTION_TYPE, ALPHA, COST_RATE, DEVICE
        )
        
        # Evaluate every epoch for precise early stopping and LR scheduler updates
        stats = evaluate(model, test_loader, K, OPTION_TYPE, ALPHA, COST_RATE, DEVICE)
        test_cvar = stats['cvar_loss']
        mean_pnl = stats['mean_pnl']
        
        # Scheduler Step
        scheduler.step(test_cvar)
        
        current_lr = optimiser.param_groups[0]['lr']
        logger.log_epoch(epoch, train_cvar=train_loss, test_cvar=test_cvar, mean_pnl=mean_pnl, lr=current_lr)

        # Check Improvement
        if test_cvar < best_test_cvar - min_delta:
            best_test_cvar = test_cvar
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            # Print update when a new best is found
            print(f"Epoch {epoch:>3}/{EPOCHS} | **New Best** | train_cvar={train_loss:.4f} | test_cvar={test_cvar:.4f} | mean_pnl={mean_pnl:.2f} | lr={current_lr:.6f}")
        else:
            epochs_no_improve += 1
            
        # Periodic normal printout (every 50 epochs)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:>3}/{EPOCHS} | train_cvar={train_loss:.4f} | test_cvar={test_cvar:.4f} | mean_pnl={mean_pnl:.2f} | lr={current_lr:.6f} | patience={epochs_no_improve}/{patience}")

        # Early Stopping check
        if early_stopping_enabled and epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}! No improvement for {patience} epochs.")
            break

    # 8. Load best model weights and Save Checkpoint
    if best_model_state is not None:
        print(f"\nLoading best model state with test_cvar={best_test_cvar:.4f}")
        model.load_state_dict(best_model_state)
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/{arch}_hedger.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "K": K, "S0": S0, "T_total_years": T_total_years,
        "dt": dt, "num_steps": num_steps,
        "sigma": SIGMA, "option_type": OPTION_TYPE,
        "architecture": arch,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "cost_rate": COST_RATE,
    }, checkpoint_path)
    print(f"Best model checkpoint saved to {checkpoint_path}")

    # Compute final metrics and log summary
    model.eval()
    with torch.no_grad():
        all_pnl = []
        for S_batch, feat_batch, _ in test_loader:
            S_batch = S_batch.to(DEVICE)
            feat_batch = feat_batch.to(DEVICE)
            deltas = run_hedger_on_batch(model, S_batch, feat_batch)
            pnl = compute_pnl(S_batch, deltas, K, OPTION_TYPE, COST_RATE)
            all_pnl.append(pnl.cpu().numpy())
        dh_pnl = np.concatenate(all_pnl)

    from python.hedging.delta_hedge import compute_bs_hedge_pnl
    S_test_np = test_ds.S_paths.numpy()
    bs_pnl, _ = compute_bs_hedge_pnl(S_test_np, K, T_total_years, R, SIGMA, OPTION_TYPE, COST_RATE)

    dh_metrics = risk_summary(dh_pnl, ALPHA, label=f"Deep Hedger {arch.upper()} (Test)")
    bs_metrics = risk_summary(bs_pnl, ALPHA, label="BS Delta Hedge (Test)")

    logger.log_summary(deep_hedger=dh_metrics, bs_hedge=bs_metrics)
    logger.save()


if __name__ == "__main__":
    main()
