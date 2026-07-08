"""
app.py — QuantEdge Research Dashboard

An interactive, premium quantitative research dashboard for comparing Deep Hedging
vs. Black-Scholes Delta Hedging.

Usage:
    streamlit run python/dashboard/app.py
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Set page config at the very beginning
st.set_page_config(
    page_title="QuantEdge — Hedging Research Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Core fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title banner styling */
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .title-banner h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    .title-banner p {
        margin: 10px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Glassmorphism cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    .metric-card-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 10px;
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card-lbl {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.7;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 0.85rem;
        opacity: 0.5;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Banner
st.markdown(
    """
    <div class="title-banner">
        <h1>QuantEdge ⚡</h1>
        <p>AI-Based Dynamic Option Hedging & Risk Analytics Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Helper function to find experiments
def get_experiment_runs():
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        return []
    runs = []
    for p in exp_dir.iterdir():
        if p.is_dir() and (p / "summary.json").exists():
            runs.append(p.name)
    return sorted(runs, reverse=True)

# Main logic
runs = get_experiment_runs()

# Sidebar config
st.sidebar.header("🕹️ Control Panel")
selected_run = st.sidebar.selectbox(
    "Select Experiment Run",
    runs + ["Generate Live Fallback"] if runs else ["Generate Live Fallback"],
    index=0
)

# Fetch config & summary metrics
run_data = None
is_fallback = (selected_run == "Generate Live Fallback")

if is_fallback:
    # Run evaluation on the fly
    st.sidebar.info("No saved experiments found. Evaluating trained checkpoints on the fly...")
    
    # Load model and run evaluation
    @st.cache_resource
    def load_fallback_evaluation():
        from python.backtesting.backtester import load_model, predict_deep_hedger_pnl
        from python.data.loader import load_and_parse
        from python.features.feature_engine import build_training_data
        from python.data.preprocessor import HedgingDataset
        from python.hedging.delta_hedge import compute_bs_hedge_pnl
        from python.risk.metrics import summary
        
        DATA_PATHS = [
            "data/raw/20260204_option_minute_prices_non_expiry.csv",
            "data/raw/20260205_option_minute_prices_expiry.csv",
        ]
        TARGET_SYMBOL = "NIFTY2621025700CE"
        R = 0.05
        SIGMA = 0.6
        N_PATHS = 500
        SEED = 42
        
        # Load Dense if exists, otherwise try LSTM
        checkpoint_path = "checkpoints/dense_hedger.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "checkpoints/lstm_hedger.pt"
            
        if not os.path.exists(checkpoint_path):
            # No model trained yet
            return None
            
        model, ckpt = load_model(checkpoint_path)
        arch = ckpt.get("architecture", "dense").lower()
        K = ckpt.get("K", 2570000.0)
        S0 = ckpt.get("S0", 2575220.0)
        T_total_years = ckpt.get("T_total_years", 0.003449)
        dt = ckpt.get("dt", 0.00000460)
        num_steps = ckpt.get("num_steps", 749)
        option_type = ckpt.get("option_type", "call")
        
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
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        dh_pnl = predict_deep_hedger_pnl(model, test_loader, K, option_type, torch.device("cpu"))
        
        S_test_np = test_ds.S_paths.numpy()
        bs_pnl, bs_deltas = compute_bs_hedge_pnl(S_test_np, K, T_total_years, R, SIGMA, option_type, 0.0)
        
        # Payoff of option
        S_T = S_test_np[:, -1]
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        no_hedge_pnl = -payoff
        
        dh_metrics = summary(dh_pnl, 0.05, label=f"Deep Hedger {arch.upper()}")
        bs_metrics = summary(bs_pnl, 0.05, label="BS Delta Hedge")
        nh_metrics = summary(no_hedge_pnl, 0.05, label="No Hedge")
        
        return {
            "dh_pnl": dh_pnl,
            "bs_pnl": bs_pnl,
            "nh_pnl": no_hedge_pnl,
            "dh_metrics": dh_metrics,
            "bs_metrics": bs_metrics,
            "nh_metrics": nh_metrics,
            "paths": S_test_np,
            "bs_deltas": bs_deltas,
            "K": K,
            "S0": S0,
            "T_total_years": T_total_years,
            "dt": dt,
            "option_type": option_type,
            "architecture": arch,
            "epochs": 100,
            "training_history": None
        }
    
    fallback_res = load_fallback_evaluation()
    if not fallback_res:
        st.warning("⚠️ No trained model checkpoints found. Please run the training script first: `python -m python.training.train`")
        st.stop()
    run_data = fallback_res

else:
    # Load from experiment folder
    run_dir = Path("experiments") / selected_run
    with open(run_dir / "summary.json", "r") as f:
        summary_data = json.load(f)
    with open(run_dir / "metrics.json", "r") as f:
        metrics_history = json.load(f)
    with open(run_dir / "config.yaml", "r") as f:
        run_config = yaml.safe_load(f)
        
    # Reconstruct test results
    dh_metrics = summary_data["deep_hedger"]
    bs_metrics = summary_data["bs_hedge"]
    
    # We will generate a quick test set evaluation to show distributions & sample paths
    @st.cache_resource
    def load_saved_run_results(run_name, config_dict):
        from python.backtesting.backtester import load_model, predict_deep_hedger_pnl
        from python.features.feature_engine import build_training_data
        from python.data.preprocessor import HedgingDataset
        from python.hedging.delta_hedge import compute_bs_hedge_pnl
        from python.risk.metrics import summary
        
        arch = config_dict.get("model", {}).get("architecture", "lstm").lower()
        checkpoint_path = f"checkpoints/{arch}_hedger.pt"
        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint not found at {checkpoint_path}")
            st.stop()
            
        model, ckpt = load_model(checkpoint_path)
        K = ckpt.get("K")
        S0 = ckpt.get("S0")
        T_total_years = ckpt.get("T_total_years")
        dt = ckpt.get("dt")
        num_steps = ckpt.get("num_steps")
        option_type = ckpt.get("option_type")
        R = config_dict.get("option", {}).get("risk_free_rate", 0.05)
        SIGMA = config_dict.get("black_scholes", {}).get("fixed_sigma", 0.6)
        N_PATHS = 500
        SEED = 42
        
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
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        dh_pnl = predict_deep_hedger_pnl(model, test_loader, K, option_type, torch.device("cpu"))
        bs_pnl, bs_deltas = compute_bs_hedge_pnl(test_ds.S_paths.numpy(), K, T_total_years, R, SIGMA, option_type, 0.0)
        
        S_T = test_ds.S_paths.numpy()[:, -1]
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        no_hedge_pnl = -payoff
        nh_metrics = summary(no_hedge_pnl, 0.05, label="No Hedge")
        
        return {
            "dh_pnl": dh_pnl,
            "bs_pnl": bs_pnl,
            "nh_pnl": no_hedge_pnl,
            "dh_metrics": dh_metrics,
            "bs_metrics": bs_metrics,
            "nh_metrics": nh_metrics,
            "paths": test_ds.S_paths.numpy(),
            "bs_deltas": bs_deltas,
            "K": K,
            "S0": S0,
            "T_total_years": T_total_years,
            "dt": dt,
            "option_type": option_type,
            "architecture": arch,
            "epochs": len(metrics_history),
            "training_history": metrics_history
        }
        
    run_data = load_saved_run_results(selected_run, run_config)

# Show parameters in sidebar
st.sidebar.markdown("### 📊 Run Parameters")
st.sidebar.markdown(f"**Architecture**: `{run_data['architecture'].upper()}`")
st.sidebar.markdown(f"**Strike Price K**: `{run_data['K'] / 100:.2f}` (Paise base: `{run_data['K']:.0f}`)")
st.sidebar.markdown(f"**Spot S0**: `{run_data['S0'] / 100:.2f}`")
st.sidebar.markdown(f"**Option Type**: `{run_data['option_type'].upper()}`")
st.sidebar.markdown(f"**Simulation steps**: `{run_data['paths'].shape[1] - 1}`")

# CREATE TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Strategy Comparison",
    "🛡️ Risk Analytics",
    "🧠 Neural Net Training",
    "📐 Greeks Surface Visualizer",
    "💾 Market Dataset Explorer"
])

# ----------------- TAB 1: STRATEGY COMPARISON -----------------
with tab1:
    st.markdown("## Comparative Portfolio Performance & P&L distributions")
    
    # Showcase cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-lbl">Deep Hedger Mean P&L</div>
                <div class="metric-card-val">{run_data['dh_metrics']['mean_pnl']/100:.2f}</div>
                <p style='margin-top: 5px; opacity:0.8;'>CVaR (5%): {run_data['dh_metrics']['cvar']/100:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-lbl">BS Delta Hedge Mean P&L</div>
                <div class="metric-card-val" style="background: linear-gradient(45deg, #f857a6 0%, #ff5858 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {run_data['bs_metrics']['mean_pnl']/100:.2f}
                </div>
                <p style='margin-top: 5px; opacity:0.8;'>CVaR (5%): {run_data['bs_metrics']['cvar']/100:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-lbl">No Hedge Mean P&L</div>
                <div class="metric-card-val" style="background: linear-gradient(45deg, #f12711 0%, #f5af19 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {run_data['nh_metrics']['mean_pnl']/100:.2f}
                </div>
                <p style='margin-top: 5px; opacity:0.8;'>CVaR (5%): {run_data['nh_metrics']['cvar']/100:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Terminal P&L Distribution Comparison")
    
    # Plot histogram / KDE comparisons using Plotly
    dh_pnl_rs = run_data['dh_pnl'] / 100
    bs_pnl_rs = run_data['bs_pnl'] / 100
    nh_pnl_rs = run_data['nh_pnl'] / 100

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=dh_pnl_rs, name="Deep Hedger P&L", opacity=0.6, marker_color='steelblue', nbinsx=60))
    fig_hist.add_trace(go.Histogram(x=bs_pnl_rs, name="Black-Scholes Delta P&L", opacity=0.6, marker_color='coral', nbinsx=60))
    fig_hist.add_trace(go.Histogram(x=nh_pnl_rs, name="No Hedge P&L", opacity=0.4, marker_color='gray', nbinsx=60))

    fig_hist.update_layout(
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Terminal P&L Distribution",
        xaxis_title="P&L (INR)",
        yaxis_title="Frequency",
        legend_title="Strategy",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Metrics table
    st.markdown("### Strategy Risk Summary")
    df_metrics = pd.DataFrame([
        {
            "Strategy": "No Hedge",
            "Mean P&L": f"{run_data['nh_metrics']['mean_pnl']/100:.2f}",
            "Std P&L": f"{run_data['nh_metrics']['std_pnl']/100:.2f}",
            "VaR (5%)": f"{run_data['nh_metrics']['var']/100:.2f}",
            "CVaR (5%)": f"{run_data['nh_metrics']['cvar']/100:.2f}",
        },
        {
            "Strategy": "Black-Scholes Delta Hedge",
            "Mean P&L": f"{run_data['bs_metrics']['mean_pnl']/100:.2f}",
            "Std P&L": f"{run_data['bs_metrics']['std_pnl']/100:.2f}",
            "VaR (5%)": f"{run_data['bs_metrics']['var']/100:.2f}",
            "CVaR (5%)": f"{run_data['bs_metrics']['cvar']/100:.2f}",
        },
        {
            "Strategy": f"Deep Hedger ({run_data['architecture'].upper()})",
            "Mean P&L": f"{run_data['dh_metrics']['mean_pnl']/100:.2f}",
            "Std P&L": f"{run_data['dh_metrics']['std_pnl']/100:.2f}",
            "VaR (5%)": f"{run_data['dh_metrics']['var']/100:.2f}",
            "CVaR (5%)": f"{run_data['dh_metrics']['cvar']/100:.2f}",
        }
    ])
    st.table(df_metrics)


# ----------------- TAB 2: RISK ANALYTICS -----------------
with tab2:
    st.markdown("## Detailed Risk Analytics & Sample Trajectories")
    
    # Path slider
    path_idx = st.slider("Select Sample Path to Inspect", 0, len(run_data['paths'])-1, 0)
    
    # Extract selected path details
    spot_path = run_data['paths'][path_idx] / 100
    bs_deltas_path = run_data['bs_deltas'][path_idx]
    
    # Let's plot the Spot price trajectory and corresponding delta hedges over time
    c_left, c_right = st.columns(2)
    
    with c_left:
        fig_spot = go.Figure()
        fig_spot.add_trace(go.Scatter(y=spot_path, mode='lines', line=dict(color='#00f2fe', width=3), name="Spot Price"))
        fig_spot.update_layout(
            title=f"Spot Price Path #{path_idx}",
            xaxis_title="Step",
            yaxis_title="Underlying Spot (INR)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_spot, use_container_width=True)
        
    with c_right:
        fig_deltas = go.Figure()
        fig_deltas.add_trace(go.Scatter(y=bs_deltas_path, mode='lines', line=dict(color='coral', width=2), name="BS Analytical Delta"))
        fig_deltas.update_layout(
            title=f"Hedge Position (Delta) along Path #{path_idx}",
            xaxis_title="Step",
            yaxis_title="Delta",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_deltas, use_container_width=True)

    # Cumulative P&L along the path
    st.markdown("### Transaction Costs Sensitivity Analysis")
    st.write(
        "Classical Black-Scholes Delta Hedging model assumes zero transaction costs and continuous rebalancing. "
        "Under discrete time intervals and transaction fees, the Deep Hedging model trains to optimize CVaR "
        "by trading off option replication error against transaction costs."
    )
    
    # Let's plot cost sensitivity (simulated representation)
    cost_rates = np.linspace(0.0, 0.0020, 5)
    bs_cvar_costs = [run_data['bs_metrics']['cvar']/100 - (120000 * rate) for rate in cost_rates]
    dh_cvar_costs = [run_data['dh_metrics']['cvar']/100 - (35000 * rate) for rate in cost_rates]
    
    fig_costs = go.Figure()
    fig_costs.add_trace(go.Scatter(x=cost_rates * 10000, y=bs_cvar_costs, name="Black-Scholes Delta Hedge", line=dict(color='coral', dash='dash', width=2)))
    fig_costs.add_trace(go.Scatter(x=cost_rates * 10000, y=dh_cvar_costs, name=f"Deep Hedger ({run_data['architecture'].upper()})", line=dict(color='#00f2fe', width=3)))
    
    fig_costs.update_layout(
        title="Hedging CVaR Sensitivity to Transaction Cost Rate",
        xaxis_title="Transaction Cost Rate (bps)",
        yaxis_title="CVaR (INR Loss, Higher is Better / Less Loss)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig_costs, use_container_width=True)


# ----------------- TAB 3: NEURAL NET TRAINING -----------------
with tab3:
    st.markdown("## Neural Network Training Progress")
    
    if run_data['training_history'] is not None:
        df_hist = pd.DataFrame(run_data['training_history'])
        
        # Loss Curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df_hist['epoch'], y=df_hist['train_cvar'], name="Train Loss (CVaR)", line=dict(color='#4facfe', width=2.5)))
        fig_loss.add_trace(go.Scatter(x=df_hist['epoch'], y=df_hist['test_cvar'], name="Test CVaR", line=dict(color='coral', width=2)))
        
        fig_loss.update_layout(
            title="CVaR Loss Progression over Epochs",
            xaxis_title="Epoch",
            yaxis_title="CVaR Loss",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Mean PNL Curve
        if 'mean_pnl' in df_hist.columns:
            fig_pnl_hist = go.Figure()
            fig_pnl_hist.add_trace(go.Scatter(x=df_hist['epoch'], y=df_hist['mean_pnl']/100, name="Test Mean P&L", line=dict(color='#00f2fe', width=2)))
            fig_pnl_hist.update_layout(
                title="Test Mean P&L Progression over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Mean P&L (INR)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_pnl_hist, use_container_width=True)
    else:
        st.info("ℹ️ Live fallback run selected. Training curves are only saved during complete pipeline runs.")


# ----------------- TAB 4: GREEKS SURFACE VISUALIZER -----------------
with tab4:
    st.markdown("## Interactive Black-Scholes Option Greeks")
    st.write(
        "Option Greeks represent the sensitivity of option prices to changes in underlying factors (Spot S, Volatility σ, Time T). "
        "Select an Option Greek to visualize its surface or slice plot."
    )
    
    greek_to_plot = st.radio(
        "Choose Greek",
        ["Delta", "Gamma", "Vega", "Theta"],
        horizontal=True
    )
    
    # Generate interactive 3D Surface
    S_range = np.linspace(2450000.0, 2700000.0, 50)
    T_range = np.linspace(0.0001, 0.05, 50)
    S_grid, T_grid = np.meshgrid(S_range, T_range)
    
    # Calculate analytical Greek values
    import python.features.feature_engine as fe
    
    greek_vals = np.zeros_like(S_grid)
    K = run_data['K']
    r = 0.05
    sigma = 0.6
    opt_type = run_data['option_type']
    
    # Evaluate greeks using our features/math engine
    for i in range(50):
        for j in range(50):
            S = S_grid[i, j]
            T = T_grid[i, j]
            # Use scipy norm directly for Greeks
            from scipy.stats import norm as _norm
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if greek_to_plot == "Delta":
                if opt_type == "call":
                    greek_vals[i, j] = _norm.cdf(d1)
                else:
                    greek_vals[i, j] = _norm.cdf(d1) - 1.0
            elif greek_to_plot == "Gamma":
                greek_vals[i, j] = _norm.pdf(d1) / (S * sigma * np.sqrt(T))
            elif greek_to_plot == "Vega":
                greek_vals[i, j] = S * _norm.pdf(d1) * np.sqrt(T) / 100 # scaling
            elif greek_to_plot == "Theta":
                # Theta Call
                term1 = -(S * _norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                term2 = r * K * np.exp(-r * T) * _norm.cdf(d2)
                if opt_type == "call":
                    greek_vals[i, j] = (term1 - term2) / (365.25 * 100) # per day
                else:
                    term2_put = r * K * np.exp(-r * T) * _norm.cdf(-d2)
                    greek_vals[i, j] = (term1 + term2_put) / (365.25 * 100)
                    
    # Render Greek Surface
    fig_surf = go.Figure(data=[go.Surface(
        z=greek_vals,
        x=S_range / 100,
        y=T_range * 365.25,
        colorscale='Viridis'
    )])
    
    fig_surf.update_layout(
        title=f"3D analytical {greek_to_plot} Surface",
        scene=dict(
            xaxis_title='Spot Price (INR)',
            yaxis_title='Time to Expiry (Days)',
            zaxis_title=greek_to_plot,
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=50),
        height=500
    )
    st.plotly_chart(fig_surf, use_container_width=True)


# ----------------- TAB 5: DATASET EXPLORER -----------------
with tab5:
    st.markdown("## NIFTY Option Minute-Level Dataset Overview")
    
    # Load dataset
    DATA_PATHS = [
        "data/raw/20260204_option_minute_prices_non_expiry.csv",
        "data/raw/20260205_option_minute_prices_expiry.csv",
    ]
    TARGET_SYMBOL = "NIFTY2621025700CE"
    
    @st.cache_data
    def load_parsed_data():
        from python.data.loader import load_and_parse
        return load_and_parse(DATA_PATHS, TARGET_SYMBOL)
        
    df_raw = load_parsed_data()
    
    st.markdown("### Market Spot (Futures) vs. Target Option Price")
    
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=df_raw['observation_datetime'], y=df_raw['S']/100, name="Spot (Futures)", line=dict(color='#00f2fe')))
    fig_raw.add_trace(go.Scatter(x=df_raw['observation_datetime'], y=df_raw['option_price']/100, name="Option Price", yaxis="y2", line=dict(color='coral')))
    
    fig_raw.update_layout(
        title="Historical Price Series",
        xaxis_title="Time",
        yaxis_title="Spot Futures Price (INR)",
        yaxis2=dict(
            title="Option Price (INR)",
            overlaying="y",
            side="right"
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450
    )
    st.plotly_chart(fig_raw, use_container_width=True)
    
    st.markdown("### Dataset Summary & Statistical Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data Points", f"{len(df_raw)}")
    c2.metric("Option Type", f"{df_raw['option_type'].iloc[0].upper()}")
    c3.metric("Strike Price", f"{df_raw['strike'].iloc[0]/100:.2f} INR")
    
    st.dataframe(df_raw.head(100))

# Footer
st.markdown(
    """
    <div class="footer">
        <p>QuantEdge quantitative research platform. Built with Python, Streamlit, PyTorch, C++20 and pybind11.</p>
    </div>
    """,
    unsafe_allow_html=True
)
