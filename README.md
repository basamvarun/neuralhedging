QuantEdge - AI-Based Dynamic Option Hedging and Risk Analytics Platform


Research Question

Can a neural network learn a dynamic hedging policy that minimizes portfolio risk better than the classical Black-Scholes delta hedge?


Overview

QuantEdge is an end-to-end quantitative research platform that compares Classical Black-Scholes Delta Hedging with an LSTM-based Deep Hedging strategy. The platform uses minute-level options market data to evaluate hedging performance under realistic conditions including transaction costs, tail risk (CVaR), and dynamic volatility.

This is a research platform, not a trading application.


Architecture

The platform is organized into three main layers.

Layer 1 - C++ Engine (compiled via pybind11)
  - Black-Scholes option pricing
  - Implied volatility solver (Bisection and Newton-Raphson)
  - Greeks computation (Delta, Gamma, Vega, Theta)
  - Geometric Brownian Motion simulation

Layer 2 - Python ML Pipeline (PyTorch)
  - Data loading and preprocessing
  - Feature engineering (moneyness, time-to-expiry, IV surface features)
  - LSTM Deep Hedger model
  - Dense Feedforward Hedger model
  - CVaR-based training loop
  - Backtesting and strategy comparison engine
  - Risk analytics (VaR, CVaR, Max Drawdown, Hedge Error)

Layer 3 - Dashboard (Streamlit)
  - Strategy comparison charts
  - Risk metrics display
  - Greeks visualization
  - Training curves
  - PnL distributions


Research Design

Strategies compared in this study:

No Hedge
Sell the option, do nothing. This is the baseline case.

Black-Scholes Delta Hedge
Classical dynamic hedging using Black-Scholes delta rebalancing at each timestep.

Deep Hedge - Dense Model (Experiment A)
Feedforward neural network with manual recurrence over the option path.

Deep Hedge - LSTM Model (Experiment B)
LSTM-based sequence model that learns the hedging policy from market state features.


Comparison Metrics

- Portfolio PnL (mean and standard deviation)
- Value at Risk (VaR at 95 percent)
- Conditional Value at Risk (CVaR at 95 percent)
- Maximum Drawdown
- Hedge Error (variance of terminal PnL)
- Turnover and Transaction Costs


Tech Stack

Pricing Engine: C++20, Eigen, OpenMP
Python Bindings: pybind11
Build System: CMake
ML Framework: PyTorch
Data Processing: NumPy, Pandas
Visualization: Plotly, Streamlit, Matplotlib, Seaborn
Configuration: YAML
Testing: pytest


Project Structure

QuantEdge/

  cpp/
    include/         Header files for all C++ modules
    src/             C++ source implementations
      black_scholes.cpp
      greeks_engine.cpp
      gbm.cpp
      bisection.cpp
      newton_raphson.cpp
    bindings/        pybind11 wrapper exposing C++ to Python
    tests/           C++ unit tests
    CMakeLists.txt

  python/
    data/            Data loading and preprocessing
    features/        Feature engineering (moneyness, TTM, IV features)
    models/          PyTorch model definitions (LSTM, Dense)
    training/        Training loop, loss functions, PnL computation
    hedging/         Classical Black-Scholes delta hedging
    backtesting/     Strategy comparison and evaluation engine
    risk/            Risk metrics (VaR, CVaR, drawdown)
    dashboard/       Streamlit interactive dashboard
    utils/           Config loader, seeding, experiment tracking

  configs/
    default.yaml             Main experiment configuration
    dense_experiment.yaml    Dense model config
    cost_experiment.yaml     Transaction cost experiment config
    transaction_costs.yaml   Cost parameter definitions

  data/
    raw/             Place your raw CSV option data files here
    processed/       Preprocessed and feature-engineered data

  experiments/       Experiment run outputs and logs
  checkpoints/       Saved model checkpoints
  results/           Final results and plots
  tests/             Python unit tests
  docs/              Project documentation

  requirements.txt
  setup.py
  README.md


Quick Start

Step 1 - Install Python dependencies

    pip install -r requirements.txt

Step 2 - Build the C++ engine

    cd cpp
    mkdir build
    cd build
    cmake ..
    make -j4
    cd ../..

Step 3 - Place your option data

    Copy your NIFTY option CSV file to data/raw/

Step 4 - Run an experiment

    python -m python.training.train --config configs/default.yaml

Step 5 - Launch the dashboard

    streamlit run python/dashboard/app.py


Datasets

Phase 1 - Included with this project

NIFTY minute-level option chain data (expiry day)
NIFTY minute-level option chain data (non-expiry day)

The raw data files are not committed to the repository due to size. Place them in data/raw/ before running.

Phase 2 - Extensible architecture

The platform is data-agnostic and supports any European-style option dataset:
- NIFTY
- BANKNIFTY
- Equity options
- Commodity options
- Currency options

Only the data loader needs to be adapted. The pricing engine, Greek calculations, and hedging logic remain unchanged.


Configuration

All experiments are controlled through YAML configuration files in the configs/ directory.

Key parameters in default.yaml:
- model type (lstm or dense)
- number of training epochs
- learning rate
- transaction cost rate
- CVaR confidence level
- sequence length for the LSTM
- feature set selection


Model Details

LSTM Hedger
The LSTM model takes a sequence of market state observations as input and outputs a hedge ratio (delta) at each timestep. It is trained using a CVaR loss function that penalizes large losses in the tail of the PnL distribution.

Dense Hedger
The dense model processes market features at each timestep independently with a manual hidden state passed between steps. It serves as a simpler baseline to compare against the LSTM.

Both models output a hedge ratio in the range 0 to 1 representing the fraction of the option notional to hold in the underlying asset.


Risk Metrics

VaR (Value at Risk)
The loss threshold not exceeded with 95 percent probability over the evaluation period.

CVaR (Conditional Value at Risk)
The expected loss in the worst 5 percent of scenarios. This is the primary training objective for the neural hedger.

Maximum Drawdown
The largest peak-to-trough decline in cumulative PnL during the evaluation period.

Hedge Error
The variance of the terminal hedging PnL. Lower variance means better risk reduction.

Turnover
The total number of rebalancing trades. High turnover increases transaction costs.


Future Extensions - Not in Version 1

The following are planned for future versions:
- Heston Stochastic Volatility Model integration
- Merton Jump Diffusion pricing
- Transformer-based hedging model
- Multi-option portfolio hedging
- Live market data feed integration
- Online learning and model updating


License

This project is for research use only.

Built by Varun | QuantEdge v1.0
