import pytest
import numpy as np
import torch
from python.hedging.delta_hedge import compute_bs_hedge_pnl, bs_delta
from python.training.pnl import compute_pnl


def test_single_point_delta():
    # ATM call delta should be around 0.5 + small positive adjustment for r
    d = bs_delta(100.0, 100.0, 0.5, 0.05, 0.2, "call")
    assert 0.5 <= d <= 0.65
    
    # ATM put delta should be negative
    dp = bs_delta(100.0, 100.0, 0.5, 0.05, 0.2, "put")
    assert -0.5 <= dp <= -0.35


def test_pnl_deterministic_path():
    # 1 path, 3 steps: S = [100, 102, 98]
    # K = 100, call option.
    # We will manually calculate terminal P&L.
    S = np.array([[100.0, 102.0, 98.0]])
    K = 100.0
    T_total = 0.1
    r = 0.05
    sigma = 0.2
    
    # Run the function
    pnl, deltas = compute_bs_hedge_pnl(S, K, T_total, r, sigma, "call", cost_rate=0.0)
    
    # Let's check manual calculation
    # step 0: S0=100. d0 = bs_delta(100, 100, 0.1, r, sigma)
    # step 1: S1=102. d1 = bs_delta(102, 100, 0.05, r, sigma)
    # step 2: S2=98. d2 = (S2 > K) -> 0.0 (expiry)
    # Cash flow from hedge = d0 * (S1 - S0) + d1 * (S2 - S1)
    # Option payoff at S2=98 is 0.0. So terminal P&L = -payoff + cash flow = 0.0 + cash flow
    d0 = bs_delta(100.0, K, T_total, r, sigma, "call")
    d1 = bs_delta(102.0, K, T_total / 2.0, r, sigma, "call")
    expected_pnl = d0 * (102.0 - 100.0) + d1 * (98.0 - 102.0)
    
    assert pytest.approx(pnl[0], abs=1e-5) == expected_pnl


def test_transaction_costs_impact():
    # 2 paths, 5 steps
    S = np.array([
        [100.0, 101.0, 102.0, 101.0, 100.0],
        [100.0, 99.0, 98.0, 99.0, 100.0]
    ])
    K = 100.0
    
    # P&L without costs
    pnl_no_costs, _ = compute_bs_hedge_pnl(S, K, 0.1, 0.05, 0.2, "call", cost_rate=0.0)
    # P&L with costs
    pnl_with_costs, _ = compute_bs_hedge_pnl(S, K, 0.1, 0.05, 0.2, "call", cost_rate=0.01)
    
    # P&L with transaction costs must be strictly lower than without
    assert np.all(pnl_with_costs < pnl_no_costs)


def test_variance_reduction():
    # Under Black-Scholes assumptions, delta hedging reduces variance of P&L significantly
    # compared to no-hedge (which has large variance because of option payoff)
    np.random.seed(42)
    n_paths = 100
    n_steps = 50
    S0 = 100.0
    K = 100.0
    T = 0.2
    r = 0.05
    sigma = 0.2
    
    dt = T / n_steps
    # Simulate paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(n_steps):
        z = np.random.normal(size=n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
    pnl_hedged, _ = compute_bs_hedge_pnl(paths, K, T, r, sigma, "call", cost_rate=0.0)
    
    # Payoffs
    payoffs = np.maximum(paths[:, -1] - K, 0.0)
    pnl_unhedged = -payoffs
    
    # Hedged variance should be much smaller than unhedged variance
    var_hedged = np.var(pnl_hedged)
    var_unhedged = np.var(pnl_unhedged)
    
    assert var_hedged < 0.1 * var_unhedged
