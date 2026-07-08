import pytest
import quantedge_cpp

OptionType = quantedge_cpp.pricing.OptionType
IVSolverConfig = quantedge_cpp.iv_solver.IVSolverConfig


def test_iv_round_trip():
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    true_sigma = 0.35
    
    # 1. Price option
    res = quantedge_cpp.pricing.black_scholes(S, K, T, r, true_sigma, OptionType.Call)
    market_price = res.price
    
    # 2. Solve IV using Bisection
    config = IVSolverConfig()
    config.vol_lower = 0.01
    config.vol_upper = 3.0
    config.tolerance = 1e-6
    config.max_iter = 100
    
    solved_bis = quantedge_cpp.iv_solver.iv_bisection(market_price, S, K, T, r, OptionType.Call, config)
    assert pytest.approx(solved_bis, abs=1e-5) == true_sigma

    # 3. Solve IV using Newton-Raphson
    solved_nr = quantedge_cpp.iv_solver.iv_newton_raphson(market_price, S, K, T, r, OptionType.Call, config)
    assert pytest.approx(solved_nr, abs=1e-5) == true_sigma


def test_solvers_agreement():
    S = 110.0
    K = 100.0
    T = 0.25
    r = 0.02
    market_price = 12.50 # Call
    
    config = IVSolverConfig()
    config.vol_lower = 0.001
    config.vol_upper = 4.0
    config.tolerance = 1e-6
    config.max_iter = 200
    
    solved_bis = quantedge_cpp.iv_solver.iv_bisection(market_price, S, K, T, r, OptionType.Call, config)
    solved_nr = quantedge_cpp.iv_solver.iv_newton_raphson(market_price, S, K, T, r, OptionType.Call, config)
    
    assert pytest.approx(solved_bis, abs=1e-5) == solved_nr


def test_iv_put():
    S = 95.0
    K = 100.0
    T = 0.5
    r = 0.05
    true_sigma = 0.22
    
    res = quantedge_cpp.pricing.black_scholes(S, K, T, r, true_sigma, OptionType.Put)
    market_price = res.price
    
    config = IVSolverConfig()
    config.vol_lower = 0.01
    config.vol_upper = 2.0
    config.tolerance = 1e-6
    config.max_iter = 100
    
    solved_bis = quantedge_cpp.iv_solver.iv_bisection(market_price, S, K, T, r, OptionType.Put, config)
    solved_nr = quantedge_cpp.iv_solver.iv_newton_raphson(market_price, S, K, T, r, OptionType.Put, config)
    
    assert pytest.approx(solved_bis, abs=1e-5) == true_sigma
    assert pytest.approx(solved_nr, abs=1e-5) == true_sigma

