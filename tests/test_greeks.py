import pytest
import quantedge_cpp
import numpy as np

OptionType = quantedge_cpp.pricing.OptionType


def test_greeks_bounds():
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.3
    
    call_res = quantedge_cpp.pricing.black_scholes(S, K, T, r, sigma, OptionType.Call)
    assert 0.0 <= call_res.delta <= 1.0
    assert call_res.gamma >= 0.0
    assert call_res.vega >= 0.0
    assert call_res.theta <= 0.0  # Theta is negative for long call
    
    put_res = quantedge_cpp.pricing.black_scholes(S, K, T, r, sigma, OptionType.Put)
    assert -1.0 <= put_res.delta <= 0.0
    assert put_res.gamma >= 0.0
    assert put_res.vega >= 0.0
    assert put_res.theta <= 0.0  # Theta is negative for long put


def test_batch_matches_individual():
    S = [100.0, 105.0, 95.0]
    K = [100.0, 100.0, 100.0]
    T = [0.5, 0.5, 0.5]
    r = 0.05
    sigma = [0.2, 0.25, 0.3]
    types = [OptionType.Call, OptionType.Put, OptionType.Call]
    
    batch_results = quantedge_cpp.greeks.compute_greeks_batch(S, K, T, r, sigma, types)
    
    for i in range(len(S)):
        indiv = quantedge_cpp.pricing.black_scholes(S[i], K[i], T[i], r, sigma[i], types[i])
        res = batch_results[i]
        
        assert pytest.approx(res.price, abs=1e-7) == indiv.price
        assert pytest.approx(res.delta, abs=1e-7) == indiv.delta
        assert pytest.approx(res.gamma, abs=1e-7) == indiv.gamma
        assert pytest.approx(res.vega, abs=1e-7) == indiv.vega
        assert pytest.approx(res.theta, abs=1e-7) == indiv.theta
        assert pytest.approx(res.rho, abs=1e-7) == indiv.rho


def test_numerical_greeks():
    # Test Delta and Gamma using finite-difference approximations
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    res = quantedge_cpp.pricing.black_scholes(S, K, T, r, sigma, OptionType.Call)
    
    dS = 1e-4
    res_up = quantedge_cpp.pricing.black_scholes(S + dS, K, T, r, sigma, OptionType.Call)
    res_dn = quantedge_cpp.pricing.black_scholes(S - dS, K, T, r, sigma, OptionType.Call)
    
    # Delta finite difference: (Price(S+dS) - Price(S-dS)) / (2 * dS)
    numerical_delta = (res_up.price - res_dn.price) / (2 * dS)
    assert pytest.approx(res.delta, abs=1e-5) == numerical_delta
    
    # Gamma finite difference: (Price(S+dS) - 2*Price(S) + Price(S-dS)) / (dS^2)
    numerical_gamma = (res_up.price - 2 * res.price + res_dn.price) / (dS ** 2)
    assert pytest.approx(res.gamma, abs=1e-5) == numerical_gamma
