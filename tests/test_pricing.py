import pytest
import numpy as np
import quantedge_cpp

OptionType = quantedge_cpp.pricing.OptionType


def test_known_analytical_call():
    # S=100, K=100, T=1.0, r=0.05, sigma=0.2, Call
    res = quantedge_cpp.pricing.black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, OptionType.Call)
    # Price should be ~10.45058
    assert pytest.approx(res.price, abs=1e-4) == 10.45058
    assert res.delta > 0.0
    assert res.gamma > 0.0
    assert res.vega > 0.0
    assert res.theta < 0.0  # Time decay is negative


def test_put_call_parity():
    S = 105.0
    K = 100.0
    T = 0.5
    r = 0.03
    sigma = 0.25
    
    call_res = quantedge_cpp.pricing.black_scholes(S, K, T, r, sigma, OptionType.Call)
    put_res = quantedge_cpp.pricing.black_scholes(S, K, T, r, sigma, OptionType.Put)
    
    # C - P = S - K * exp(-r * T)
    lhs = call_res.price - put_res.price
    rhs = S - K * np.exp(-r * T)
    assert pytest.approx(lhs, abs=1e-6) == rhs


def test_itm_otm_boundary():
    # Deep OTM Call (K = 500, S = 100) -> price should be close to 0
    otm = quantedge_cpp.pricing.black_scholes(100.0, 500.0, 0.5, 0.05, 0.2, OptionType.Call)
    assert otm.price < 1e-4
    assert pytest.approx(otm.delta, abs=1e-4) == 0.0

    # Deep ITM Call (K = 20, S = 100) -> price should be close to intrinsic: S - K * exp(-r*T)
    itm = quantedge_cpp.pricing.black_scholes(100.0, 20.0, 0.5, 0.05, 0.2, OptionType.Call)
    intrinsic = 100.0 - 20.0 * np.exp(-0.05 * 0.5)
    assert pytest.approx(itm.price, abs=1e-4) == intrinsic
    assert pytest.approx(itm.delta, abs=1e-4) == 1.0


def test_time_boundary():
    # T -> 0 Call price -> max(S - K, 0)
    res_itm = quantedge_cpp.pricing.black_scholes(110.0, 100.0, 1e-9, 0.05, 0.2, OptionType.Call)
    assert pytest.approx(res_itm.price, abs=1e-4) == 10.0
    
    res_otm = quantedge_cpp.pricing.black_scholes(90.0, 100.0, 1e-9, 0.05, 0.2, OptionType.Call)
    assert pytest.approx(res_otm.price, abs=1e-4) == 0.0
