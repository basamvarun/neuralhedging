import pytest
import numpy as np
from scipy.stats import norm
from python.risk.metrics import var, cvar, summary


def test_deterministic_var_cvar():
    # 10 elements from -100 to -10: -100, -90, -80, -70, -60, -50, -40, -30, -20, -10
    pnl = np.array([-100.0, -90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0])
    
    # alpha = 0.2 means the worst 20%
    # np.percentile with alpha*100 = 20: index 1.8.
    val_var = var(pnl, alpha=0.2)
    # The tail values <= var must contain the worst 2 elements: -100, -90 (or more, depending on exact interpolation)
    val_cvar = cvar(pnl, alpha=0.2)
    
    # CVaR must be worse (more negative) than or equal to VaR
    assert val_cvar <= val_var


def test_normal_distribution_var_cvar():
    # Verify risk metrics on a standard normal distribution
    np.random.seed(123)
    pnl = np.random.normal(0.0, 1.0, size=50000)
    
    alpha = 0.05
    # Theoretical VaR at 5% is norm.ppf(0.05) ≈ -1.64485
    theoretical_var = norm.ppf(alpha)
    sample_var = var(pnl, alpha=alpha)
    assert pytest.approx(sample_var, abs=0.03) == theoretical_var
    
    # Theoretical CVaR (Expected Shortfall) for N(0, 1) is -pdf(ppf(alpha)) / alpha ≈ -2.0627
    theoretical_cvar = -norm.pdf(theoretical_var) / alpha
    sample_cvar = cvar(pnl, alpha=alpha)
    assert pytest.approx(sample_cvar, abs=0.03) == theoretical_cvar


def test_summary_dictionary():
    pnl = np.array([-10.0, 0.0, 10.0])
    res = summary(pnl, alpha=0.33, label="Test Strategy")
    
    assert res["label"] == "Test Strategy"
    assert res["mean_pnl"] == 0.0
    assert pytest.approx(res["std_pnl"], abs=1e-4) == np.std(pnl)
    assert "var" in res
    assert "cvar" in res
