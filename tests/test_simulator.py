import pytest
import numpy as np
import quantedge_cpp

GBMConfig = quantedge_cpp.simulator.GBMConfig


def test_simulator_shape():
    # 5 paths, 10 steps, S0=100.0, mu=0.05, sigma=0.2, dt=0.01, seed=42
    config = GBMConfig()
    config.S0 = 100.0
    config.mu = 0.05
    config.sigma = 0.2
    config.dt = 0.01
    config.num_steps = 10
    config.num_paths = 5
    config.seed = 42
    
    paths = quantedge_cpp.simulator.simulate_gbm(config)
    
    assert len(paths) == 5
    for path in paths:
        assert len(path) == 11  # num_steps + 1
        assert path[0] == 100.0


def test_simulator_positivity():
    # Extreme volatility should still yield positive prices
    config = GBMConfig()
    config.S0 = 10.0
    config.mu = 0.1
    config.sigma = 2.5
    config.dt = 0.1
    config.num_steps = 50
    config.num_paths = 10
    config.seed = 123
    
    paths = quantedge_cpp.simulator.simulate_gbm(config)
    for path in paths:
        for val in path:
            assert val > 0.0


def test_simulator_reproducibility():
    config1 = GBMConfig()
    config1.S0 = 100.0
    config1.mu = 0.05
    config1.sigma = 0.2
    config1.dt = 0.01
    config1.num_steps = 10
    config1.num_paths = 5
    config1.seed = 42

    config2 = GBMConfig()
    config2.S0 = 100.0
    config2.mu = 0.05
    config2.sigma = 0.2
    config2.dt = 0.01
    config2.num_steps = 10
    config2.num_paths = 5
    config2.seed = 42

    config3 = GBMConfig()
    config3.S0 = 100.0
    config3.mu = 0.05
    config3.sigma = 0.2
    config3.dt = 0.01
    config3.num_steps = 10
    config3.num_paths = 5
    config3.seed = 43

    paths1 = quantedge_cpp.simulator.simulate_gbm(config1)
    paths2 = quantedge_cpp.simulator.simulate_gbm(config2)
    paths3 = quantedge_cpp.simulator.simulate_gbm(config3)

    assert paths1 == paths2
    assert paths1 != paths3


def test_simulator_statistics():
    # Simulating 500 paths to check statistical convergence
    S0 = 100.0
    mu = 0.08
    sigma = 0.2
    dt = 0.01
    num_steps = 100
    T = dt * num_steps  # 1.0 year
    
    config = GBMConfig()
    config.S0 = S0
    config.mu = mu
    config.sigma = sigma
    config.dt = dt
    config.num_steps = num_steps
    config.num_paths = 500
    config.seed = 42
    
    paths = np.array(quantedge_cpp.simulator.simulate_gbm(config))
    
    S_T = paths[:, -1]
    
    # Expected mean: S0 * exp(mu * T)
    expected_mean = S0 * np.exp(mu * T)
    sample_mean = np.mean(S_T)
    
    # Check within 5% tolerance for 500 paths
    assert pytest.approx(sample_mean, rel=0.05) == expected_mean

