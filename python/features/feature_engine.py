"""
feature_engine.py — GBM Path Generation and BS Delta Computation

Matches the reference notebook exactly:
  - S0 = first futures price from real data
  - num_steps = actual number of minute bars in the real data
  - T_total = time span from first to last observation in real data (NOT time to expiry)
  - dt = T_total / num_steps
  - mu = r = 0.05 (risk-neutral drift)
  - sigma = 0.6 (constant)
  - num_paths = 1000

BS delta at step t uses T_remaining = T_total_years - (t * dt)
(remaining time within the simulation window)
"""

import numpy as np
from scipy.stats import norm


def bs_delta(S, K, T, r, sigma, option_type):
    """Black-Scholes delta for one option at one point in time."""
    if T <= 0 or sigma <= 1e-8:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if option_type == "call" else float(norm.cdf(d1) - 1.0)


def generate_gbm_paths(S0, mu, sigma, dt, num_steps, num_paths, seed=42, use_cpp=True):
    """
    Generate GBM paths matching the notebook.
    Uses high-performance C++ simulator if available and use_cpp=True.

    Returns: (num_paths, num_steps+1)
    """
    if use_cpp:
        try:
            import quantedge_cpp.simulator as cpp_sim
            config = cpp_sim.GBMConfig()
            config.S0 = float(S0)
            config.mu = float(mu)
            config.sigma = float(sigma)
            config.dt = float(dt)
            config.num_steps = int(num_steps)
            config.num_paths = int(num_paths)
            config.seed = int(seed)
            return np.array(cpp_sim.simulate_gbm(config))
        except (ImportError, AttributeError) as e:
            # Silently fallback or log
            pass

    np.random.seed(seed)
    Z = np.random.normal(0.0, 1.0, size=(num_paths, num_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = S0
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(log_ret[:, t - 1])
    return paths


def compute_bs_deltas_for_paths(paths, K, T_total_years, dt, r, sigma, option_type):
    """
    Compute BS delta at every step for every path — fully vectorised.
    T_remaining = T_total_years - (t_step * dt) — matches notebook exactly.

    Returns: (num_paths, num_steps+1)
    """
    num_paths, num_steps_plus1 = paths.shape

    # T_remaining for each timestep: shape (1, n_steps+1) — broadcast over paths
    t_steps = np.arange(num_steps_plus1, dtype=np.float64)
    T_rem = np.maximum(0.0, T_total_years - t_steps * dt)  # (n_steps+1,)

    # Mask where T_rem > 0; use closed-form formula vectorised over all paths & steps
    valid = T_rem > 0  # (n_steps+1,)

    # Broadcast paths (n_paths, n_steps+1) against T_rem (n_steps+1,)
    S = paths                              # (n_paths, n_steps+1)
    T = T_rem[np.newaxis, :]              # (1, n_steps+1)

    deltas = np.zeros_like(paths)
    if np.any(valid):
        S_v = S[:, valid]
        T_v = T[:, valid]
        d1 = (np.log(S_v / K) + (r + 0.5 * sigma**2) * T_v) / (sigma * np.sqrt(T_v))
        from scipy.stats import norm as _norm
        if option_type == "call":
            deltas[:, valid] = _norm.cdf(d1)
        else:
            deltas[:, valid] = _norm.cdf(d1) - 1.0

    # At expiry (T_rem == 0): intrinsic delta
    exp_mask = ~valid
    if np.any(exp_mask):
        S_exp = S[:, exp_mask]
        if option_type == "call":
            deltas[:, exp_mask] = (S_exp > K).astype(np.float64)
        else:
            deltas[:, exp_mask] = -(S_exp < K).astype(np.float64)

    return deltas


def compute_all_greeks_for_paths(paths, K, T_total_years, dt, r, sigma, option_type):
    """
    Compute BS deltas, gammas, vegas, and thetas for all paths and timesteps.
    Uses C++ batch Greeks computation module.

    Returns:
        deltas, gammas, vegas, thetas: np.ndarrays of shape (num_paths, num_steps+1)
    """
    import quantedge_cpp
    num_paths, num_steps_plus1 = paths.shape

    t_steps = np.arange(num_steps_plus1, dtype=np.float64)
    T_rem = np.maximum(0.0, T_total_years - t_steps * dt)

    S_flat = paths.flatten()
    T_flat = np.tile(T_rem, num_paths)

    N = len(S_flat)
    K_flat = [float(K)] * N
    sigma_flat = [float(sigma)] * N

    cpp_type = quantedge_cpp.pricing.OptionType.Call if option_type == "call" else quantedge_cpp.pricing.OptionType.Put
    types_flat = [cpp_type] * N

    results = quantedge_cpp.greeks.compute_greeks_batch(
        S_flat.tolist(), K_flat, T_flat.tolist(), float(r), sigma_flat, types_flat
    )

    deltas = np.array([res.delta for res in results]).reshape(num_paths, num_steps_plus1)
    gammas = np.array([res.gamma for res in results]).reshape(num_paths, num_steps_plus1)
    vegas  = np.array([res.vega for res in results]).reshape(num_paths, num_steps_plus1)
    thetas = np.array([res.theta for res in results]).reshape(num_paths, num_steps_plus1)

    return deltas, gammas, vegas, thetas


def build_training_data(S0, K, T_total_years, dt, r, sigma, option_type,
                        num_steps, num_paths, seed=42, use_cpp=True, return_all_greeks=False):
    """
    Full pipeline matching notebook Steps 1-2:
        GBM paths -> BS deltas (using T_remaining within simulation window)

    Args:
        S0:             First futures price from real data
        K:              Strike price
        T_total_years:  Time span of real data (start to end, NOT to expiry)
        dt:             T_total_years / num_steps
        r:              Risk-free rate (0.05)
        sigma:          Constant volatility (0.6)
        option_type:    'call' or 'put'
        num_steps:      Actual number of minute bars in real data
        num_paths:      1000
        use_cpp:        Whether to use high-performance C++ simulator
        return_all_greeks: Whether to compute and return gammas, vegas, thetas

    Returns dict with keys: paths, bs_deltas, S0, K, T_total_years, dt, and optionally bs_gammas, bs_vegas, bs_thetas
    """
    paths = generate_gbm_paths(S0, mu=r, sigma=sigma, dt=dt,
                               num_steps=num_steps, num_paths=num_paths, seed=seed, use_cpp=use_cpp)
    if return_all_greeks:
        bs_deltas, bs_gammas, bs_vegas, bs_thetas = compute_all_greeks_for_paths(
            paths, K, T_total_years, dt, r, sigma, option_type
        )
        return dict(paths=paths, bs_deltas=bs_deltas, bs_gammas=bs_gammas,
                    bs_vegas=bs_vegas, bs_thetas=bs_thetas,
                    S0=S0, K=K, T_total_years=T_total_years, dt=dt)
    else:
        bs_deltas = compute_bs_deltas_for_paths(paths, K, T_total_years, dt,
                                                r, sigma, option_type)
        return dict(paths=paths, bs_deltas=bs_deltas,
                    S0=S0, K=K, T_total_years=T_total_years, dt=dt)

