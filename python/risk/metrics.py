"""
metrics.py — Risk Metrics: VaR and CVaR

Computes standard risk metrics on a distribution of P&L values.
Used to compare the deep hedger vs. the BS delta hedger.

Metrics:
    VaR_alpha  = alpha-th percentile of P&L (threshold of worst losses)
    CVaR_alpha = expected P&L given it falls below VaR_alpha
    mean_pnl   = average P&L across all paths
    std_pnl    = standard deviation of P&L
"""

import numpy as np


def var(pnl_values: np.ndarray, alpha: float = 0.05) -> float:
    """
    Value at Risk at level alpha.
    Returns the alpha-th percentile of the P&L distribution.
    A negative VaR means expected losses in the tail.
    """
    return float(np.percentile(pnl_values, alpha * 100))


def cvar(pnl_values: np.ndarray, alpha: float = 0.05) -> float:
    """
    Conditional Value at Risk (Expected Shortfall) at level alpha.
    Returns the mean P&L of the worst alpha-fraction of paths.
    A more negative CVaR = worse tail risk.
    """
    threshold = var(pnl_values, alpha)
    tail = pnl_values[pnl_values <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(tail.mean())


def summary(pnl_values: np.ndarray, alpha: float = 0.05,
            label: str = "") -> dict:
    """
    Full risk summary for a P&L distribution.

    Returns dict with: label, mean, std, var, cvar
    """
    return {
        "label":    label,
        "mean_pnl": float(np.mean(pnl_values)),
        "std_pnl":  float(np.std(pnl_values)),
        "var":      var(pnl_values, alpha),
        "cvar":     cvar(pnl_values, alpha),
        "alpha":    alpha,
    }


def print_summary(metrics: dict) -> None:
    """Pretty-print a metrics summary dict."""
    label = metrics.get("label", "")
    alpha = metrics.get("alpha", 0.05)
    print(f"\n{'─' * 45}")
    if label:
        print(f"  {label}")
    print(f"  Mean P&L : {metrics['mean_pnl']:>12.2f}")
    print(f"  Std  P&L : {metrics['std_pnl']:>12.2f}")
    print(f"  VaR  ({alpha:.0%}): {metrics['var']:>12.2f}")
    print(f"  CVaR ({alpha:.0%}): {metrics['cvar']:>12.2f}")
    print(f"{'─' * 45}")
