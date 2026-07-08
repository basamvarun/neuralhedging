/**
 * @file newton_raphson.cpp
 * @brief Newton-Raphson method for implied volatility.
 *
 * Uses the analytic Vega (dPrice/dSigma) as the derivative for fast convergence.
 * Falls back to a bisection-style clamp when Vega is near-zero (deep OTM/ITM)
 * to prevent the Newton step from diverging outside [vol_lower, vol_upper].
 */

#include "quantedge/iv_solver.hpp"
#include <cmath>

namespace quantedge {

double iv_newton_raphson(
    double observed_price,
    double S,
    double K,
    double T,
    double r,
    OptionType type,
    const IVSolverConfig& config
) {
    if (T <= 0.0) return 0.0;

    // Warm-start from the midpoint of the volatility search range
    double sigma = (config.vol_lower + config.vol_upper) / 2.0;

    for (int i = 0; i < config.max_iter; ++i) {
        BSResult bs = black_scholes(S, K, T, r, sigma, type);

        double price_error = bs.price - observed_price;

        if (std::abs(price_error) < config.tolerance) {
            return sigma;
        }

        // Guard against near-zero Vega (deep OTM/ITM) which would cause a huge step
        if (bs.vega < 1e-10) {
            // Gradient is unusable — nudge sigma toward observed price direction
            sigma += (price_error < 0.0) ? 0.01 : -0.01;
        } else {
            sigma -= price_error / bs.vega;
        }

        // Clamp sigma to the valid search range
        if (sigma < config.vol_lower) sigma = config.vol_lower;
        if (sigma > config.vol_upper) sigma = config.vol_upper;
    }

    return sigma;
}

} // namespace quantedge
