/**
 * @file iv_solver.hpp
 * @brief Implied Volatility solvers.
 *
 * Given an observed market price, find σ such that BS(S, K, T, r, σ) = observed_price.
 * Methods: Bisection, Newton-Raphson, (future: Brent)
 */

#pragma once

#include "quantedge/black_scholes.hpp"

namespace quantedge {

struct IVSolverConfig {
    double tolerance    = 1e-5;
    int    max_iter     = 100;
    double vol_lower    = 0.001;
    double vol_upper    = 5.0;
};

double iv_bisection(double observed_price, double S, double K, double T,
                    double r, OptionType type, const IVSolverConfig& config = {});

double iv_newton_raphson(double observed_price, double S, double K, double T,
                         double r, OptionType type, const IVSolverConfig& config = {});

} // namespace quantedge
