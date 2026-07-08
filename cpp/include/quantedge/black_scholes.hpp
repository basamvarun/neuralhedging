/**
 * @file black_scholes.hpp
 * @brief Black-Scholes option pricing model.
 *
 * Implements closed-form European option pricing and Greeks
 * computation without any external finance libraries.
 */

#pragma once

namespace quantedge {

enum class OptionType { Call, Put };

struct BSResult {
    double price;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
};

/**
 * @brief Compute Black-Scholes price and all Greeks.
 *
 * @param S     Spot / Futures price
 * @param K     Strike price
 * @param T     Time to expiry (years)
 * @param r     Risk-free rate
 * @param sigma Volatility
 * @param type  Call or Put
 * @return BSResult containing price, delta, gamma, theta, vega, rho
 */
BSResult black_scholes(double S, double K, double T, double r, double sigma, OptionType type);

} // namespace quantedge
