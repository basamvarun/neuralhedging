/**
 * @file gbm.hpp
 * @brief Geometric Brownian Motion path simulator.
 *
 * Generates synthetic underlying price paths for deep hedging training.
 * Future extensions: Jump Diffusion, Heston.
 */

#pragma once

#include <vector>
#include <cstdint>

namespace quantedge {

struct GBMConfig {
    double S0;           // Initial price
    double mu;           // Drift (risk-neutral = risk-free rate)
    double sigma;        // Volatility
    double dt;           // Time increment per step (in years)
    int    num_steps;    // Number of time steps per path
    int    num_paths;    // Number of paths to generate
    uint64_t seed;       // Random seed for reproducibility
};

std::vector<std::vector<double>> simulate_gbm(const GBMConfig& config);

} // namespace quantedge
