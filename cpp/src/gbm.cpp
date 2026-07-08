/**
 * @file gbm.cpp
 * @brief Geometric Brownian Motion path simulation.
 *
 * Implements the exact GBM log-normal step:
 *   S(t+dt) = S(t) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
 * where Z ~ N(0,1).
 *
 * This is the "exact" discretisation — no Euler approximation error.
 * Output layout: results[path_index][step_index], step 0 = S0.
 */

#include "quantedge/gbm.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace quantedge {

std::vector<std::vector<double>> simulate_gbm(const GBMConfig& config) {
    if (config.num_paths <= 0 || config.num_steps <= 0) {
        throw std::invalid_argument("num_paths and num_steps must be positive.");
    }
    if (config.S0 <= 0.0) {
        throw std::invalid_argument("Initial price S0 must be positive.");
    }
    if (config.sigma < 0.0) {
        throw std::invalid_argument("Volatility sigma must be non-negative.");
    }

    // Allocate result: [num_paths][num_steps + 1]  (includes S0 at step 0)
    const int total_steps = config.num_steps + 1;
    std::vector<std::vector<double>> paths(
        config.num_paths, std::vector<double>(total_steps, 0.0)
    );

    // Pre-compute constants shared across all paths and steps
    const double drift   = (config.mu - 0.5 * config.sigma * config.sigma) * config.dt;
    const double vol_dt  = config.sigma * std::sqrt(config.dt);

    // Seeded Mersenne Twister for reproducible draws
    std::mt19937_64 rng(config.seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    for (int p = 0; p < config.num_paths; ++p) {
        paths[p][0] = config.S0;
        for (int t = 1; t < total_steps; ++t) {
            paths[p][t] = paths[p][t - 1] * std::exp(drift + vol_dt * Z(rng));
        }
    }

    return paths;
}

} // namespace quantedge
