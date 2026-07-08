/**
 * @file greeks_engine.hpp
 * @brief Batch Greeks computation engine with OpenMP parallelism.
 */

#pragma once

#include "quantedge/black_scholes.hpp"
#include <vector>

namespace quantedge {

std::vector<BSResult> compute_greeks_batch(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& T,
    double r,
    const std::vector<double>& sigma,
    const std::vector<OptionType>& types
);

} // namespace quantedge
