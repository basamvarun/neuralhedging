/**
 * @file greeks_engine.cpp
 * @brief Batch Greeks computation with OpenMP parallelism.
 *
 * Wraps black_scholes() in a parallelised loop over a portfolio of options.
 * Each iteration is independent, making this embarrassingly parallel.
 * Compile with -fopenmp to enable multi-threading; falls back to serial otherwise.
 */

#include "quantedge/greeks_engine.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace quantedge {

std::vector<BSResult> compute_greeks_batch(
    const std::vector<double>&     S,
    const std::vector<double>&     K,
    const std::vector<double>&     T,
    double                         r,
    const std::vector<double>&     sigma,
    const std::vector<OptionType>& types
) {
    const size_t n = K.size();
    std::vector<BSResult> results(n);

    // Each option is priced independently — no data dependencies between iterations.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        results[i] = black_scholes(S[i], K[i], T[i], r, sigma[i], types[i]);
    }

    return results;
}

} // namespace quantedge
