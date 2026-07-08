/**
 * @file bind_greeks.cpp
 * @brief Python bindings for batch Greeks computation.
 *
 * Exposes:
 *   quantedge_cpp.greeks.compute_greeks_batch(S, K, T, r, sigma, types)
 *       -> list[BSResult]
 *
 * pybind11/stl.h handles automatic std::vector <-> Python list conversion.
 * BSResult and OptionType are registered by bind_pricing (called first in
 * module.cpp), so no re-registration is needed here.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quantedge/greeks_engine.hpp"

namespace py = pybind11;

void bind_greeks(py::module_& m) {

    m.def("compute_greeks_batch",
        &quantedge::compute_greeks_batch,
        py::arg("S"),
        py::arg("K"),
        py::arg("T"),
        py::arg("r"),
        py::arg("sigma"),
        py::arg("types"),
        "Compute Greeks for a batch of options.\n"
        "Returns list[BSResult]. All input lists must be the same length."
    );
}
