/**
 * @file bind_pricing.cpp
 * @brief Python bindings for Black-Scholes pricing.
 *
 * Exposes:
 *   quantedge_cpp.pricing.OptionType  — enum  { Call, Put }
 *   quantedge_cpp.pricing.BSResult    — struct { price, delta, gamma, theta, vega, rho }
 *   quantedge_cpp.pricing.black_scholes(S, K, T, r, sigma, type) -> BSResult
 */

#include <pybind11/pybind11.h>
#include "quantedge/black_scholes.hpp"

namespace py = pybind11;

void bind_pricing(py::module_& m) {
    // --- OptionType enum ---
     py::enum_<quantedge::OptionType>(m, "OptionType")
        .value("Call", quantedge::OptionType::Call)
        .value("Put",  quantedge::OptionType::Put)
        .export_values();
    // --- BSResult struct ---
    py::class_<quantedge::BSResult>(m, "BSResult")
        .def_readonly("price", &quantedge::BSResult::price)
        .def_readonly("delta", &quantedge::BSResult::delta)
        .def_readonly("gamma", &quantedge::BSResult::gamma)
        .def_readonly("theta", &quantedge::BSResult::theta)
        .def_readonly("vega",  &quantedge::BSResult::vega)
        .def_readonly("rho",   &quantedge::BSResult::rho)
        .def("__repr__", [](const quantedge::BSResult& r) {
            return "BSResult(price=" + std::to_string(r.price) +
                   ", delta=" + std::to_string(r.delta) +
                   ", gamma=" + std::to_string(r.gamma) +
                   ", vega="  + std::to_string(r.vega)  + ")";
        });
    // --- black_scholes() function ---
    m.def("black_scholes",
        &quantedge::black_scholes,
        py::arg("S"),
        py::arg("K"),
        py::arg("T"),
        py::arg("r"),
        py::arg("sigma"),
        py::arg("type"),
        "Compute Black-Scholes price and all Greeks for a European option."
    );
}
