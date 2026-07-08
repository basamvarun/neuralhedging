/**
 * @file bind_iv_solver.cpp
 * @brief Python bindings for IV solvers.
 *
 * Exposes:
 *   quantedge_cpp.iv_solver.IVSolverConfig  — config struct with defaults
 *   quantedge_cpp.iv_solver.iv_bisection(market_price, S, K, T, r, type, config) -> float
 *   quantedge_cpp.iv_solver.iv_newton_raphson(market_price, S, K, T, r, type, config) -> float
 */

#include <pybind11/pybind11.h>
#include "quantedge/iv_solver.hpp"

namespace py = pybind11;

void bind_iv_solver(py::module_& m) {

    // --- IVSolverConfig struct ---
    py::class_<quantedge::IVSolverConfig>(m, "IVSolverConfig")
        .def(py::init<>())
        .def_readwrite("vol_lower",  &quantedge::IVSolverConfig::vol_lower)
        .def_readwrite("vol_upper",  &quantedge::IVSolverConfig::vol_upper)
        .def_readwrite("tolerance",  &quantedge::IVSolverConfig::tolerance)
        .def_readwrite("max_iter",   &quantedge::IVSolverConfig::max_iter)
        .def("__repr__", [](const quantedge::IVSolverConfig& c) {
            return "IVSolverConfig(vol_lower=" + std::to_string(c.vol_lower) +
                   ", vol_upper=" + std::to_string(c.vol_upper) +
                   ", tolerance=" + std::to_string(c.tolerance) +
                   ", max_iter="  + std::to_string(c.max_iter) + ")";
        });

    // --- iv_bisection() ---
    m.def("iv_bisection",
        &quantedge::iv_bisection,
        py::arg("market_price"),
        py::arg("S"),
        py::arg("K"),
        py::arg("T"),
        py::arg("r"),
        py::arg("type"),
        py::arg("config") = quantedge::IVSolverConfig{},
        "Solve implied volatility using bisection. Returns sigma as float."
    );

    // --- iv_newton_raphson() ---
    m.def("iv_newton_raphson",
        &quantedge::iv_newton_raphson,
        py::arg("market_price"),
        py::arg("S"),
        py::arg("K"),
        py::arg("T"),
        py::arg("r"),
        py::arg("type"),
        py::arg("config") = quantedge::IVSolverConfig{},
        "Solve implied volatility using Newton-Raphson. Returns sigma as float."
    );
}
