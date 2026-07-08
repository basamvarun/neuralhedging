/**
 * @file bind_simulator.cpp
 * @brief Python bindings for market path simulation.
 *
 * Exposes:
 *   quantedge_cpp.simulator.GBMConfig   — simulation config struct
 *   quantedge_cpp.simulator.simulate_gbm(config) -> list[list[float]]
 *
 * Return layout: results[path_index][step_index], step 0 = S0.
 * pybind11/stl.h converts vector<vector<double>> to list[list[float]] automatically.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quantedge/gbm.hpp"

namespace py = pybind11;

void bind_simulator(py::module_& m) {

    // --- GBMConfig struct ---
    py::class_<quantedge::GBMConfig>(m, "GBMConfig")
        .def(py::init<>())
        .def_readwrite("S0",      &quantedge::GBMConfig::S0)
        .def_readwrite("mu",      &quantedge::GBMConfig::mu)
        .def_readwrite("sigma",   &quantedge::GBMConfig::sigma)
        .def_readwrite("dt",        &quantedge::GBMConfig::dt)
        .def_readwrite("num_steps", &quantedge::GBMConfig::num_steps)
        .def_readwrite("num_paths", &quantedge::GBMConfig::num_paths)
        .def_readwrite("seed",    &quantedge::GBMConfig::seed)
        .def("__repr__", [](const quantedge::GBMConfig& c) {
            return "GBMConfig(S0=" + std::to_string(c.S0) +
                   ", mu="     + std::to_string(c.mu) +
                   ", sigma="  + std::to_string(c.sigma) +
                   ", dt="      + std::to_string(c.dt) +
                   ", num_steps=" + std::to_string(c.num_steps) +
                   ", num_paths=" + std::to_string(c.num_paths) + ")";
        });

    // --- simulate_gbm() ---
    m.def("simulate_gbm",
        &quantedge::simulate_gbm,
        py::arg("config"),
        "Simulate GBM paths.\n"
        "Returns list[list[float]] with shape [n_paths][n_steps + 1].\n"
        "Index [i][0] is always S0 for every path i."
    );
}
