/**
 * @file module.cpp
 * @brief pybind11 module entry point.
 *
 * Aggregates all submodule bindings into a single `quantedge_cpp` Python module.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_pricing(py::module_& m);
void bind_iv_solver(py::module_& m);
void bind_greeks(py::module_& m);
void bind_simulator(py::module_& m);

PYBIND11_MODULE(quantedge_cpp, m) {
    m.doc() = "QuantEdge C++ Computational Finance Engine";

    auto pricing_mod = m.def_submodule("pricing", "Black-Scholes pricing");
    bind_pricing(pricing_mod);

    auto iv_mod = m.def_submodule("iv_solver", "Implied Volatility solvers");
    bind_iv_solver(iv_mod);

    auto greeks_mod = m.def_submodule("greeks", "Greeks computation");
    bind_greeks(greeks_mod);

    auto sim_mod = m.def_submodule("simulator", "Market path simulation");
    bind_simulator(sim_mod);
}
