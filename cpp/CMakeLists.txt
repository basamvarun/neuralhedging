cmake_minimum_required(VERSION 3.20)
project(QuantEdgeEngine
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "C++ Computational Finance Engine for QuantEdge"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Options ---
option(BUILD_TESTS "Build C++ unit tests" ON)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" ON)
option(USE_OPENMP "Enable OpenMP for parallelism" ON)

# --- OpenMP ---
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found — building without parallelism")
    endif()
endif()

# --- Eigen (header-only) ---
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found — will use bundled or system headers if available")
endif()

# --- Core Library ---
add_library(quantedge_core STATIC
    src/black_scholes.cpp
    src/bisection.cpp
    src/newton_raphson.cpp
    src/greeks_engine.cpp
    src/gbm.cpp
)

target_include_directories(quantedge_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(quantedge_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(Eigen3_FOUND)
    target_link_libraries(quantedge_core PUBLIC Eigen3::Eigen)
endif()

# --- pybind11 Python Module ---
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(quantedge_cpp
        bindings/module.cpp
        bindings/bind_pricing.cpp
        bindings/bind_iv_solver.cpp
        bindings/bind_greeks.cpp
        bindings/bind_simulator.cpp
    )

    target_link_libraries(quantedge_cpp PRIVATE quantedge_core)
endif()

# --- Tests ---
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
