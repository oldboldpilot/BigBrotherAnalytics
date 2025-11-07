cmake_minimum_required(VERSION 3.20)
project(BigBrotherAnalytics VERSION 1.0.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI)

# Include Intel MKL if available
find_package(MKL CONFIG)
if(MKL_FOUND)
    message(STATUS "Intel MKL found: ${MKL_VERSION}")
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
else()
    # Fallback to standard BLAS/LAPACK
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
)

# Options Pricing Engine Library
add_library(options_pricing SHARED
    src/correlation_engine/cpp/options_pricing.cpp
    src/correlation_engine/cpp/black_scholes.cpp
    src/correlation_engine/cpp/binomial_tree.cpp
    src/correlation_engine/cpp/greeks.cpp
    src/correlation_engine/cpp/implied_volatility.cpp
)

target_link_libraries(options_pricing
    PUBLIC
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Python bindings for Options Pricing
pybind11_add_module(options_pricing_py
    src/correlation_engine/cpp/bindings/options_pricing_bindings.cpp
)

target_link_libraries(options_pricing_py
    PRIVATE
    options_pricing
    pybind11::module
)

# Correlation Engine Library
add_library(correlation_engine SHARED
    src/correlation_engine/cpp/correlation.cpp
    src/correlation_engine/cpp/pearson.cpp
    src/correlation_engine/cpp/spearman.cpp
    src/correlation_engine/cpp/time_lagged.cpp
)

target_link_libraries(correlation_engine
    PUBLIC
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Add MPI support if available
if(MPI_FOUND)
    target_link_libraries(correlation_engine PUBLIC MPI::MPI_CXX)
    target_compile_definitions(correlation_engine PRIVATE USE_MPI)
endif()

# Python bindings for Correlation Engine
pybind11_add_module(correlation_engine_py
    src/correlation_engine/cpp/bindings/correlation_bindings.cpp
)

target_link_libraries(correlation_engine_py
    PRIVATE
    correlation_engine
    pybind11::module
)

# Schwab API Client Library (C++23)
add_library(schwab_api SHARED
    src/schwab_api/cpp/auth.cpp
    src/schwab_api/cpp/market_data.cpp
    src/schwab_api/cpp/orders.cpp
    src/schwab_api/cpp/websocket.cpp
)

target_link_libraries(schwab_api
    PUBLIC
    OpenMP::OpenMP_CXX
)

# Find libcurl for HTTP requests
find_package(CURL REQUIRED)
target_link_libraries(schwab_api PUBLIC CURL::libcurl)

# Python bindings for Schwab API
pybind11_add_module(schwab_api_py
    src/schwab_api/cpp/bindings/schwab_bindings.cpp
)

target_link_libraries(schwab_api_py
    PRIVATE
    schwab_api
    pybind11::module
)

# Set output directories
set_target_properties(
    options_pricing
    correlation_engine
    schwab_api
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(
    options_pricing_py
    correlation_engine_py
    schwab_api_py
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/correlation_engine
)

# Installation
install(TARGETS options_pricing correlation_engine schwab_api
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS options_pricing_py correlation_engine_py schwab_api_py
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

# Testing
enable_testing()
add_subdirectory(tests/cpp)

# Print configuration summary
message(STATUS "")
message(STATUS "BigBrotherAnalytics C++ Configuration Summary")
message(STATUS "==========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Python: ${Python3_VERSION}")
message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
if(MPI_FOUND)
    message(STATUS "MPI: ${MPI_CXX_VERSION}")
else()
    message(STATUS "MPI: Not found (optional)")
endif()
if(MKL_FOUND)
    message(STATUS "Intel MKL: ${MKL_VERSION}")
else()
    message(STATUS "BLAS/LAPACK: System libraries")
endif()
message(STATUS "==========================================")
message(STATUS "")
