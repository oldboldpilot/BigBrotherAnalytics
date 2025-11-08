cmake_minimum_required(VERSION 3.28)  # Required for C++23 modules support
project(BigBrotherAnalytics VERSION 1.0.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable C++23 modules
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for Clang 21
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
# Removed -ffast-math due to isnan/infinity usage in Greeks calculations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Clang-specific flags for C++23 modules
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprebuilt-module-path=${CMAKE_BINARY_DIR}/modules)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(MPI)
find_package(Threads REQUIRED)

# Find CURL for HTTP requests
find_package(CURL REQUIRED)

# Find Python for optional bindings
find_package(Python3 COMPONENTS Interpreter Development)

# Intel MKL (optional, for optimized BLAS/LAPACK)
find_package(MKL CONFIG)
if(MKL_FOUND)
    message(STATUS "Intel MKL found: ${MKL_VERSION}")
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${BLAS_INCLUDE_DIRS}
)

# ============================================================================
# External Dependencies (to be installed separately)
# ============================================================================
# These will need to be installed on the system:
# - DuckDB C++ library
# - ONNX Runtime C++ library
# - spdlog (logging)
# - nlohmann/json (JSON parsing)
# - yaml-cpp (config files)
# - websocketpp (WebSocket client)

# Try to find optional libraries
find_library(DUCKDB_LIB duckdb)
find_path(DUCKDB_INCLUDE duckdb.hpp)

find_library(ONNXRUNTIME_LIB onnxruntime)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h)

find_package(spdlog)
find_package(nlohmann_json)
find_package(yaml-cpp)

# ============================================================================
# Core Libraries
# ============================================================================

# Utils Library (logging, config, database connections)
add_library(utils SHARED
    src/utils/logger.cpp
    src/utils/config.cpp
    src/utils/database.cpp
    src/utils/timer.cpp
)

# C++23 modules using Clang 21 approach (global module fragment)
# Following C++ Core Guidelines and Clang 21 standards
# See: https://releases.llvm.org/21.1.0/tools/clang/docs/StandardCPlusPlusModules.html
# See: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
target_sources(utils
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/utils/types.cppm
            src/utils/logger.cppm
            src/utils/config.cppm
)

# Module compilation flags for Clang 21
target_compile_options(utils PRIVATE -fmodule-output)

target_link_libraries(utils
    PUBLIC
    Threads::Threads
)

if(spdlog_FOUND)
    target_link_libraries(utils PUBLIC spdlog::spdlog)
endif()

if(yaml-cpp_FOUND)
    target_link_libraries(utils PUBLIC yaml-cpp)
endif()

if(DUCKDB_LIB AND DUCKDB_INCLUDE)
    target_link_libraries(utils PUBLIC ${DUCKDB_LIB})
    target_include_directories(utils PUBLIC ${DUCKDB_INCLUDE})
    target_compile_definitions(utils PUBLIC HAS_DUCKDB)
endif()

# ============================================================================
# Market Intelligence Engine
# ============================================================================

add_library(market_intelligence SHARED
    src/market_intelligence/data_fetcher.cpp
    src/market_intelligence/market_data_client.cpp
    src/market_intelligence/news_client.cpp
    src/market_intelligence/sentiment_analyzer.cpp
    src/market_intelligence/entity_recognizer.cpp
)

target_link_libraries(market_intelligence
    PUBLIC
    utils
    CURL::libcurl
    OpenMP::OpenMP_CXX
)

if(nlohmann_json_FOUND)
    target_link_libraries(market_intelligence PUBLIC nlohmann_json::nlohmann_json)
endif()

if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
    target_link_libraries(market_intelligence PUBLIC ${ONNXRUNTIME_LIB})
    target_include_directories(market_intelligence PUBLIC ${ONNXRUNTIME_INCLUDE})
    target_compile_definitions(market_intelligence PUBLIC HAS_ONNXRUNTIME)
endif()

# ============================================================================
# Correlation Engine
# ============================================================================

add_library(correlation_engine SHARED
    src/correlation_engine/correlation.cpp
    src/correlation_engine/pearson.cpp
    src/correlation_engine/spearman.cpp
    src/correlation_engine/time_lagged.cpp
    src/correlation_engine/rolling_window.cpp
)

target_link_libraries(correlation_engine
    PUBLIC
    utils
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

if(MPI_FOUND)
    target_link_libraries(correlation_engine PUBLIC MPI::MPI_CXX)
    target_compile_definitions(correlation_engine PRIVATE USE_MPI)
    message(STATUS "Correlation Engine: MPI support enabled")
endif()

# ============================================================================
# Options Pricing Engine
# ============================================================================

add_library(options_pricing SHARED
    src/correlation_engine/options_pricing.cpp
    src/correlation_engine/black_scholes.cpp
    src/correlation_engine/binomial_tree.cpp
    src/correlation_engine/trinomial_tree.cpp
    src/correlation_engine/greeks.cpp
    src/correlation_engine/implied_volatility.cpp
    src/correlation_engine/iv_surface.cpp
)

target_link_libraries(options_pricing
    PUBLIC
    utils
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ============================================================================
# Trading Decision Engine
# ============================================================================

add_library(trading_decision SHARED
    src/trading_decision/strategy_base.cpp
    src/trading_decision/signal_aggregator.cpp
    src/trading_decision/strategy_straddle.cpp
    src/trading_decision/strategy_strangle.cpp
    src/trading_decision/strategy_volatility_arb.cpp
    src/trading_decision/ml_predictor.cpp
    src/trading_decision/portfolio_optimizer.cpp
)

# C++23 module for Iron Condor strategy
target_sources(trading_decision
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/trading_decision/strategy_iron_condor.cppm
)

target_link_libraries(trading_decision
    PUBLIC
    utils
    correlation_engine
    options_pricing
    OpenMP::OpenMP_CXX
)

if(ONNXRUNTIME_LIB)
    target_link_libraries(trading_decision PUBLIC ${ONNXRUNTIME_LIB})
endif()

# ============================================================================
# Risk Management System
# ============================================================================

add_library(risk_management SHARED
    src/risk_management/position_sizer.cpp
    src/risk_management/stop_loss.cpp
    src/risk_management/portfolio_constraints.cpp
    src/risk_management/kelly_criterion.cpp
    src/risk_management/monte_carlo.cpp
)

target_link_libraries(risk_management
    PUBLIC
    utils
    options_pricing
    OpenMP::OpenMP_CXX
)

# ============================================================================
# Schwab API Client
# ============================================================================

add_library(schwab_api SHARED
    src/schwab_api/auth.cpp
    src/schwab_api/token_manager.cpp
    src/schwab_api/market_data.cpp
    src/schwab_api/options_chain.cpp
    src/schwab_api/orders.cpp
    src/schwab_api/account.cpp
    src/schwab_api/websocket_client.cpp
)

target_link_libraries(schwab_api
    PUBLIC
    utils
    CURL::libcurl
    OpenMP::OpenMP_CXX
    Threads::Threads
)

if(nlohmann_json_FOUND)
    target_link_libraries(schwab_api PUBLIC nlohmann_json::nlohmann_json)
endif()

# ============================================================================
# Explainability Layer
# ============================================================================

add_library(explainability SHARED
    src/explainability/decision_logger.cpp
    src/explainability/feature_importance.cpp
    src/explainability/trade_analyzer.cpp
)

target_link_libraries(explainability
    PUBLIC
    utils
)

# ============================================================================
# Main Trading Application
# ============================================================================

add_executable(bigbrother
    src/main.cpp
    src/trading_engine.cpp
)

target_link_libraries(bigbrother
    PRIVATE
    utils
    market_intelligence
    correlation_engine
    options_pricing
    trading_decision
    risk_management
    schwab_api
    explainability
)

# ============================================================================
# Backtesting Application
# ============================================================================

add_executable(backtest
    src/backtest_main.cpp
    src/backtesting/backtest_engine.cpp
    src/backtesting/order_simulator.cpp
    src/backtesting/performance_metrics.cpp
)

target_link_libraries(backtest
    PRIVATE
    utils
    correlation_engine
    options_pricing
    trading_decision
    risk_management
    explainability
)

# ============================================================================
# Python Bindings (Optional)
# ============================================================================

if(Python3_FOUND)
    find_package(pybind11 CONFIG)

    if(pybind11_FOUND)
        message(STATUS "Building Python bindings")

        pybind11_add_module(bigbrother_py
            src/python_bindings/bigbrother_bindings.cpp
        )

        target_link_libraries(bigbrother_py
            PRIVATE
            utils
            correlation_engine
            options_pricing
            trading_decision
            risk_management
        )

        set_target_properties(bigbrother_py
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        )
    endif()
endif()

# ============================================================================
# Testing
# ============================================================================

enable_testing()

find_package(GTest)
if(GTest_FOUND)
    add_subdirectory(tests/cpp)
    message(STATUS "C++ tests enabled")
else()
    message(STATUS "Google Test not found - C++ tests disabled")
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS bigbrother backtest
    RUNTIME DESTINATION bin
)

install(TARGETS
    utils
    market_intelligence
    correlation_engine
    options_pricing
    trading_decision
    risk_management
    schwab_api
    explainability
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════╗")
message(STATUS "║     BigBrotherAnalytics C++ Configuration Summary          ║")
message(STATUS "╚════════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  C++ Standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler            : ${CMAKE_CXX_COMPILER}")
message(STATUS "")
message(STATUS "Core Dependencies:")
message(STATUS "  OpenMP              : ${OpenMP_CXX_VERSION}")
message(STATUS "  Threads             : Found")
message(STATUS "  CURL                : ${CURL_VERSION_STRING}")
if(MPI_FOUND)
    message(STATUS "  MPI                 : ${MPI_CXX_VERSION} ✓")
else()
    message(STATUS "  MPI                 : Not found (optional)")
endif()
message(STATUS "")
message(STATUS "Math Libraries:")
if(MKL_FOUND)
    message(STATUS "  Intel MKL           : ${MKL_VERSION} ✓")
else()
    message(STATUS "  BLAS/LAPACK         : System libraries")
endif()
message(STATUS "")
message(STATUS "Optional Dependencies:")
if(DUCKDB_LIB)
    message(STATUS "  DuckDB              : Found ✓")
else()
    message(STATUS "  DuckDB              : Not found (install libduckdb-dev)")
endif()
if(ONNXRUNTIME_LIB)
    message(STATUS "  ONNX Runtime        : Found ✓")
else()
    message(STATUS "  ONNX Runtime        : Not found (install onnxruntime)")
endif()
if(spdlog_FOUND)
    message(STATUS "  spdlog              : Found ✓")
else()
    message(STATUS "  spdlog              : Not found (install libspdlog-dev)")
endif()
if(nlohmann_json_FOUND)
    message(STATUS "  nlohmann/json       : Found ✓")
else()
    message(STATUS "  nlohmann/json       : Not found (install nlohmann-json3-dev)")
endif()
if(yaml-cpp_FOUND)
    message(STATUS "  yaml-cpp            : Found ✓")
else()
    message(STATUS "  yaml-cpp            : Not found (install libyaml-cpp-dev)")
endif()
message(STATUS "")
if(Python3_FOUND AND pybind11_FOUND)
    message(STATUS "Python Bindings:")
    message(STATUS "  Python              : ${Python3_VERSION} ✓")
    message(STATUS "  pybind11            : Found ✓")
    message(STATUS "")
endif()
if(GTest_FOUND)
    message(STATUS "Testing:")
    message(STATUS "  Google Test         : Found ✓")
    message(STATUS "")
endif()
message(STATUS "Executables:")
message(STATUS "  bigbrother          : Main trading application")
message(STATUS "  backtest            : Backtesting engine")
message(STATUS "")
message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "")
