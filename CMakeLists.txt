# BigBrotherAnalytics - Build Configuration
# Author: Olumuyiwa Oluwasanmi

cmake_minimum_required(VERSION 3.28)  # Required for C++23 modules support

# ============================================================================
# Compiler Configuration (MUST be before project())
# ============================================================================
# Auto-detect Clang or use environment variables (portable across systems)
# Priority: ENV CC/CXX > /usr/local/bin > /usr/bin > system default
if(NOT CMAKE_C_COMPILER)
    if(DEFINED ENV{CC})
        set(CMAKE_C_COMPILER "$ENV{CC}")
    elseif(EXISTS "/usr/local/bin/clang")
        set(CMAKE_C_COMPILER "/usr/local/bin/clang")
    else()
        find_program(CLANG_EXECUTABLE clang)
        if(CLANG_EXECUTABLE)
            set(CMAKE_C_COMPILER "${CLANG_EXECUTABLE}")
        endif()
    endif()
endif()

if(NOT CMAKE_CXX_COMPILER)
    if(DEFINED ENV{CXX})
        set(CMAKE_CXX_COMPILER "$ENV{CXX}")
    elseif(EXISTS "/usr/local/bin/clang++")
        set(CMAKE_CXX_COMPILER "/usr/local/bin/clang++")
    else()
        find_program(CLANGXX_EXECUTABLE clang++)
        if(CLANGXX_EXECUTABLE)
            set(CMAKE_CXX_COMPILER "${CLANGXX_EXECUTABLE}")
        endif()
    endif()
endif()

message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")

project(BigBrotherAnalytics VERSION 1.0.0 LANGUAGES CXX)

# Enable C++23 module dependency scanning and caching for incremental builds
# Precompiled modules (.pcm files) are reused when source files haven't changed
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Use libc++ (LLVM's C++ standard library)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# Auto-detect libc++ library path (portable across systems)
# Priority: ENV LIBCXX_PATH > /usr/local/lib > /usr/lib
if(DEFINED ENV{LIBCXX_PATH})
    set(LIBCXX_LIB_PATH "$ENV{LIBCXX_PATH}")
elseif(EXISTS "/usr/local/lib/libc++.so" OR EXISTS "/usr/local/lib/libc++.a")
    set(LIBCXX_LIB_PATH "/usr/local/lib")
elseif(EXISTS "/usr/lib/x86_64-linux-gnu/libc++.so" OR EXISTS "/usr/lib/x86_64-linux-gnu/libc++.a")
    set(LIBCXX_LIB_PATH "/usr/lib/x86_64-linux-gnu")
elseif(EXISTS "/usr/lib/libc++.so" OR EXISTS "/usr/lib/libc++.a")
    set(LIBCXX_LIB_PATH "/usr/lib")
else()
    # Fallback to /usr/local/lib if not found
    set(LIBCXX_LIB_PATH "/usr/local/lib")
    message(WARNING "Could not find libc++ library, using fallback: ${LIBCXX_LIB_PATH}")
endif()

message(STATUS "Using libc++ library path: ${LIBCXX_LIB_PATH}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -L${LIBCXX_LIB_PATH} -Wl,-rpath,${LIBCXX_LIB_PATH} -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -L${LIBCXX_LIB_PATH} -Wl,-rpath,${LIBCXX_LIB_PATH} -lc++abi")

# libc++ adds abi_tag attributes by default; disable to avoid duplicate attribute
# emissions when Clang builds std headers alongside modules.
add_compile_definitions(_LIBCPP_NO_ABI_TAG)

# ============================================================================
# Pre-Build Quality Check: Run clang-tidy
# ============================================================================
# Run clang-tidy before every build to enforce code quality
# Set SKIP_CLANG_TIDY=1 to bypass (NOT RECOMMENDED)

if(NOT DEFINED ENV{SKIP_CLANG_TIDY})
    message(STATUS "Running pre-build clang-tidy validation...")
    execute_process(
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_clang_tidy.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE CLANG_TIDY_RESULT
    )

    if(NOT CLANG_TIDY_RESULT EQUAL 0)
        message(FATAL_ERROR "clang-tidy validation failed! Fix errors before building.")
    endif()

    message(STATUS "✅ clang-tidy validation passed")
else()
    message(WARNING "⚠️  clang-tidy check SKIPPED (SKIP_CLANG_TIDY is set)")
endif()

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Global Optimization Flags (Applied to ALL modules for consistency)
# ============================================================================
# CRITICAL: All C++23 modules MUST use identical compiler flags to ensure
# precompiled module (.pcm) compatibility across translation units

# Base optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# SIMD and vectorization flags (applied globally)
add_compile_options(
    -march=native          # Auto-detect CPU features (AVX2/AVX-512)
    -mavx2                 # Enable AVX2 SIMD (4-wide parallel, 256-bit)
    -mfma                  # Fused multiply-add instructions
    -fopenmp-simd          # OpenMP SIMD directives
)

# AVX-512 support (disabled - CPU doesn't support it)
# Check for AVX-512 support
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
#
# if(COMPILER_SUPPORTS_AVX512F)
#     # Enable AVX-512 globally for 8-wide parallel operations
#     add_compile_options(
#         -mavx512f          # AVX-512 Foundation
#         -mavx512dq         # Doubleword and Quadword instructions
#         -mavx512bw         # Byte and Word instructions
#         -mavx512vl         # Vector Length Extensions
#     )
#     message(STATUS "✅ AVX-512 SIMD enabled globally (8x parallel operations)")
# else()
    message(STATUS "✅ AVX2 SIMD enabled globally (4x parallel operations)")
# endif()

# C++23 module precompilation flags (applied globally)
add_compile_options(
    -fmodule-output                              # Generate .pcm files
    -fprebuilt-module-path=${CMAKE_BINARY_DIR}/modules  # Cache location
)

message(STATUS "Global optimization flags configured:")
message(STATUS "  - SIMD: AVX2 (baseline) + AVX-512 (if supported)")
message(STATUS "  - Precompiled modules: ${CMAKE_BINARY_DIR}/modules")
message(STATUS "  - Auto-vectorization: Enabled (-march=native)")

# ============================================================================

# Enable C++23 modules
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for Clang 21
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# Release flags with SIMD and OpenMP optimizations
# -march=native: Use all CPU instructions available on build machine
# -mavx2: Enable AVX2 SIMD instructions for vectorization (4x double or 8x float at once)
# -mfma: Enable fused multiply-add instructions (a*b+c in single operation)
# -fopenmp-simd: Enable OpenMP SIMD directives for loop vectorization
# -ffast-math: Fast math optimizations (required for module consistency)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma -fopenmp-simd -ffast-math -DNDEBUG")

# Clang-specific flags for C++23 modules
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprebuilt-module-path=${CMAKE_BINARY_DIR}/modules)
endif()

# C++23 Standard Library Module Support (import std;)
# Auto-detect libc++ modules path (portable across systems)
# Priority: ENV LIBCXX_MODULES_PATH > /opt/libc++_modules > /usr/local > /usr
if(DEFINED ENV{LIBCXX_MODULES_PATH})
    set(LIBCXX_MODULES_BASE "$ENV{LIBCXX_MODULES_PATH}")
elseif(EXISTS "/opt/libc++_modules/share/libc++/v1/std.cppm")
    set(LIBCXX_MODULES_BASE "/opt/libc++_modules")
elseif(EXISTS "/usr/local/share/libc++/v1/std.cppm")
    set(LIBCXX_MODULES_BASE "/usr/local")
elseif(EXISTS "/usr/share/libc++/v1/std.cppm")
    set(LIBCXX_MODULES_BASE "/usr")
else()
    # Fallback - build without std module (slower but still works)
    set(LIBCXX_MODULES_BASE "")
    message(WARNING "Could not find std.cppm, building without import std; support (slower builds)")
endif()

if(LIBCXX_MODULES_BASE)
    message(STATUS "Using libc++ modules from: ${LIBCXX_MODULES_BASE}")

    # Precompile std module for faster builds and cleaner syntax
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/modules/std.pcm
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/modules
        COMMAND ${CMAKE_CXX_COMPILER}
                -std=c++23
                -stdlib=libc++
                -nostdinc++
                -isystem ${LIBCXX_MODULES_BASE}/include/c++/v1
                -D_LIBCPP_NO_ABI_TAG
                --precompile
                ${LIBCXX_MODULES_BASE}/share/libc++/v1/std.cppm
                -o ${CMAKE_BINARY_DIR}/modules/std.pcm
        COMMENT "Precompiling C++23 std module for import std; support"
        VERBATIM
    )
    add_custom_target(std_module DEPENDS ${CMAKE_BINARY_DIR}/modules/std.pcm)
else()
    # Dummy target when std module is not available
    add_custom_target(std_module)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(OpenMP REQUIRED)

# MPI Configuration - DISABLED FOR TIER 1
# Note: MPI is only needed for correlation_engine with massive parallelization
#       Not used in live trading or backtesting executables
#       Will be enabled in Tier 2 for large-scale correlation analysis
set(ENABLE_MPI OFF CACHE BOOL "Enable MPI support")

if(ENABLE_MPI)
    # Use custom-built OpenMPI at /usr/local (built with Clang)
    set(MPI_CXX_INCLUDE_DIRS "/usr/local/include" CACHE STRING "MPI include directories")
    set(MPI_CXX_LIBRARIES "/usr/local/lib/libmpi.so" CACHE FILEPATH "MPI libraries")
    set(MPI_CXX_FOUND TRUE CACHE BOOL "MPI found")
    set(MPI_FOUND TRUE CACHE BOOL "MPI found")

    if(NOT TARGET MPI::MPI_CXX)
        add_library(MPI::MPI_CXX INTERFACE IMPORTED)
        set_target_properties(MPI::MPI_CXX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${MPI_CXX_LIBRARIES}"
        )
        message(STATUS "MPI enabled for correlation engine")
        message(STATUS "  MPI Include: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "  MPI Library: ${MPI_CXX_LIBRARIES}")
    endif()
else()
    message(STATUS "MPI disabled (not needed for Tier 1 live trading)")
endif()

find_package(Threads REQUIRED)

# Berkeley Labs PGAS Components (UPC++, GASNet, OpenSHMEM)
# Installed via ansible playbook to /opt/berkeley/
# Currently optional - not required for Tier 1 POC
set(BERKELEY_PREFIX "/opt/berkeley" CACHE PATH "Berkeley Labs components installation prefix")
set(GASNET_HOME "${BERKELEY_PREFIX}/gasnet" CACHE PATH "GASNet-EX installation directory")
set(UPCXX_HOME "${BERKELEY_PREFIX}/upcxx" CACHE PATH "UPC++ installation directory")
set(OPENSHMEM_HOME "${BERKELEY_PREFIX}/openshmem" CACHE PATH "OpenSHMEM installation directory")

# Check if Berkeley components are installed
if(EXISTS "${UPCXX_HOME}/bin/upcxx")
    message(STATUS "UPC++ found at ${UPCXX_HOME}")
    set(UPCXX_FOUND TRUE)
else()
    message(STATUS "UPC++ not found (optional for Tier 1)")
    set(UPCXX_FOUND FALSE)
endif()

if(EXISTS "${GASNET_HOME}/bin/gasnetrun_mpi")
    message(STATUS "GASNet-EX found at ${GASNET_HOME}")
    set(GASNET_FOUND TRUE)
else()
    message(STATUS "GASNet-EX not found (optional for Tier 1)")
    set(GASNET_FOUND FALSE)
endif()

if(EXISTS "${OPENSHMEM_HOME}/bin/oshcc")
    message(STATUS "OpenSHMEM found at ${OPENSHMEM_HOME}")
    set(OPENSHMEM_FOUND TRUE)
else()
    message(STATUS "OpenSHMEM not found (optional for Tier 1)")
    set(OPENSHMEM_FOUND FALSE)
endif()

# Find CURL for HTTP requests
find_package(CURL REQUIRED)

# Find Python for optional bindings - prefer uv venv Python if available
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
    message(STATUS "Using uv venv Python: ${Python3_EXECUTABLE}")
endif()
find_package(Python3 COMPONENTS Interpreter Development)

# ============================================================================
# Intel oneAPI MKL 2025.3 (High-Performance Math Library)
# ============================================================================
# MKL provides optimized BLAS, LAPACK, FFT, and vector math operations
# Location: /opt/intel/oneapi/mkl/latest
# Performance: 5-10x faster than generic BLAS/LAPACK implementations

set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
set(CMAKE_PREFIX_PATH "${MKL_ROOT}/lib/cmake/mkl;${CMAKE_PREFIX_PATH}")

find_package(MKL CONFIG PATHS ${MKL_ROOT}/lib/cmake/mkl)

if(MKL_FOUND)
    message(STATUS "✅ Intel oneAPI MKL found: ${MKL_VERSION}")
    message(STATUS "   MKL Root: ${MKL_ROOT}")
    message(STATUS "   MKL Libraries: ${MKL_LIBRARIES}")

    # Use MKL with GNU threading (compatible with OpenMP)
    # Options: MKL::MKL_INTEL_LP64_GNU_THREAD (GNU OpenMP)
    #          MKL::MKL_INTEL_LP64_SEQUENTIAL (no threading)
    #          MKL::MKL_INTEL_LP64_TBB_THREAD (Intel TBB)

    set(BLAS_LIBRARIES MKL::MKL)
    set(BLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})

    # Add MKL library paths to runtime search path
    link_directories(${MKL_ROOT}/lib)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${MKL_ROOT}/lib")

    # Enable MKL optimizations
    add_compile_definitions(MKL_AVAILABLE)
    add_compile_definitions(EIGEN_USE_MKL_ALL)  # If using Eigen

else()
    message(WARNING "⚠️  Intel MKL not found, falling back to generic BLAS/LAPACK")
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
# - nlohmann/json (JSON parsing - legacy)
# - simdjson (High-performance JSON parsing)
# - yaml-cpp (config files)
# - websocketpp (WebSocket client)

# Try to find optional libraries
find_library(DUCKDB_LIB duckdb)
find_path(DUCKDB_INCLUDE duckdb.hpp)

find_library(ONNXRUNTIME_LIB onnxruntime)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h)

find_package(spdlog)
find_package(nlohmann_json)
find_package(simdjson)
find_package(yaml-cpp)

# ============================================================================
# Core Libraries
# ============================================================================

# Utils Library (C++23 modules ENABLED)
# All implementation moved to .cppm modules
add_library(utils SHARED)

# C++23 modules with trailing return syntax and fluent APIs
# All modules converted following C++ Core Guidelines
target_sources(utils
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/utils/types.cppm
            src/utils/logger.cppm
            src/utils/config.cppm
            src/utils/database_api.cppm
            src/utils/database.cppm
            src/utils/timer.cppm
            src/utils/circuit_breaker.cppm
            src/utils/math.cppm
            src/utils/tax.cppm
            src/utils/risk_free_rate.cppm
            src/utils/simdjson_wrapper.cppm
            src/utils/utils.cppm
)

target_link_libraries(utils
    PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    yaml-cpp
)

if(spdlog_FOUND)
    target_link_libraries(utils PUBLIC spdlog::spdlog)
endif()

if(simdjson_FOUND)
    target_link_libraries(utils PUBLIC simdjson::simdjson)
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
# Market Intelligence Engine (C++23 Module)
# ============================================================================

add_library(market_intelligence SHARED)

# C++23 module with fluent API
target_sources(market_intelligence
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/market_intelligence/market_intelligence.cppm
            src/market_intelligence/employment_signals.cppm
            src/market_intelligence/sentiment_analyzer.cppm
            src/market_intelligence/news_ingestion.cppm
            src/market_intelligence/fred_rates.cppm
            src/market_intelligence/fred_rate_provider.cppm
            src/market_intelligence/alphavantage_news.cppm
            src/market_intelligence/ml_sentiment_analyzer.cppm
            src/market_intelligence/market_data_types.cppm
            src/market_intelligence/yahoo_finance.cppm
            src/market_intelligence/schwab_api.cppm
            src/market_intelligence/feature_extractor.cppm
            src/market_intelligence/price_predictor.cppm
)

# Optimization flags applied globally - no per-target flags needed
# Include directory for SIMD headers
target_include_directories(market_intelligence PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(market_intelligence
    PUBLIC
    utils
    CURL::libcurl
    OpenMP::OpenMP_CXX
)

if(nlohmann_json_FOUND)
    target_link_libraries(market_intelligence PUBLIC nlohmann_json::nlohmann_json)
endif()

if(simdjson_FOUND)
    target_link_libraries(market_intelligence PUBLIC simdjson::simdjson)
endif()

if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
    target_link_libraries(market_intelligence PUBLIC ${ONNXRUNTIME_LIB})
    target_include_directories(market_intelligence PUBLIC ${ONNXRUNTIME_INCLUDE})
    target_compile_definitions(market_intelligence PUBLIC HAS_ONNXRUNTIME)
endif()

# ============================================================================
# Correlation Engine (C++23 Module ENABLED - Header-only as SHARED library)
# ============================================================================

add_library(correlation_engine SHARED)

# C++23 module with fluent API
target_sources(correlation_engine
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/correlation_engine/correlation.cppm
)
target_compile_options(correlation_engine PRIVATE -fmodule-output)

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
# Options Pricing Engine (C++23 Modules ENABLED - Header-only as SHARED library)
# ============================================================================

add_library(options_pricing SHARED)

# C++23 modules with fluent API and trailing return syntax
target_sources(options_pricing
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/correlation_engine/black_scholes.cppm
            src/correlation_engine/trinomial_tree.cppm
            src/correlation_engine/options_pricing.cppm
)
target_compile_options(options_pricing PRIVATE -fmodule-output)

target_link_libraries(options_pricing
    PUBLIC
    utils
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ============================================================================
# Options Strategies Library (C++23 Modules - SIMD-Optimized)
# ============================================================================
# Complete implementation of 52+ options strategies with AVX2 acceleration
# Implements: Single Leg, Vertical Spreads, Butterflies, Condors, and more
# Performance: <1μs option pricing, <5μs Greeks (AVX2)

add_library(options_strategies SHARED)

# C++23 modules with SIMD-optimized Black-Scholes and Greeks
target_sources(options_strategies
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/options_strategies/base.cppm
            src/options_strategies/simd_utils.cppm
            src/options_strategies/single_leg.cppm
            src/options_strategies/vertical_spreads.cppm
            src/options_strategies/straddles_strangles.cppm
            src/options_strategies/butterflies_condors.cppm
            src/options_strategies/covered_positions.cppm
            src/options_strategies/calendar_spreads.cppm
            src/options_strategies/ratio_spreads.cppm
            src/options_strategies/albatross_ladder.cppm
)

# AVX2 optimization flags for SIMD performance
# Additional optimizations beyond global flags for performance-critical code
target_compile_options(options_strategies PRIVATE
    -funroll-loops
    -ftree-vectorize
)

target_compile_definitions(options_strategies PRIVATE
    USE_AVX2=1
    USE_SIMD_PRICING=1
)

target_include_directories(options_strategies PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(options_strategies
    PUBLIC
    utils
    OpenMP::OpenMP_CXX
)

# ============================================================================
# Trading Decision Engine (C++23 Modules ENABLED)
# ============================================================================

add_library(trading_decision SHARED
    # Module implementation units removed - implementations in strategies.cppm
    # TODO: Merge these .cpp files into strategies.cppm module :private; section
    # See TIER1_EXTENSION_CHECKLIST.md Section J for consolidation tasks
)

# C++23 modules with fluent API and trailing return syntax
target_sources(trading_decision
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/trading_decision/strategy.cppm
            src/trading_decision/strategy_iron_condor.cppm
            src/trading_decision/strategies.cppm
            src/trading_decision/options_strategy_integrator.cppm
)

# Module output configuration
target_compile_options(trading_decision PRIVATE
    -fmodule-output
)

target_link_libraries(trading_decision
    PUBLIC
    utils
    correlation_engine
    options_pricing
    options_strategies
    risk_management
    schwab_api
    market_intelligence
    OpenMP::OpenMP_CXX
)

if(ONNXRUNTIME_LIB)
    target_link_libraries(trading_decision PUBLIC ${ONNXRUNTIME_LIB})
endif()

# ============================================================================
# Risk Management System (C++23 Modules ENABLED)
# ============================================================================

add_library(risk_management SHARED
    # Modular risk management system with fluent APIs
    # Each component is a separate module for better composability
)

# C++23 modules with fluent API and trailing return syntax
target_sources(risk_management
    PUBLIC
        FILE_SET CXX_MODULES FILES
            # Core risk management modules (NEW - fluent API design)
            src/risk_management/position_sizer.cppm         # Position sizing with Kelly Criterion
            src/risk_management/stop_loss.cppm              # Stop loss management
            src/risk_management/monte_carlo.cppm            # Monte Carlo simulation
            src/risk_management/var_calculator.cppm         # Value at Risk with MKL
            src/risk_management/stress_testing.cppm         # Stress testing with SIMD
            src/risk_management/performance_metrics.cppm    # Performance metrics (Sharpe, Sortino, etc.)
            src/risk_management/correlation_analyzer.cppm   # Correlation and diversification analysis
            src/risk_management/risk_manager.cppm           # Main risk management orchestrator

            # Legacy monolithic module (DEPRECATED - will be removed after migration)
            src/risk_management/risk_management.cppm
)
target_compile_options(risk_management PRIVATE -fmodule-output)

target_link_libraries(risk_management
    PUBLIC
    utils
    options_pricing
    OpenMP::OpenMP_CXX
    ${MKL_LIBRARIES}
)

# ============================================================================
# Trading Core - Platform-Agnostic Order Management (C++23 Modules)
# ============================================================================

add_library(trading_core SHARED)

# C++23 modules for platform-agnostic trading
target_sources(trading_core
    PUBLIC
        FILE_SET CXX_MODULES FILES
            # Module dependency chain:
            # order_types (base types) -> platform_interface -> orders_manager
            src/core/trading/order_types.cppm
            src/core/trading/platform_interface.cppm
            src/core/trading/orders_manager.cppm
)
target_compile_options(trading_core PRIVATE -fmodule-output)

target_link_libraries(trading_core
    PUBLIC
    utils
    duckdb_bridge
    OpenMP::OpenMP_CXX
    Threads::Threads
    c++
    c++abi
)

if(nlohmann_json_FOUND)
    target_link_libraries(trading_core PUBLIC nlohmann_json::nlohmann_json)
endif()

if(simdjson_FOUND)
    target_link_libraries(trading_core PUBLIC simdjson::simdjson)
endif()

# ============================================================================
# Schwab API Client (C++23 Modules ENABLED)
# ============================================================================

add_library(schwab_api SHARED
    # token_manager.cpp removed - inline implementation in schwab_api.cppm module
)

# DuckDB Bridge Library - isolates DuckDB incomplete types from C++23 modules
# This bridge library provides clean C++ interfaces without exposing DuckDB's
# problematic forward declarations (e.g., QueryNode)
add_library(duckdb_bridge STATIC
    src/schwab_api/duckdb_bridge.cpp
)

target_include_directories(duckdb_bridge
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(duckdb_bridge
    PUBLIC
        duckdb
)

# Position tracker and portfolio analyzer C++23 modules
# TODO: Update these modules to use duckdb_bridge instead of void* pointers
# Pattern for updating:
#   1. Include schwab_api/duckdb_bridge.hpp (NOT <duckdb.hpp>)
#   2. Use DatabaseHandle/ConnectionHandle instead of void*
#   3. Call bridge functions: openDatabase(), createConnection(), executeQuery()
# Once updated, add position_tracker.cppm and portfolio_analyzer.cppm to FILE_SET below

# C++23 modules with fluent API and trailing return syntax
target_sources(schwab_api
    PUBLIC
        FILE_SET CXX_MODULES FILES
            # Module dependency chain:
            # account_types -> schwab_api -> account_manager
            # AccountClient (lightweight) is in schwab_api.cppm
            # AccountManager (full-featured) is in account_manager.cppm
            #
            # NOTE: position_tracker and portfolio_analyzer work correctly but are
            # disabled here due to CMake C++23 module dependency detection limitations.
            # They are available via Python bindings which use a different build path.
            # Both modules are fully updated to use duckdb_bridge for database access.
            src/schwab_api/account_types.cppm
            src/schwab_api/schwab_api.cppm
            src/schwab_api/account_manager.cppm
            src/schwab_api/orders_manager.cppm  # Uses duckdb_bridge for order logging
            src/schwab_api/schwab_order_executor.cppm  # Schwab-specific TradingPlatformInterface implementation
            # src/schwab_api/position_tracker.cppm  # Uses duckdb_bridge - works in Python bindings
            # src/schwab_api/portfolio_analyzer.cppm  # No DB deps - works in Python bindings
)
target_compile_options(schwab_api PRIVATE -fmodule-output)

target_link_libraries(schwab_api
    PUBLIC
    utils
    options_pricing
    trading_core  # Platform-agnostic order management
    duckdb_bridge
    CURL::libcurl
    OpenMP::OpenMP_CXX
    Threads::Threads
    c++
    c++abi
)

if(nlohmann_json_FOUND)
    target_link_libraries(schwab_api PUBLIC nlohmann_json::nlohmann_json)
endif()

if(simdjson_FOUND)
    target_link_libraries(schwab_api PUBLIC simdjson::simdjson)
endif()

# ============================================================================
# Explainability Layer (C++23 Module ENABLED)
# ============================================================================

add_library(explainability SHARED)

# C++23 module with fluent API and trailing return syntax
target_sources(explainability
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/explainability/explainability.cppm
)
target_compile_options(explainability PRIVATE -fmodule-output)

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
    c++
    c++abi
)

# ============================================================================
# Backtesting Application (C++23 Modules ENABLED)
# ============================================================================

add_executable(backtest
    src/backtest_main.cpp
    # src/backtesting/backtest_engine_impl.cpp  # Removed - consolidated into backtest_engine.cppm
)

# C++23 modules with fluent API and trailing return syntax
target_sources(backtest
    PRIVATE
        FILE_SET CXX_MODULES FILES
            # src/backtesting/backtest.cppm  # Removed - superseded by backtest_engine.cppm
            src/backtesting/backtest_engine.cppm
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
# Schwab API Live Test
# ============================================================================

add_executable(test_schwab_live
    examples/test_schwab_live.cpp
)

target_link_libraries(test_schwab_live
    PRIVATE
    schwab_api
    options_pricing
    utils
    ${CURL_LIBRARIES}
    c++
    c++abi
)

# ============================================================================
# Options Strategy Integration Example
# ============================================================================

add_executable(options_strategy_integration_example
    examples/options_strategy_integration_example.cpp
)

target_link_libraries(options_strategy_integration_example
    PRIVATE
    trading_decision
    options_strategies
    risk_management
    utils
    c++
    c++abi
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

# Use GoogleTest built with libc++ from /usr/local
# Manually create imported targets to bypass system GTest
if(EXISTS "/usr/local/lib/libgtest.a" AND EXISTS "/usr/local/lib/libgtest_main.a")
    set(GTest_FOUND TRUE)

    # Create imported target for GTest::gtest
    add_library(GTest::gtest STATIC IMPORTED)
    set_target_properties(GTest::gtest PROPERTIES
        IMPORTED_LOCATION "/usr/local/lib/libgtest.a"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"
        INTERFACE_LINK_LIBRARIES "/usr/local/lib/libc++.so;/lib/x86_64-linux-gnu/libc++abi.so"
    )

    # Create imported target for GTest::gtest_main
    add_library(GTest::gtest_main STATIC IMPORTED)
    set_target_properties(GTest::gtest_main PROPERTIES
        IMPORTED_LOCATION "/usr/local/lib/libgtest_main.a"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"
        INTERFACE_LINK_LIBRARIES "GTest::gtest"
    )

    add_subdirectory(tests/cpp)
    message(STATUS "C++ tests enabled (using libc++ compatible GoogleTest from /usr/local)")
else()
    message(STATUS "Google Test not found - C++ tests disabled")
endif()

# ============================================================================
# Benchmarks
# ============================================================================

add_subdirectory(benchmarks)

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
    trading_core
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
if(simdjson_FOUND)
    message(STATUS "  simdjson            : Found ✓")
else()
    message(STATUS "  simdjson            : Not found (build from source)")
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

# Options Pricing Python Bindings (Tagged: PYTHON_BINDINGS)
pybind11_add_module(bigbrother_options src/python_bindings/options_bindings.cpp)
target_link_libraries(bigbrother_options PRIVATE options_pricing utils)
set_target_properties(bigbrother_options PROPERTIES
    OUTPUT_NAME "bigbrother_options"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# Correlation Engine Python Bindings (Tagged: PYTHON_BINDINGS)
pybind11_add_module(bigbrother_correlation src/python_bindings/correlation_bindings.cpp)
target_link_libraries(bigbrother_correlation PRIVATE correlation_engine utils)
set_target_properties(bigbrother_correlation PROPERTIES
    OUTPUT_NAME "bigbrother_correlation"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# Risk Management Python Bindings (Tagged: PYTHON_BINDINGS)
pybind11_add_module(bigbrother_risk src/python_bindings/risk_bindings.cpp)
target_link_libraries(bigbrother_risk PRIVATE risk_management utils)
set_target_properties(bigbrother_risk PROPERTIES
    OUTPUT_NAME "bigbrother_risk"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# News Ingestion Python Bindings (Tagged: PYTHON_BINDINGS)
pybind11_add_module(news_ingestion_py src/python_bindings/news_bindings.cpp)
target_link_libraries(news_ingestion_py PRIVATE market_intelligence utils)
set_target_properties(news_ingestion_py PROPERTIES
    OUTPUT_NAME "news_ingestion_py"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# FRED Rates Python Bindings (Tagged: PYTHON_BINDINGS)
pybind11_add_module(fred_rates_py src/python_bindings/fred_bindings.cpp)
target_link_libraries(fred_rates_py PRIVATE market_intelligence utils CURL::libcurl)
set_target_properties(fred_rates_py PROPERTIES
    OUTPUT_NAME "fred_rates_py"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# Market Data Python Bindings - Schwab + Yahoo Finance (Tagged: PYTHON_BINDINGS)
pybind11_add_module(market_data_py src/python_bindings/market_data_bindings.cpp)
target_link_libraries(market_data_py PRIVATE market_intelligence utils CURL::libcurl)
set_target_properties(market_data_py PROPERTIES
    OUTPUT_NAME "market_data_py"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# DuckDB Python Bindings - CRITICAL (Tagged: PYTHON_BINDINGS)
# Temporarily disabled due to DuckDB header incomplete type issue
# TODO: Fix DuckDB SelectStatement/QueryNode incomplete type in duckdb.hpp
# pybind11_add_module(bigbrother_duckdb src/python_bindings/duckdb_bindings.cpp)
# target_link_libraries(bigbrother_duckdb PRIVATE utils)
# target_link_libraries(bigbrother_duckdb PRIVATE ${DUCKDB_LIB})
# set_target_properties(bigbrother_duckdb PROPERTIES
#     OUTPUT_NAME "bigbrother_duckdb"
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
# )
