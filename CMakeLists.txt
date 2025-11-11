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
            src/utils/utils.cppm
)
target_compile_options(utils PRIVATE -fmodule-output)

target_link_libraries(utils
    PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
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
)
target_compile_options(market_intelligence PRIVATE -fmodule-output)

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
)
target_compile_options(trading_decision PRIVATE -fmodule-output)

target_link_libraries(trading_decision
    PUBLIC
    utils
    correlation_engine
    options_pricing
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
    # Module implementation units removed - implementations in risk_management.cppm
    # TODO: Merge these .cpp files into risk_management.cppm module :private; section
    # See TIER1_EXTENSION_CHECKLIST.md Section J for consolidation tasks
)

# C++23 modules with fluent API and trailing return syntax
target_sources(risk_management
    PUBLIC
        FILE_SET CXX_MODULES FILES
            # src/risk_management/risk.cppm  # Removed - superseded by risk_management.cppm
            src/risk_management/risk_management.cppm
)
target_compile_options(risk_management PRIVATE -fmodule-output)

target_link_libraries(risk_management
    PUBLIC
    utils
    options_pricing
    OpenMP::OpenMP_CXX
)

# ============================================================================
# Schwab API Client (C++23 Modules ENABLED)
# ============================================================================

add_library(schwab_api SHARED
    # token_manager.cpp removed - inline implementation in schwab_api.cppm module
)

# C++23 modules with fluent API and trailing return syntax
target_sources(schwab_api
    PUBLIC
        FILE_SET CXX_MODULES FILES
            # src/schwab_api/schwab.cppm  # Removed - superseded by schwab_api.cppm
            src/schwab_api/schwab_api.cppm
)
target_compile_options(schwab_api PRIVATE -fmodule-output)

target_link_libraries(schwab_api
    PUBLIC
    utils
    options_pricing
    CURL::libcurl
    OpenMP::OpenMP_CXX
    Threads::Threads
    c++
    c++abi
)

if(nlohmann_json_FOUND)
    target_link_libraries(schwab_api PUBLIC nlohmann_json::nlohmann_json)
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
