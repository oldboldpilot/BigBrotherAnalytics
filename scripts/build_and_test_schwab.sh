#!/usr/bin/env bash

#==============================================================================
# Schwab API Build and Test Script
#==============================================================================
# This script performs a comprehensive build and test cycle for the Schwab API
# integration, including all related components and end-to-end workflows.
#
# Usage:
#   ./scripts/build_and_test_schwab.sh [options]
#
# Options:
#   --clean         Force a clean build (removes build directory)
#   --no-tests      Build only, skip running tests
#   --verbose       Enable verbose output
#   --help          Show this help message
#
# Exit Codes:
#   0 - Success (all tests passed)
#   1 - Build failed
#   2 - Tests failed
#   3 - Configuration failed
#==============================================================================

set -euo pipefail

#==============================================================================
# Color Definitions for Output
#==============================================================================

# Check if we're in a terminal that supports colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    RESET=''
fi

#==============================================================================
# Logging Functions
#==============================================================================

log_header() {
    echo -e "\n${BOLD}${CYAN}========================================${RESET}"
    echo -e "${BOLD}${CYAN}$1${RESET}"
    echo -e "${BOLD}${CYAN}========================================${RESET}\n"
}

log_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

log_step() {
    echo -e "\n${BOLD}${MAGENTA}>>> $1${RESET}\n"
}

#==============================================================================
# Configuration Variables
#==============================================================================

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Default options
FORCE_CLEAN=false
RUN_TESTS=true
VERBOSE=false

# Build targets
BUILD_TARGETS=(
    "schwab_api"
    "test_order_manager_integration"
    "test_account_manager_integration"
    "test_schwab_e2e_workflow"
)

# Test executables
TEST_EXECUTABLES=(
    "test_order_manager_integration"
    "test_account_manager_integration"
    "test_schwab_e2e_workflow"
)

#==============================================================================
# Parse Command Line Arguments
#==============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                FORCE_CLEAN=true
                shift
                ;;
            --no-tests)
                RUN_TESTS=false
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                head -n 30 "$0" | grep -E '^#' | sed 's/^# //g' | sed 's/^#//g'
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 3
                ;;
        esac
    done
}

#==============================================================================
# Environment Setup
#==============================================================================

setup_environment() {
    log_header "Setting Up Build Environment"

    # Set compiler to Clang
    export CC=/usr/local/bin/clang
    export CXX=/usr/local/bin/clang++

    # Skip clang-tidy for faster builds
    export SKIP_CLANG_TIDY=1

    # Set library path for runtime linking
    export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

    log_info "CC          = ${CC}"
    log_info "CXX         = ${CXX}"
    log_info "SKIP_CLANG_TIDY = ${SKIP_CLANG_TIDY}"
    log_info "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"

    # Verify compilers exist
    if [[ ! -x "${CC}" ]]; then
        log_error "C compiler not found: ${CC}"
        exit 3
    fi

    if [[ ! -x "${CXX}" ]]; then
        log_error "C++ compiler not found: ${CXX}"
        exit 3
    fi

    log_success "Environment configured successfully"
}

#==============================================================================
# Clean Build Directory
#==============================================================================

clean_build() {
    if [[ "${FORCE_CLEAN}" == true ]]; then
        log_step "Cleaning build directory"

        if [[ -d "${BUILD_DIR}" ]]; then
            log_info "Removing existing build directory: ${BUILD_DIR}"
            rm -rf "${BUILD_DIR}"
            log_success "Build directory cleaned"
        else
            log_info "Build directory does not exist, skipping clean"
        fi
    fi
}

#==============================================================================
# Configure CMake
#==============================================================================

configure_cmake() {
    log_header "Configuring CMake"

    # Create build directory
    if [[ ! -d "${BUILD_DIR}" ]]; then
        log_info "Creating build directory: ${BUILD_DIR}"
        mkdir -p "${BUILD_DIR}"
    fi

    cd "${BUILD_DIR}"

    log_step "Running CMake configuration"

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER="${CC}"
        -DCMAKE_CXX_COMPILER="${CXX}"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    )

    if [[ "${VERBOSE}" == true ]]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi

    log_info "CMake arguments: ${CMAKE_ARGS[*]}"

    if ! cmake "${CMAKE_ARGS[@]}" ..; then
        log_error "CMake configuration failed"
        exit 3
    fi

    log_success "CMake configuration completed successfully"
}

#==============================================================================
# Build Targets
#==============================================================================

build_targets() {
    log_header "Building Schwab API Module and Tests"

    cd "${BUILD_DIR}"

    # Determine number of parallel jobs
    local num_jobs
    num_jobs=$(nproc 2>/dev/null || echo 4)
    log_info "Building with ${num_jobs} parallel jobs"

    # Build each target sequentially to ensure proper dependency resolution
    for target in "${BUILD_TARGETS[@]}"; do
        log_step "Building target: ${target}"

        local build_cmd="cmake --build . --target ${target} -j ${num_jobs}"

        if [[ "${VERBOSE}" == true ]]; then
            build_cmd="${build_cmd} --verbose"
        fi

        log_info "Build command: ${build_cmd}"

        if ! eval "${build_cmd}"; then
            log_error "Failed to build target: ${target}"
            exit 1
        fi

        log_success "Target '${target}' built successfully"
    done

    log_success "All targets built successfully"
}

#==============================================================================
# Run Tests
#==============================================================================

run_tests() {
    if [[ "${RUN_TESTS}" != true ]]; then
        log_warning "Test execution skipped (--no-tests flag)"
        return 0
    fi

    log_header "Running Schwab API Integration Tests"

    cd "${BUILD_DIR}"

    local test_results=()
    local failed_tests=()
    local total_tests=0
    local passed_tests=0

    for test_exe in "${TEST_EXECUTABLES[@]}"; do
        local test_path="${BUILD_DIR}/bin/${test_exe}"

        if [[ ! -x "${test_path}" ]]; then
            log_error "Test executable not found: ${test_path}"
            failed_tests+=("${test_exe} (NOT FOUND)")
            ((total_tests++)) || true
            continue
        fi

        log_step "Running test: ${test_exe}"
        ((total_tests++)) || true

        # Run test with colored output
        if "${test_path}" --gtest_color=yes; then
            log_success "Test '${test_exe}' PASSED"
            test_results+=("${GREEN}✓${RESET} ${test_exe}")
            ((passed_tests++)) || true
        else
            log_error "Test '${test_exe}' FAILED"
            test_results+=("${RED}✗${RESET} ${test_exe}")
            failed_tests+=("${test_exe}")
        fi
    done

    # Display test summary
    log_header "Test Summary"

    echo -e "${BOLD}Test Results:${RESET}"
    for result in "${test_results[@]}"; do
        echo -e "  ${result}"
    done

    echo ""
    echo -e "${BOLD}Statistics:${RESET}"
    echo -e "  Total Tests:  ${total_tests}"
    echo -e "  Passed:       ${GREEN}${passed_tests}${RESET}"
    echo -e "  Failed:       ${RED}$((total_tests - passed_tests))${RESET}"

    if [[ ${#failed_tests[@]} -gt 0 ]]; then
        echo ""
        log_error "The following tests failed:"
        for failed_test in "${failed_tests[@]}"; do
            echo -e "  ${RED}✗${RESET} ${failed_test}"
        done
        return 2
    else
        echo ""
        log_success "All tests passed!"
        return 0
    fi
}

#==============================================================================
# Display Build Information
#==============================================================================

display_build_info() {
    log_header "Build Information"

    log_info "Project Root: ${PROJECT_ROOT}"
    log_info "Build Directory: ${BUILD_DIR}"
    log_info "C Compiler: $(${CC} --version | head -n 1)"
    log_info "C++ Compiler: $(${CXX} --version | head -n 1)"

    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        local build_type
        build_type=$(grep CMAKE_BUILD_TYPE:STRING "${BUILD_DIR}/CMakeCache.txt" | cut -d'=' -f2)
        log_info "Build Type: ${build_type}"
    fi
}

#==============================================================================
# Main Execution Flow
#==============================================================================

main() {
    local start_time
    start_time=$(date +%s)

    # Print banner
    echo -e "${BOLD}${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║       Schwab API Build and Test Script                       ║
║       BigBrotherAnalytics Trading System                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${RESET}"

    # Parse command line arguments
    parse_args "$@"

    # Display build information
    display_build_info

    # Execute build steps
    setup_environment
    clean_build
    configure_cmake
    build_targets

    # Run tests if enabled
    local test_exit_code=0
    if ! run_tests; then
        test_exit_code=$?
    fi

    # Calculate elapsed time
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    # Final summary
    log_header "Build Complete"

    echo -e "${BOLD}Time Elapsed:${RESET} ${minutes}m ${seconds}s"

    if [[ ${test_exit_code} -eq 0 ]]; then
        echo -e "\n${BOLD}${GREEN}✓ Build and test cycle completed successfully!${RESET}\n"
        exit 0
    elif [[ ${test_exit_code} -eq 2 ]]; then
        echo -e "\n${BOLD}${RED}✗ Build succeeded but tests failed${RESET}\n"
        exit 2
    else
        echo -e "\n${BOLD}${RED}✗ Build or configuration failed${RESET}\n"
        exit 1
    fi
}

# Trap errors and provide helpful message
trap 'log_error "Script failed at line $LINENO"' ERR

# Execute main function with all arguments
main "$@"
