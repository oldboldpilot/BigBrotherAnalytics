#!/bin/bash
################################################################################
# Build and Test Script for Schwab API Implementation
#
# This script builds and runs all Schwab API integration tests with proper
# error handling and progress indicators. It skips clang-tidy during testing
# to speed up the build process.
#
# Usage:
#   ./scripts/build_schwab_tests.sh [clean]
#
# Arguments:
#   clean    - Remove build directory before building
#
# Builds:
#   - schwab_api library
#   - test_order_manager_integration
#   - test_account_manager_integration
#   - test_schwab_e2e_workflow
#
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-09
################################################################################

set -e  # Exit on any error
set -o pipefail  # Catch errors in pipes

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="/home/muyiwa/Development/BigBrotherAnalytics"
BUILD_DIR="${PROJECT_ROOT}/build"
CLANG_PATH="/usr/local/bin/clang++"
CMAKE_PATH="cmake"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Progress indicators
CHECKMARK="${GREEN}✓${NC}"
CROSSMARK="${RED}✗${NC}"
ARROW="${BLUE}➜${NC}"
BUILDING="${CYAN}⚙${NC}"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${ARROW} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${CHECKMARK} ${GREEN}$1${NC}"
}

print_error() {
    echo -e "${CROSSMARK} ${RED}$1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}⚠  $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ  $1${NC}"
}

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    print_error "Build failed at line ${line_number} with exit code ${exit_code}"
    echo ""
    echo -e "${YELLOW}Troubleshooting tips:${NC}"
    echo "  1. Check compiler version: clang++ --version"
    echo "  2. Verify dependencies are installed"
    echo "  3. Check CMake configuration in build/CMakeCache.txt"
    echo "  4. Review error messages above"
    exit ${exit_code}
}

trap 'handle_error ${LINENO}' ERR

# ============================================================================
# Parse Arguments
# ============================================================================

CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [clean]"
            echo ""
            echo "Options:"
            echo "  clean    Remove build directory before building"
            echo "  -h       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Build Process
# ============================================================================

print_header "Schwab API Integration Tests Build & Run"

# Show configuration
print_info "Project Root: ${PROJECT_ROOT}"
print_info "Build Directory: ${BUILD_DIR}"
print_info "Compiler: ${CLANG_PATH}"
print_info "Skipping clang-tidy for faster builds"
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Clean build directory if requested
if [ "${CLEAN_BUILD}" = true ]; then
    print_step "Cleaning build directory..."
    if [ -d "${BUILD_DIR}" ]; then
        rm -rf "${BUILD_DIR}"
        print_success "Build directory cleaned"
    else
        print_info "Build directory doesn't exist, skipping clean"
    fi
fi

# Create build directory
print_step "Creating build directory..."
mkdir -p "${BUILD_DIR}"
print_success "Build directory ready"

# Export environment variables
export SKIP_CLANG_TIDY=1
export CXX="${CLANG_PATH}"
export CC="${CLANG_PATH%++}"  # Remove ++ to get clang

print_step "Environment configured"
print_info "SKIP_CLANG_TIDY=1 (clang-tidy disabled)"
print_info "CXX=${CXX}"
echo ""

# ============================================================================
# CMake Configuration
# ============================================================================

print_header "CMake Configuration"

cd "${BUILD_DIR}"

print_step "Configuring project with CMake..."
if "${CMAKE_PATH}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER="${CLANG_PATH}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    .. > cmake_output.log 2>&1; then
    print_success "CMake configuration completed"
else
    print_error "CMake configuration failed"
    echo ""
    echo "Last 20 lines of cmake_output.log:"
    tail -n 20 cmake_output.log
    exit 1
fi

# ============================================================================
# Build Libraries
# ============================================================================

print_header "Building Schwab API Library"

print_step "Building schwab_api library..."
if "${CMAKE_PATH}" --build . --target schwab_api -j$(nproc) 2>&1 | tee schwab_api_build.log; then
    print_success "schwab_api library built successfully"
else
    print_error "Failed to build schwab_api library"
    exit 1
fi

# ============================================================================
# Build Test Executables
# ============================================================================

print_header "Building Test Executables"

declare -a TEST_TARGETS=(
    "test_order_manager_integration"
    "test_account_manager_integration"
    "test_schwab_e2e_workflow"
)

FAILED_BUILDS=()

for target in "${TEST_TARGETS[@]}"; do
    print_step "Building ${target}..."
    if "${CMAKE_PATH}" --build . --target "${target}" -j$(nproc) 2>&1 | tee "${target}_build.log"; then
        print_success "${target} built successfully"
    else
        print_error "Failed to build ${target}"
        FAILED_BUILDS+=("${target}")
    fi
    echo ""
done

# Check if any builds failed
if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    print_error "Failed to build ${#FAILED_BUILDS[@]} test(s):"
    for failed in "${FAILED_BUILDS[@]}"; do
        echo "  - ${failed}"
    done
    exit 1
fi

# ============================================================================
# Run Tests
# ============================================================================

print_header "Running Integration Tests"

declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

for target in "${TEST_TARGETS[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    print_step "Running ${target}..."
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    TEST_EXECUTABLE="${BUILD_DIR}/bin/${target}"

    if [ ! -f "${TEST_EXECUTABLE}" ]; then
        print_error "Test executable not found: ${TEST_EXECUTABLE}"
        TEST_RESULTS["${target}"]="NOT_FOUND"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        continue
    fi

    # Run test with color output
    if "${TEST_EXECUTABLE}" --gtest_color=yes 2>&1 | tee "${target}_output.log"; then
        print_success "${target} PASSED"
        TEST_RESULTS["${target}"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "${target} FAILED"
        TEST_RESULTS["${target}"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    echo ""
done

# ============================================================================
# Test Summary
# ============================================================================

print_header "Test Results Summary"

echo -e "${BOLD}Individual Test Results:${NC}"
echo ""

for target in "${TEST_TARGETS[@]}"; do
    result="${TEST_RESULTS[${target}]}"
    case "${result}" in
        PASSED)
            echo -e "  ${CHECKMARK} ${target}: ${GREEN}PASSED${NC}"
            ;;
        FAILED)
            echo -e "  ${CROSSMARK} ${target}: ${RED}FAILED${NC}"
            ;;
        NOT_FOUND)
            echo -e "  ${CROSSMARK} ${target}: ${RED}NOT FOUND${NC}"
            ;;
        *)
            echo -e "  ${CROSSMARK} ${target}: ${YELLOW}UNKNOWN${NC}"
            ;;
    esac
done

echo ""
echo -e "${BOLD}Overall Statistics:${NC}"
echo -e "  Total Tests:  ${TOTAL_TESTS}"
echo -e "  Passed:       ${GREEN}${PASSED_TESTS}${NC}"
echo -e "  Failed:       ${RED}${FAILED_TESTS}${NC}"
echo ""

# ============================================================================
# Final Status
# ============================================================================

if [ ${FAILED_TESTS} -eq 0 ]; then
    print_header "✓ All Tests Passed"
    echo -e "${GREEN}${BOLD}SUCCESS: All Schwab API integration tests passed!${NC}"
    echo ""
    echo -e "${CYAN}Test artifacts saved in:${NC}"
    echo "  ${BUILD_DIR}/*_build.log"
    echo "  ${BUILD_DIR}/*_output.log"
    echo ""
    exit 0
else
    print_header "✗ Some Tests Failed"
    echo -e "${RED}${BOLD}FAILURE: ${FAILED_TESTS} test(s) failed${NC}"
    echo ""
    echo -e "${YELLOW}Review the following logs for details:${NC}"
    for target in "${TEST_TARGETS[@]}"; do
        if [ "${TEST_RESULTS[${target}]}" != "PASSED" ]; then
            echo "  ${BUILD_DIR}/${target}_output.log"
        fi
    done
    echo ""
    exit 1
fi
