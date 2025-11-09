#!/bin/bash
################################################################################
# Quick Test Script for Schwab API (Pre-built Tests)
#
# This script runs the Schwab API integration tests using already-built
# executables. Use this when you've already built the tests and just want
# to run them quickly without rebuilding.
#
# Usage:
#   ./scripts/quick_test_schwab.sh [test_name]
#
# Arguments:
#   test_name  - Optional: Run specific test only
#                (order_manager | account_manager | e2e_workflow | all)
#                Default: all
#
# Examples:
#   ./scripts/quick_test_schwab.sh                    # Run all tests
#   ./scripts/quick_test_schwab.sh order_manager      # Run order manager tests only
#   ./scripts/quick_test_schwab.sh e2e_workflow       # Run E2E workflow tests only
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
BIN_DIR="${BUILD_DIR}/bin"

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

# ============================================================================
# Validate Environment
# ============================================================================

validate_environment() {
    if [ ! -d "${BUILD_DIR}" ]; then
        print_error "Build directory not found: ${BUILD_DIR}"
        echo ""
        echo "Please build the project first using:"
        echo "  ./scripts/build_schwab_tests.sh"
        exit 1
    fi

    if [ ! -d "${BIN_DIR}" ]; then
        print_error "Binary directory not found: ${BIN_DIR}"
        echo ""
        echo "Tests haven't been built yet. Please run:"
        echo "  ./scripts/build_schwab_tests.sh"
        exit 1
    fi
}

# ============================================================================
# Test Definitions
# ============================================================================

declare -A TEST_MAP=(
    ["order_manager"]="test_order_manager_integration"
    ["account_manager"]="test_account_manager_integration"
    ["e2e_workflow"]="test_schwab_e2e_workflow"
)

# ============================================================================
# Parse Arguments
# ============================================================================

SELECTED_TESTS=()

if [ $# -eq 0 ]; then
    # No arguments - run all tests
    for key in "${!TEST_MAP[@]}"; do
        SELECTED_TESTS+=("${TEST_MAP[${key}]}")
    done
else
    case "$1" in
        order_manager|account_manager|e2e_workflow)
            SELECTED_TESTS+=("${TEST_MAP[$1]}")
            ;;
        all)
            for key in "${!TEST_MAP[@]}"; do
                SELECTED_TESTS+=("${TEST_MAP[${key}]}")
            done
            ;;
        -h|--help)
            echo "Usage: $0 [test_name]"
            echo ""
            echo "Arguments:"
            echo "  test_name    Optional test to run (default: all)"
            echo ""
            echo "Available tests:"
            echo "  order_manager     - Order Manager integration tests"
            echo "  account_manager   - Account Manager integration tests"
            echo "  e2e_workflow      - End-to-end workflow tests"
            echo "  all               - Run all tests (default)"
            echo ""
            echo "Examples:"
            echo "  $0                        # Run all tests"
            echo "  $0 order_manager          # Run order manager tests only"
            echo "  $0 e2e_workflow           # Run E2E workflow tests only"
            exit 0
            ;;
        *)
            print_error "Unknown test: $1"
            echo ""
            echo "Available tests: order_manager, account_manager, e2e_workflow, all"
            echo "Use -h for help"
            exit 1
            ;;
    esac
fi

# ============================================================================
# Main Execution
# ============================================================================

print_header "Schwab API Quick Test Runner"

# Validate environment
validate_environment

print_info "Build Directory: ${BUILD_DIR}"
print_info "Tests to run: ${#SELECTED_TESTS[@]}"
echo ""

# Change to build directory for cleaner output
cd "${BUILD_DIR}"

# ============================================================================
# Run Tests
# ============================================================================

declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

for test_exec in "${SELECTED_TESTS[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    TEST_PATH="${BIN_DIR}/${test_exec}"

    # Check if test executable exists
    if [ ! -f "${TEST_PATH}" ]; then
        print_error "Test executable not found: ${test_exec}"
        print_warning "Please rebuild tests using: ./scripts/build_schwab_tests.sh"
        TEST_RESULTS["${test_exec}"]="NOT_FOUND"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo ""
        continue
    fi

    # Check if executable is up-to-date (not older than 1 day)
    if [ -n "$(find "${TEST_PATH}" -mtime +1 2>/dev/null)" ]; then
        print_warning "${test_exec} is older than 1 day - consider rebuilding"
    fi

    # Run the test
    print_step "Running ${test_exec}..."
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Execute with colored output
    if "${TEST_PATH}" --gtest_color=yes 2>&1; then
        print_success "${test_exec} PASSED"
        TEST_RESULTS["${test_exec}"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "${test_exec} FAILED"
        TEST_RESULTS["${test_exec}"]="FAILED"
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

for test_exec in "${SELECTED_TESTS[@]}"; do
    result="${TEST_RESULTS[${test_exec}]}"

    # Get friendly name for display
    friendly_name="${test_exec#test_}"  # Remove test_ prefix
    friendly_name="${friendly_name//_/ }"  # Replace underscores with spaces

    case "${result}" in
        PASSED)
            echo -e "  ${CHECKMARK} ${friendly_name}: ${GREEN}PASSED${NC}"
            ;;
        FAILED)
            echo -e "  ${CROSSMARK} ${friendly_name}: ${RED}FAILED${NC}"
            ;;
        NOT_FOUND)
            echo -e "  ${CROSSMARK} ${friendly_name}: ${RED}NOT FOUND${NC}"
            ;;
        *)
            echo -e "  ${CROSSMARK} ${friendly_name}: ${YELLOW}UNKNOWN${NC}"
            ;;
    esac
done

echo ""
echo -e "${BOLD}Overall Statistics:${NC}"
echo -e "  Total Tests:  ${TOTAL_TESTS}"
echo -e "  Passed:       ${GREEN}${PASSED_TESTS}${NC}"
echo -e "  Failed:       ${RED}${FAILED_TESTS}${NC}"
echo ""

# Calculate success rate
if [ ${TOTAL_TESTS} -gt 0 ]; then
    SUCCESS_RATE=$((100 * PASSED_TESTS / TOTAL_TESTS))
    echo -e "  Success Rate: ${SUCCESS_RATE}%"
    echo ""
fi

# ============================================================================
# Final Status
# ============================================================================

if [ ${FAILED_TESTS} -eq 0 ]; then
    print_header "✓ All Tests Passed"
    echo -e "${GREEN}${BOLD}SUCCESS: All selected tests passed!${NC}"
    echo ""
    exit 0
else
    print_header "✗ Some Tests Failed"
    echo -e "${RED}${BOLD}FAILURE: ${FAILED_TESTS} test(s) failed${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Review test output above for specific failures"
    echo "  2. Run individual tests for detailed debugging:"
    for test_exec in "${SELECTED_TESTS[@]}"; do
        if [ "${TEST_RESULTS[${test_exec}]}" != "PASSED" ]; then
            echo "     ${BIN_DIR}/${test_exec} --gtest_filter=*"
        fi
    done
    echo "  3. Rebuild with full output:"
    echo "     ./scripts/build_schwab_tests.sh clean"
    echo ""
    exit 1
fi
