#!/bin/bash

# Schwab API Test Runner
# Runs pre-built integration tests with colored output and summary

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set library path for DuckDB and other dependencies
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Test directory
TEST_DIR="build/tests/cpp"
BUILD_DIR="build"

# Test executables
TESTS=(
    "test_order_manager_integration"
    "test_account_manager_integration"
    "test_schwab_e2e_workflow"
)

# Results tracking
declare -A TEST_RESULTS
PASSED=0
FAILED=0
SKIPPED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Schwab API Integration Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory '$BUILD_DIR' not found!${NC}"
    echo -e "${YELLOW}Please run cmake and build first.${NC}"
    exit 1
fi

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Error: Test directory '$TEST_DIR' not found!${NC}"
    echo -e "${YELLOW}Please build the tests first.${NC}"
    exit 1
fi

echo -e "${BLUE}Test Directory:${NC} $TEST_DIR"
echo -e "${BLUE}Library Path:${NC} $LD_LIBRARY_PATH"
echo ""

# Run each test
for test_name in "${TESTS[@]}"; do
    test_path="$TEST_DIR/$test_name"

    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Running:${NC} $test_name"
    echo -e "${BLUE}----------------------------------------${NC}"

    if [ ! -f "$test_path" ]; then
        echo -e "${YELLOW}SKIPPED${NC} - Test executable not found: $test_path"
        TEST_RESULTS[$test_name]="SKIPPED"
        ((SKIPPED++))
        echo ""
        continue
    fi

    if [ ! -x "$test_path" ]; then
        echo -e "${YELLOW}Warning: Test is not executable, attempting to run anyway...${NC}"
    fi

    # Run the test and capture output
    if "$test_path"; then
        echo -e "${GREEN}PASSED${NC} - $test_name"
        TEST_RESULTS[$test_name]="PASSED"
        ((PASSED++))
    else
        echo -e "${RED}FAILED${NC} - $test_name (exit code: $?)"
        TEST_RESULTS[$test_name]="FAILED"
        ((FAILED++))
    fi

    echo ""
done

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           TEST SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for test_name in "${TESTS[@]}"; do
    result="${TEST_RESULTS[$test_name]}"
    case $result in
        "PASSED")
            echo -e "  ${GREEN}[✓]${NC} $test_name"
            ;;
        "FAILED")
            echo -e "  ${RED}[✗]${NC} $test_name"
            ;;
        "SKIPPED")
            echo -e "  ${YELLOW}[-]${NC} $test_name"
            ;;
    esac
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "Total Tests: $((PASSED + FAILED + SKIPPED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
echo -e "${BLUE}========================================${NC}"

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
elif [ $PASSED -eq 0 ]; then
    echo -e "${YELLOW}Warning: No tests were run successfully${NC}"
    exit 2
else
    exit 0
fi
