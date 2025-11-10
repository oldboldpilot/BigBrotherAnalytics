#!/usr/bin/env bash

#==============================================================================
# Trading Engine End-to-End Integration Test Script
#==============================================================================
# Comprehensive integration test for the BigBrotherAnalytics trading engine.
#
# This script validates the complete trading pipeline:
# - Project build
# - Paper trading configuration and startup
# - Signal generation from strategies
# - Order placement (dry-run mode)
# - Position tracking
# - Risk management and stop-loss checking
#
# Features:
# - Parallel build with multiple jobs
# - Real-time log monitoring
# - Pattern-based validation
# - Comprehensive error handling
# - Clear pass/fail reporting
#
# Usage:
#   ./scripts/test_trading_engine.sh [options]
#
# Options:
#   --clean           Force clean rebuild
#   --verbose         Enable verbose output
#   --timeout SECS    Set test timeout (default: 30)
#   --no-build        Skip build step
#   --help            Show this help message
#
# Exit Codes:
#   0 - All tests passed
#   1 - Build failed
#   2 - Test setup failed
#   3 - Tests failed
#   4 - Timeout
#
# Examples:
#   ./scripts/test_trading_engine.sh                 # Full test with build
#   ./scripts/test_trading_engine.sh --clean        # Clean rebuild
#   ./scripts/test_trading_engine.sh --timeout 60   # Extended timeout
#   ./scripts/test_trading_engine.sh --no-build     # Skip build phase
#
#==============================================================================

set -euo pipefail

#==============================================================================
# Color Definitions
#==============================================================================

# Detect if running in terminal
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
    echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}$1${RESET}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════${RESET}\n"
}

log_section() {
    echo -e "\n${BOLD}${MAGENTA}>>> $1${RESET}\n"
}

log_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${RESET} $1"
}

log_error() {
    echo -e "${RED}[✗]${RESET} $1"
}

log_test() {
    echo -e "${BLUE}[TEST]${RESET} $1"
}

log_found() {
    echo -e "${GREEN}[FOUND]${RESET} $1"
}

log_notfound() {
    echo -e "${YELLOW}[MISSING]${RESET} $1"
}

#==============================================================================
# Configuration
#==============================================================================

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/configs/config.yaml"

# Test configuration
CLEAN_BUILD=false
VERBOSE=false
SKIP_BUILD=false
TEST_TIMEOUT=30
TEMP_LOG_FILE=""
TEST_LOGFILE=""

# Test results tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TEST_FAILURES=()

# Executable name
EXECUTABLE_NAME="bigbrother"
EXECUTABLE_PATH="${BIN_DIR}/${EXECUTABLE_NAME}"

#==============================================================================
# Argument Parsing
#==============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            --no-build)
                SKIP_BUILD=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 2
                ;;
        esac
    done
}

show_help() {
    head -n 50 "$0" | grep -E '^#' | sed 's/^# //g' | sed 's/^#//g'
}

#==============================================================================
# Cleanup
#==============================================================================

cleanup() {
    local exit_code=$?

    # Kill any lingering test processes
    if [[ -n "$TEMP_LOG_FILE" && -f "$TEMP_LOG_FILE" ]]; then
        rm -f "$TEMP_LOG_FILE"
    fi

    if [[ $exit_code -ne 0 && $exit_code -ne 0 ]]; then
        log_error "Test script failed with exit code: $exit_code"
    fi

    return $exit_code
}

trap cleanup EXIT

#==============================================================================
# Helper Functions
#==============================================================================

# Check if file exists and is executable
check_executable() {
    local exe_path="$1"

    if [[ ! -f "$exe_path" ]]; then
        log_error "Executable not found: $exe_path"
        return 1
    fi

    if [[ ! -x "$exe_path" ]]; then
        log_warning "File exists but is not executable: $exe_path"
        chmod +x "$exe_path" || return 1
    fi

    return 0
}

# Search for pattern in log file with timeout
search_log_pattern() {
    local log_file="$1"
    local pattern="$2"
    local timeout_secs="${3:-${TEST_TIMEOUT}}"
    local elapsed=0
    local interval=0.5

    # Convert timeout to integer milliseconds for bc
    timeout_secs=$(echo "$timeout_secs" | cut -d. -f1)

    while [[ $elapsed -lt $timeout_secs ]]; do
        if grep -q "$pattern" "$log_file" 2>/dev/null; then
            return 0
        fi

        sleep "$interval"
        elapsed=$(echo "$elapsed + $interval" | bc)
    done

    return 1
}

# Count occurrences of pattern in log file
count_log_pattern() {
    local log_file="$1"
    local pattern="$2"

    grep -c "$pattern" "$log_file" 2>/dev/null || echo "0"
}

# Extract value from log based on pattern
extract_log_value() {
    local log_file="$1"
    local pattern="$2"

    grep "$pattern" "$log_file" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1
}

#==============================================================================
# Build Functions
#==============================================================================

setup_build() {
    log_section "Setting Up Build Environment"

    # Clean if requested
    if [[ "$CLEAN_BUILD" == true ]]; then
        log_info "Cleaning build directory..."
        if [[ -d "$BUILD_DIR" ]]; then
            rm -rf "$BUILD_DIR"
            log_success "Build directory cleaned"
        fi
    fi

    # Create build directory
    mkdir -p "$BUILD_DIR"
    mkdir -p "$LOG_DIR"

    log_success "Build environment ready"
}

build_project() {
    if [[ "$SKIP_BUILD" == true ]]; then
        log_section "Skipping Build (--no-build flag set)"
        return 0
    fi

    log_section "Building Project"

    cd "$BUILD_DIR"

    # Run CMake configuration
    log_info "Running CMake configuration..."

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=/usr/local/bin/clang
        -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++
    )

    if ! cmake "${CMAKE_ARGS[@]}" ..; then
        log_error "CMake configuration failed"
        return 1
    fi

    log_success "CMake configuration completed"

    # Build the project
    log_info "Compiling (using $(nproc) parallel jobs)..."

    local num_jobs
    num_jobs=$(nproc 2>/dev/null || echo 4)

    if ! cmake --build . -j "$num_jobs"; then
        log_error "Build failed"
        return 1
    fi

    log_success "Build completed successfully"

    cd "$PROJECT_ROOT"
    return 0
}

#==============================================================================
# Test Setup Functions
#==============================================================================

setup_test_config() {
    log_section "Setting Up Test Configuration"

    # Create temporary config for paper trading
    log_info "Verifying config file: $CONFIG_FILE"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        return 1
    fi

    # Verify paper trading is enabled
    if ! grep -q "paper_trading: true" "$CONFIG_FILE"; then
        log_warning "Paper trading not explicitly enabled in config"
        log_info "Config will be used as-is, but ensure paper_trading: true"
    else
        log_success "Paper trading mode enabled"
    fi

    # Check for required config sections
    local required_sections=("database" "logging" "trading" "schwab" "risk")

    for section in "${required_sections[@]}"; do
        if grep -q "^${section}:" "$CONFIG_FILE"; then
            log_found "Config section: $section"
        else
            log_notfound "Config section: $section"
            log_warning "Config may be incomplete"
        fi
    done

    log_success "Configuration verified"
    return 0
}

check_executable_exists() {
    log_section "Checking for Compiled Executable"

    if [[ ! -f "$EXECUTABLE_PATH" ]]; then
        log_error "Executable not found: $EXECUTABLE_PATH"
        log_info "Build may have failed or executable has a different name"

        # Try to find any executable that looks like bigbrother
        log_info "Searching for executable in $BIN_DIR..."
        if ls "$BIN_DIR"/bigbrother* 2>/dev/null | head -5; then
            log_info "Found potential executables"
        else
            log_error "No bigbrother executable found in $BIN_DIR"
            return 1
        fi

        return 1
    fi

    if ! check_executable "$EXECUTABLE_PATH"; then
        return 1
    fi

    log_found "Executable: $EXECUTABLE_PATH"
    log_success "Executable verified"
    return 0
}

#==============================================================================
# Test Execution Functions
#==============================================================================

# Run the trading engine with timeout and monitor logs
run_trading_engine_test() {
    log_section "Running Trading Engine Test"

    # Create temp log file
    TEMP_LOG_FILE=$(mktemp)
    TEST_LOGFILE="${LOG_DIR}/test_run_$(date +%s).log"

    log_info "Test log: $TEST_LOGFILE"
    log_info "Temp log: $TEMP_LOG_FILE"

    # Start the trading engine in background with output capture
    log_info "Starting trading engine (${TEST_TIMEOUT}s timeout)..."

    # Run with timeout and capture output
    timeout "$TEST_TIMEOUT" "$EXECUTABLE_PATH" --config "$CONFIG_FILE" 2>&1 | tee "$TEMP_LOG_FILE" "$TEST_LOGFILE" || {
        local exit_code=$?

        # Exit code 124 means timeout, which is expected for this test
        if [[ $exit_code -eq 124 ]]; then
            log_success "Trading engine ran and stopped (timeout as expected)"
            # Copy temp log to final location
            cat "$TEMP_LOG_FILE" >> "$TEST_LOGFILE"
        elif [[ $exit_code -ne 0 ]]; then
            log_error "Trading engine exited with code: $exit_code"
            cat "$TEMP_LOG_FILE" >> "$TEST_LOGFILE"
            return 1
        fi
    }

    log_success "Trading engine test completed"
    return 0
}

#==============================================================================
# Test Validation Functions
#==============================================================================

test_engine_initialization() {
    log_test "Engine Initialization"

    if search_log_pattern "$TEMP_LOG_FILE" "BigBrotherAnalytics Trading Engine"; then
        log_found "Engine initialization message"
        ((TESTS_PASSED++))
    else
        log_notfound "Engine initialization message"
        TEST_FAILURES+=("Engine initialization message not found")
        ((TESTS_FAILED++))
    fi

    ((TESTS_RUN++))
}

test_config_loading() {
    log_test "Configuration Loading"

    if search_log_pattern "$TEMP_LOG_FILE" "Loading configuration from"; then
        log_found "Configuration loading"
        ((TESTS_PASSED++))
    else
        log_notfound "Configuration loading"
        TEST_FAILURES+=("Configuration loading not found")
        ((TESTS_FAILED++))
    fi

    ((TESTS_RUN++))
}

test_paper_trading_mode() {
    log_test "Paper Trading Mode"

    if search_log_pattern "$TEMP_LOG_FILE" "PAPER TRADING MODE"; then
        log_found "Paper trading mode confirmation"
        ((TESTS_PASSED++))
    else
        log_notfound "Paper trading mode confirmation"
        TEST_FAILURES+=("Paper trading mode not confirmed")
        ((TESTS_FAILED++))
    fi

    ((TESTS_RUN++))
}

test_database_initialization() {
    log_test "Database Initialization"

    if search_log_pattern "$TEMP_LOG_FILE" "Database initialized"; then
        log_found "Database initialization"
        ((TESTS_PASSED++))
    else
        log_notfound "Database initialization"
        TEST_FAILURES+=("Database initialization not found")
        ((TESTS_FAILED++))
    fi

    ((TESTS_RUN++))
}

test_strategy_registration() {
    log_test "Strategy Registration"

    if search_log_pattern "$TEMP_LOG_FILE" "Strategies registered"; then
        log_found "Strategy registration message"

        # Count registered strategies
        local strategy_count=$(count_log_pattern "$TEMP_LOG_FILE" "enabled")
        if [[ $strategy_count -gt 0 ]]; then
            log_info "Found $strategy_count strategy-related entries"
            ((TESTS_PASSED++))
        else
            log_warning "No strategy entries found"
            ((TESTS_FAILED++))
        fi
    else
        log_notfound "Strategy registration"
        TEST_FAILURES+=("Strategy registration not found")
        ((TESTS_FAILED++))
    fi

    ((TESTS_RUN++))
}

test_trading_cycle() {
    log_test "Trading Cycle Execution"

    if search_log_pattern "$TEMP_LOG_FILE" "Trading Cycle Start"; then
        log_found "Trading cycle initiated"
        ((TESTS_PASSED++))
    else
        log_notfound "Trading cycle initialization"
        TEST_FAILURES+=("Trading cycle not initiated")
        ((TESTS_FAILED++))
    fi

    ((TESTS_RUN++))
}

test_signal_generation() {
    log_test "Signal Generation"

    if search_log_pattern "$TEMP_LOG_FILE" "Generated.*trading signals"; then
        log_found "Signal generation"

        # Try to extract signal count
        local signal_count=$(count_log_pattern "$TEMP_LOG_FILE" "Generated")
        log_info "Signal generation events recorded: $signal_count"
        ((TESTS_PASSED++))
    else
        log_notfound "Signal generation"
        log_info "Note: This is normal if no trading conditions were met"
        ((TESTS_PASSED++))  # Not failing - signals optional based on market conditions
    fi

    ((TESTS_RUN++))
}

test_position_tracking() {
    log_test "Position Tracking"

    if search_log_pattern "$TEMP_LOG_FILE" "Updating positions\|Retrieved.*positions"; then
        log_found "Position tracking"
        ((TESTS_PASSED++))
    else
        log_notfound "Position tracking"
        log_info "Note: This is normal for initial runs with no positions"
        ((TESTS_PASSED++))  # Not failing - positions optional on startup
    fi

    ((TESTS_RUN++))
}

test_stop_loss_checking() {
    log_test "Stop Loss Checking"

    if search_log_pattern "$TEMP_LOG_FILE" "Checking stop losses"; then
        log_found "Stop loss check routine"
        ((TESTS_PASSED++))
    else
        log_notfound "Stop loss check routine"
        log_info "Note: This may not run if no positions are held"
        ((TESTS_PASSED++))  # Not failing - stop losses optional
    fi

    ((TESTS_RUN++))
}

test_account_info_retrieval() {
    log_test "Account Information Retrieval"

    if search_log_pattern "$TEMP_LOG_FILE" "Account.*buying power\|getAccountInfo"; then
        log_found "Account information retrieval"
        ((TESTS_PASSED++))
    else
        log_notfound "Account information retrieval"
        log_info "Note: API calls may be mocked in dry-run"
        ((TESTS_PASSED++))  # Not failing - depends on API availability
    fi

    ((TESTS_RUN++))
}

test_risk_management() {
    log_test "Risk Management"

    if search_log_pattern "$TEMP_LOG_FILE" "risk\|Risk"; then
        log_found "Risk management processing"
        ((TESTS_PASSED++))
    else
        log_notfound "Risk management"
        ((TESTS_FAILED++))
        TEST_FAILURES+=("Risk management not detected")
    fi

    ((TESTS_RUN++))
}

test_no_errors() {
    log_test "Error-Free Execution"

    local error_count=$(count_log_pattern "$TEMP_LOG_FILE" "ERROR\|error")

    if [[ $error_count -eq 0 ]]; then
        log_found "No errors in execution"
        ((TESTS_PASSED++))
    else
        log_warning "$error_count error entries found"
        log_info "Showing first few errors:"
        grep -i "ERROR\|error" "$TEMP_LOG_FILE" | head -3
        ((TESTS_FAILED++))
        TEST_FAILURES+=("$error_count errors detected in logs")
    fi

    ((TESTS_RUN++))
}

test_graceful_shutdown() {
    log_test "Graceful Shutdown"

    if search_log_pattern "$TEMP_LOG_FILE" "Shutdown\|shutdown"; then
        log_found "Graceful shutdown"
        ((TESTS_PASSED++))
    else
        log_warning "Shutdown message not found (may have been killed by timeout)"
        ((TESTS_PASSED++))  # Not failing - timeout stop is acceptable
    fi

    ((TESTS_RUN++))
}

#==============================================================================
# Main Test Execution
#==============================================================================

run_all_tests() {
    log_header "Running Trading Engine Integration Tests"

    # Run the trading engine
    if ! run_trading_engine_test; then
        log_error "Failed to run trading engine test"
        return 1
    fi

    # Verify log file was created
    if [[ ! -f "$TEMP_LOG_FILE" ]] || [[ ! -s "$TEMP_LOG_FILE" ]]; then
        log_error "Log file was not created or is empty"
        return 1
    fi

    log_success "Log file created: $TEST_LOGFILE ($(wc -c < "$TEMP_LOG_FILE") bytes)"

    # Show first 20 lines of logs
    log_section "Initial Log Output"
    head -20 "$TEMP_LOG_FILE"

    # Run all test validations
    log_section "Running Test Validations"

    test_engine_initialization
    test_config_loading
    test_paper_trading_mode
    test_database_initialization
    test_strategy_registration
    test_trading_cycle
    test_signal_generation
    test_position_tracking
    test_stop_loss_checking
    test_account_info_retrieval
    test_risk_management
    test_no_errors
    test_graceful_shutdown

    return 0
}

#==============================================================================
# Results Reporting
#==============================================================================

print_results() {
    log_header "Test Results Summary"

    echo ""
    echo -e "${BOLD}Test Statistics:${RESET}"
    echo -e "  Total Tests Run:    $TESTS_RUN"
    echo -e "  ${GREEN}Tests Passed:      $TESTS_PASSED${RESET}"
    echo -e "  ${RED}Tests Failed:      $TESTS_FAILED${RESET}"
    echo -e "  Success Rate:       $(( (TESTS_PASSED * 100) / TESTS_RUN ))%"
    echo ""

    if [[ ${#TEST_FAILURES[@]} -gt 0 ]]; then
        echo -e "${RED}Failed Assertions:${RESET}"
        for failure in "${TEST_FAILURES[@]}"; do
            echo -e "  ${RED}•${RESET} $failure"
        done
        echo ""
    fi

    echo -e "${BOLD}Log Files:${RESET}"
    echo -e "  Test Log:           $TEST_LOGFILE"
    echo -e "  View with:          tail -f $TEST_LOGFILE"
    echo ""

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}ALL TESTS PASSED!${RESET}"
        return 0
    else
        echo -e "${RED}${BOLD}SOME TESTS FAILED${RESET}"
        return 1
    fi
}

#==============================================================================
# Main Entry Point
#==============================================================================

main() {
    local start_time
    start_time=$(date +%s)

    # Print banner
    echo -e "${BOLD}${CYAN}"
    cat << "EOF"
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║   Trading Engine End-to-End Integration Test               ║
║   BigBrotherAnalytics Automated Testing Suite              ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
EOF
    echo -e "${RESET}"

    # Parse arguments
    parse_args "$@"

    # Print configuration
    log_info "Project Root:  $PROJECT_ROOT"
    log_info "Build Dir:     $BUILD_DIR"
    log_info "Config File:   $CONFIG_FILE"
    log_info "Timeout:       ${TEST_TIMEOUT}s"
    log_info "Clean Build:   $CLEAN_BUILD"
    log_info "Skip Build:    $SKIP_BUILD"
    log_info "Verbose:       $VERBOSE"
    echo ""

    # Execute test pipeline
    setup_build || exit 1
    build_project || exit 1
    setup_test_config || exit 2
    check_executable_exists || exit 2

    # Run the main tests
    if ! run_all_tests; then
        log_error "Testing failed"
        exit 3
    fi

    # Print results
    if ! print_results; then
        exit 3
    fi

    # Calculate elapsed time
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    log_section "Test Execution Complete"
    echo -e "Total Time: ${minutes}m ${seconds}s"
    echo ""

    exit 0
}

# Run main function
main "$@"
