#!/bin/bash
#
# BigBrotherAnalytics - Token Receiver Demo Script
#
# This script automates testing of the token receiver module.
# It starts the receiver, sends test tokens, and monitors the results.
#
# Usage:
#   ./token_receiver_demo.sh         # Interactive demo
#   ./token_receiver_demo.sh --auto  # Automated test
#
# Author: Olumuyiwa Oluwasanmi
# Date: November 13, 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/muyiwa/Development/BigBrotherAnalytics"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"
LOGS_DIR="${PROJECT_ROOT}/logs"
SOCKET_PATH="/tmp/bigbrother_token.sock"

# Functions
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"

    # Check if receiver binary exists
    if [ ! -f "${BIN_DIR}/test_token_receiver" ]; then
        print_error "test_token_receiver not found!"
        echo "Please build it first:"
        echo "  cd ${BUILD_DIR}"
        echo "  cmake --build . --target test_token_receiver"
        exit 1
    fi
    print_success "test_token_receiver binary found"

    # Check if Python script exists
    if [ ! -f "${PROJECT_ROOT}/scripts/send_test_token.py" ]; then
        print_error "send_test_token.py not found!"
        exit 1
    fi
    print_success "send_test_token.py script found"

    # Check if nc (netcat) is available
    if ! command -v nc &> /dev/null; then
        print_error "nc (netcat) not found!"
        echo "Please install netcat:"
        echo "  sudo dnf install nmap-ncat  # Fedora/RHEL"
        echo "  sudo apt install netcat     # Debian/Ubuntu"
        exit 1
    fi
    print_success "netcat available"

    # Create logs directory if it doesn't exist
    mkdir -p "${LOGS_DIR}"
    print_success "Logs directory ready: ${LOGS_DIR}"

    echo
}

cleanup() {
    print_info "Cleaning up..."

    # Kill receiver if running
    if [ -n "${RECEIVER_PID}" ]; then
        kill ${RECEIVER_PID} 2>/dev/null || true
        wait ${RECEIVER_PID} 2>/dev/null || true
    fi

    # Remove socket file
    rm -f "${SOCKET_PATH}"

    print_success "Cleanup complete"
}

trap cleanup EXIT

start_receiver() {
    print_header "Starting Token Receiver"

    # Clean up old socket
    rm -f "${SOCKET_PATH}"

    # Start receiver in background
    "${BIN_DIR}/test_token_receiver" > "${LOGS_DIR}/receiver_test.log" 2>&1 &
    RECEIVER_PID=$!

    # Wait for receiver to start
    sleep 2

    # Check if receiver is still running
    if ! kill -0 ${RECEIVER_PID} 2>/dev/null; then
        print_error "Receiver failed to start!"
        cat "${LOGS_DIR}/receiver_test.log"
        exit 1
    fi

    # Check if socket exists
    if [ ! -S "${SOCKET_PATH}" ]; then
        print_error "Socket file not created!"
        print_info "Waiting a bit longer..."
        sleep 3
        if [ ! -S "${SOCKET_PATH}" ]; then
            print_error "Socket still not available!"
            exit 1
        fi
    fi

    print_success "Token Receiver started (PID: ${RECEIVER_PID})"
    print_info "Socket: ${SOCKET_PATH}"
    echo
}

test_unix_socket() {
    print_header "Test 1: Unix Domain Socket"

    local token="test_token_unix_$(date +%s)"

    print_info "Sending token via Unix socket..."
    echo "${token}" | nc -U "${SOCKET_PATH}"

    sleep 1

    # Check logs
    if grep -q "${token}" "${LOGS_DIR}/receiver_test.log"; then
        print_success "Token received via Unix socket!"
    else
        print_error "Token not found in logs!"
        return 1
    fi

    echo
}

test_tcp_socket() {
    print_header "Test 2: TCP Socket"

    local token="test_token_tcp_$(date +%s)"

    print_info "Sending token via TCP socket..."
    echo "${token}" | nc localhost 9999

    sleep 1

    # Check logs
    if grep -q "${token}" "${LOGS_DIR}/receiver_test.log"; then
        print_success "Token received via TCP socket!"
    else
        print_error "Token not found in logs!"
        return 1
    fi

    echo
}

test_python_script() {
    print_header "Test 3: Python Test Script"

    local token="test_token_python_$(date +%s)"

    print_info "Sending token via Python script..."
    "${PROJECT_ROOT}/scripts/send_test_token.py" "${token}"

    sleep 1

    # Check logs
    if grep -q "${token}" "${LOGS_DIR}/receiver_test.log"; then
        print_success "Token received via Python script!"
    else
        print_error "Token not found in logs!"
        return 1
    fi

    echo
}

test_continuous_mode() {
    print_header "Test 4: Continuous Mode"

    print_info "Sending 3 tokens with 2-second intervals..."

    "${PROJECT_ROOT}/scripts/send_test_token.py" --continuous --interval 2 > /dev/null 2>&1 &
    local sender_pid=$!

    # Wait for 3 tokens (6 seconds + buffer)
    sleep 7

    # Stop sender
    kill ${sender_pid} 2>/dev/null || true

    # Count tokens received
    local count=$(grep -c "Token received" "${LOGS_DIR}/receiver_test.log" || echo "0")

    if [ ${count} -ge 3 ]; then
        print_success "Continuous mode works! (${count} tokens received)"
    else
        print_error "Expected at least 3 tokens, got ${count}"
        return 1
    fi

    echo
}

show_statistics() {
    print_header "Statistics"

    local total_tokens=$(grep -c "Token received" "${LOGS_DIR}/receiver_test.log" || echo "0")
    local unix_tokens=$(grep -c "Unix socket" "${LOGS_DIR}/receiver_test.log" || echo "0")
    local tcp_tokens=$(grep -c "TCP socket" "${LOGS_DIR}/receiver_test.log" || echo "0")

    echo "Total tokens received:    ${total_tokens}"
    echo "Unix socket connections:  ${unix_tokens}"
    echo "TCP socket connections:   ${tcp_tokens}"
    echo

    print_info "View full logs:"
    echo "  cat ${LOGS_DIR}/receiver_test.log"
    echo
}

run_automated_tests() {
    check_dependencies
    start_receiver

    local failed=0

    test_unix_socket || ((failed++))
    test_tcp_socket || ((failed++))
    test_python_script || ((failed++))
    test_continuous_mode || ((failed++))

    show_statistics

    print_header "Test Results"

    if [ ${failed} -eq 0 ]; then
        print_success "All tests passed! 🎉"
        return 0
    else
        print_error "${failed} test(s) failed!"
        return 1
    fi
}

run_interactive_demo() {
    check_dependencies

    print_header "Interactive Token Receiver Demo"
    echo
    echo "This demo will:"
    echo "  1. Start the token receiver"
    echo "  2. Show you how to send tokens"
    echo "  3. Display received tokens in real-time"
    echo
    read -p "Press Enter to continue..."
    echo

    start_receiver

    print_header "Receiver is Running!"
    echo
    echo "You can now send tokens using:"
    echo
    echo "1. Unix socket (recommended):"
    echo "   ${GREEN}echo \"my_test_token\" | nc -U ${SOCKET_PATH}${NC}"
    echo
    echo "2. TCP socket:"
    echo "   ${GREEN}echo \"my_test_token\" | nc localhost 9999${NC}"
    echo
    echo "3. Python script:"
    echo "   ${GREEN}${PROJECT_ROOT}/scripts/send_test_token.py \"my_test_token\"${NC}"
    echo
    echo "4. Continuous mode:"
    echo "   ${GREEN}${PROJECT_ROOT}/scripts/send_test_token.py --continuous --interval 5${NC}"
    echo
    echo "Receiver logs: ${LOGS_DIR}/receiver_test.log"
    echo
    echo "Press Ctrl+C to stop the demo"
    echo

    # Tail logs in real-time
    tail -f "${LOGS_DIR}/receiver_test.log"
}

# Main
main() {
    if [ "$1" = "--auto" ]; then
        run_automated_tests
    else
        run_interactive_demo
    fi
}

main "$@"
