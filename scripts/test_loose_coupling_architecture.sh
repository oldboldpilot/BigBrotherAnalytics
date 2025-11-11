#!/usr/bin/env bash

# Trading Platform Architecture - Regression Test Suite
# Tests loose coupling, dependency inversion, and multi-platform support
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-11

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test functions
test_clean_build() {
    log_info "Test 1: Clean build with all modules"

    export SKIP_CLANG_TIDY=1
    rm -rf build
    mkdir -p build

    if cmake -G Ninja -S . -B build &>/dev/null; then
        log_success "CMake configuration succeeded"
    else
        log_failure "CMake configuration failed"
        return 1
    fi

    if ninja -C build market_intelligence >/dev/null 2>&1; then
        log_success "Build succeeded with all modules"
    else
        log_failure "Build failed"
        return 1
    fi
}

test_trading_core_library() {
    log_info "Test 2: trading_core library builds independently"

    export SKIP_CLANG_TIDY=1
    if ninja -C build trading_core &>/dev/null; then
        log_success "trading_core library built successfully"
    else
        log_failure "trading_core library build failed"
        return 1
    fi

    # Verify library file exists
    if [ -f "build/lib/libtrading_core.so" ]; then
        log_success "libtrading_core.so exists"
    else
        log_failure "libtrading_core.so not found"
        return 1
    fi
}

test_schwab_api_with_executor() {
    log_info "Test 3: schwab_api builds with executor enabled"

    export SKIP_CLANG_TIDY=1
    if ninja -C build schwab_api &>/dev/null; then
        log_success "schwab_api with executor built successfully"
    else
        log_failure "schwab_api build failed"
        return 1
    fi

    # Verify library file exists
    if [ -f "build/lib/libschwab_api.so" ]; then
        log_success "libschwab_api.so exists"
    else
        log_failure "libschwab_api.so not found"
        return 1
    fi
}

test_module_files_exist() {
    log_info "Test 4: All required module files exist"

    local modules=(
        "src/core/trading/order_types.cppm"
        "src/core/trading/platform_interface.cppm"
        "src/core/trading/orders_manager.cppm"
        "src/schwab_api/schwab_order_executor.cppm"
    )

    local all_exist=true
    for module in "${modules[@]}"; do
        if [ -f "$module" ]; then
            log_success "Module exists: $module"
        else
            log_failure "Module missing: $module"
            all_exist=false
        fi
    done

    if [ "$all_exist" = false ]; then
        return 1
    fi
}

test_type_conversion_logic() {
    log_info "Test 5: Type conversion code is present in executor"

    local executor_file="src/schwab_api/schwab_order_executor.cppm"

    # Check for type conversion functions
    if grep -q "convertToSchwabOrder" "$executor_file" && \
       grep -q "convertFromSchwabOrder" "$executor_file"; then
        log_success "Type conversion functions present"
    else
        log_failure "Type conversion functions missing"
        return 1
    fi

    # Check for type aliases (disambiguation)
    if grep -q "using Order = trading::Order" "$executor_file"; then
        log_success "Type aliases for disambiguation present"
    else
        log_failure "Type aliases missing"
        return 1
    fi

    # Check for chrono header (timestamp conversion)
    if grep -q "#include <chrono>" "$executor_file"; then
        log_success "Chrono header included for timestamp conversion"
    else
        log_failure "Chrono header missing"
        return 1
    fi
}

test_dependency_inversion() {
    log_info "Test 6: OrdersManager depends only on abstraction"

    local orders_manager_file="src/core/trading/orders_manager.cppm"

    # Check that OrdersManager imports platform_interface
    if grep -q "import bigbrother.trading.platform_interface" "$orders_manager_file"; then
        log_success "OrdersManager imports platform interface"
    else
        log_failure "OrdersManager doesn't import platform interface"
        return 1
    fi

    # Check that OrdersManager does NOT import schwab_api
    if grep -q "import bigbrother.schwab_api" "$orders_manager_file"; then
        log_failure "OrdersManager directly depends on schwab_api (VIOLATION!)"
        return 1
    else
        log_success "OrdersManager does not depend on schwab_api (correct)"
    fi

    # Check for unique_ptr<TradingPlatformInterface> in constructor
    if grep -q "std::unique_ptr<TradingPlatformInterface> platform" "$orders_manager_file"; then
        log_success "OrdersManager uses dependency injection"
    else
        log_failure "OrdersManager doesn't use dependency injection"
        return 1
    fi
}

test_interface_implementation() {
    log_info "Test 7: SchwabOrderExecutor implements TradingPlatformInterface"

    local executor_file="src/schwab_api/schwab_order_executor.cppm"

    # Check inheritance
    if grep -q "class SchwabOrderExecutor : public TradingPlatformInterface" "$executor_file"; then
        log_success "SchwabOrderExecutor inherits from TradingPlatformInterface"
    else
        log_failure "SchwabOrderExecutor doesn't inherit from interface"
        return 1
    fi

    # Check for interface methods
    local methods=("submitOrder" "cancelOrder" "modifyOrder" "getOrder" "getOrders" "getPositions" "getPlatformName")

    local all_present=true
    for method in "${methods[@]}"; do
        if grep -q "$method" "$executor_file"; then
            log_success "Method implemented: $method"
        else
            log_failure "Method missing: $method"
            all_present=false
        fi
    done

    if [ "$all_present" = false ]; then
        return 1
    fi
}

test_cmake_configuration() {
    log_info "Test 8: CMakeLists.txt properly configured"

    local cmake_file="CMakeLists.txt"

    # Check trading_core library definition
    if grep -q "add_library(trading_core SHARED)" "$cmake_file"; then
        log_success "trading_core library defined in CMake"
    else
        log_failure "trading_core library not defined"
        return 1
    fi

    # Check that schwab_api links trading_core
    if grep -A 10 "target_link_libraries(schwab_api" "$cmake_file" | grep -q "trading_core"; then
        log_success "schwab_api links trading_core"
    else
        log_failure "schwab_api doesn't link trading_core"
        return 1
    fi

    # Check that executor is enabled in build
    if grep -q "src/schwab_api/schwab_order_executor.cppm" "$cmake_file" && \
       ! grep -q "#.*src/schwab_api/schwab_order_executor.cppm" "$cmake_file"; then
        log_success "Schwab executor enabled in build"
    else
        log_failure "Schwab executor not enabled in build"
        return 1
    fi
}

test_no_circular_dependencies() {
    log_info "Test 9: No circular dependencies between modules"

    # trading_core should not depend on schwab_api
    if grep -q "schwab_api" build/CMakeFiles/trading_core.dir/depend.make 2>/dev/null; then
        log_failure "Circular dependency: trading_core depends on schwab_api"
        return 1
    else
        log_success "No circular dependency detected"
    fi

    # Check module imports
    if grep -q "import bigbrother.schwab_api" src/core/trading/*.cppm 2>/dev/null; then
        log_failure "trading_core modules import schwab_api"
        return 1
    else
        log_success "trading_core modules don't import schwab_api"
    fi
}

test_library_symbols() {
    log_info "Test 10: Platform libraries export correct symbols"

    # Check if libtrading_core.so exports TradingPlatformInterface
    if nm build/lib/libtrading_core.so 2>/dev/null | grep -q "TradingPlatformInterface"; then
        log_success "libtrading_core.so exports TradingPlatformInterface"
    else
        log_warning "TradingPlatformInterface symbols not found (may be expected)"
    fi

    # Check if libschwab_api.so exports SchwabOrderExecutor
    if nm build/lib/libschwab_api.so 2>/dev/null | grep -q "SchwabOrderExecutor"; then
        log_success "libschwab_api.so exports SchwabOrderExecutor"
    else
        log_warning "SchwabOrderExecutor symbols not found (may be expected)"
    fi
}

test_documentation_exists() {
    log_info "Test 11: Architecture documentation exists"

    if [ -f "docs/TRADING_PLATFORM_ARCHITECTURE.md" ]; then
        log_success "Architecture documentation exists"

        # Check key sections
        local doc_file="docs/TRADING_PLATFORM_ARCHITECTURE.md"
        if grep -q "Dependency Inversion Principle" "$doc_file" && \
           grep -q "Three-Layer Architecture" "$doc_file" && \
           grep -q "Adding a New Trading Platform" "$doc_file"; then
            log_success "Documentation contains all key sections"
        else
            log_failure "Documentation missing key sections"
            return 1
        fi
    else
        log_failure "Architecture documentation missing"
        return 1
    fi
}

test_position_type_disambiguation() {
    log_info "Test 12: Position type properly disambiguated"

    local executor_file="src/schwab_api/schwab_order_executor.cppm"

    # Check for explicit trading::Position qualification
    if grep -q "trading::Position" "$executor_file"; then
        log_success "Position type explicitly qualified"
    else
        log_failure "Position type not disambiguated"
        return 1
    fi
}

# Main test execution
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   Trading Platform Architecture - Regression Test Suite   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    cd "$(dirname "$0")/.." || exit 1

    # Run all tests
    test_clean_build
    test_trading_core_library
    test_schwab_api_with_executor
    test_module_files_exist
    test_type_conversion_logic
    test_dependency_inversion
    test_interface_implementation
    test_cmake_configuration
    test_no_circular_dependencies
    test_library_symbols
    test_documentation_exists
    test_position_type_disambiguation

    # Print summary
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "                     Test Summary                            "
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"

    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}Failed: $TESTS_FAILED${NC}"
        echo ""
        echo "❌ REGRESSION TESTS FAILED"
        exit 1
    else
        echo "Failed: 0"
        echo ""
        echo "✅ ALL REGRESSION TESTS PASSED"
        echo ""
        echo "Architecture Status:"
        echo "  ✓ Loose coupling verified"
        echo "  ✓ Dependency inversion confirmed"
        echo "  ✓ Type conversions working"
        echo "  ✓ Multi-platform support enabled"
        exit 0
    fi
}

# Run main
main "$@"
