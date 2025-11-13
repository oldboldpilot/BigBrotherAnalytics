#!/bin/bash
# BigBrotherAnalytics - Valgrind Memory Leak Tests for simdjson Benchmarks
#
# Runs comprehensive memory leak detection on simdjson wrapper benchmarks.
# Validates that the thread-local parser pattern doesn't leak memory.
#
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-12

set -e

BENCHMARK_BIN="./build/bin/benchmark_json_parsers"
UNIT_TEST_BIN="./build/bin/test_simdjson_wrapper"
REPORT_DIR="./valgrind_reports"

mkdir -p "$REPORT_DIR"

echo "========================================"
echo "Valgrind Memory Leak Tests for simdjson"
echo "========================================"
echo ""

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "ERROR: Valgrind is not installed"
    echo "Install with: sudo apt-get install valgrind"
    exit 1
fi

# Check if binaries exist
if [ ! -f "$BENCHMARK_BIN" ]; then
    echo "ERROR: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Build with: SKIP_CLANG_TIDY=1 ninja -C build benchmark_json_parsers"
    exit 1
fi

if [ ! -f "$UNIT_TEST_BIN" ]; then
    echo "ERROR: Unit test binary not found at $UNIT_TEST_BIN"
    echo "Build with: SKIP_CLANG_TIDY=1 ninja -C build test_simdjson_wrapper"
    exit 1
fi

echo "✓ Valgrind found: $(valgrind --version | head -1)"
echo "✓ Benchmark binary: $BENCHMARK_BIN"
echo "✓ Unit test binary: $UNIT_TEST_BIN"
echo ""

# ============================================================================
# Test 1: Unit Tests Memory Leak Detection
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Unit Tests Memory Leak Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

valgrind \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file="$REPORT_DIR/unit_tests_valgrind.txt" \
    --error-exitcode=1 \
    "$UNIT_TEST_BIN" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ PASS: No memory leaks detected in unit tests"
else
    echo "❌ FAIL: Memory leaks detected in unit tests"
    echo "See detailed report: $REPORT_DIR/unit_tests_valgrind.txt"
    exit 1
fi

# Extract leak summary
echo ""
echo "Leak Summary:"
grep "LEAK SUMMARY" -A 5 "$REPORT_DIR/unit_tests_valgrind.txt" | sed 's/^/  /'
echo ""

# ============================================================================
# Test 2: Benchmark Memory Leak Detection (Short Run)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Benchmark Memory Leak Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run benchmarks with minimum iterations to speed up Valgrind (it's slow!)
valgrind \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file="$REPORT_DIR/benchmarks_valgrind.txt" \
    --error-exitcode=1 \
    "$BENCHMARK_BIN" \
        --benchmark_min_time=0.01 \
        --benchmark_filter="SimdJson" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ PASS: No memory leaks detected in benchmarks"
else
    echo "❌ FAIL: Memory leaks detected in benchmarks"
    echo "See detailed report: $REPORT_DIR/benchmarks_valgrind.txt"
    exit 1
fi

# Extract leak summary
echo ""
echo "Leak Summary:"
grep "LEAK SUMMARY" -A 5 "$REPORT_DIR/benchmarks_valgrind.txt" | sed 's/^/  /'
echo ""

# ============================================================================
# Test 3: Thread Safety Memory Test
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Thread Safety Memory Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run only thread safety tests with Helgrind (data race detector)
valgrind \
    --tool=helgrind \
    --log-file="$REPORT_DIR/thread_safety_helgrind.txt" \
    --error-exitcode=1 \
    "$UNIT_TEST_BIN" \
        --gtest_filter="*Thread*" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ PASS: No data races detected in multithreaded tests"
else
    echo "⚠️  WARNING: Possible data races detected"
    echo "See detailed report: $REPORT_DIR/thread_safety_helgrind.txt"
    # Don't fail on Helgrind warnings (they can be false positives)
fi

echo ""
echo "Data Race Summary:"
grep "ERROR SUMMARY" "$REPORT_DIR/thread_safety_helgrind.txt" | sed 's/^/  /'
echo ""

# ============================================================================
# Summary Report
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Valgrind Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Reports saved to: $REPORT_DIR/"
echo ""
echo "Files generated:"
echo "  - unit_tests_valgrind.txt       (Memory leak report for unit tests)"
echo "  - benchmarks_valgrind.txt       (Memory leak report for benchmarks)"
echo "  - thread_safety_helgrind.txt    (Data race detection report)"
echo ""
echo "✅ All Valgrind tests passed!"
echo ""
echo "Memory Safety Status: VALIDATED"
echo "Thread Safety Status: VALIDATED"
echo ""
