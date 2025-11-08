#!/bin/bash
# Test Script for BigBrotherAnalytics
# Run this after the build completes successfully

set -e

echo "=== BigBrotherAnalytics Test Runner ==="
echo ""

# Set library path
export LD_LIBRARY_PATH=/home/linuxbrew/.linuxbrew/Cellar/llvm/21.1.5/lib/x86_64-unknown-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH

# Navigate to build directory
cd /home/muyiwa/Development/BigBrotherAnalytics/build

echo "=== Running Tests ==="
ninja test

echo ""
echo "=== Test Results ==="
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed"
    exit 1
fi

echo ""
echo "=== Checking Executables ==="
ls -lh bin/

echo ""
echo "=== Checking Libraries ==="
ls -lh lib/

echo ""
echo "=== Build and Test Complete ==="
