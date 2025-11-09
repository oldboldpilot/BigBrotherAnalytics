#!/bin/bash
# Build and test employment signals module
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-09

set -e

echo "BigBrotherAnalytics - Employment Signals Test"
echo "=============================================="
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Build directory not found. Creating..."
    mkdir -p build
fi

cd build

echo "Configuring with CMake..."
SKIP_CLANG_TIDY=1 cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++

echo ""
echo "Building market_intelligence library..."
cmake --build . --target market_intelligence -j$(nproc)

echo ""
echo "Building test executable..."
/usr/local/bin/clang++ -std=c++23 -stdlib=libc++ \
    -fprebuilt-module-path=modules \
    -I../src \
    -c ../tests/test_employment_signals.cpp \
    -o test_employment_signals.o

/usr/local/bin/clang++ -std=c++23 -stdlib=libc++ \
    test_employment_signals.o \
    -L./lib -lmarket_intelligence -lutils \
    -lcurl -lpthread -lomp \
    -o test_employment_signals

echo ""
echo "Running test..."
echo ""
./test_employment_signals

echo ""
echo "Test completed!"
