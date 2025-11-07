#!/bin/bash

# Build script for BigBrotherAnalytics C++ components

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BUILD_TYPE="Release"
CLEAN=false
RUN_TESTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        debug|Debug|DEBUG)
            BUILD_TYPE="Debug"
            shift
            ;;
        clean)
            CLEAN=true
            shift
            ;;
        test)
            RUN_TESTS=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [debug|clean|test]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}BigBrotherAnalytics C++ Build Script${NC}"
echo "Build Type: $BUILD_TYPE"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

# Build
echo -e "${YELLOW}Building...${NC}"
NPROC=$(nproc)
echo "Using $NPROC parallel jobs"
make -j$NPROC

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    ctest --output-on-failure || {
        echo -e "${RED}Tests failed!${NC}"
        exit 1
    }
    echo -e "${GREEN}All tests passed!${NC}"
fi

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Python modules are available at:"
echo "  - src/correlation_engine/options_pricing_py.so"
echo "  - src/correlation_engine/correlation_engine_py.so"
echo "  - src/schwab_api/schwab_api_py.so"
echo ""
echo "To use in Python:"
echo "  uv run python -c 'from src.correlation_engine import options_pricing_py'"
