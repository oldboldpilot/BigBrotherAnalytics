#!/bin/bash
# BigBrotherAnalytics - Pre-Build clang-tidy Check
# Runs comprehensive clang-tidy analysis before every build
#
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-08
#
# This script is called automatically by CMake before building

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  clang-tidy Pre-Build Validation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Find all C++ source files (exclude tests and external)
CPP_FILES=$(find src -name "*.cpp" -o -name "*.cppm" | grep -v "test\|external\|third_party" || true)

if [ -z "$CPP_FILES" ]; then
    echo -e "${GREEN}✅ No C++ files to check${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$CPP_FILES" | wc -l)
echo "Files to check: $FILE_COUNT"
echo ""

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo -e "${RED}❌ clang-tidy not found!${NC}"
    echo ""
    echo "clang-tidy is REQUIRED for building BigBrotherAnalytics."
    echo ""
    echo "Install:"
    echo "  Ubuntu/Debian: sudo apt-get install clang-tidy"
    echo "  macOS: brew install llvm"
    echo "  RHEL/Fedora: sudo dnf install clang-tools-extra"
    echo ""
    exit 1
fi

echo -e "${BLUE}Running clang-tidy with comprehensive checks:${NC}"
echo "  - cppcoreguidelines-* (C++ Core Guidelines)"
echo "  - cert-* (CERT C++ Secure Coding)"
echo "  - concurrency-* (Thread safety, race conditions)"
echo "  - performance-* (Optimization opportunities)"
echo "  - portability-* (Cross-platform compatibility)"
echo "  - openmp-* (OpenMP parallelization safety)"
echo "  - mpi-* (MPI message passing correctness)"
echo ""
echo "Note: Module import errors ignored (false positives before compilation)"
echo ""

errors=0
warnings=0
files_checked=0

# Run clang-tidy on each file
for file in $CPP_FILES; do
    echo "Checking: $file"

    output=$(clang-tidy "$file" -p=./build -- -std=c++23 -I./src 2>&1 || true)

    # Filter out module import errors (false positives before modules are built)
    filtered_output=$(echo "$output" | grep -v "module.*not found" || true)

    file_errors=$(echo "$filtered_output" | grep -c "error:" || true)
    file_warnings=$(echo "$filtered_output" | grep -c "warning:" || true)

    if [ "$file_errors" -gt 0 ]; then
        echo -e "${RED}  ❌ $file_errors errors${NC}"
        echo "$filtered_output" | grep "error:" | head -3
        errors=$((errors + file_errors))
    fi

    if [ "$file_warnings" -gt 10 ]; then
        echo -e "${YELLOW}  ⚠️  $file_warnings warnings${NC}"
        warnings=$((warnings + file_warnings))
    fi

    files_checked=$((files_checked + 1))
done

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  clang-tidy Results${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Files checked: $files_checked"
echo "Total errors: $errors"
echo "Total warnings: $warnings"
echo ""

if [ $errors -gt 0 ]; then
    echo -e "${RED}❌ BUILD BLOCKED: $errors clang-tidy errors found${NC}"
    echo ""
    echo "Fix all errors before building."
    echo "See .clang-tidy for configuration details."
    echo ""
    echo "To bypass (NOT RECOMMENDED):"
    echo "  export SKIP_CLANG_TIDY=1"
    echo "  ninja"
    exit 1
elif [ $warnings -gt 50 ]; then
    echo -e "${YELLOW}⚠️  Warning: $warnings clang-tidy warnings${NC}"
    echo "Consider fixing warnings for better code quality"
    echo ""
    echo -e "${GREEN}✅ clang-tidy passed (no errors)${NC}"
    exit 0
else
    echo -e "${GREEN}✅ clang-tidy passed (no errors, low warnings)${NC}"
    exit 0
fi
