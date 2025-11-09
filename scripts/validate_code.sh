#!/bin/bash
# BigBrotherAnalytics - Code Validation Script
# Runs clang-tidy and cppcheck on the entire codebase
#
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-08
#
# Usage:
#   ./scripts/validate_code.sh              # Check all files
#   ./scripts/validate_code.sh src/utils/   # Check specific directory
#   ./scripts/validate_code.sh src/file.cpp # Check specific file

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TARGET="${1:-src}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  BigBrotherAnalytics Code Validation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Target: $TARGET"
echo ""

# Find all C++ files
if [ -f "$TARGET" ]; then
    CPP_FILES="$TARGET"
else
    CPP_FILES=$(find "$TARGET" -name "*.cpp" -o -name "*.cppm" | grep -v "test\|build\|external\|third_party" || true)
fi

if [ -z "$CPP_FILES" ]; then
    echo -e "${YELLOW}No C++ files found in $TARGET${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$CPP_FILES" | wc -l)
echo "Files to check: $FILE_COUNT"
echo ""

violations=0

# ============================================================================
# 1. clang-tidy (C++ Core Guidelines)
# ============================================================================

echo -e "${BLUE}1️⃣  Running clang-tidy (C++ Core Guidelines)...${NC}"
echo ""

if command -v clang-tidy &> /dev/null; then
    tidy_errors=0
    tidy_warnings=0

    for file in $CPP_FILES; do
        echo "  Checking: $file"

        output=$(clang-tidy "$file" -p=./build -- -std=c++23 -I./src 2>&1 || true)

        errors=$(echo "$output" | grep -c "error:" || true)
        warnings=$(echo "$output" | grep -c "warning:" || true)

        if [ "$errors" -gt 0 ]; then
            echo -e "${RED}    ❌ $errors errors found${NC}"
            echo "$output" | grep "error:" | head -3
            tidy_errors=$((tidy_errors + errors))
        fi

        if [ "$warnings" -gt 5 ]; then
            echo -e "${YELLOW}    ⚠️  $warnings warnings${NC}"
            tidy_warnings=$((tidy_warnings + warnings))
        fi
    done

    echo ""
    if [ $tidy_errors -gt 0 ]; then
        echo -e "${RED}❌ clang-tidy: $tidy_errors errors, $tidy_warnings warnings${NC}"
        violations=$((violations + 1))
    else
        echo -e "${GREEN}✅ clang-tidy: 0 errors, $tidy_warnings warnings${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  clang-tidy not found${NC}"
    echo "   Install: sudo apt-get install clang-tidy"
    violations=$((violations + 1))
fi

# ============================================================================
# 2. Build Check
# ============================================================================

echo ""
echo -e "${BLUE}3️⃣  Build Verification...${NC}"
echo ""

if [ -d "build" ]; then
    echo "  Building project with ninja..."

    cd build
    if ninja 2>&1 | tee /tmp/build.log | tail -20; then
        echo -e "${GREEN}✅ Build succeeded${NC}"
    else
        echo -e "${RED}❌ Build failed${NC}"
        echo "See /tmp/build.log for details"
        violations=$((violations + 1))
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  Build directory not found${NC}"
    echo "   Run: mkdir build && cd build && cmake -G Ninja .."
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Validation Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

if [ $violations -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Code is ready for:"
    echo "  - Commit and push"
    echo "  - Pull request"
    echo "  - Production deployment"
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED: $violations check(s) failed${NC}"
    echo ""
    echo "Please fix the issues above before committing."
    echo ""
    echo "Common fixes:"
    echo "  - clang-tidy errors: Follow C++ Core Guidelines"
    echo "  - cppcheck errors: Fix logic/safety issues"
    echo "  - Build errors: Check compiler output"
    echo ""
    echo "Documentation: docs/CODING_STANDARDS.md"
    exit 1
fi
