#!/bin/bash
# Setup Git Hooks for BigBrotherAnalytics
# Configures local enforcement of coding standards
#
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-08

set -e

echo "🔧 Setting up Git hooks for BigBrotherAnalytics..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.githooks"
GIT_HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

# Configure git to use .githooks directory
git config core.hooksPath .githooks

# Make hooks executable
chmod +x "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/post-commit"

echo "✅ Git hooks configured!"
echo ""
echo "📋 Installed hooks:"
echo "  - pre-commit: Enforces C++23 standards, trailing return syntax, module structure"
echo "  - post-commit: Displays helpful tips"
echo ""
echo "🔍 Pre-commit checks will run automatically before each commit:"
echo "  1. Trailing return type syntax"
echo "  2. [[nodiscard]] attributes on getters"
echo "  3. C++23 module structure"
echo "  4. Documentation completeness"
echo "  5. Code formatting (if clang-format available)"
echo ""
echo "💡 To bypass hooks (not recommended): git commit --no-verify"
echo "💡 To run checks manually: .githooks/pre-commit"
echo ""
echo "📚 See docs/CODING_STANDARDS.md for complete guidelines"
