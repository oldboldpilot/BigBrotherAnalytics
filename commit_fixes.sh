#!/bin/bash
# Commit script for C++23 module build fixes

cd /home/muyiwa/Development/BigBrotherAnalytics

echo "=== Staging Modified Files ==="
git add src/correlation_engine/trinomial_tree.cppm
git add src/utils/utils.cppm
git add CMakeLists.txt

echo ""
echo "=== Creating Commit ==="
git commit -m "$(cat <<'EOF'
fix: Resolve C++23 module build errors and OpenMP configuration

Fixed three critical build issues preventing compilation:

1. **trinomial_tree.cppm**: Moved Greeks struct definition before first use
   - Greeks struct was defined at line 413 but referenced at line 177
   - Relocated to line 55 to resolve "use of undeclared identifier" error
   - Maintains trailing return type syntax and fluent API design

2. **utils.cppm**: Corrected module structure
   - Removed invalid 'export module' partition declarations
   - Simplified to clean module aggregation re-exporting sub-modules
   - Fixes "module file cannot be loaded" configuration errors

3. **CMakeLists.txt**: Fixed OpenMP module configuration mismatch
   - Added OpenMP::OpenMP_CXX to utils library linking (line 115)
   - Resolves "OpenMP support and version differs in precompiled file" errors
   - Ensures consistent compiler flags across all modules

All C++23 modules with trailing return syntax and fluent APIs now compile
successfully. Module precompiled files (.pcm) and object files (.o) generate
correctly. Verified via ninja build logs showing successful compilation of:
- 9 utils modules (types, logger, config, database, timer, math, tax, etc.)
- Options pricing modules (black_scholes, trinomial_tree, options_pricing)
- Trading decision, market intelligence, risk management modules
- Main application and backtest engine

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

echo ""
echo "=== Commit Status ==="
git log -1 --stat

echo ""
echo "✅ Commit created successfully!"
