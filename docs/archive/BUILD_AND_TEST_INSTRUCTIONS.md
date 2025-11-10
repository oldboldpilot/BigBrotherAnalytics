# Build and Test Instructions

**Author:** Olumuyiwa Oluwasanmi
**Updated:** 2025-11-08

## MANDATORY: Code Validation Before Build

**ALWAYS run validation first:**
```bash
./scripts/validate_code.sh
```

This runs:
1. ✅ clang-tidy (C++ Core Guidelines)
2. ✅ cppcheck (Static analysis)
3. ✅ Build verification
4. ✅ Module structure checks

**For specific files:**
```bash
./scripts/validate_code.sh src/utils/logger.cpp
./scripts/validate_code.sh src/risk_management/
```

## Build Fixes Applied

Three critical build errors have been fixed:

### 1. **trinomial_tree.cppm** (Line 55)
- **Issue**: Greeks struct was defined after first use (line 413 vs 177)
- **Fix**: Moved Greeks struct definition to line 55, before TrinomialPricer class
- **File**: `src/correlation_engine/trinomial_tree.cppm`

### 2. **utils.cppm** (Line 19-27)
- **Issue**: Invalid module partition declarations causing module configuration errors
- **Fix**: Simplified to clean module aggregation file
- **File**: `src/utils/utils.cppm`

### 3. **CMakeLists.txt** (Line 115)
- **Issue**: OpenMP configuration mismatch between utils and other libraries
- **Fix**: Added `OpenMP::OpenMP_CXX` to utils library link dependencies
- **File**: `CMakeLists.txt`

## Build Instructions

### Step 1: Complete the Build

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/build
ninja -v
```

**Expected Output**: All modules should compile successfully and link into shared libraries and executables in `bin/` and `lib/` directories.

### Step 2: Run Tests

```bash
chmod +x /home/muyiwa/Development/BigBrotherAnalytics/run_tests.sh
/home/muyiwa/Development/BigBrotherAnalytics/run_tests.sh
```

Or manually:

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/build
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ninja test
```

### Step 3: Commit the Fixes

```bash
chmod +x /home/muyiwa/Development/BigBrotherAnalytics/commit_fixes.sh
/home/muyiwa/Development/BigBrotherAnalytics/commit_fixes.sh
```

Or manually:

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
git add src/correlation_engine/trinomial_tree.cppm src/utils/utils.cppm CMakeLists.txt
git commit -m "fix: Resolve C++23 module build errors and OpenMP configuration"
```

## Verification

### Successful Compilation Indicators

Based on the ninja log, these files compiled successfully:

**Utils Modules (9 modules):**
- ✅ `bigbrother.utils.types.pcm`
- ✅ `bigbrother.utils.logger.pcm`
- ✅ `bigbrother.utils.config.pcm`
- ✅ `bigbrother.utils.database.pcm`
- ✅ `bigbrother.utils.database.api.pcm`
- ✅ `bigbrother.utils.timer.pcm`
- ✅ `bigbrother.utils.math.pcm`
- ✅ `bigbrother.utils.tax.pcm`
- ✅ `bigbrother.utils.risk_free_rate.pcm`

**Options Pricing Modules:**
- ✅ `bigbrother.pricing.trinomial_tree.pcm`
- ✅ `bigbrother.options.pricing.pcm`
- ✅ Black-Scholes module

**Other Modules:**
- ✅ Market Intelligence
- ✅ Explainability
- ✅ Risk Management
- ✅ Schwab API
- ✅ Trading Decision

**Main Applications:**
- ✅ `main.cpp` (bigbrother executable)
- ✅ `backtest_main.cpp` (backtest executable)

## Troubleshooting

If the build still fails, check:

1. **Linker errors**: Ensure all required libraries are installed
2. **Library paths**: Verify `LD_LIBRARY_PATH` includes all necessary directories
3. **Dependencies**: Check that all optional dependencies (DuckDB, ONNX Runtime, etc.) are properly installed

## Summary of Changes

All changes preserve the C++23 module structure with:
- ✅ Trailing return type syntax (`auto func() -> ReturnType`)
- ✅ Fluent API patterns (builder pattern with method chaining)
- ✅ `[[nodiscard]]` attributes
- ✅ Modern C++23 features (modules, concepts, ranges)

The fixes only addressed compilation and linking configuration issues without modifying the API design or functionality.
