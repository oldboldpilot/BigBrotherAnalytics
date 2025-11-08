# Build Fixes Needed - Quick Reference

## Status: 80% Building (113/139 files compiled successfully)

Remaining errors: **48 compilation errors** across 4-5 files

---

## Quick Fixes Required (Est. 15-20 minutes)

### 1. Logging Macro Issues in main.cpp & backtest_main.cpp

**Problem:** LOG_INFO, LOG_ERROR, LOG_WARN macros expanding incorrectly
**Error:** `expected unqualified-id`

**Files affected:**
- `src/main.cpp`
- `src/backtest_main.cpp`

**Fix:**
Add at top of files:
```cpp
using bigbrother::utils::LOG_INFO;
using bigbrother::utils::LOG_ERROR;
using bigbrother::utils::LOG_WARN;
```

Or change:
```cpp
utils::LOG_INFO("message");
// to
LOG_INFO("message");
```

**Estimate:** 5 minutes

---

### 2. Timer Namespace Issues in pearson.cpp

**Problem:** `use of undeclared identifier 'Timer'`

**File:** `src/correlation_engine/pearson.cpp`

**Fix:**
Change:
```cpp
Timer timer;
// to
utils::Timer timer;
```

**Estimate:** 2 minutes

---

### 3. Strategy Constructor Issues (MOSTLY FIXED)

**Status:** ✅ Added default constructors for:
- DeltaNeutralStraddleStrategy
- DeltaNeutralStrangleStrategy
- VolatilityArbitrageStrategy
- MeanReversionStrategy

**Remaining:** May need to add implementations for stub methods

**Estimate:** 5 minutes

---

### 4. Missing Include: <functional> in some files

**Error:** Use of std::function without including <functional>

**Fix:** Add `#include <functional>` where needed

**Estimate:** 3 minutes

---

## Alternative: Use Stub Implementations for Now

Since this is Tier 1 POC, we can temporarily stub out unimplemented methods to get the build working, then implement properly during Tier 1:

**Create:** `src/trading_decision/strategy_stubs.cpp`
```cpp
// Temporary stubs for unimplemented strategies

#include "strategy_straddle.hpp"
#include "strategy_volatility_arb.hpp"

namespace bigbrother::strategy {

// Stub implementations - TODO: Implement in Tier 1
auto DeltaNeutralStrangleStrategy::generateSignals(StrategyContext const&)
    -> std::vector<TradingSignal> { return {}; }

auto VolatilityArbitrageStrategy::generateSignals(StrategyContext const&)
    -> std::vector<TradingSignal> { return {}; }

auto MeanReversionStrategy::generateSignals(StrategyContext const&)
    -> std::vector<TradingSignal> { return {}; }

// Add getParameters, setActive for each...

} // namespace
```

**Estimate:** 10 minutes to create comprehensive stubs

---

## Recommended Approach

**Option A: Fix All Errors Now (20 minutes)**
- Proper fix for all compilation errors
- Clean build
- Ready for Tier 1 implementation

**Option B: Defer to Next Session (0 minutes now)**
- Document all remaining errors
- Start fresh next session
- Fix during first Tier 1 implementation session

**Option C: Stub Everything (10 minutes)**
- Add stub implementations
- Get build working
- Implement properly during Tier 1 weeks

---

## Build Success Criteria

When these are fixed, you'll have:
- ✅ All 139 files compiling
- ✅ All libraries linking
- ✅ Executables built: `bigbrother`, `backtest`
- ✅ Tests ready to run
- ✅ Python bindings built

---

## Context: What's Already Done

**Major Success:**
- ✅ LLVM/Clang 21 + Flang built & installed (2 hours, 7,034 targets)
- ✅ OpenMPI 5.0.7 built
- ✅ CMake configured for C++23 modules
- ✅ 6,500+ lines of documentation
- ✅ std::map → std::unordered_map optimizations
- ✅ 113/139 files (80%) compiling successfully

**Remaining:**
- Fix 48 compilation errors (mostly trivial)
- Get 100% build success
- Run tests
- Begin Tier 1 implementation

---

## Next Session Start

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/build
ninja 2>&1 | grep "error:" | head -20  # See current errors
# Fix each category one by one
# Should take 15-20 minutes total
```

The toolchain foundation is solid. These are just normal C++ development issues that every project encounters.
