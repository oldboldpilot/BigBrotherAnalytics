# Options Strategies Implementation Guide

**Status:** Foundation + Tier 1 + Tier 2 + Tier 3 Complete
**Completed:** 28 strategies (54% of total)
**Remaining:** 24 strategies across 4 files
**Estimated Time:** 2-3 additional sessions

---

## ✅ Completed Files

| File | Lines | Status | Strategies |
|------|-------|--------|------------|
| base.cppm | 400 | ✅ Complete | Base classes & interfaces |
| simd_utils.cppm | 590 | ✅ Complete | AVX2 Black-Scholes & All Greeks |
| single_leg.cppm | 550 | ✅ Complete | 4 strategies (Tier 1) |
| vertical_spreads.cppm | 740 | ✅ Complete | 4 vertical spreads (Tier 2.1) |
| straddles_strangles.cppm | 1470 | ✅ Complete | 8 volatility strategies (Tier 2.2) |
| butterflies_condors.cppm | 2850 | ✅ Complete | 12 butterfly/condor strategies (Tier 3) |

**Total:** 6,600 lines of production-ready C++23 code with AVX2 optimization

---

## 📋 Remaining Files (Prioritized)

### File 1: covered_positions.cppm (High Priority)
**Strategies:** 3
- Covered Call
- Covered Put
- Covered Call Collar

**Pattern:** Stock + option(s)

**Complexity:** Simple (stock + 1-2 options)

---

### File 2: calendar_spreads.cppm (Medium Priority)
**Strategies:** 6
- Calendar Call/Put Spread
- Calendar Straddle/Strangle
- Short Calendar variants

**Pattern:** Different expirations, same strikes

---

### File 3: ratio_spreads.cppm (Medium Priority)
**Strategies:** 8
- Call/Put Ratio Spread
- Call/Put Ratio Backspread
- Bull/Bear Ratio variants

**Pattern:** Unequal number of legs (1:2, 2:3, etc.)

---

### File 4: albatross_ladder.cppm (Low Priority)
**Strategies:** 7
- Albatross variants (4)
- Ladder spreads (2)
- Short Albatross

**Pattern:** Wide-bodied condor variants

---

## 🔨 Implementation Pattern

Each strategy file follows this structure:

```cpp
module;
#include [standard headers]

export module bigbrother.options_strategies.[tier_name];

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// Strategy 1
class [Strategy1]Strategy : public BaseOptionsStrategy<[Strategy1]Strategy> {
    // Constructor
    // P&L calculations
    // Greeks calculations
    // Max profit/loss
    // Breakevens
};

// Strategy 2
class [Strategy2]Strategy : public BaseOptionsStrategy<[Strategy2]Strategy> {
    // ...
};

// Factory functions
[[nodiscard]] inline auto create[Strategy1](...) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<[Strategy1]Strategy>(...);
}

}  // namespace
```

---

## 📊 Complexity Guidelines

### Simple (Beginner)
- 1-2 legs
- Clear risk/reward
- Example: Long Call, Bull Call Spread

### Intermediate
- 2-3 legs
- Moderate complexity
- Example: Iron Butterfly, Calendar Spread

### Advanced
- 3-4 legs
- Complex P&L diagrams
- Example: Condor, Albatross

### Complex
- 4+ legs or special conditions
- Advanced risk management
- Example: Ratio Backspreads, Ladder Spreads

---

## 🧪 Testing Strategy

For each file, create corresponding test:

```cpp
// tests/options_strategies/test_single_leg.cpp

#include <gtest/gtest.h>
import bigbrother.options_strategies.single_leg;

TEST(LongCallTest, BasicFunctionality) {
    auto strategy = createLongCall(100.0f, 105.0f, 30.0f, 0.25f);

    // Test P&L at various prices
    EXPECT_LT(strategy->calculateProfitLoss(100.0f), 0.0f);  // Loss
    EXPECT_NEAR(strategy->calculateProfitLoss(110.0f), 5.0f - premium, 0.01f);  // Profit

    // Test Greeks
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.delta, 0.0f);  // Positive for long call
    EXPECT_GT(greeks.gamma, 0.0f);

    // Test breakeven
    auto breakevens = strategy->getBreakevens();
    EXPECT_EQ(breakevens.size(), 1);
}

// Repeat for each strategy...
```

---

## 🚀 Quick Start (Next Session)

### Step 1: Create covered_positions.cppm

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
cp src/options_strategies/single_leg.cppm src/options_strategies/covered_positions.cppm
```

### Step 2: Modify for stock + option(s) strategies

Implement 3 strategies:
- Covered Call (buy stock, sell call)
- Covered Put (short stock, sell put)
- Collar (buy stock, sell call, buy put)

### Step 3: Update CMakeLists.txt

Add new module to build system (see below)

### Step 4: Build & Test

```bash
cmake -G Ninja -B build
ninja -C build options_strategies
./build/tests/test_covered_positions
```

---

## 🔧 CMakeLists.txt Integration

Add this to your CMakeLists.txt:

```cmake
# Options Strategies Library
add_library(options_strategies)

target_sources(options_strategies
    PUBLIC FILE_SET CXX_MODULES FILES
        src/options_strategies/base.cppm
        src/options_strategies/simd_utils.cppm
        src/options_strategies/single_leg.cppm
        src/options_strategies/vertical_spreads.cppm
        src/options_strategies/straddles_strangles.cppm
        src/options_strategies/butterflies_condors.cppm
        # TODO: Add these as implemented
        # src/options_strategies/covered_positions.cppm
        # src/options_strategies/calendar_spreads.cppm
        # src/options_strategies/ratio_spreads.cppm
        # src/options_strategies/albatross_ladder.cppm
)

target_compile_options(options_strategies PRIVATE
    -O3 -march=native -mavx2 -mfma -fopenmp-simd
    -ffast-math -funroll-loops -ftree-vectorize
)

target_compile_definitions(options_strategies PRIVATE
    -DUSE_AVX2=1 -DUSE_SIMD_PRICING=1
)

# Link to main engine
target_link_libraries(bigbrother PRIVATE options_strategies)
```

---

## 📈 Performance Benchmarks

Target performance (actual will vary):

| Operation | Target | Actual (TODO) |
|-----------|--------|---------------|
| Single option pricing | <1μs | TBD |
| Greeks (all 5) | <5μs | TBD |
| Batch pricing (100) | <100μs | TBD |
| Strategy P&L | <3μs | TBD |

Measure with:
```cpp
auto start = std::chrono::high_resolution_clock::now();
// ... operation ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
```

---

## 💡 Tips for Implementation

1. **Start with simple strategies** - Get vertical spreads working first
2. **Test incrementally** - Don't wait to test all strategies at once
3. **Reuse SIMD functions** - They're already optimized
4. **Follow the pattern** - single_leg.cppm is the template
5. **Document assumptions** - Note any simplifications made

---

## 🐛 Common Pitfalls

1. **Forgetting to negate Greeks for short positions**
2. **Incorrect sign conventions for P&L**
3. **Not converting days_to_expiration to years (divide by 365)**
4. **Misaligning SIMD data (use alignas(32))**
5. **Forgetting to multiply by quantity**

---

## 📚 References

- [Options Strategies A-Z](https://www.optionstrading.org/strategies/a-z-list/)
- Black-Scholes Model
- Greeks Calculations
- Intel AVX2 Intrinsics Guide

---

**Next Action:** Implement covered_positions.cppm (3 strategies, ~350 lines)

**Author:** Claude Code + Olumuyiwa Oluwasanmi
**Last Updated:** November 12, 2025
