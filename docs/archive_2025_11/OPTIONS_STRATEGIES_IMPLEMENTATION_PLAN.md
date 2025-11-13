# Options Strategies Implementation Plan - Complete A-Z List

**Date:** November 12, 2025
**Scope:** 52+ options strategies with SIMD optimizations
**Target:** C++23 modules with AVX2/AVX-512 intrinsics
**Source:** https://www.optionstrading.org/strategies/a-z-list/

---

## Architecture Overview

### Core Design Principles

1. **Modular C++23 Architecture**
   - Base `OptionsStrategy` class with virtual methods
   - Each strategy as derived class in separate module
   - SIMD-optimized Greeks calculations
   - Compile-time polymorphism where possible

2. **SIMD Optimization Strategy**
   - AVX2 intrinsics (256-bit, 8 floats parallel)
   - AVX-512 intrinsics (512-bit, 16 floats parallel) when available
   - Compiler flags: `-O3 -march=native -mavx2 -mfma -fopenmp-simd`
   - Batch processing for multiple strikes/expirations

3. **Performance Targets**
   - Single strategy pricing: <10μs
   - Greeks calculation: <50μs (all 5 Greeks)
   - Batch pricing (100 scenarios): <500μs
   - Zero heap allocations in hot path

---

## Strategy Taxonomy (52 Strategies)

### Tier 1: Single Leg Strategies (4) - **PRIORITY 1**
*Simple, foundational, already partially implemented*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Long Call | Single | Bullish | Beginner | ⏳ TODO |
| Long Put | Single | Bearish | Beginner | ⏳ TODO |
| Short Call | Single | Bearish | Beginner | ⏳ TODO |
| Short Put | Single | Bullish | Beginner | ⏳ TODO |

**Implementation File:** `src/options_strategies/single_leg.cppm`

---

### Tier 2: Basic Spreads (12) - **PRIORITY 2**
*Most commonly used, suitable for beginners/intermediate*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Bull Call Spread | Vertical | Bullish | Beginner | ⏳ TODO |
| Bull Put Spread | Vertical | Bullish | Intermediate | ⏳ TODO |
| Bear Call Spread | Vertical | Bearish | Intermediate | ⏳ TODO |
| Bear Put Spread | Vertical | Bearish | Beginner | ⏳ TODO |
| Straddle (Long) | Combination | Volatile | Beginner | ✅ PARTIAL |
| Strangle (Long) | Combination | Volatile | Beginner | ✅ PARTIAL |
| Short Straddle | Combination | Neutral | Intermediate | ⏳ TODO |
| Short Strangle | Combination | Neutral | Intermediate | ⏳ TODO |
| Covered Call | Combo | Neutral | Beginner | ⏳ TODO |
| Covered Put | Combo | Neutral | Intermediate | ⏳ TODO |
| Covered Call Collar | Combo | Neutral | Beginner | ⏳ TODO |
| Long Gut | Combination | Volatile | Beginner | ⏳ TODO |

**Implementation Files:**
- `src/options_strategies/vertical_spreads.cppm`
- `src/options_strategies/straddles_strangles.cppm` (expand existing)
- `src/options_strategies/covered_positions.cppm`

---

### Tier 3: Butterfly & Condor Spreads (12) - **PRIORITY 3**
*Advanced strategies for neutral markets*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Butterfly Spread | 4-Leg | Neutral | Advanced | ⏳ TODO |
| Bull Butterfly | 4-Leg | Bullish | Complex | ⏳ TODO |
| Bear Butterfly | 4-Leg | Bearish | Complex | ⏳ TODO |
| Short Butterfly | 4-Leg | Volatile | Complex | ⏳ TODO |
| Condor Spread | 4-Leg | Neutral | Advanced | ⏳ TODO |
| Bull Condor | 4-Leg | Bullish | Complex | ⏳ TODO |
| Short Condor | 4-Leg | Volatile | Advanced | ⏳ TODO |
| Iron Butterfly | 4-Leg | Neutral | Advanced | ✅ PARTIAL |
| Reverse Iron Butterfly | 4-Leg | Volatile | Complex | ⏳ TODO |
| Iron Condor | 4-Leg | Neutral | Advanced | ✅ PARTIAL |
| Reverse Iron Condor | 4-Leg | Volatile | Advanced | ⏳ TODO |
| Short Gut | Combination | Neutral | Simple | ⏳ TODO |

**Implementation File:** `src/options_strategies/butterflies_condors.cppm`

---

### Tier 4: Calendar Spreads (6) - **PRIORITY 4**
*Time-based strategies using different expirations*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Calendar Call Spread | Time | Neutral | Simple | ⏳ TODO |
| Calendar Put Spread | Time | Neutral | Simple | ⏳ TODO |
| Calendar Straddle | Time | Neutral | Advanced | ⏳ TODO |
| Calendar Strangle | Time | Neutral | Advanced | ⏳ TODO |
| Short Calendar Call | Time | Volatile | Advanced | ⏳ TODO |
| Short Calendar Put | Time | Volatile | Complex | ⏳ TODO |

**Implementation File:** `src/options_strategies/calendar_spreads.cppm`

---

### Tier 5: Ratio Spreads (8) - **PRIORITY 5**
*Unequal number of legs for specific risk/reward profiles*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Call Ratio Spread | Ratio | Neutral | Advanced | ⏳ TODO |
| Put Ratio Spread | Ratio | Neutral | Advanced | ⏳ TODO |
| Call Ratio Backspread | Ratio | Volatile/Bullish | Complex | ⏳ TODO |
| Put Ratio Backspread | Ratio | Volatile/Bearish | Complex | ⏳ TODO |
| Bear Ratio Spread | Ratio | Bearish | Complex | ⏳ TODO |
| Bull Ratio Spread | Ratio | Bullish | Complex | ⏳ TODO |
| Short Bear Ratio | Ratio | Bearish | Complex | ⏳ TODO |
| Short Bull Ratio | Ratio | Bullish | Complex | ⏳ TODO |

**Implementation File:** `src/options_strategies/ratio_spreads.cppm`

---

### Tier 6: Strap/Strip Variations (4) - **PRIORITY 6**
*Straddle/Strangle variants with unequal legs*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Strap Straddle | Combo | Volatile | Beginner | ⏳ TODO |
| Strip Straddle | Combo | Volatile | Beginner | ⏳ TODO |
| Strap Strangle | Combo | Volatile | Beginner | ⏳ TODO |
| Strip Strangle | Combo | Volatile | Beginner | ⏳ TODO |

**Implementation File:** `src/options_strategies/strap_strip.cppm`

---

### Tier 7: Albatross Spreads (3) - **PRIORITY 7**
*Wide-bodied condor variants*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Albatross Spread | 4-Leg | Neutral | Advanced | ⏳ TODO |
| Iron Albatross | 4-Leg | Neutral | Advanced | ⏳ TODO |
| Reverse Iron Albatross | 4-Leg | Volatile | Complex | ⏳ TODO |
| Short Albatross | 4-Leg | Volatile | Complex | ⏳ TODO |

**Implementation File:** `src/options_strategies/albatross_spreads.cppm`

---

### Tier 8: Ladder Spreads (2) - **PRIORITY 8**
*Multiple strike spreads with ladder-like P&L*

| Strategy | Type | Outlook | Complexity | Status |
|----------|------|---------|------------|--------|
| Bull Call Ladder | Ladder | Bullish | Complex | ⏳ TODO |
| Bear Put Ladder | Ladder | Bearish | Complex | ⏳ TODO |

**Implementation File:** `src/options_strategies/ladder_spreads.cppm`

---

## Implementation Architecture

### Base Strategy Class

```cpp
export module bigbrother.options_strategies.base;

import <span>;
import <array>;
import <optional>;
import <immintrin.h>;  // AVX2/AVX-512

export namespace bigbrother::options_strategies {

// Strategy metadata
enum class StrategyType {
    SINGLE_LEG,
    VERTICAL_SPREAD,
    HORIZONTAL_SPREAD,
    DIAGONAL_SPREAD,
    BUTTERFLY,
    CONDOR,
    RATIO,
    COMBINATION
};

enum class MarketOutlook {
    BULLISH,
    BEARISH,
    NEUTRAL,
    VOLATILE,
    BULLISH_VOLATILE,
    BEARISH_VOLATILE
};

enum class ComplexityLevel {
    BEGINNER,
    INTERMEDIATE,
    ADVANCED,
    COMPLEX
};

// Greeks structure with SIMD alignment
struct alignas(32) Greeks {
    float delta{0.0f};
    float gamma{0.0f};
    float theta{0.0f};
    float vega{0.0f};
    float rho{0.0f};
    float _padding[3];  // Align to 32 bytes for AVX2
};

// Position leg
struct OptionLeg {
    bool is_call{true};
    bool is_long{true};
    float strike{0.0f};
    float quantity{1.0f};
    float days_to_expiration{30.0f};
    float implied_volatility{0.20f};
};

// Base strategy interface
class IOptionsStrategy {
public:
    virtual ~IOptionsStrategy() = default;

    // Strategy metadata
    [[nodiscard]] virtual auto getName() const -> std::string_view = 0;
    [[nodiscard]] virtual auto getType() const -> StrategyType = 0;
    [[nodiscard]] virtual auto getOutlook() const -> MarketOutlook = 0;
    [[nodiscard]] virtual auto getComplexity() const -> ComplexityLevel = 0;

    // Pricing and Greeks (SIMD-optimized)
    [[nodiscard]] virtual auto calculateProfitLoss(
        float underlying_price,
        float current_price) const -> float = 0;

    [[nodiscard]] virtual auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks = 0;

    // Batch calculations (AVX2: 8 prices at once)
    [[nodiscard]] virtual auto calculateProfitLossBatch(
        std::span<float const> underlying_prices,
        float current_price) const -> std::vector<float> = 0;

    // Position management
    [[nodiscard]] virtual auto getLegs() const -> std::span<OptionLeg const> = 0;
    [[nodiscard]] virtual auto getMaxProfit() const -> std::optional<float> = 0;
    [[nodiscard]] virtual auto getMaxLoss() const -> std::optional<float> = 0;
    [[nodiscard]] virtual auto getBreakevens() const -> std::vector<float> = 0;
};

} // namespace
```

### SIMD Optimization Utilities

```cpp
export module bigbrother.options_strategies.simd_utils;

import <immintrin.h>;
import <span>;
import <cmath>;

export namespace bigbrother::options_strategies::simd {

// Black-Scholes pricing with AVX2 (8 options at once)
inline auto blackScholesCallBatch(
    __m256 S,       // 8 underlying prices
    __m256 K,       // 8 strike prices
    __m256 T,       // 8 times to expiration
    __m256 r,       // 8 risk-free rates
    __m256 sigma    // 8 volatilities
) -> __m256 {
    // d1 = (ln(S/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = _mm256_log_ps(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 r_plus_half_sigma_sq = _mm256_add_ps(r, half_sigma_sq);
    __m256 numerator = _mm256_add_ps(ln_S_K, _mm256_mul_ps(r_plus_half_sigma_sq, T));
    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);

    // d2 = d1 - sigma*sqrt(T)
    __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);

    // N(d1) and N(d2) - cumulative normal distribution (approximation)
    __m256 Nd1 = normalCDFAvx2(d1);
    __m256 Nd2 = normalCDFAvx2(d2);

    // Call = S*N(d1) - K*exp(-r*T)*N(d2)
    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = _mm256_exp_ps(neg_rT);
    __m256 discounted_K = _mm256_mul_ps(K, exp_neg_rT);

    __m256 term1 = _mm256_mul_ps(S, Nd1);
    __m256 term2 = _mm256_mul_ps(discounted_K, Nd2);

    return _mm256_sub_ps(term1, term2);
}

// Normal CDF approximation (Abramowitz & Stegun)
inline auto normalCDFAvx2(__m256 x) -> __m256 {
    __m256 const a1 = _mm256_set1_ps(0.254829592f);
    __m256 const a2 = _mm256_set1_ps(-0.284496736f);
    __m256 const a3 = _mm256_set1_ps(1.421413741f);
    __m256 const a4 = _mm256_set1_ps(-1.453152027f);
    __m256 const a5 = _mm256_set1_ps(1.061405429f);
    __m256 const p = _mm256_set1_ps(0.3275911f);

    __m256 sign = _mm256_set1_ps(1.0f);
    __m256 mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    sign = _mm256_blendv_ps(sign, _mm256_set1_ps(-1.0f), mask);

    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);

    __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f),
                              _mm256_add_ps(_mm256_set1_ps(1.0f),
                                           _mm256_mul_ps(p, abs_x)));

    __m256 y = _mm256_add_ps(a1, _mm256_mul_ps(t, a2));
    y = _mm256_add_ps(y, _mm256_mul_ps(t, _mm256_mul_ps(t, a3)));
    y = _mm256_add_ps(y, _mm256_mul_ps(t, _mm256_mul_ps(_mm256_mul_ps(t, t), a4)));
    y = _mm256_add_ps(y, _mm256_mul_ps(t, _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(t, t), t), a5)));

    __m256 exp_term = _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(-0.5f),
                                                    _mm256_mul_ps(x, x)));

    y = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(y, exp_term));

    return _mm256_mul_ps(_mm256_set1_ps(0.5f),
                         _mm256_add_ps(_mm256_set1_ps(1.0f),
                                      _mm256_mul_ps(sign, y)));
}

} // namespace
```

---

## Implementation Timeline

### Phase 1: Foundation (Day 1-2)
- ✅ Architecture design
- [ ] Base strategy class module
- [ ] SIMD utilities module
- [ ] Build system integration (CMakeLists.txt)
- [ ] Unit test framework setup

### Phase 2: Tier 1-2 (Day 3-5)
- [ ] Single leg strategies (4)
- [ ] Basic spreads (12)
- [ ] Greeks calculations with AVX2
- [ ] Performance benchmarks

### Phase 3: Tier 3-4 (Day 6-8)
- [ ] Butterfly & Condor spreads (12)
- [ ] Calendar spreads (6)
- [ ] Integration tests

### Phase 4: Tier 5-8 (Day 9-12)
- [ ] Ratio spreads (8)
- [ ] Strap/Strip variants (4)
- [ ] Albatross spreads (4)
- [ ] Ladder spreads (2)

### Phase 5: Testing & Optimization (Day 13-14)
- [ ] Comprehensive unit tests (52 strategies)
- [ ] Performance profiling
- [ ] AVX-512 optimizations (if available)
- [ ] Documentation

---

## Performance Benchmarks (Target)

| Operation | Scalar | AVX2 | AVX-512 | Target |
|-----------|--------|------|---------|--------|
| Single option pricing | 2.5μs | 0.8μs | 0.4μs | <1μs |
| Greeks (5 values) | 12μs | 4μs | 2μs | <5μs |
| Batch pricing (100) | 250μs | 80μs | 40μs | <100μs |
| Strategy P&L | 5μs | 2μs | 1μs | <3μs |

---

## File Structure

```
src/options_strategies/
├── base.cppm                    # Base interfaces
├── simd_utils.cppm              # SIMD helpers
├── single_leg.cppm              # Tier 1 (4 strategies)
├── vertical_spreads.cppm        # Tier 2 (4 strategies)
├── straddles_strangles.cppm     # Tier 2 (8 strategies)
├── covered_positions.cppm       # Tier 2 (3 strategies)
├── butterflies_condors.cppm     # Tier 3 (12 strategies)
├── calendar_spreads.cppm        # Tier 4 (6 strategies)
├── ratio_spreads.cppm           # Tier 5 (8 strategies)
├── strap_strip.cppm             # Tier 6 (4 strategies)
├── albatross_spreads.cppm       # Tier 7 (4 strategies)
└── ladder_spreads.cppm          # Tier 8 (2 strategies)

tests/options_strategies/
├── test_single_leg.cpp
├── test_spreads.cpp
├── test_butterflies.cpp
├── test_simd_performance.cpp
└── test_greeks_accuracy.cpp
```

---

## Compiler Optimization Flags

```cmake
# CMakeLists.txt
add_library(options_strategies)

target_compile_options(options_strategies PRIVATE
    -O3                    # Maximum optimization
    -march=native          # Use all CPU features
    -mavx2                 # Enable AVX2 instructions
    -mfma                  # Enable FMA (fused multiply-add)
    -fopenmp-simd          # Enable OpenMP SIMD directives
    -ffast-math            # Aggressive floating-point optimizations
    -funroll-loops         # Loop unrolling
    -ftree-vectorize       # Auto-vectorization
)

target_compile_definitions(options_strategies PRIVATE
    -DUSE_AVX2=1
    -DUSE_SIMD_PRICING=1
)

# Optional AVX-512 support (detect at runtime)
if(AVX512_SUPPORTED)
    target_compile_options(options_strategies PRIVATE
        -mavx512f
        -mavx512dq
    )
    target_compile_definitions(options_strategies PRIVATE
        -DUSE_AVX512=1
    )
endif()
```

---

## Next Steps

1. **Immediate:** Implement base strategy class and SIMD utilities
2. **Phase 1:** Complete Tier 1-2 strategies (16 total)
3. **Phase 2:** Add Tier 3-4 strategies (18 total)
4. **Phase 3:** Complete remaining strategies (18 total)
5. **Integration:** Wire all strategies into trading engine

**Total Implementation Time:** 14 days (aggressive) | 21 days (realistic)

---

**Author:** Claude Code + Olumuyiwa Oluwasanmi
**Last Updated:** November 12, 2025
**Status:** Phase 1 - Architecture Design
