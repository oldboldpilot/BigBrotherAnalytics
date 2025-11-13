# C++23 Module Migration Plan

**Date**: November 7, 2025
**Status**: In Progress
**Target**: Complete conversion of all C++ code to C++23 modules with trailing return syntax and fluent APIs

## Overview

This document outlines the systematic migration of BigBrotherAnalytics from traditional header-based C++ to modern C++23 modules, following C++ Core Guidelines and best practices from Clang 21 standards.

## Design Principles

### 1. C++23 Module Structure
```cpp
// Global module fragment - for standard library includes only
module;

#include <string>
#include <vector>
#include <expected>

// Module declaration
export module bigbrother.<subsystem>.<component>;

// Optional: Import other modules
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Export namespace
export namespace bigbrother::<subsystem> {
    // All exported types, functions, classes
}
```

### 2. Trailing Return Type Syntax
All functions use trailing return types:
```cpp
// ✅ Correct
auto calculatePrice(PricingParams const& params) -> Result<Price>;
[[nodiscard]] auto getSymbol() const noexcept -> std::string const&;

// ❌ Avoid
Price calculatePrice(PricingParams const& params);
std::string const& getSymbol() const noexcept;
```

### 3. Fluent API Pattern
Builder-style interfaces for complex operations:
```cpp
auto result = OptionBuilder()
    .call()
    .american()
    .spot(150.0)
    .strike(155.0)
    .volatility(0.25)
    .price();
```

### 4. C++ Core Guidelines Compliance
- **C.1**: Use `struct` for passive data, `class` when invariants exist
- **C.2**: Private data with public interface when invariants required
- **C.21**: Define or delete all default operations as a group
- **C.41**: Constructors establish class invariants
- **C.47**: Initialize members in declaration order
- **E.**: Use `std::expected<T, Error>` for functions that can fail
- **F.1**: Meaningful, descriptive names
- **F.4**: Use `constexpr` for compile-time evaluation
- **F.6**: Use `noexcept` where no exceptions possible
- **F.16**: Pass cheap types by value, others by `const&`
- **F.20**: Prefer return values to output parameters
- **I.10**: Use exceptions or `std::expected` for error handling
- **P.4**: Type safety over primitives (strong types)

## Module Hierarchy

```
bigbrother
├── utils
│   ├── types          ✅ (complete)
│   ├── logger         ✅ (complete)
│   ├── config         ✅ (complete)
│   ├── database_api   ✅ (complete)
│   ├── timer          🔄 (pending)
│   └── math           🔄 (pending)
├── options
│   ├── black_scholes  ✅ (complete)
│   ├── trinomial_tree ✅ (complete)
│   ├── binomial_tree  🔄 (pending)
│   ├── greeks         🔄 (pending)
│   ├── implied_vol    🔄 (pending)
│   ├── iv_surface     🔄 (pending)
│   └── pricing        🔄 (pending - main module)
├── correlation
│   ├── correlation    🔄 (pending - main module)
│   ├── pearson        🔄 (pending)
│   ├── spearman       🔄 (pending)
│   ├── time_lagged    🔄 (pending)
│   ├── rolling_window 🔄 (pending)
│   └── parallel       🔄 (pending)
├── risk
│   ├── manager        🔄 (pending - main module)
│   ├── position_sizer 🔄 (pending)
│   ├── stop_loss      🔄 (pending)
│   ├── kelly          🔄 (pending)
│   ├── monte_carlo    🔄 (pending)
│   └── constraints    🔄 (pending)
├── strategy
│   ├── base           🔄 (pending)
│   ├── iron_condor    ✅ (complete)
│   ├── straddle       🔄 (pending)
│   ├── strangle       🔄 (pending)
│   ├── volatility_arb 🔄 (pending)
│   ├── signal_aggregator 🔄 (pending)
│   └── ml_predictor   🔄 (pending)
├── schwab
│   ├── client         🔄 (pending - main module)
│   ├── auth           🔄 (pending)
│   ├── token_manager  🔄 (pending)
│   ├── market_data    🔄 (pending)
│   ├── options_chain  🔄 (pending)
│   ├── orders         🔄 (pending)
│   ├── account        🔄 (pending)
│   └── websocket      🔄 (pending)
├── intelligence
│   ├── data_fetcher   🔄 (pending)
│   ├── market_data    🔄 (pending)
│   ├── news           🔄 (pending)
│   ├── sentiment      🔄 (pending)
│   └── entities       🔄 (pending)
├── explainability
│   ├── decision_logger 🔄 (pending)
│   ├── feature_importance 🔄 (pending)
│   └── trade_analyzer 🔄 (pending)
└── backtest
    ├── engine         🔄 (pending - main module)
    ├── order_simulator 🔄 (pending)
    └── metrics        🔄 (pending)
```

## Migration Phases

### Phase 1: Complete Utils Library ✅→🔄
**Status**: 50% complete
**Dependencies**: None
**Files**:
- ✅ `types.cppm` (complete)
- ✅ `logger.cppm` (complete)
- ✅ `config.cppm` (complete)
- ✅ `database_api.cppm` (complete)
- 🔄 `timer.cppm` (convert from timer.hpp/cpp)
- 🔄 `math.cppm` (convert from math.hpp)

**Priority**: HIGH - Foundation for all other modules

### Phase 2: Options Pricing Engine ✅→🔄
**Status**: 25% complete
**Dependencies**: utils
**Files**:
- ✅ `black_scholes.cppm` (complete)
- ✅ `trinomial_tree.cppm` (complete)
- 🔄 `binomial_tree.cppm`
- 🔄 `greeks.cppm`
- 🔄 `implied_volatility.cppm`
- 🔄 `iv_surface.cppm`
- 🔄 `pricing.cppm` (main module - integrate fluent API)

**Priority**: CRITICAL - Core trading feature

### Phase 3: Risk Management
**Status**: 0% complete
**Dependencies**: utils, options
**Files**:
- 🔄 `risk_manager.cppm` (main module with fluent API)
- 🔄 `position_sizer.cppm`
- 🔄 `stop_loss.cppm`
- 🔄 `kelly_criterion.cppm`
- 🔄 `monte_carlo.cppm`
- 🔄 `portfolio_constraints.cppm`

**Priority**: CRITICAL - Capital protection

### Phase 4: Correlation Engine
**Status**: 0% complete
**Dependencies**: utils
**Files**:
- 🔄 `correlation.cppm` (main module with fluent API)
- 🔄 `pearson.cppm`
- 🔄 `spearman.cppm`
- 🔄 `time_lagged.cppm`
- 🔄 `rolling_window.cppm`
- 🔄 `parallel_correlation.cppm`

**Priority**: HIGH - Key differentiator

### Phase 5: Trading Strategies
**Status**: 14% complete (1/7)
**Dependencies**: utils, options, risk, correlation
**Files**:
- 🔄 `strategy_base.cppm`
- ✅ `strategy_iron_condor.cppm` (complete)
- 🔄 `strategy_straddle.cppm`
- 🔄 `strategy_strangle.cppm`
- 🔄 `strategy_volatility_arb.cppm`
- 🔄 `signal_aggregator.cppm`
- 🔄 `ml_predictor.cppm`

**Priority**: HIGH - Core trading logic

### Phase 6: Schwab API Client
**Status**: 0% complete
**Dependencies**: utils
**Files**:
- 🔄 `schwab_client.cppm` (main module with fluent API)
- 🔄 `auth.cppm`
- 🔄 `token_manager.cppm`
- 🔄 `market_data.cppm`
- 🔄 `options_chain.cppm`
- 🔄 `orders.cppm`
- 🔄 `account.cppm`
- 🔄 `websocket_client.cppm`

**Priority**: CRITICAL - Market connectivity

### Phase 7: Market Intelligence
**Status**: 0% complete
**Dependencies**: utils
**Files**:
- 🔄 `data_fetcher.cppm`
- 🔄 `market_data_client.cppm`
- 🔄 `news_client.cppm`
- 🔄 `sentiment_analyzer.cppm`
- 🔄 `entity_recognizer.cppm`

**Priority**: MEDIUM - Future enhancement

### Phase 8: Explainability & Backtesting
**Status**: 0% complete
**Dependencies**: utils, options, risk, strategy
**Files**:
- 🔄 `decision_logger.cppm`
- 🔄 `feature_importance.cppm`
- 🔄 `trade_analyzer.cppm`
- 🔄 `backtest_engine.cppm` (main module with fluent API)
- 🔄 `order_simulator.cppm`
- 🔄 `performance_metrics.cppm`

**Priority**: HIGH - Validation & compliance

## Fluent API Examples

### Options Pricing Builder
```cpp
auto result = OptionBuilder()
    .call()
    .american()
    .spot(150.0)
    .strike(155.0)
    .daysToExpiration(30)
    .volatility(0.25)
    .riskFreeRate(0.05)
    .useBlackScholes()
    .priceWithGreeks();

if (result) {
    auto [price, greeks] = *result;
    std::println("Price: ${}, Delta: {}", price, greeks.delta);
}
```

### Risk Assessment Builder
```cpp
auto risk = RiskAssessor()
    .symbol("AAPL")
    .positionSize(1000.0)
    .entryPrice(150.0)
    .stopPrice(145.0)
    .targetPrice(160.0)
    .winProbability(0.65)
    .useKellyCriterion()
    .assess();

if (risk->isApproved()) {
    // Execute trade
}
```

### Correlation Analysis Builder
```cpp
auto correlations = CorrelationAnalyzer()
    .addSymbol("NVDA")
    .addSymbol("AMD")
    .addSymbol("TSM")
    .timeframe(30) // days
    .usePearson()
    .withLags(0, 5)
    .parallel()
    .calculate();
```

### Backtest Builder
```cpp
auto results = BacktestBuilder()
    .strategy(IronCondorStrategy{})
    .startDate("2020-01-01")
    .endDate("2024-01-01")
    .initialCapital(30000.0)
    .commission(0.65)
    .slippage(0.05)
    .run();

std::println("Total Return: {}%", results->totalReturn());
std::println("Sharpe Ratio: {}", results->sharpeRatio());
```

## CMake Integration

Update `CMakeLists.txt` to support module compilation:

```cmake
# Enable C++23 modules
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_STANDARD 23)

# Module compilation flags for Clang 21
target_compile_options(target_name PRIVATE -fmodule-output)

# Define module sources
target_sources(library_name
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/module1.cppm
            src/module2.cppm
)
```

## Testing Strategy

1. **Module-by-Module Testing**: Test each converted module independently
2. **Integration Testing**: Verify module imports work correctly
3. **Performance Validation**: Ensure no performance regression
4. **Build Time**: Monitor module compilation time

## Success Criteria

- ✅ All `.hpp` + `.cpp` files converted to `.cppm` modules
- ✅ 100% trailing return type syntax
- ✅ Fluent APIs implemented for all major subsystems
- ✅ Full C++ Core Guidelines compliance
- ✅ All tests passing
- ✅ Build successful with Clang 21
- ✅ No performance regression
- ✅ Clean module dependency graph

## Timeline

- **Phase 1** (Utils): ~2 hours
- **Phase 2** (Options): ~3 hours
- **Phase 3** (Risk): ~2 hours
- **Phase 4** (Correlation): ~2 hours
- **Phase 5** (Strategies): ~2 hours
- **Phase 6** (Schwab API): ~3 hours
- **Phase 7** (Intelligence): ~2 hours
- **Phase 8** (Backtest/Explain): ~2 hours

**Total Estimated Time**: ~18 hours

## Third-Party Library Integration: DuckDB Bridge Pattern

### Problem: Incomplete Types in Module Interfaces

Some C++ libraries (like DuckDB) export incomplete types that cannot be forward-declared, causing compilation errors when used in C++23 modules:

```cpp
// ❌ FAILS - DuckDB's QueryNode cannot be forward-declared
export module bigbrother.utils.database;
namespace duckdb { class QueryNode; }  // ERROR: incomplete type

// Forces inclusion of full headers:
#include <duckdb.hpp>  // 5000+ lines, pollutes module interface!
```

### Solution: Bridge Pattern with Opaque Handles

Implement a bridge library that isolates third-party incomplete types:

```cpp
// duckdb_bridge.hpp - Module-safe interface
export class DatabaseHandle {
  private:
    struct Impl;  // Defined only in .cpp, hides DuckDB types
    std::unique_ptr<Impl> pImpl_;
};

auto openDatabase(std::string const& path) -> std::unique_ptr<DatabaseHandle>;

// Module can now safely import:
export module bigbrother.utils.resilient_database;
#include "duckdb_bridge.hpp"  // ✅ Works! Only opaque types
```

### Implementation Pattern

**Files:**
- `src/schwab_api/duckdb_bridge.hpp` (146 lines) - Public opaque interface
- `src/schwab_api/duckdb_bridge.cpp` (413 lines) - Implementation using DuckDB C API

**Key Components:**
- `DatabaseHandle` - Wraps `duckdb_database`
- `ConnectionHandle` - Wraps `duckdb_connection`
- `PreparedStatementHandle` - Wraps `duckdb_prepared_statement`
- `QueryResultHandle` - Wraps `duckdb_result`

**Benefits:**
- ✅ No incomplete types in module interface
- ✅ Zero runtime overhead (opaque handles are zero-cost abstractions)
- ✅ 2.6x faster compilation (no 5000+ line DuckDB headers)
- ✅ Clean module boundaries (third-party types hidden)
- ✅ Single integration point for DuckDB API changes

**Usage in Modules:**
```cpp
export module bigbrother.utils.resilient_database;
#include "duckdb_bridge.hpp"

export auto executeQuery(std::string const& sql) -> std::vector<std::string> {
    auto db = duckdb_bridge::openDatabase("data/bigbrother.duckdb");
    auto conn = duckdb_bridge::createConnection(*db);
    auto result = duckdb_bridge::executeQueryWithResults(*conn, sql);

    // Safe value extraction - never exposes DuckDB internals
    std::vector<std::string> values;
    for (size_t i = 0; i < duckdb_bridge::getRowCount(*result); ++i) {
        values.push_back(duckdb_bridge::getValueAsString(*result, 0, i));
    }
    return values;
}
```

### Recommended Pattern for Third-Party Integration

When integrating third-party C++ libraries into C++23 modules:

1. **Check for incomplete types:** Does the library export types that cannot be forward-declared?
2. **Use Bridge pattern if needed:** Create opaque handle wrappers in `.cpp` files
3. **Prefer C API:** If available, use the library's C API (more stable, no incomplete types)
4. **Hide implementation:** Keep all third-party includes in `.cpp` files only
5. **Test module isolation:** Verify modules can be imported without exposing internals

### Complete Example: DuckDB Bridge (November 2025)

**Status:** ✅ Complete and validated (9/9 regression tests passed)

See [DUCKDB_BRIDGE_INTEGRATION.md](../DUCKDB_BRIDGE_INTEGRATION.md) for:
- Complete architecture documentation
- Technical deep dive and implementation details
- Performance analysis and testing results
- Usage guide for module developers
- Maintenance and extension patterns

---

## References

- [Clang 21 C++ Modules Documentation](https://releases.llvm.org/21.1.0/tools/clang/docs/StandardCPlusPlusModules.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [C++23 Standard](https://en.cppreference.com/w/cpp/23)
- [DuckDB Bridge Integration](../DUCKDB_BRIDGE_INTEGRATION.md) - Bridge pattern implementation
- [Pimpl Idiom (Pointer to Implementation)](https://en.cppreference.com/w/cpp/pimpl) - Pattern used for opaque handles
