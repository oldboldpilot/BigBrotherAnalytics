# Fluent API Pattern Guide - Strategy Framework

## Overview

The BigBrotherAnalytics trading strategy framework now includes comprehensive fluent API support for building, configuring, and executing trading strategies. This enables readable, chainable method calls following the builder pattern.

## Key Features

- **C++23 Trailing Return Syntax**: Modern C++ with clear method return types
- **Thread-Safe**: Mutex protection for concurrent access
- **Type-Safe**: Strong typing with optional values
- **[[nodiscard]]**: Compiler warnings for discarded results
- **Backward Compatible**: Traditional methods still available

---

## StrategyManager - Fluent Configuration

### Adding Strategies

```cpp
StrategyManager mgr;

// Fluent API - method chaining
mgr.addStrategy(std::make_unique<StraddleStrategy>())
   .addStrategy(std::make_unique<StrangleStrategy>())
   .addStrategy(std::make_unique<VolatilityArbStrategy>())
   .addStrategy(SectorRotationStrategy::builder()
       .withEmploymentWeight(0.65)
       .topNOverweight(4)
       .build());

// Or use traditional approach
auto strategy = std::make_unique<StraddleStrategy>();
mgr.addStrategy(std::move(strategy));
```

### Managing Strategy State

```cpp
// Enable/disable individual strategies
mgr.setStrategyActive("Long Straddle", true)
   .setStrategyActive("Long Strangle", false);

// Enable/disable all strategies
mgr.enableAll();
mgr.disableAll();

// Query strategy count
size_t count = mgr.getStrategyCount();

// Get specific strategy
if (auto strategy = mgr.getStrategy("Sector Rotation (Multi-Signal)")) {
    auto params = strategy->getParameters();
}
```

---

## SignalBuilder - Fluent Signal Generation

Generate and filter trading signals with chainable configuration.

### Basic Usage

```cpp
auto context = StrategyContext::builder()
    .withAccountValue(100000.0)
    .withAvailableCapital(20000.0)
    .withQuotes(quotes_map)
    .withEmploymentSignals(employment_signals)
    .build();

auto signals = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.70)
    .limitTo(10)
    .generate();
```

### Advanced Filtering

```cpp
// Filter by multiple criteria
auto high_quality_signals = mgr.signalBuilder()
    .forContext(context)
    .fromStrategies({"SectorRotation", "Volatility Arbitrage"})
    .withMinConfidence(0.75)
    .withMinRiskRewardRatio(2.0)
    .onlyActionable(true)
    .limitTo(5)
    .generate();

// Process signals
for (auto const& signal : high_quality_signals) {
    double reward = signal.expected_return;
    double risk = signal.max_risk;
    double confidence = signal.confidence;
    std::string rationale = signal.rationale;

    if (signal.isActionable()) {
        // Execute trade
    }
}
```

### SignalBuilder Methods

| Method | Type | Purpose |
|--------|------|---------|
| `forContext(ctx)` | StrategyContext | Set market data context |
| `fromStrategies(names)` | std::vector\<string\> | Filter by strategy names |
| `withMinConfidence(conf)` | double (0.0-1.0) | Minimum confidence threshold |
| `withMinRiskRewardRatio(ratio)` | double | Minimum risk/reward ratio |
| `onlyActionable(bool)` | bool | Filter for actionable signals only |
| `limitTo(count)` | int | Maximum signals to return |
| `generate()` | Terminal | Execute and return signals |

---

## StrategyContext::Builder - Fluent Context Construction

Build market data contexts with fluent configuration.

### Complete Example

```cpp
// Build comprehensive strategy context
auto context = StrategyContext::builder()
    // Account settings
    .withAccountValue(500000.0)
    .withAvailableCapital(50000.0)
    .withCurrentTime(getCurrentTimestamp())

    // Market data
    .withQuotes(current_quotes_map)
    .withOptions(options_chains_map)
    .withPositions(current_positions_vector)

    // Employment signals
    .withEmploymentSignals(employment_signals_vector)
    .withRotationSignals(rotation_signals_vector)
    .withJoblessClaims(jobless_claims_alert)

    .build();
```

### Incremental Building

```cpp
auto builder = StrategyContext::builder()
    .withAccountValue(100000.0)
    .withAvailableCapital(10000.0);

// Add quotes one by one
builder.addQuote("SPY", spy_quote)
       .addQuote("QQQ", qqq_quote)
       .addQuote("IWM", iwm_quote);

// Add positions
builder.addPosition(position1)
       .addPosition(position2);

// Add employment signals
builder.addEmploymentSignal(signal1)
       .addEmploymentSignal(signal2);

auto final_context = builder.build();
```

### ContextBuilder Methods

| Method | Type | Purpose |
|--------|------|---------|
| `withAccountValue(val)` | double | Total account value |
| `withAvailableCapital(cap)` | double | Liquid capital available |
| `withCurrentTime(ts)` | Timestamp | Current market time |
| `withQuotes(map)` | unordered_map | All market quotes |
| `withOptions(map)` | unordered_map | Options chains |
| `withPositions(vec)` | vector | Current positions |
| `withEmploymentSignals(vec)` | vector | Employment data |
| `withRotationSignals(vec)` | vector | Sector rotation data |
| `withJoblessClaims(alert)` | optional | Jobless claims warning |
| `addQuote(symbol, quote)` | Quote | Add single quote |
| `addPosition(pos)` | Position | Add single position |
| `addEmploymentSignal(sig)` | EmploymentSignal | Add single signal |
| `build()` | Terminal | Construct StrategyContext |

---

## Strategy Builders - Fluent Strategy Configuration

### SectorRotationStrategy Builder

```cpp
auto sector_rotation = SectorRotationStrategy::builder()
    // Signal weighting
    .withEmploymentWeight(0.60)
    .withSentimentWeight(0.30)
    .withMomentumWeight(0.10)

    // Sector selection
    .topNOverweight(3)
    .bottomNUnderweight(2)

    // Signal thresholds
    .minCompositeScore(0.60)
    .rotationThreshold(0.70)

    // Position sizing
    .maxSectorAllocation(0.25)
    .minSectorAllocation(0.05)

    // Rebalancing
    .rebalanceFrequency(30)

    // Data sources
    .withDatabasePath("data/bigbrother.duckdb")
    .withScriptsPath("scripts")

    .build();

mgr.addStrategy(std::move(sector_rotation));
```

### StraddleStrategy Builder

```cpp
auto straddle = StraddleStrategy::builder()
    .withMinIVRank(0.70)
    .withMaxDistance(0.10)
    .build();
```

### StrangleStrategy Builder

```cpp
auto strangle = StrangleStrategy::builder()
    .withMinIVRank(0.65)
    .withWingWidth(0.20)
    .build();
```

### VolatilityArbStrategy Builder

```cpp
auto vol_arb = VolatilityArbStrategy::builder()
    .withMinIVHVSpread(0.15)
    .withLookbackPeriod(30)
    .build();
```

---

## PerformanceQueryBuilder - Analyze Strategy Performance

Query and analyze strategy performance metrics.

```cpp
// Single strategy analysis
auto perf = mgr.performanceBuilder()
    .forStrategy("Sector Rotation (Multi-Signal)")
    .inPeriod(start_timestamp, end_timestamp)
    .minTradeCount(10)
    .calculate();

if (perf) {
    std::cout << "Strategy: " << perf->strategy_name << "\n";
    std::cout << "Signals Generated: " << perf->signals_generated << "\n";
    std::cout << "Win Rate: " << (perf->win_rate * 100) << "%\n";
    std::cout << "Sharpe Ratio: " << perf->sharpe_ratio << "\n";
    std::cout << "Max Drawdown: " << perf->max_drawdown << "\n";
    std::cout << "Total P&L: $" << perf->total_pnl << "\n";
}
```

### PerformanceQueryBuilder Methods

| Method | Type | Purpose |
|--------|------|---------|
| `forStrategy(name)` | string | Target strategy name |
| `inPeriod(start, end)` | Timestamp pair | Analysis time period |
| `minTradeCount(count)` | int | Minimum trades required |
| `calculate()` | Terminal | Execute query and return metrics |

---

## ReportBuilder - Generate Strategy Reports

Create comprehensive strategy performance reports.

```cpp
// Report all strategies
std::string report = mgr.reportBuilder()
    .allStrategies()
    .withMetrics({"sharpe", "win_rate", "max_drawdown", "profit_factor"})
    .sortBy("sharpe_ratio")
    .descending(true)
    .generate();

std::cout << report << "\n";

// Report specific strategies
std::string filtered_report = mgr.reportBuilder()
    .forStrategy("SectorRotation")
    .forStrategy("Volatility Arbitrage")
    .withMetrics({"sharpe", "win_rate"})
    .sortBy("win_rate")
    .generate();
```

### ReportBuilder Methods

| Method | Type | Purpose |
|--------|------|---------|
| `allStrategies()` | - | Include all strategies |
| `forStrategy(name)` | string | Add specific strategy |
| `withMetrics(list)` | vector\<string\> | Performance metrics to include |
| `sortBy(field)` | string | Sort results by field |
| `descending(bool)` | bool | Sort direction (default: ascending) |
| `generate()` | Terminal | Create and return report string |

---

## Complete Example: Multi-Strategy Setup

```cpp
#include "bigbrother.strategy"

using namespace bigbrother::strategy;
using namespace bigbrother::strategies;

int main() {
    // 1. Create manager
    StrategyManager mgr;

    // 2. Add strategies with fluent configuration
    mgr.addStrategy(SectorRotationStrategy::builder()
        .withEmploymentWeight(0.65)
        .withSentimentWeight(0.25)
        .topNOverweight(4)
        .bottomNUnderweight(3)
        .rotationThreshold(0.75)
        .build())

    .addStrategy(StraddleStrategy::builder()
        .withMinIVRank(0.75)
        .build())

    .addStrategy(VolatilityArbStrategy::builder()
        .withMinIVHVSpread(0.15)
        .withLookbackPeriod(20)
        .build());

    // 3. Build market context
    auto context = StrategyContext::builder()
        .withAccountValue(500000.0)
        .withAvailableCapital(50000.0)
        .withCurrentTime(getCurrentTimestamp())
        .withQuotes(fetchCurrentQuotes())
        .withEmploymentSignals(fetchEmploymentData())
        .build();

    // 4. Generate high-quality signals
    auto signals = mgr.signalBuilder()
        .forContext(context)
        .fromStrategies({"Sector Rotation (Multi-Signal)", "Volatility Arbitrage"})
        .withMinConfidence(0.70)
        .withMinRiskRewardRatio(2.0)
        .onlyActionable(true)
        .limitTo(10)
        .generate();

    // 5. Process signals
    for (auto const& signal : signals) {
        executeSignal(signal);
    }

    // 6. Generate performance report
    auto report = mgr.reportBuilder()
        .allStrategies()
        .withMetrics({"sharpe", "win_rate", "max_drawdown"})
        .sortBy("sharpe_ratio")
        .descending(true)
        .generate();

    std::cout << report << "\n";

    return 0;
}
```

---

## Design Patterns Used

### Builder Pattern
- **StrategyContext::builder()** - Flexible context construction
- **SectorRotationStrategy::builder()** - Strategy configuration
- **SignalBuilder** - Signal filtering and generation

### Fluent Interface
- Method chaining with `-> StrategyManager&` returns
- All intermediate methods marked with `[[nodiscard]]`
- Terminal operations (`.generate()`, `.build()`, `.calculate()`)

### Strategy Pattern
- `IStrategy` base interface
- Pluggable strategy implementations
- Thread-safe management

### Optional Values
- `std::optional` for conditional filtering
- Elegant handling of optional configuration

---

## Thread Safety

All StrategyManager operations are thread-safe:

```cpp
// Safe to call from multiple threads
std::thread t1([&mgr, &ctx]() {
    auto signals = mgr.signalBuilder()
        .forContext(ctx)
        .generate();
});

std::thread t2([&mgr]() {
    mgr.setStrategyActive("SectorRotation", false);
});

t1.join();
t2.join();
```

---

## Performance Considerations

### Memory Efficiency
- Move semantics throughout for large data structures
- No unnecessary copies in signal filtering
- Efficient range-based operations

### Computational Efficiency
- Lazy evaluation in builders
- Single-pass filtering
- Early termination with `limitTo()`

### Example: Filtering Optimization

```cpp
// Efficient: Single pass with early termination
auto signals = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.75)      // First filter
    .withMinRiskRewardRatio(2.0)   // Second filter
    .limitTo(5)                     // Early termination
    .generate();
```

---

## Backward Compatibility

Traditional methods still available:

```cpp
// Old API still works
mgr.addStrategy(std::make_unique<StraddleStrategy>());
auto all_signals = mgr.generateSignals(context);
auto strategies = mgr.getStrategies();

// New fluent API
mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.70)
    .generate();
```

---

## Type Safety

All builder methods are type-safe with proper return types:

```cpp
// Compiler error - wrong type
mgr.signalBuilder()
    .withMinConfidence("0.70")  // ERROR: expects double
    .generate();

// Correct
mgr.signalBuilder()
    .withMinConfidence(0.70)    // OK: double
    .generate();
```

---

## Future Extensions

The fluent API framework supports easy extension:

```cpp
// Add new filter methods to SignalBuilder
[[nodiscard]] auto minExpectedReturn(double ret) -> SignalBuilder& {
    min_return_ = ret;
    return *this;
}

[[nodiscard]] auto maxMaxRisk(double risk) -> SignalBuilder& {
    max_risk_ = risk;
    return *this;
}

// Use in chains
auto signals = mgr.signalBuilder()
    .forContext(context)
    .minExpectedReturn(100.0)
    .maxMaxRisk(50.0)
    .generate();
```

---

## Summary

The fluent API pattern in BigBrotherAnalytics provides:

1. **Readable Code** - Clear intention through method chaining
2. **Type Safety** - Compile-time checking of configuration
3. **Flexibility** - Mix and match filters and configurations
4. **Performance** - Efficient filtering and early termination
5. **Maintainability** - Clear structure and easy extensions
6. **Thread Safety** - Safe concurrent access
7. **Backward Compatibility** - Old code still works

This makes the strategy framework both powerful and easy to use.
