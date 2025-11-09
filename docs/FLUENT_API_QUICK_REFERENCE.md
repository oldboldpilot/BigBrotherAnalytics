# Fluent API Quick Reference

## At a Glance

Complete fluent API reference for the BigBrotherAnalytics strategy framework.

---

## StrategyManager Fluent Methods

### Adding & Removing Strategies
```cpp
mgr.addStrategy(std::make_unique<StraddleStrategy>())
   .addStrategy(std::make_unique<StrangleStrategy>())
   .removeStrategy("Long Straddle");
```

### Managing Strategy State
```cpp
mgr.setStrategyActive("SectorRotation", true)
   .enableAll()
   .disableAll();
```

### Getting Information
```cpp
size_t count = mgr.getStrategyCount();
auto strategy = mgr.getStrategy("Long Straddle");
auto all = mgr.getStrategies();
```

### Builder Accessors
```cpp
auto signals = mgr.signalBuilder();           // SignalBuilder
auto perf = mgr.performanceBuilder();         // PerformanceQueryBuilder
auto report = mgr.reportBuilder();            // ReportBuilder
```

---

## SignalBuilder - Signal Generation & Filtering

### Basic Usage
```cpp
auto signals = mgr.signalBuilder()
    .forContext(context)
    .generate();
```

### With Filters
```cpp
auto signals = mgr.signalBuilder()
    .forContext(context)                       // Required
    .fromStrategies({"SectorRotation"})       // Optional
    .withMinConfidence(0.70)                   // Optional
    .withMinRiskRewardRatio(2.0)              // Optional
    .onlyActionable(true)                      // Optional
    .limitTo(10)                               // Optional
    .generate();                               // Terminal
```

### Available Methods
| Method | Parameter(s) | Returns |
|--------|------------|---------|
| `forContext()` | StrategyContext const& | SignalBuilder& |
| `fromStrategies()` | vector\<string\> | SignalBuilder& |
| `withMinConfidence()` | double (0.0-1.0) | SignalBuilder& |
| `withMinRiskRewardRatio()` | double | SignalBuilder& |
| `onlyActionable()` | bool | SignalBuilder& |
| `limitTo()` | int | SignalBuilder& |
| `generate()` | - | vector\<TradingSignal\> |

---

## StrategyContext::builder() - Context Construction

### Full Example
```cpp
auto context = StrategyContext::builder()
    // Account settings
    .withAccountValue(500000.0)
    .withAvailableCapital(50000.0)
    .withCurrentTime(timestamp)
    // Market data
    .withQuotes(quotes_map)
    .withOptions(options_map)
    .withPositions(positions_vec)
    // Employment signals
    .withEmploymentSignals(emp_signals)
    .withRotationSignals(rotation_signals)
    .withJoblessClaims(jobless_alert)
    .build();
```

### Incremental Building
```cpp
auto builder = StrategyContext::builder()
    .withAccountValue(100000.0)
    .withAvailableCapital(10000.0);

builder.addQuote("SPY", quote1)
       .addQuote("QQQ", quote2)
       .addPosition(pos1)
       .addEmploymentSignal(sig1);

auto context = builder.build();
```

### Available Methods
| Method | Parameter(s) | Returns |
|--------|------------|---------|
| `withAccountValue()` | double | ContextBuilder& |
| `withAvailableCapital()` | double | ContextBuilder& |
| `withCurrentTime()` | Timestamp | ContextBuilder& |
| `withQuotes()` | map\<string, Quote\> | ContextBuilder& |
| `withOptions()` | map\<string, OptionsChainData\> | ContextBuilder& |
| `withPositions()` | vector\<Position\> | ContextBuilder& |
| `withEmploymentSignals()` | vector\<EmploymentSignal\> | ContextBuilder& |
| `withRotationSignals()` | vector\<SectorRotationSignal\> | ContextBuilder& |
| `withJoblessClaims()` | optional\<EmploymentSignal\> | ContextBuilder& |
| `addQuote()` | string, Quote | ContextBuilder& |
| `addPosition()` | Position | ContextBuilder& |
| `addEmploymentSignal()` | EmploymentSignal | ContextBuilder& |
| `build()` | - | StrategyContext |

---

## SectorRotationStrategy::builder()

### Configuration
```cpp
auto strategy = SectorRotationStrategy::builder()
    // Signal weighting
    .withEmploymentWeight(0.60)
    .withSentimentWeight(0.30)
    .withMomentumWeight(0.10)
    // Sector selection
    .topNOverweight(3)
    .bottomNUnderweight(2)
    // Thresholds
    .minCompositeScore(0.60)
    .rotationThreshold(0.70)
    // Sizing
    .maxSectorAllocation(0.25)
    .minSectorAllocation(0.05)
    // Frequency
    .rebalanceFrequency(30)
    // Paths
    .withDatabasePath("data/bigbrother.duckdb")
    .withScriptsPath("scripts")
    .build();  // Returns std::unique_ptr<IStrategy>
```

### Available Methods
| Method | Parameter | Default |
|--------|-----------|---------|
| `withEmploymentWeight()` | double | 0.60 |
| `withSentimentWeight()` | double | 0.30 |
| `withMomentumWeight()` | double | 0.10 |
| `topNOverweight()` | int | 3 |
| `bottomNUnderweight()` | int | 2 |
| `minCompositeScore()` | double | 0.60 |
| `rotationThreshold()` | double | 0.70 |
| `maxSectorAllocation()` | double | 0.25 |
| `minSectorAllocation()` | double | 0.05 |
| `rebalanceFrequency()` | int | 30 |
| `withDatabasePath()` | string | "data/bigbrother.duckdb" |
| `withScriptsPath()` | string | "scripts" |

---

## StraddleStrategy::builder()

```cpp
auto strategy = StraddleStrategy::builder()
    .withMinIVRank(0.70)
    .withMaxDistance(0.10)
    .build();
```

| Method | Parameter | Default |
|--------|-----------|---------|
| `withMinIVRank()` | double | 0.70 |
| `withMaxDistance()` | double | 0.10 |

---

## StrangleStrategy::builder()

```cpp
auto strategy = StrangleStrategy::builder()
    .withMinIVRank(0.65)
    .withWingWidth(0.20)
    .build();
```

| Method | Parameter | Default |
|--------|-----------|---------|
| `withMinIVRank()` | double | 0.65 |
| `withWingWidth()` | double | 0.20 |

---

## VolatilityArbStrategy::builder()

```cpp
auto strategy = VolatilityArbStrategy::builder()
    .withMinIVHVSpread(0.15)
    .withLookbackPeriod(30)
    .build();
```

| Method | Parameter | Default |
|--------|-----------|---------|
| `withMinIVHVSpread()` | double | 0.10 |
| `withLookbackPeriod()` | int | 30 |

---

## PerformanceQueryBuilder - Performance Analysis

### Usage
```cpp
auto perf = mgr.performanceBuilder()
    .forStrategy("SectorRotation")
    .inPeriod(start_time, end_time)
    .minTradeCount(10)
    .calculate();

if (perf) {
    std::cout << "Win Rate: " << (perf->win_rate * 100) << "%\n";
    std::cout << "Sharpe: " << perf->sharpe_ratio << "\n";
}
```

### Available Methods
| Method | Parameter(s) | Returns |
|--------|------------|---------|
| `forStrategy()` | string | PerformanceQueryBuilder& |
| `inPeriod()` | Timestamp, Timestamp | PerformanceQueryBuilder& |
| `minTradeCount()` | int | PerformanceQueryBuilder& |
| `calculate()` | - | optional\<PerformanceMetrics\> |

### PerformanceMetrics Fields
```cpp
struct PerformanceMetrics {
    std::string strategy_name;
    int signals_generated;
    int trades_executed;
    int winning_trades;
    int losing_trades;
    double total_pnl;
    double win_rate;           // 0.0 to 1.0
    double sharpe_ratio;
    double max_drawdown;
    double profit_factor;
    Timestamp period_start;
    Timestamp period_end;
};
```

---

## ReportBuilder - Strategy Reports

### Usage
```cpp
auto report = mgr.reportBuilder()
    .allStrategies()
    .withMetrics({"sharpe", "win_rate", "max_drawdown"})
    .sortBy("sharpe_ratio")
    .descending(true)
    .generate();

std::cout << report << "\n";
```

### Available Methods
| Method | Parameter(s) | Returns |
|--------|------------|---------|
| `allStrategies()` | - | ReportBuilder& |
| `forStrategy()` | string | ReportBuilder& |
| `withMetrics()` | vector\<string\> | ReportBuilder& |
| `sortBy()` | string | ReportBuilder& |
| `descending()` | bool | ReportBuilder& |
| `generate()` | - | string |

### Available Metrics
- "sharpe" or "sharpe_ratio"
- "win_rate"
- "max_drawdown"
- "profit_factor"
- "pnl"
- "signals_generated"
- "trades_executed"

---

## Complete Example

```cpp
#include <iostream>
// Import modules when available
// import bigbrother.strategy;
// import bigbrother.strategies;

int main() {
    // 1. Create manager
    StrategyManager mgr;

    // 2. Add strategies with builders
    mgr.addStrategy(SectorRotationStrategy::builder()
        .withEmploymentWeight(0.65)
        .topNOverweight(4)
        .build())
    .addStrategy(StraddleStrategy::builder()
        .withMinIVRank(0.75)
        .build())
    .addStrategy(VolatilityArbStrategy::builder()
        .withMinIVHVSpread(0.15)
        .build());

    // 3. Build context
    auto context = StrategyContext::builder()
        .withAccountValue(500000.0)
        .withAvailableCapital(50000.0)
        .withCurrentTime(getCurrentTimestamp())
        .withQuotes(fetchQuotes())
        .withEmploymentSignals(fetchEmploymentData())
        .build();

    // 4. Generate signals
    auto signals = mgr.signalBuilder()
        .forContext(context)
        .withMinConfidence(0.70)
        .limitTo(10)
        .generate();

    // 5. Process signals
    for (auto const& signal : signals) {
        executeSignal(signal);
    }

    // 6. Analyze performance
    if (auto perf = mgr.performanceBuilder()
        .forStrategy("SectorRotation")
        .calculate()) {
        std::cout << "Win Rate: " << (perf->win_rate * 100) << "%\n";
    }

    // 7. Generate report
    auto report = mgr.reportBuilder()
        .allStrategies()
        .withMetrics({"sharpe", "win_rate"})
        .generate();

    std::cout << report << "\n";

    return 0;
}
```

---

## Key Features

### Thread Safety
All StrategyManager operations are thread-safe:
```cpp
// Safe to use from multiple threads
std::thread t1([&mgr] {
    mgr.signalBuilder().forContext(ctx).generate();
});
std::thread t2([&mgr] {
    mgr.setStrategyActive("Strategy", false);
});
```

### Type Safety
```cpp
mgr.signalBuilder()
    .withMinConfidence(0.70)      // OK: double
    .withMinConfidence("0.70");   // ERROR: expects double
```

### Move Semantics
All builders use move semantics for efficiency:
```cpp
std::vector<Quote> quotes = fetchQuotes();
auto context = StrategyContext::builder()
    .withQuotes(std::move(quotes))  // No copy!
    .build();
```

### Error Prevention
```cpp
// Compiler warning - forgot to chain
mgr.addStrategy(strat);

// Correct - explicit chaining
mgr.addStrategy(strat)
   .addStrategy(strat2);
```

---

## Comparison: Old vs New API

### Old API (Still Works)
```cpp
std::unique_ptr<IStrategy> s1 = std::make_unique<StraddleStrategy>();
std::unique_ptr<IStrategy> s2 = std::make_unique<StrangleStrategy>();

mgr.addStrategy(std::move(s1));
mgr.addStrategy(std::move(s2));

auto all_signals = mgr.generateSignals(context);

// Filter manually
std::vector<TradingSignal> filtered;
for (auto const& sig : all_signals) {
    if (sig.confidence >= 0.70) {
        filtered.push_back(sig);
    }
}
```

### New Fluent API
```cpp
mgr.addStrategy(StraddleStrategy::builder().build())
   .addStrategy(StrangleStrategy::builder().build());

auto signals = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.70)
    .generate();
```

---

## Performance Tips

### Efficient Signal Filtering
```cpp
// GOOD: Filters applied in order, early termination
auto signals = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.75)        // Strictest first
    .withMinRiskRewardRatio(2.0)    // Then risk/reward
    .onlyActionable(true)            // Then actionability
    .limitTo(5)                      // Early termination
    .generate();

// LESS EFFICIENT: Processes all signals
auto all = mgr.generateSignals(context);  // All 1000 signals
// Then filter...
```

### Reusing Contexts
```cpp
// Build once, use multiple times
auto context = StrategyContext::builder()
    .withAccountValue(500000.0)
    .withQuotes(quotes)
    .build();

// Generate different signal sets
auto high_confidence = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.80)
    .generate();

auto all_signals = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.50)
    .generate();
```

---

## Troubleshooting

### Compiler Error: "Ignoring return value"
**Problem:**
```cpp
mgr.addStrategy(strat);  // WARNING: ignoring return value
```

**Solution:**
```cpp
mgr.addStrategy(strat)
   .addStrategy(strat2);  // Chain properly
```

**Or disable for that line:**
```cpp
(void)mgr.addStrategy(strat);
```

### Optional with No Value
**Problem:**
```cpp
auto perf = mgr.performanceBuilder()
    .calculate();

std::cout << perf->win_rate;  // CRASH if !perf
```

**Solution:**
```cpp
if (perf) {
    std::cout << perf->win_rate << "\n";
}

// Or with value_or
double rate = perf ? perf->win_rate : 0.0;
```

### Context Building Errors
**Problem:**
```cpp
auto ctx = StrategyContext::builder()
    .build();  // Empty context!
```

**Solution:**
```cpp
auto ctx = StrategyContext::builder()
    .withAccountValue(100000.0)
    .withAvailableCapital(10000.0)
    .withQuotes(my_quotes)
    .build();
```

---

## See Also

- **Full Guide:** `FLUENT_API_GUIDE.md`
- **Implementation Details:** `FLUENT_API_IMPLEMENTATION_SUMMARY.md`
- **Tests:** `tests/test_fluent_api.cpp`

