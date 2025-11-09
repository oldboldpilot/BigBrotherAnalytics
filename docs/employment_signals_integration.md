# Employment Signals Integration in StrategyContext

## Overview

Employment signals from the BLS (Bureau of Labor Statistics) have been successfully integrated into the `StrategyContext` structure in `/home/muyiwa/Development/BigBrotherAnalytics/src/trading_decision/strategy.cppm`. This allows all trading strategies to access real-time employment data for sector-based decision making.

## Changes Made

### 1. Module Dependencies

**File**: `src/trading_decision/strategy.cppm`

Added employment signals module import:
```cpp
import bigbrother.employment.signals;
```

Added namespace usage:
```cpp
using namespace bigbrother::employment;
```

### 2. StrategyContext Fields

Three new fields added to `StrategyContext`:

```cpp
struct StrategyContext {
    // ... existing fields (quotes, options_chains, positions, etc.) ...

    // Employment Signals (from BLS data)
    std::vector<EmploymentSignal> employment_signals;
    std::vector<SectorRotationSignal> rotation_signals;
    std::optional<EmploymentSignal> jobless_claims_alert;

    // ... helper methods ...
};
```

#### Field Descriptions

1. **`employment_signals`** - Individual sector employment signals
   - Type: `std::vector<EmploymentSignal>`
   - Contains: Detailed employment metrics per sector
   - Fields per signal:
     - `sector_code` (GICS sector code)
     - `sector_name` (e.g., "Information Technology")
     - `confidence` (0.0 to 1.0)
     - `employment_change` (% change)
     - `signal_strength` (-1.0 to +1.0, where negative = bearish, positive = bullish)
     - `bullish`/`bearish` flags
     - `rationale` (human-readable explanation)

2. **`rotation_signals`** - Sector rotation recommendations
   - Type: `std::vector<SectorRotationSignal>`
   - Contains: Overweight/Underweight recommendations per sector
   - Fields per signal:
     - `sector_code`, `sector_name`, `sector_etf`
     - `employment_score`, `sentiment_score`, `technical_score`
     - `composite_score` (weighted average)
     - `action` (Overweight, Neutral, Underweight)
     - `target_allocation` (% of portfolio)

3. **`jobless_claims_alert`** - Recession warning
   - Type: `std::optional<EmploymentSignal>`
   - Contains: Alert if jobless claims spike >10% (recession indicator)
   - Usage: Check with `context.hasRecessionWarning()`

### 3. Helper Methods

Five utility methods added to `StrategyContext` for easy access to employment data:

#### `getEmploymentSignalsForSector(sector_name)`

Get all employment signals for a specific sector.

```cpp
auto signals = context.getEmploymentSignalsForSector("Information Technology");
for (auto const& signal : signals) {
    // Use signal.signal_strength, signal.confidence, etc.
}
```

#### `getRotationSignalForSector(sector_name)`

Get rotation recommendation for a specific sector.

```cpp
auto rotation = context.getRotationSignalForSector("Financials");
if (rotation.has_value() && rotation->isStrongSignal()) {
    if (rotation->action == SectorRotationSignal::Action::Overweight) {
        // Generate buy signal for rotation->sector_etf
    }
}
```

#### `hasRecessionWarning()`

Check if there's an active recession warning from jobless claims.

```cpp
if (context.hasRecessionWarning()) {
    // Reduce risk exposure, avoid new positions
}
```

#### `getAggregateEmploymentScore()`

Get overall employment health across all sectors.

```cpp
double overall_score = context.getAggregateEmploymentScore();
// Returns: -1.0 (very negative) to +1.0 (very positive)

if (overall_score < -0.5) {
    // Economy deteriorating - consider defensive positioning
}
```

#### `getStrongestEmploymentSignals(limit = 5)`

Get top N actionable employment signals, sorted by strength.

```cpp
auto top_signals = context.getStrongestEmploymentSignals(3);
for (auto const& signal : top_signals) {
    // Process strongest signals first
}
```

## Usage Examples for Strategy Developers

### Example 1: Sector Rotation Strategy

```cpp
auto SectorRotationStrategy::generateSignals(StrategyContext const& context)
    -> std::vector<TradingSignal> {

    std::vector<TradingSignal> signals;

    // Use rotation signals directly
    for (auto const& rotation : context.rotation_signals) {
        if (!rotation.isStrongSignal()) continue;

        if (rotation.action == SectorRotationSignal::Action::Overweight) {
            // Generate buy signal
            TradingSignal signal;
            signal.symbol = rotation.sector_etf;
            signal.type = SignalType::Buy;
            signal.confidence = std::abs(rotation.composite_score);
            signal.rationale = "Sector rotation: Overweight " + rotation.sector_name;
            signals.push_back(signal);
        }
        else if (rotation.action == SectorRotationSignal::Action::Underweight) {
            // Generate sell signal
            TradingSignal signal;
            signal.symbol = rotation.sector_etf;
            signal.type = SignalType::Sell;
            signal.confidence = std::abs(rotation.composite_score);
            signal.rationale = "Sector rotation: Underweight " + rotation.sector_name;
            signals.push_back(signal);
        }
    }

    return signals;
}
```

### Example 2: Risk-Adjusted Strategy with Employment Filter

```cpp
auto VolatilityArbStrategy::generateSignals(StrategyContext const& context)
    -> std::vector<TradingSignal> {

    // Check for recession warning
    if (context.hasRecessionWarning()) {
        // Reduce position sizes or skip signals entirely
        Logger::getInstance().warn("Recession warning active - reducing exposure");
        return {}; // No new positions during recession warning
    }

    // Get overall employment health
    double employment_score = context.getAggregateEmploymentScore();

    std::vector<TradingSignal> signals;

    for (auto const& [symbol, chain] : context.options_chains) {
        // ... generate volatility arbitrage signals ...

        // Adjust confidence based on employment trends
        signal.confidence *= (1.0 + employment_score * 0.2);

        // Check sector-specific employment
        auto sector_signals = context.getEmploymentSignalsForSector(
            getSectorForSymbol(symbol)
        );

        for (auto const& emp_signal : sector_signals) {
            if (emp_signal.bearish && emp_signal.signal_strength < -0.7) {
                // Reduce position size for weak sectors
                signal.max_risk *= 0.5;
            }
        }

        signals.push_back(signal);
    }

    return signals;
}
```

### Example 3: Defensive Strategy Trigger

```cpp
auto IronCondorStrategy::generateSignals(StrategyContext const& context)
    -> std::vector<TradingSignal> {

    // Get top employment signals
    auto top_signals = context.getStrongestEmploymentSignals(3);

    int bearish_count = 0;
    for (auto const& signal : top_signals) {
        if (signal.bearish) bearish_count++;
    }

    // If majority of top signals are bearish, widen strikes for safety
    double strike_width_multiplier = (bearish_count >= 2) ? 1.5 : 1.0;

    // ... generate iron condor signals with adjusted strikes ...
}
```

### Example 4: Employment-Driven Sector Selection

```cpp
auto generateSignals(StrategyContext const& context) -> std::vector<TradingSignal> {
    std::vector<TradingSignal> signals;

    // Focus on sectors with strong employment trends
    for (auto const& emp_signal : context.employment_signals) {
        if (!emp_signal.isActionable()) continue;

        if (emp_signal.bullish && emp_signal.signal_strength > 0.7) {
            // Look for opportunities in this sector's ETF
            auto rotation = context.getRotationSignalForSector(emp_signal.sector_name);

            if (rotation.has_value()) {
                std::string etf = rotation->sector_etf;

                // Check if we have options data for this ETF
                if (context.options_chains.contains(etf)) {
                    // Generate strategy-specific signals for this ETF
                    // ...
                }
            }
        }
    }

    return signals;
}
```

## Populating Employment Signals

To populate the employment signals in `StrategyContext`, use the `EmploymentSignalGenerator`:

```cpp
#include "market_intelligence/employment_signals.cppm"

using namespace bigbrother::employment;

// Create generator
EmploymentSignalGenerator generator("scripts", "data/bigbrother.duckdb");

// Generate signals
auto employment_signals = generator.generateSignals();
auto rotation_signals = generator.generateRotationSignals();
auto jobless_alert = generator.checkJoblessClaimsSpike();

// Populate context
StrategyContext context;
context.employment_signals = std::move(employment_signals);
context.rotation_signals = std::move(rotation_signals);
context.jobless_claims_alert = jobless_alert;

// ... populate other context fields (quotes, options, etc.) ...

// Pass to strategies
auto signals = strategy_manager.generateSignals(context);
```

## Data Flow

```
BLS Employment Data (DuckDB)
    ↓
scripts/employment_signals.py (Python analysis)
    ↓
EmploymentSignalGenerator (C++ wrapper)
    ↓
StrategyContext (employment_signals, rotation_signals)
    ↓
Trading Strategies (use signals in decision logic)
    ↓
TradingSignal (Buy/Sell recommendations)
```

## Backward Compatibility

**Breaking Changes**: NONE

All existing strategies continue to work without modification. The new employment signal fields are optional:
- Default-initialized to empty vectors/nullopt
- Strategies can ignore them if not needed
- No changes required to existing strategy implementations

Strategies can opt-in to using employment signals by accessing the new fields when they become available.

## Build System Changes

Updated `CMakeLists.txt` to link `market_intelligence` module to `trading_decision`:

```cmake
target_link_libraries(trading_decision
    PUBLIC
    # ... existing dependencies ...
    market_intelligence  # NEW: Added for employment signals
    OpenMP::OpenMP_CXX
)
```

## Testing Recommendations

1. **Unit Tests**: Test each helper method in `StrategyContext`
   ```cpp
   TEST(StrategyContext, GetEmploymentSignalsForSector) {
       StrategyContext ctx;
       ctx.employment_signals = {
           {/* sector_name = "Tech" */},
           {/* sector_name = "Financials" */}
       };
       auto tech_signals = ctx.getEmploymentSignalsForSector("Tech");
       EXPECT_EQ(tech_signals.size(), 1);
   }
   ```

2. **Integration Tests**: Test full signal generation flow
   ```cpp
   TEST(EmploymentIntegration, FullPipeline) {
       EmploymentSignalGenerator generator;
       auto signals = generator.generateSignals();

       StrategyContext ctx;
       ctx.employment_signals = signals;

       EXPECT_GT(ctx.getAggregateEmploymentScore(), -1.0);
       EXPECT_LT(ctx.getAggregateEmploymentScore(), 1.0);
   }
   ```

3. **Strategy Tests**: Verify strategies use employment data correctly
   ```cpp
   TEST(SectorRotationStrategy, UsesEmploymentSignals) {
       SectorRotationStrategy strategy;
       StrategyContext ctx;
       ctx.rotation_signals = /* ... mock data ... */;

       auto signals = strategy.generateSignals(ctx);
       EXPECT_GT(signals.size(), 0);
   }
   ```

## Performance Considerations

- **Memory**: Employment signals are stored by value in `StrategyContext`
  - Typical size: ~11 sectors × 2 signals = ~22 small structs
  - Total memory: < 5KB
  - No performance impact

- **Copy Overhead**: `StrategyContext` is passed by const reference to strategies
  - No copying of employment signals during strategy evaluation
  - Helper methods return vectors by value (RVO applies)

- **Lookup Performance**:
  - `getEmploymentSignalsForSector()`: O(n) linear search, n ≤ 11
  - `getRotationSignalForSector()`: O(n) linear search, n ≤ 11
  - Negligible impact (< 1μs per call)

## Sector Mappings

GICS Sector codes and corresponding ETFs used in signals:

| Sector Code | Sector Name                | ETF  |
|-------------|----------------------------|------|
| 10          | Energy                     | XLE  |
| 15          | Materials                  | XLB  |
| 20          | Industrials                | XLI  |
| 25          | Consumer Discretionary     | XLY  |
| 30          | Consumer Staples           | XLP  |
| 35          | Health Care                | XLV  |
| 40          | Financials                 | XLF  |
| 45          | Information Technology     | XLK  |
| 50          | Communication Services     | XLC  |
| 55          | Utilities                  | XLU  |
| 60          | Real Estate                | XLRE |

## Future Enhancements

1. **Signal Caching**: Cache employment signals with TTL (e.g., refresh hourly)
2. **Historical Trends**: Add trend analysis over multiple periods
3. **Correlation Analysis**: Correlate employment signals with actual sector performance
4. **ML Integration**: Use employment signals as features in ML models
5. **Custom Scoring**: Allow strategies to define custom employment scoring functions

## References

- Employment Signals Module: `/home/muyiwa/Development/BigBrotherAnalytics/src/market_intelligence/employment_signals.cppm`
- Strategy Module: `/home/muyiwa/Development/BigBrotherAnalytics/src/trading_decision/strategy.cppm`
- Python Signal Generator: `/home/muyiwa/Development/BigBrotherAnalytics/scripts/employment_signals.py`
- Sector Rotation Strategy Example: `/home/muyiwa/Development/BigBrotherAnalytics/src/trading_decision/strategies.cppm`
