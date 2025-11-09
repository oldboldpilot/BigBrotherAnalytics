# Employment Signals Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BLS Employment Data Sources                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Job Openings │  │   Jobless    │  │    Sector    │              │
│  │   & Labor    │  │   Claims     │  │  Employment  │              │
│  │   Turnover   │  │   (Weekly)   │  │   (Monthly)  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                        ┌───────────────┐
                        │   DuckDB      │
                        │   Database    │
                        └───────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Python Analysis Layer (scripts/)                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  employment_signals.py                                     │    │
│  │  ────────────────────                                      │    │
│  │  • generateSignals() → Employment trends by sector         │    │
│  │  • rotationSignals() → Overweight/Underweight recs         │    │
│  │  • checkJoblessClaims() → Recession warning detection      │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                         (JSON output)
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│          C++ Employment Signals Module (market_intelligence/)      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  EmploymentSignalGenerator                                 │    │
│  │  ────────────────────────                                  │    │
│  │  • generateSignals() → vector<EmploymentSignal>            │    │
│  │  • generateRotationSignals() → vector<SectorRotationSignal>│   │
│  │  • checkJoblessClaimsSpike() → optional<EmploymentSignal>  │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                         (C++ objects)
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│           StrategyContext (trading_decision/)                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Fields:                                                    │    │
│  │  • employment_signals (vector)                              │    │
│  │  • rotation_signals (vector)                                │    │
│  │  • jobless_claims_alert (optional)                          │    │
│  │                                                             │    │
│  │  Helper Methods:                                            │    │
│  │  • getEmploymentSignalsForSector(name)                      │    │
│  │  • getRotationSignalForSector(name)                         │    │
│  │  • hasRecessionWarning()                                    │    │
│  │  • getAggregateEmploymentScore()                            │    │
│  │  • getStrongestEmploymentSignals(N)                         │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                    (Passed to all strategies)
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Trading Strategies                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Sector Rotation  │  │  Volatility Arb  │  │  Iron Condor     │  │
│  │                  │  │                  │  │                  │  │
│  │ Uses rotation    │  │ Uses employment  │  │ Uses recession   │  │
│  │ signals directly │  │ as risk filter   │  │ warning to       │  │
│  │ for ETF trades   │  │                  │  │ widen strikes    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                    ┌─────────────────────┐
                    │  TradingSignals     │
                    │  (Buy/Sell/Hold)    │
                    └─────────────────────┘
```

## Data Structures

### EmploymentSignal
```cpp
struct EmploymentSignal {
    EmploymentSignalType type;           // JoblessClaimsSpike, SectorLayoffs, etc.
    int sector_code;                     // GICS code (10-60)
    string sector_name;                  // "Information Technology"
    double confidence;                   // 0.0 to 1.0
    double employment_change;            // % change in employment
    string rationale;                    // Human-readable explanation
    Timestamp timestamp;

    // Trading implications
    bool bullish;                        // Positive for sector
    bool bearish;                        // Negative for sector
    double signal_strength;              // -1.0 (bearish) to +1.0 (bullish)

    bool isActionable() const;           // confidence > 0.60 && |strength| > 0.50
};
```

### SectorRotationSignal
```cpp
struct SectorRotationSignal {
    int sector_code;                     // GICS code
    string sector_name;                  // "Financials"
    string sector_etf;                   // "XLF"

    // Signal components
    double employment_score;             // From BLS data
    double sentiment_score;              // From news (future)
    double technical_score;              // From price action (future)
    double composite_score;              // Weighted average

    // Recommendation
    enum Action { Overweight, Neutral, Underweight };
    Action action;
    double target_allocation;            // % of portfolio

    bool isStrongSignal() const;         // |composite_score| > 0.70
};
```

### StrategyContext
```cpp
struct StrategyContext {
    // Existing fields
    unordered_map<string, Quote> current_quotes;
    unordered_map<string, OptionsChainData> options_chains;
    vector<Position> current_positions;
    double account_value;
    double available_capital;
    Timestamp current_time;

    // NEW: Employment signals
    vector<EmploymentSignal> employment_signals;
    vector<SectorRotationSignal> rotation_signals;
    optional<EmploymentSignal> jobless_claims_alert;

    // NEW: Helper methods
    vector<EmploymentSignal> getEmploymentSignalsForSector(string) const;
    optional<SectorRotationSignal> getRotationSignalForSector(string) const;
    bool hasRecessionWarning() const;
    double getAggregateEmploymentScore() const;
    vector<EmploymentSignal> getStrongestEmploymentSignals(int limit=5) const;
};
```

## Module Dependencies

```
bigbrother.strategy (trading_decision/strategy.cppm)
    ├── import bigbrother.utils.types
    ├── import bigbrother.options.pricing
    ├── import bigbrother.risk_management
    ├── import bigbrother.schwab_api
    └── import bigbrother.employment.signals  ← NEW

bigbrother.employment.signals (market_intelligence/employment_signals.cppm)
    └── import bigbrother.utils.types
```

## CMake Dependencies

```cmake
# trading_decision library
target_link_libraries(trading_decision
    PUBLIC
    utils
    correlation_engine
    options_pricing
    risk_management
    schwab_api
    market_intelligence  ← NEW: Added for employment signals
    OpenMP::OpenMP_CXX
)
```

## Usage Flow in Trading Application

```cpp
// 1. Initialize generator
EmploymentSignalGenerator generator("scripts", "data/bigbrother.duckdb");

// 2. Generate signals
auto employment_signals = generator.generateSignals();
auto rotation_signals = generator.generateRotationSignals();
auto jobless_alert = generator.checkJoblessClaimsSpike();

// 3. Populate context
StrategyContext context;
context.employment_signals = std::move(employment_signals);
context.rotation_signals = std::move(rotation_signals);
context.jobless_claims_alert = jobless_alert;

// ... populate market data (quotes, options chains, etc.) ...

// 4. Generate trading signals
StrategyManager strategy_manager;
strategy_manager.addStrategy(std::make_unique<SectorRotationStrategy>());
strategy_manager.addStrategy(std::make_unique<VolatilityArbStrategy>());

auto trading_signals = strategy_manager.generateSignals(context);

// 5. Each strategy can now access employment data
//    - SectorRotationStrategy uses rotation_signals
//    - VolatilityArbStrategy uses employment as risk filter
//    - All strategies can check hasRecessionWarning()
```

## Strategy Access Patterns

### Pattern 1: Direct Signal Iteration
```cpp
for (auto const& signal : context.employment_signals) {
    if (signal.isActionable()) {
        // Use signal.sector_name, signal.signal_strength, etc.
    }
}
```

### Pattern 2: Sector-Specific Lookup
```cpp
auto tech_signals = context.getEmploymentSignalsForSector("Information Technology");
if (!tech_signals.empty() && tech_signals[0].bullish) {
    // Generate buy signals for tech stocks/ETFs
}
```

### Pattern 3: Rotation Signal Usage
```cpp
auto rotation = context.getRotationSignalForSector("Energy");
if (rotation.has_value() && rotation->action == Action::Overweight) {
    // Buy XLE (Energy sector ETF)
}
```

### Pattern 4: Aggregate Health Check
```cpp
double employment_health = context.getAggregateEmploymentScore();
if (employment_health < -0.5) {
    // Economy deteriorating - reduce position sizes
    position_size *= 0.5;
}
```

### Pattern 5: Top Signals Only
```cpp
auto top_signals = context.getStrongestEmploymentSignals(3);
for (auto const& signal : top_signals) {
    // Focus on strongest signals (positive or negative)
    if (signal.signal_strength > 0.7) {
        // Strong bullish signal
    }
    else if (signal.signal_strength < -0.7) {
        // Strong bearish signal
    }
}
```

## Sector ETF Mappings

The system tracks 11 GICS sectors with corresponding liquid ETFs:

| Code | Sector                     | ETF  | Description                           |
|------|----------------------------|------|---------------------------------------|
| 10   | Energy                     | XLE  | Oil, gas, energy services             |
| 15   | Materials                  | XLB  | Chemicals, mining, metals             |
| 20   | Industrials                | XLI  | Manufacturing, aerospace, defense     |
| 25   | Consumer Discretionary     | XLY  | Retail, restaurants, leisure          |
| 30   | Consumer Staples           | XLP  | Food, beverages, household products   |
| 35   | Health Care                | XLV  | Pharmaceuticals, biotech, healthcare  |
| 40   | Financials                 | XLF  | Banks, insurance, brokers             |
| 45   | Information Technology     | XLK  | Software, hardware, semiconductors    |
| 50   | Communication Services     | XLC  | Telecom, media, entertainment         |
| 55   | Utilities                  | XLU  | Electric, gas, water utilities        |
| 60   | Real Estate                | XLRE | REITs, real estate investment         |

## Signal Generation Frequency

- **Employment Signals**: Monthly (BLS data releases)
- **Rotation Signals**: Monthly (based on employment data)
- **Jobless Claims**: Weekly (Thursday releases)

Recommended refresh strategy:
- Query employment signals once per day
- Cache results for intraday strategy evaluation
- Refresh immediately after BLS data releases

## Performance Characteristics

### Memory Footprint
```
StrategyContext employment data:
  - 11 sectors × 1-2 employment signals = ~22 signals
  - Each EmploymentSignal ≈ 150 bytes
  - Each SectorRotationSignal ≈ 200 bytes
  - Total: < 5KB per context

Impact: Negligible
```

### Computational Complexity
```
Helper Methods:
  - getEmploymentSignalsForSector(): O(n), n ≤ 22
  - getRotationSignalForSector(): O(n), n ≤ 11
  - hasRecessionWarning(): O(1)
  - getAggregateEmploymentScore(): O(n), n ≤ 22
  - getStrongestEmploymentSignals(): O(n log n), n ≤ 22

Typical execution time: < 1μs per call
Impact: Negligible
```

### Data Flow Performance
```
Python signal generation: ~100-500ms (database queries)
C++ signal parsing: ~1-5ms (JSON parsing)
Context population: ~0.1ms (vector copies)
Strategy evaluation: ~0.001ms per signal check

Bottleneck: Python database queries (acceptable for monthly refresh)
```

## Error Handling

### Empty Employment Data
```cpp
// All helper methods handle empty data gracefully:
context.employment_signals = {};  // Empty vector
auto score = context.getAggregateEmploymentScore();  // Returns 0.0
auto top = context.getStrongestEmploymentSignals(5);  // Returns empty vector
```

### Missing Sector Data
```cpp
auto signals = context.getEmploymentSignalsForSector("NonExistent");
// Returns empty vector (no exception)

auto rotation = context.getRotationSignalForSector("NonExistent");
// Returns std::nullopt (check with has_value())
```

### No Recession Warning
```cpp
if (!context.hasRecessionWarning()) {
    // Normal operation - jobless_claims_alert is std::nullopt
}
```

## Testing Strategy

### Unit Tests
1. Test each StrategyContext helper method
2. Test EmploymentSignalGenerator parsing
3. Test signal filtering and sorting logic

### Integration Tests
1. Test full pipeline (DB → Python → C++ → Strategy)
2. Test strategy signal generation with employment data
3. Test backward compatibility with empty employment data

### Performance Tests
1. Measure context population overhead
2. Measure helper method execution time
3. Ensure < 1% overhead in strategy evaluation

## Future Enhancements

### Phase 1: Core Integration (COMPLETE)
- ✓ Add employment signals to StrategyContext
- ✓ Implement helper methods
- ✓ Integrate with build system
- ✓ Document API and usage patterns

### Phase 2: Signal Enhancement (Future)
- Add sentiment_score from news analysis
- Add technical_score from price action
- Implement composite scoring algorithm
- Add signal confidence calibration

### Phase 3: Historical Analysis (Future)
- Track employment signal accuracy over time
- Correlate signals with actual sector performance
- Build ML model for signal weighting
- Implement adaptive scoring

### Phase 4: Real-time Updates (Future)
- WebSocket feed for jobless claims
- Intraday employment news monitoring
- Real-time signal strength adjustment
- Event-driven signal updates

## Conclusion

The employment signals integration provides a robust, efficient, and easy-to-use mechanism for incorporating BLS employment data into trading decisions. The design prioritizes:

1. **Simplicity**: Clean API with intuitive helper methods
2. **Performance**: Minimal overhead, efficient data structures
3. **Flexibility**: Multiple access patterns for different use cases
4. **Compatibility**: Zero breaking changes to existing strategies
5. **Extensibility**: Easy to add new signal types and scoring methods

All trading strategies can now leverage fundamental employment data alongside technical and options-based signals for more informed decision-making.
