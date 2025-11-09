# Employment Signals Module - Implementation Documentation

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-09
**Module:** `bigbrother.employment.signals`

## Overview

The Employment Signals module generates trading signals from BLS (Bureau of Labor Statistics) employment data stored in DuckDB. It provides three main capabilities:

1. **Employment Signal Generation** - Detects significant employment changes by sector
2. **Sector Rotation Signals** - Recommends portfolio allocation adjustments
3. **Jobless Claims Spike Detection** - Warns of recession indicators (future)

## Architecture

### Hybrid Python/C++ Design

The module uses a hybrid architecture for optimal performance and flexibility:

- **Python Backend** (`scripts/employment_signals.py`):
  - Queries DuckDB employment data
  - Calculates employment trends (3-month, 6-month)
  - Generates signals using threshold-based rules
  - Returns JSON for C++ consumption

- **C++ Frontend** (`src/market_intelligence/employment_signals.cppm`):
  - Provides clean API for trading engine integration
  - Calls Python backend via subprocess (`popen`)
  - Parses JSON responses into C++ structs
  - Integrates with existing C++23 module system

### Why This Design?

1. **DuckDB Access**: Python has mature DuckDB bindings, simpler than C++ integration
2. **Rapid Iteration**: Python allows quick algorithm changes without C++ recompilation
3. **Type Safety**: C++ provides compile-time type checking for trading engine
4. **Performance**: Python subprocess overhead negligible for signal generation (~1/day)

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DuckDB Database                          │
│  sector_employment_raw: 2,128 records (Jan 2021 - Aug 2025)    │
│  - report_date, employment_count, series_id                     │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│          Python Backend (employment_signals.py)                  │
│                                                                  │
│  1. Query latest employment data by BLS series                  │
│  2. Calculate % change over 3-month and 6-month periods         │
│  3. Apply thresholds (>5% for employment signals)               │
│  4. Calculate composite scores for rotation signals             │
│  5. Generate JSON output                                        │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│        C++ Frontend (EmploymentSignalGenerator)                  │
│                                                                  │
│  1. Execute Python script via popen()                           │
│  2. Capture JSON stdout                                         │
│  3. Parse JSON into C++ structs                                 │
│  4. Return vector<EmploymentSignal> or vector<RotationSignal>   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│              Trading Decision Engine                             │
│  - Consumes signals for sector rotation strategies              │
│  - Adjusts portfolio allocations based on recommendations       │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Employment Signal Generation

**Function:** `EmploymentSignalGenerator::generateSignals()`

**Algorithm:**
```
For each BLS series mapped to GICS sectors:
  1. Calculate 3-month employment change %
  2. Calculate 6-month employment change %
  3. If |change| > 5%:
     - Generate EmploymentImproving or EmploymentDeclining signal
     - Set confidence: 0.70 (base) or 0.85 (if both periods agree)
     - Calculate signal_strength: normalized to [-1.0, +1.0]
```

**Output Structure:**
```cpp
struct EmploymentSignal {
    EmploymentSignalType type;      // EmploymentImproving/Declining
    int sector_code;                // GICS sector (10-60)
    string sector_name;             // "Energy", "Financials", etc.
    double confidence;              // 0.70 or 0.85
    double employment_change;       // % change
    string rationale;               // Human-readable explanation
    Timestamp timestamp;            // Unix timestamp
    bool bullish/bearish;           // Direction flags
    double signal_strength;         // -1.0 to +1.0
};
```

**Example Signal:**
```json
{
  "type": "EmploymentImproving",
  "sector_code": 35,
  "sector_name": "Health Care",
  "confidence": 0.85,
  "employment_change": 7.2,
  "rationale": "Employment increased by 7.2% over 3 months",
  "timestamp": 1731135600,
  "bullish": true,
  "bearish": false,
  "signal_strength": 0.36
}
```

### 2. Sector Rotation Signals

**Function:** `EmploymentSignalGenerator::generateRotationSignals()`

**Algorithm:**
```
For each GICS sector:
  1. Calculate employment_score: weighted avg of 3m (60%) & 6m (40%) trends
  2. Normalize to [-1.0, +1.0] scale
  3. Calculate composite_score: employment_score * 1.0 (for now)
  4. Determine action:
     - If composite > +0.30: Overweight (target: 10-15%)
     - If composite < -0.30: Underweight (target: 3-7%)
     - Else: Neutral (equal weight: ~9%)
```

**Output Structure:**
```cpp
struct SectorRotationSignal {
    int sector_code;
    string sector_name;
    string sector_etf;              // XLE, XLF, XLV, etc.
    double employment_score;        // From BLS data
    double sentiment_score;         // Future: from news
    double technical_score;         // Future: from price action
    double composite_score;         // Weighted average
    Action action;                  // Overweight/Neutral/Underweight
    double target_allocation;       // % of portfolio
};
```

**Example Signal:**
```json
{
  "sector_code": 40,
  "sector_name": "Financials",
  "sector_etf": "XLF",
  "employment_score": 0.45,
  "sentiment_score": 0.0,
  "technical_score": 0.0,
  "composite_score": 0.45,
  "action": "Overweight",
  "target_allocation": 12.25
}
```

### 3. Jobless Claims Spike Detection

**Function:** `EmploymentSignalGenerator::checkJoblessClaimsSpike()`

**Status:** Placeholder implementation (jobless claims data not yet in database)

**Future Algorithm:**
```
1. Query latest weekly jobless claims from economic_data table
2. Calculate 4-week moving average
3. If current_week > 1.10 * moving_average:
   - Return RecessionWarning signal
   - Set high confidence (0.90+)
```

## BLS Series to GICS Sector Mapping

The module maps BLS employment series to GICS sectors:

| BLS Series      | Industry              | GICS Sectors              |
|-----------------|----------------------|---------------------------|
| CES1000000001   | Mining/Logging       | 10 (Energy), 15 (Materials) |
| CES2000000001   | Construction         | 20 (Industrials), 60 (Real Estate) |
| CES3000000001   | Manufacturing        | 15 (Materials), 20 (Industrials) |
| CES4200000001   | Retail Trade         | 25 (Cons. Disc.), 30 (Cons. Staples) |
| CES4300000001   | Transportation       | 20 (Industrials) |
| CES4422000001   | Utilities            | 55 (Utilities) |
| CES5000000001   | Information          | 45 (IT), 50 (Communications) |
| CES5500000001   | Financial Activities | 40 (Financials) |
| CES6500000001   | Education/Health     | 35 (Health Care) |
| CES7000000001   | Leisure/Hospitality  | 25 (Consumer Discretionary) |

## Signal Thresholds and Parameters

### Employment Signals
- **Threshold:** ±5% change required to generate signal
- **Base Confidence:** 0.70
- **High Confidence:** 0.85 (when 3-month and 6-month trends agree)
- **Actionability:** confidence > 0.60 AND |signal_strength| > 0.50

### Rotation Signals
- **Overweight Threshold:** composite_score > +0.30
- **Underweight Threshold:** composite_score < -0.30
- **Strong Signal:** |composite_score| > 0.70
- **Target Allocations:**
  - Overweight: 10-15% (scales with score)
  - Neutral: ~9.09% (equal weight)
  - Underweight: 3-7% (scales with score)

## Usage Examples

### C++ Integration

```cpp
import bigbrother.employment.signals;

using namespace bigbrother::employment;

// Create generator
EmploymentSignalGenerator generator;

// Generate employment signals
auto signals = generator.generateSignals();
for (const auto& signal : signals) {
    if (signal.isActionable()) {
        std::cout << signal.sector_name << ": "
                  << signal.rationale << "\n";
    }
}

// Generate rotation signals
auto rotation_signals = generator.generateRotationSignals();
for (const auto& signal : rotation_signals) {
    if (signal.action == SectorRotationSignal::Action::Overweight) {
        std::cout << "Overweight " << signal.sector_etf
                  << " to " << signal.target_allocation << "%\n";
    }
}

// Check for recession warning
auto spike_signal = generator.checkJoblessClaimsSpike();
if (spike_signal.has_value()) {
    std::cout << "WARNING: " << spike_signal->rationale << "\n";
}
```

### Python CLI

```bash
# Generate employment signals
uv run python scripts/employment_signals.py generate_signals

# Generate rotation signals
uv run python scripts/employment_signals.py rotation_signals

# Check jobless claims
uv run python scripts/employment_signals.py check_jobless_claims

# Use custom database path
uv run python scripts/employment_signals.py generate_signals /path/to/db.duckdb
```

## Testing

### Run C++ Integration Test
```bash
chmod +x scripts/test_employment_signals.sh
./scripts/test_employment_signals.sh
```

### Run Python Unit Test
```bash
# Test signal generation
uv run python scripts/employment_signals.py generate_signals | jq .

# Test rotation signals
uv run python scripts/employment_signals.py rotation_signals | jq .
```

## Database Schema

### Current Schema (sector_employment_raw)
```sql
CREATE TABLE sector_employment_raw (
    report_date DATE,
    employment_count INTEGER,
    series_id VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Current Data:**
- 2,128 records
- Date range: Jan 2021 - Aug 2025
- 19 BLS series
- Monthly granularity

### Future Enhancement (jobless_claims)
```sql
CREATE TABLE jobless_claims (
    report_date DATE,
    initial_claims INTEGER,
    continuing_claims INTEGER,
    series_id VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Performance Characteristics

### Python Backend
- **Query Time:** ~50ms (DuckDB on 2K records)
- **Calculation Time:** ~10ms (trend analysis)
- **Total Time:** ~100ms per command

### C++ Frontend
- **Subprocess Spawn:** ~20ms
- **JSON Parsing:** ~5ms (simple parser)
- **Total Overhead:** ~25ms

**Total Latency:** ~125ms for signal generation (acceptable for daily signals)

## Future Enhancements

### 1. Sentiment Score Integration
Add news sentiment analysis to rotation signals:
```python
def calculate_sentiment_score(sector_code: int) -> float:
    # Query news API for sector-related headlines
    # Run sentiment analysis (VADER/FinBERT)
    # Return score [-1.0, +1.0]
```

### 2. Technical Score Integration
Add price action analysis:
```python
def calculate_technical_score(sector_etf: str) -> float:
    # Query sector ETF price data
    # Calculate momentum indicators (RSI, MACD, MA crossovers)
    # Return score [-1.0, +1.0]
```

### 3. Composite Score Weighting
Update rotation signal calculation:
```python
composite_score = (
    0.50 * employment_score +
    0.30 * sentiment_score +
    0.20 * technical_score
)
```

### 4. Jobless Claims Data Collection
Add weekly jobless claims scraping:
```python
# Collect from FRED API (ICSA series)
# Store in jobless_claims table
# Update checkJoblessClaimsSpike() implementation
```

### 5. Machine Learning Enhancement
Train ML model on historical signals:
```python
# Features: employment trends, sector correlations, macro indicators
# Target: forward sector returns
# Model: XGBoost or LightGBM
# Output: confidence scores for signals
```

## Integration Points

### Trading Decision Engine
The employment signals integrate with the trading decision engine:

```cpp
// In strategy evaluation
auto employment_signals = generator.generateSignals();
auto rotation_signals = generator.generateRotationSignals();

// Adjust sector allocations based on rotation signals
for (const auto& signal : rotation_signals) {
    if (signal.action == SectorRotationSignal::Action::Overweight) {
        increaseSectorExposure(signal.sector_etf, signal.target_allocation);
    }
}

// Avoid sectors with negative employment signals
for (const auto& signal : employment_signals) {
    if (signal.bearish && signal.isActionable()) {
        reduceSectorExposure(signal.sector_code);
    }
}
```

### Risk Management System
Employment signals can trigger risk adjustments:

```cpp
auto spike_signal = generator.checkJoblessClaimsSpike();
if (spike_signal.has_value()) {
    // Recession warning - reduce overall exposure
    reducePortfolioRisk();
    increaseDefensiveSectors(); // XLU, XLP, XLV
}
```

## Monitoring and Logging

### Recommended Logging
```cpp
// Log signal generation
logger.info("Generated {} employment signals", signals.size());
for (const auto& signal : signals) {
    logger.info("Signal: {} {} ({:.1f}% change, confidence: {:.2f})",
                signal.sector_name,
                signal.bullish ? "improving" : "declining",
                signal.employment_change,
                signal.confidence);
}
```

### Metrics to Track
- Number of signals generated per day
- Average signal confidence
- Actionable signal percentage
- Signal accuracy (requires backtesting)
- Python backend execution time

## Known Limitations

1. **Current Employment Changes Small:** Recent data shows <1% changes, below 5% threshold
2. **No Jobless Claims Data:** Spike detection not yet functional
3. **Simple JSON Parser:** C++ parser fragile to JSON format changes
4. **No Sentiment/Technical Scores:** Rotation signals use only employment data
5. **Monthly Granularity:** BLS data monthly, signals update monthly

## Troubleshooting

### Common Issues

**Issue:** No signals generated
- **Cause:** Employment changes below 5% threshold
- **Solution:** Normal during stable periods; adjust threshold if needed

**Issue:** Python script fails to execute
- **Cause:** `uv` not installed or Python dependencies missing
- **Solution:** Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Issue:** Database not found
- **Cause:** Wrong database path in generator constructor
- **Solution:** Verify path: `ls data/bigbrother.duckdb`

**Issue:** C++ compile errors
- **Cause:** Missing C++23 module support
- **Solution:** Use Clang 21+: `clang++ --version`

## References

- BLS Employment Data: https://www.bls.gov/ces/
- GICS Sector Classification: https://www.msci.com/gics
- DuckDB Python API: https://duckdb.org/docs/api/python/overview
- C++23 Modules: https://en.cppreference.com/w/cpp/language/modules

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Next Review:** 2025-12-09
