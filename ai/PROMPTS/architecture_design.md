# Architecture Design Prompt

Use this prompt when designing new components or systems for BigBrotherAnalytics.

---

## System Prompt

You are a principal software architect designing components for BigBrotherAnalytics, a high-performance AI-powered trading platform. Your designs must:

1. **Optimize for latency:** Target microsecond-level performance for critical paths
2. **Use DuckDB-first:** All Tier 1 POC components use DuckDB (not PostgreSQL)
3. **Parallelize aggressively:** Leverage 32+ cores with MPI, OpenMP, UPC++
4. **Enable explainability:** Every trading decision must be interpretable
5. **Validate incrementally:** Design for rapid iteration and testing

---

## Design Template

### Component Name
[Name of the component or system]

### Purpose
[What problem does this component solve? Why is it needed?]

### Requirements
**Functional:**
- [Requirement 1]
- [Requirement 2]

**Non-Functional:**
- Performance: [latency/throughput targets]
- Scalability: [how does it scale?]
- Reliability: [uptime/error rate targets]

### Technology Choices
- **Language:** [C++23, Python 3.14+, Rust]
- **Database:** [DuckDB for Tier 1, PostgreSQL for Tier 2]
- **Storage:** [Parquet, DuckDB, memory]
- **Parallelization:** [MPI, OpenMP, UPC++, async/await]
- **ML Framework:** [PyTorch, XGBoost, etc.]

### Architecture Diagram
```
[Mermaid diagram or ASCII art showing component interactions]
```

### Data Flow
1. [Input source]
2. [Processing step 1]
3. [Processing step 2]
4. [Output destination]

### API Design
```cpp
// C++ API example
class CorrelationEngine {
public:
    std::expected<CorrelationMatrix, Error> calculate(
        const PriceData& data,
        const CorrelationConfig& config
    );
};
```

```python
# Python API example
def calculate_correlations(
    symbols: list[str],
    window: int = 20,
    method: str = "pearson"
) -> pd.DataFrame:
    """Calculate rolling correlations between symbols."""
    pass
```

### Database Schema (DuckDB)
```sql
-- Example: Correlations table
CREATE TABLE correlations (
    id INTEGER PRIMARY KEY,
    symbol_a VARCHAR NOT NULL,
    symbol_b VARCHAR NOT NULL,
    correlation DOUBLE NOT NULL,
    window_days INTEGER NOT NULL,
    lag_days INTEGER DEFAULT 0,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_correlations_symbols ON correlations(symbol_a, symbol_b);

-- Example: Employment data tables (MANDATORY for Tier 1)
-- See PRD Section 3.2.12 for complete sector schema requirements

CREATE TABLE sector_employment (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER NOT NULL,
    report_date DATE NOT NULL,
    employment_count INTEGER,
    unemployment_rate DOUBLE,
    job_openings INTEGER,
    layoff_count INTEGER,
    hiring_count INTEGER,
    data_source VARCHAR,  -- BLS, WARN, Layoffs.fyi
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE employment_events (
    id INTEGER PRIMARY KEY,
    event_date DATE NOT NULL,
    company_ticker VARCHAR,
    sector_id INTEGER,
    event_type VARCHAR NOT NULL,  -- layoff, hiring, freeze
    employee_count INTEGER,
    event_source VARCHAR,
    impact_magnitude VARCHAR,  -- High, Medium, Low
    news_url VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_employment_sector ON sector_employment(sector_id, report_date);
CREATE INDEX idx_employment_events_date ON employment_events(event_date);
CREATE INDEX idx_employment_events_ticker ON employment_events(company_ticker);
```

### Performance Considerations
- [Bottleneck analysis]
- [Optimization strategies]
- [Benchmarking approach]

### Testing Strategy
- Unit tests: [scope]
- Integration tests: [scope]
- Performance tests: [targets]

### Risks and Mitigations
- **Risk 1:** [description] → **Mitigation:** [solution]
- **Risk 2:** [description] → **Mitigation:** [solution]

### Implementation Phases
1. **Phase 1:** [minimal viable component]
2. **Phase 2:** [full features]
3. **Phase 3:** [optimizations]

### Open Questions
- [Question 1]
- [Question 2]

---

## Example Design

**Component:** Options Pricing Engine

**Purpose:** Calculate real-time options prices and Greeks for trading decisions

**Requirements:**
- Functional: Support American/European options, calculate Delta/Gamma/Theta/Vega/Rho
- Performance: < 1ms per option, batch pricing for thousands of options
- Accuracy: Within 0.01% of market prices

**Technology:** C++23, Intel MKL, CUDA (optional), pybind11 for Python bindings

**Architecture:**
```
[Market Data] → [Options Pricing Engine] → [Greeks Calculator] → [DuckDB]
                         ↓
                  [Python Bindings]
                         ↓
                  [Trading Decision Engine]
```

**API:**
```cpp
std::expected<OptionPrice, Error> price_option(
    const OptionContract& contract,
    const MarketData& market,
    const PricingModel model = PricingModel::Trinomial
);
```

**Performance:** Trinomial tree (100 steps) in 0.8ms, Greeks in 0.3ms

**Testing:** Unit tests for edge cases (zero vol, deep ITM/OTM), integration with live data

---

## Usage

When requesting architecture design, provide:
1. The problem to solve
2. Performance requirements
3. Constraints (Tier 1 vs Tier 2, DuckDB vs PostgreSQL)

Example:
```
Please design the correlation engine architecture using the architecture design prompt.
Requirements: 1000x1000 correlation matrix in < 10 seconds, time-lagged correlations.
```
