# Tier 1 Extension - Detailed Implementation Checklist

**Author:** Olumuyiwa Oluwasanmi
**Created:** 2025-11-08
**Timeline:** Weeks 5-6
**Estimated Duration:** 2 weeks (80-100 hours)

---

## Overview

This checklist provides detailed tasks for Tier 1 Extension, including:
- Department of Labor (DoL) & employment data integration
- 11 GICS business sector analysis framework
- Sector rotation trading strategies
- Code quality compliance
- Python bindings with pybind11

**Success Criteria:**
- [ ] Employment data flowing into DuckDB
- [ ] All 11 sectors classified and tracked
- [ ] Sector rotation signals operational
- [ ] 100% clang-tidy compliance
- [ ] Python bindings functional
- [ ] Backtest shows improved performance with employment signals

---

## Section A: BLS API Integration (Employment Data)

**Estimated Time:** 8-12 hours
**Status:** Framework complete, deployment needed

### Database Initialization
- [ ] Run database schema creation
  ```bash
  duckdb data/bigbrother.duckdb < scripts/database_schema_employment.sql
  ```
- [ ] Verify all 8 tables created
  ```sql
  -- Run in DuckDB
  SHOW TABLES;
  SELECT * FROM sectors;  -- Should show 11 GICS sectors
  ```
- [ ] Verify indexes created
- [ ] Test views (latest_sector_employment, recent_employment_events, sector_employment_trends)

### BLS Data Collection
- [x] ✅ BLS API key configured in api_keys.yaml
- [ ] Test BLS API connection
  ```bash
  python -c "from scripts.data_collection.bls_employment import BLSEmploymentCollector; c = BLSEmploymentCollector(); print('API Key loaded:', bool(c.api_key))"
  ```
- [ ] Run initial data collection (5 years historical)
  ```bash
  python scripts/data_collection/bls_employment.py
  ```
- [ ] Verify data in database
  ```sql
  SELECT COUNT(*) FROM sector_employment_raw;
  SELECT COUNT(*) FROM jobless_claims_raw;
  ```

### Employment Series Configuration (19 series)
- [ ] Total nonfarm employment (CES0000000001)
- [ ] Total private employment (CES0500000001)
- [ ] Mining/logging (CES1000000001) → Energy/Materials sectors
- [ ] Construction (CES2000000001) → Industrials sector
- [ ] Manufacturing (CES3000000001) → Industrials/Materials
- [ ] Durable goods (CES3100000001)
- [ ] Nondurable goods (CES3200000001)
- [ ] Trade/transport/utilities (CES4000000001)
- [ ] Wholesale trade (CES4142000001)
- [ ] Retail trade (CES4200000001) → Consumer Discretionary
- [ ] Transport/warehousing (CES4300000001) → Industrials
- [ ] Utilities (CES4422000001) → Utilities sector
- [ ] Information (CES5000000001) → Technology/Communication
- [ ] Financial activities (CES5500000001) → Financials sector
- [ ] Professional/business (CES6000000001) → IT Services
- [ ] Education/health (CES6500000001) → Health Care sector
- [ ] Leisure/hospitality (CES7000000001) → Consumer Discretionary
- [ ] Other services (CES8000000001)
- [ ] Government (CES9000000001)

### Weekly Jobless Claims
- [ ] Fetch initial claims (ICSA) - leading indicator
- [ ] Fetch continued claims (CCSA)
- [ ] Calculate 4-week moving average
- [ ] Set up Thursday 8:30 AM ET alerts
- [ ] Track week-over-week changes
- [ ] Alert on >10% spike (recession warning)

### Monthly Jobs Report
- [ ] Fetch nonfarm payrolls (first Friday monthly)
- [ ] Track unemployment rate
- [ ] Track labor force participation
- [ ] Compare actual vs consensus estimates
- [ ] Calculate surprise factor
- [ ] Store market impact assessment

---

## Section B: Private Sector Job Data

**Estimated Time:** 12-16 hours

### Layoffs.fyi Integration
- [ ] Research Layoffs.fyi API or scraping method
- [ ] Implement tech sector layoff tracker
- [ ] Parse company, date, employee count, location
- [ ] Map companies to stock tickers
- [ ] Store in `employment_events` table
- [ ] Create daily update cron job
- [ ] Test with recent layoff data

### WARN Act Database
- [ ] Identify state-level WARN databases (start with CA, NY, TX)
- [ ] Implement web scraper for WARN notices
- [ ] Parse company, date, employee count, facility
- [ ] Map to stock tickers where possible
- [ ] Store in `employment_events` table
- [ ] Set up weekly update
- [ ] Validate 60-day advance notice timeline

### Company Hiring Announcements
- [ ] Use NewsAPI to search for hiring announcements
- [ ] Keywords: "hiring", "workforce expansion", "new jobs", "hiring freeze"
- [ ] Parse company, job count, location
- [ ] Classify event type (hiring, freeze, expansion)
- [ ] Store in `employment_events` table
- [ ] Calculate sentiment impact

### Event Classification
- [ ] Define event types (layoff, hiring, freeze, expansion, restructuring)
- [ ] Define impact magnitude (High: >1000 employees, Medium: 100-1000, Low: <100)
- [ ] Map events to sectors
- [ ] Create event timeline visualization

---

## Section C: Sector Analysis Module (11 GICS Sectors)

**Estimated Time:** 16-20 hours

### Sector Master Data

**11 GICS Sectors to Implement:**

#### 1. Energy (Sector Code: 10)
- [ ] Populate sector record in database
- [ ] ETF: XLE
- [ ] Category: Cyclical
- [ ] Map companies: XOM, CVX, COP
- [ ] BLS series: Mining/Logging (CES1000000001)

#### 2. Materials (Sector Code: 15)
- [ ] Populate sector record
- [ ] ETF: XLB
- [ ] Category: Cyclical
- [ ] Map companies: DOW, DD, FCX, NEM
- [ ] BLS series: Manufacturing (CES3000000001)

#### 3. Industrials (Sector Code: 20)
- [ ] Populate sector record
- [ ] ETF: XLI
- [ ] Category: Sensitive
- [ ] Map companies: BA, LMT, CAT, DE, UPS, FDX
- [ ] BLS series: Construction + Manufacturing + Transport

#### 4. Consumer Discretionary (Sector Code: 25)
- [ ] Populate sector record
- [ ] ETF: XLY
- [ ] Category: Sensitive
- [ ] Map companies: AMZN, TSLA, HD, MCD, NKE
- [ ] BLS series: Retail Trade + Leisure/Hospitality

#### 5. Consumer Staples (Sector Code: 30)
- [ ] Populate sector record
- [ ] ETF: XLP
- [ ] Category: Defensive
- [ ] Map companies: PG, KO, PEP, WMT, COST
- [ ] BLS series: Retail Trade

#### 6. Health Care (Sector Code: 35)
- [ ] Populate sector record
- [ ] ETF: XLV
- [ ] Category: Defensive
- [ ] Map companies: JNJ, PFE, UNH, ABBV, LLY
- [ ] BLS series: Education/Health Services (CES6500000001)

#### 7. Financials (Sector Code: 40)
- [ ] Populate sector record
- [ ] ETF: XLF
- [ ] Category: Sensitive
- [ ] Map companies: JPM, BAC, WFC, GS, MS, BRK.B
- [ ] BLS series: Financial Activities (CES5500000001)

#### 8. Information Technology (Sector Code: 45)
- [ ] Populate sector record
- [ ] ETF: XLK
- [ ] Category: Sensitive
- [ ] Map companies: AAPL, MSFT, NVDA, GOOGL, META
- [ ] BLS series: Information (CES5000000001)

#### 9. Communication Services (Sector Code: 50)
- [ ] Populate sector record
- [ ] ETF: XLC
- [ ] Category: Sensitive
- [ ] Map companies: META, GOOGL, DIS, NFLX, T, VZ
- [ ] BLS series: Information (CES5000000001)

#### 10. Utilities (Sector Code: 55)
- [ ] Populate sector record
- [ ] ETF: XLU
- [ ] Category: Defensive
- [ ] Map companies: NEE, DUK, SO
- [ ] BLS series: Utilities (CES4422000001)

#### 11. Real Estate (Sector Code: 60)
- [ ] Populate sector record
- [ ] ETF: XLRE
- [ ] Category: Sensitive
- [ ] Map companies: AMT, PLD, CCI
- [ ] BLS series: Construction (CES2000000001)

### Company-to-Sector Mapping
- [ ] Create mapping for all 24 current stock symbols
- [ ] Populate `company_sectors` table
- [ ] Include industry and sub-industry classifications
- [ ] Add market cap categories (Large, Mid, Small)
- [ ] Verify mapping accuracy
- [ ] Create sector summary report

### Sector Employment Analysis
- [ ] Calculate employment trends by sector (3-month, 6-month)
- [ ] Identify sectors with improving employment
- [ ] Identify sectors with declining employment
- [ ] Calculate employment momentum (rate of change)
- [ ] Create sector employment dashboard

### Sector News Sentiment
- [ ] Integrate NewsAPI for sector-specific news
- [ ] Classify news by affected sector(s)
- [ ] Calculate sector sentiment scores (-1.0 to 1.0)
- [ ] Track major events by sector
- [ ] Create sector sentiment timeline

### Sector Performance Tracking
- [ ] Fetch sector ETF prices (XLE, XLB, XLI, etc.)
- [ ] Calculate daily returns by sector
- [ ] Calculate relative strength vs S&P 500
- [ ] Identify sector rotation patterns
- [ ] Track sector leadership changes

---

## Section D: Decision Engine Integration

**Estimated Time:** 12-16 hours

### Employment Signal Integration
- [ ] Create EmploymentSignal class/struct
- [ ] Define signal types (jobless_claims_spike, sector_layoffs, sector_hiring, etc.)
- [ ] Set signal thresholds (e.g., claims >10% = warning)
- [ ] Integrate with existing trading decision logic
- [ ] Weight employment signals vs other factors
- [ ] Backtest employment signal effectiveness

### Sector-Based Filters
- [ ] Implement sector exposure tracking
- [ ] Set max exposure per sector (default: 30%)
- [ ] Block trades that would exceed sector limits
- [ ] Create sector diversification requirements
- [ ] Add sector correlation checks
- [ ] Log sector exposure in decision rationale

### Sector Rotation Strategy
- [ ] Create SectorRotationStrategy class
- [ ] Define rotation rules:
  - Employment improving + Strong sentiment → Overweight
  - Employment declining + Weak sentiment → Underweight
  - Defensive rotation on recession signals
  - Cyclical rotation on recovery signals
- [ ] Implement sector ETF trading (XLE, XLF, XLK, etc.)
- [ ] Create sector pairs trades (Long strong / Short weak)
- [ ] Set position sizing by sector strength
- [ ] Add stop-loss for sector positions

### Trading Decision Enhancement
- [ ] Modify signal generation to include employment data
- [ ] Add sector context to all trade decisions
- [ ] Include employment trends in risk assessment
- [ ] Factor sector rotation into position sizing
- [ ] Update explainability to show employment influence

### Alerts and Notifications
- [ ] Jobless claims spike alert (>10% increase)
- [ ] Major sector layoff alert (>1000 employees)
- [ ] Sector rotation signal alert
- [ ] Employment report preview (day before release)
- [ ] Post-report impact analysis
- [ ] Sector rebalancing recommendations

---

## Section E: Database Schema Implementation

**Estimated Time:** 4-6 hours

### Table Population
- [x] ✅ Schema SQL file created
- [ ] Run schema initialization
- [ ] Verify `sectors` table has 11 rows
- [ ] Populate `company_sectors` with current holdings
- [ ] Create test data for validation
- [ ] Run sample queries to verify structure
- [ ] Create database backup

### Index Verification
- [ ] Verify all indexes created successfully
- [ ] Test query performance with indexes
- [ ] Check index usage with EXPLAIN
- [ ] Optimize slow queries if needed

### Views Testing
- [ ] Test `latest_sector_employment` view
- [ ] Test `recent_employment_events` view
- [ ] Test `sector_employment_trends` view
- [ ] Verify view performance (<100ms queries)

### Data Integrity
- [ ] Set up foreign key constraints validation
- [ ] Test cascade deletes (if applicable)
- [ ] Verify unique constraints
- [ ] Test data type validations
- [ ] Create data quality checks

---

## Section F: Configuration & Testing

**Estimated Time:** 6-8 hours

### API Configuration
- [x] ✅ BLS API key configured in api_keys.yaml
- [x] ✅ News API key configured in api_keys.yaml
- [ ] Verify FRED API key still valid
- [ ] Test all API connections
- [ ] Set up API rate limit tracking
- [ ] Configure retry logic for failed requests
- [ ] Set up API error monitoring

### BLS API Integration Testing
- [ ] Test authenticated API calls (500/day limit)
- [ ] Test unauthenticated fallback (25/day limit)
- [ ] Verify employment data parsing
- [ ] Test time series data retrieval
- [ ] Validate data accuracy vs BLS website
- [ ] Test error handling (invalid series, network issues)
- [ ] Measure API response times

### News API Testing
- [ ] Test NewsAPI connection
- [ ] Fetch sector-specific news
- [ ] Test sentiment analysis
- [ ] Verify article classification
- [ ] Test rate limits (100 requests/day free tier)
- [ ] Implement caching to reduce API calls

### Sector Classification Validation
- [ ] Verify all 24 stocks mapped to correct sectors
- [ ] Cross-check sector assignments vs public data
- [ ] Validate ETF-to-sector mappings
- [ ] Test sector aggregation queries
- [ ] Verify sector employment calculations

### Backtesting with Employment Signals
- [ ] Add employment signals to backtest framework
- [ ] Run backtest with sector rotation strategy
- [ ] Compare performance vs baseline (no employment data)
- [ ] Measure signal effectiveness (precision, recall)
- [ ] Calculate incremental return from employment signals
- [ ] Analyze which signals are most profitable
- [ ] Document backtest results

### Performance Measurement
- [ ] Measure employment data fetch time
- [ ] Measure sector calculation time
- [ ] Measure signal generation latency
- [ ] Verify <100ms decision time (including employment)
- [ ] Profile database query performance
- [ ] Optimize slow operations

---

## Section G: Code Quality & Standards

**Estimated Time:** 16-20 hours
**Priority:** HIGH (blocks Tier 2)

### clang-tidy Configuration Complete
- [x] ✅ Enabled ALL cppcoreguidelines-* checks
- [x] ✅ Added cert-* (CERT C++ Secure Coding)
- [x] ✅ Added concurrency-* (thread safety)
- [x] ✅ Added performance-* (optimization)
- [x] ✅ Added portability-* (cross-platform)
- [x] ✅ Added openmp-* (OpenMP safety)
- [x] ✅ Added mpi-* (MPI safety)
- [x] ✅ Trailing return syntax enforced as ERROR
- [x] ✅ Standardized on clang-tidy (cppcheck removed)

### Full Codebase Validation
- [ ] Run validation on entire codebase
  ```bash
  ./scripts/validate_code.sh src/
  ```
- [ ] Review all clang-tidy errors
- [ ] Review all clang-tidy warnings
- [ ] Create prioritized fix list
- [ ] Document intentional suppressions

### Fix All Violations (Critical)

**By Category:**

#### Trailing Return Syntax
- [ ] Scan for old-style functions
- [ ] Convert all to `auto func() -> ReturnType`
- [ ] Verify constructors/destructors exempt
- [ ] Run: `grep -r "^[[:space:]]*[a-zA-Z_].*(" src/ | grep -v auto`

#### Rule of Five (C.21)
- [ ] Identify classes missing special member functions
- [ ] Define or delete all 5 members as group
- [ ] Fix classes with mutex members (delete move ops)
- [ ] Fix classes with atomic members (delete move ops)
- [ ] Document why move ops deleted

#### [[nodiscard]] Attributes
- [ ] Add [[nodiscard]] to all getters
- [ ] Add [[nodiscard]] to all query functions
- [ ] Add [[nodiscard]] to functions returning resources
- [ ] Verify no ignored return values

#### Memory Management (R.1, R.3)
- [ ] Replace any malloc/free with RAII
- [ ] Use smart pointers for ownership
- [ ] Use references for non-owning access
- [ ] Verify no memory leaks

#### Concurrency Issues
- [ ] Review mutex usage for correctness
- [ ] Check for potential race conditions
- [ ] Verify atomic operations used correctly
- [ ] Test thread-safe classes with multiple threads
- [ ] Fix any data races detected

#### OpenMP Safety
- [ ] Review all #pragma omp parallel regions
- [ ] Verify private/shared variable classifications
- [ ] Check for race conditions in parallel loops
- [ ] Validate reduction operations
- [ ] Test with ThreadSanitizer if available

#### MPI Safety
- [ ] Review MPI_Send/MPI_Recv buffer management
- [ ] Verify no buffer overlaps
- [ ] Check communicator usage
- [ ] Validate collective operations
- [ ] Test with multiple MPI ranks

#### Performance Issues
- [ ] Fix unnecessary copies
- [ ] Use move semantics where appropriate
- [ ] Use const& for expensive parameters
- [ ] Prefer emplace over insert
- [ ] Review container choices (vector vs list vs unordered_map)

#### Portability Issues
- [ ] Fix platform-specific code
- [ ] Use portable integer types (int32_t, uint64_t)
- [ ] Check endianness assumptions
- [ ] Verify cross-platform compatibility

### Verification
- [ ] Re-run validation after fixes
- [ ] Achieve zero clang-tidy errors
- [ ] Reduce warnings to <10 per file
- [ ] Document any remaining warnings
- [ ] Get clean bill of health

### C++23 Module Validation
- [ ] Verify all .cppm files have proper structure
- [ ] Check global module fragment (module;) usage
- [ ] Verify export module declarations
- [ ] Check export namespace patterns
- [ ] Validate module dependency order
- [ ] Test module compilation isolation

---

## Section H: Python Bindings with pybind11

**Estimated Time:** 20-24 hours
**Priority:** HIGH (enables ML/AI integration)

### Setup & Infrastructure
- [x] ✅ pybind11 in ansible playbook
- [x] ✅ Python bindings file tagged: PYTHON_BINDINGS
- [ ] Verify pybind11 installed: `python -c "import pybind11; print(pybind11.__version__)"`
- [ ] Check C++ headers accessible
- [ ] Configure CMake for pybind11 module
- [ ] Test basic binding compilation

### 1. DuckDB C++ API Bindings (CRITICAL)

**Why Critical:** DuckDB built from source, C++ headers available for direct access

- [ ] Locate DuckDB C++ headers (likely /usr/local/include/duckdb)
- [ ] Create `src/python_bindings/duckdb_bindings.cpp`
- [ ] Tag file: PYTHON_BINDINGS
- [ ] Expose DuckDB::Database class
  ```cpp
  py::class_<duckdb::Database>(m, "Database")
      .def(py::init<std::string>())
      .def("connect", &duckdb::Database::connect);
  ```
- [ ] Expose DuckDB::Connection class
- [ ] Expose query execution methods
- [ ] Expose result set handling
- [ ] Expose transaction management
- [ ] Enable zero-copy NumPy array transfer
- [ ] Test basic queries from Python
- [ ] Benchmark vs Python DuckDB library
- [ ] Document performance improvements

### 2. Options Pricing Bindings

- [ ] Create `src/python_bindings/options_bindings.cpp`
- [ ] Tag file: PYTHON_BINDINGS
- [ ] Expose Black-Scholes pricing
  ```python
  # Target API:
  price = bigbrother.black_scholes(spot=150, strike=155, vol=0.25, ...)
  ```
- [ ] Expose Trinomial Tree pricing
- [ ] Expose Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- [ ] Expose OptionBuilder fluent API
- [ ] Test from Python with real option data
- [ ] Benchmark: Compare vs QuantLib Python
- [ ] Add NumPy array batch processing
- [ ] Document API with examples

### 3. Correlation Engine Bindings

- [ ] Create `src/python_bindings/correlation_bindings.cpp`
- [ ] Tag file: PYTHON_BINDINGS
- [ ] Expose Pearson correlation
  ```python
  corr = bigbrother.pearson(x, y)
  ```
- [ ] Expose Spearman correlation
- [ ] Expose time-lagged correlation
- [ ] Expose CorrelationAnalyzer fluent API
- [ ] Support pandas DataFrames as input
- [ ] Support NumPy arrays as input
- [ ] Benchmark vs pandas.corr() and scipy.stats
- [ ] Add parallel batch correlation (OpenMP)

### 4. Risk Management Bindings

- [ ] Create `src/python_bindings/risk_bindings.cpp`
- [ ] Tag file: PYTHON_BINDINGS
- [ ] Expose Kelly Criterion calculator
  ```python
  kelly = bigbrother.kelly_criterion(win_prob=0.65, win_loss_ratio=2.0)
  ```
- [ ] Expose position sizing functions
- [ ] Expose Monte Carlo simulation
- [ ] Expose RiskAssessor fluent API
- [ ] Test with portfolio data
- [ ] Benchmark vs pure Python implementations
- [ ] Verify GIL-free execution

### 5. Tax Calculator Bindings

- [ ] Create `src/python_bindings/tax_bindings.cpp`
- [ ] Tag file: PYTHON_BINDINGS
- [ ] Expose tax calculation functions
- [ ] Expose wash sale detection
  ```python
  result = bigbrother.calculate_taxes(trades, federal_rate=0.24)
  ```
- [ ] Expose TaxCalculatorBuilder
- [ ] Test with trade history
- [ ] Verify IRS-compliant calculations

### 6. Backtesting Engine Bindings

- [ ] Expose BacktestRunner fluent API
- [ ] Support pandas DataFrame for price data
- [ ] Return results as pandas DataFrame
- [ ] Test with historical data
- [ ] Benchmark vs pure Python backtest

### Integration & Testing
- [ ] Create comprehensive test suite for all bindings
- [ ] Test GIL-free execution with threading
- [ ] Measure performance improvements
- [ ] Create Python usage examples
- [ ] Write documentation for each binding
- [ ] Create Jupyter notebook examples

### Performance Targets
- [ ] Options pricing: >50x speedup vs Python
- [ ] Correlation: >100x speedup vs pandas
- [ ] Risk calculations: >20x speedup
- [ ] Tax calculations: >10x speedup
- [ ] DuckDB queries: >5x speedup vs Python DuckDB

---

## Section I: Documentation & Communication

**Estimated Time:** 4-6 hours

### Implementation Documentation
- [ ] Update README with employment data features
- [ ] Document 11 GICS sectors in user guide
- [ ] Create sector rotation strategy guide
- [ ] Document employment signal interpretation
- [ ] Add Python bindings usage guide
- [ ] Create API reference for bindings

### Architecture Updates
- [ ] Update systems integration diagram with sectors
- [ ] Document employment data flow
- [ ] Add sector analysis to architecture docs
- [ ] Document Python bindings architecture
- [ ] Update component interaction diagrams

### User Guide
- [ ] How to interpret employment signals
- [ ] How to use sector rotation
- [ ] How to configure alerts
- [ ] How to use Python bindings
- [ ] Performance optimization tips

---

## Summary Statistics

### Total Tasks: 200+

**By Section:**
- A. BLS API Integration: 25 tasks (8-12 hours)
- B. Private Sector Job Data: 28 tasks (12-16 hours)
- C. Sector Analysis: 50 tasks (16-20 hours)
- D. Decision Engine: 25 tasks (12-16 hours)
- E. Database Schema: 12 tasks (4-6 hours)
- F. Configuration & Testing: 20 tasks (6-8 hours)
- G. Code Quality: 30 tasks (16-20 hours)
- H. Python Bindings: 40 tasks (20-24 hours)
- I. Documentation: 10 tasks (4-6 hours)

**Total Estimated Time:** 100-128 hours (2.5-3 weeks full-time)

### Completion Status
- ✅ Completed: ~15 tasks (infrastructure setup)
- 🔄 In Progress: ~5 tasks (partial implementation)
- ⏳ Not Started: ~180 tasks

### Priority Order
1. **HIGH:** Section C (Sectors), G (Code Quality), H (Python Bindings)
2. **MEDIUM:** Section A (BLS), D (Decision Engine)
3. **LOW:** Section B (Private data), F (Testing), I (Documentation)

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** 2025-11-08
**Version:** 1.0.0
