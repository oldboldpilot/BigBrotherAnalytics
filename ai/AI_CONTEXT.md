# AI Context - BigBrotherAnalytics System Architecture

**Document Version:** 1.0 | **Last Updated:** November 12, 2025 | **Status:** ✅ Phase 5+ Production Ready

This document provides AI agents with a comprehensive understanding of the BigBrotherAnalytics system architecture, critical patterns, and implementation guidelines.

## System Overview

**BigBrotherAnalytics** is a high-performance AI-powered options trading platform combining:
- Real-time market data ingestion (price, news, economic indicators)
- Advanced ML-based price prediction (56.3-56.6% accuracy)
- Risk management with automated position sizing
- 52+ options trading strategies with dynamic signal generation
- Live paper trading with full audit trail

**Architecture Paradigm:** Highly optimized C++23 core with Python orchestration and dashboard.

---

## Technology Stack

### Languages & Compilers
- **C++23** (primary) - Core trading engine, SIMD-optimized algorithms
  - **Compiler:** Clang 21.1.5 + libc++
  - **Standard:** `-std=c++2c` (C++23)
  - **Build System:** CMake 4.1.2+ with Ninja generator

- **Python 3.13** (orchestration) - Dashboard, ML training, reporting
  - **Package Manager:** `uv` (10-100x faster than pip)
  - **Execution:** Always use `uv run python script.py` (never direct python)

- **C++/Python Integration** - pybind11 for performance-critical paths
  - **Memory Management:** `std::shared_ptr` holder pattern (NO raw new/delete)
  - **Examples:** Risk analytics (8 modules), news ingestion, FRED rates, price prediction

### Database Layer

#### DuckDB (Primary - Tier 1 POC)
- **Location:** `data/bigbrother.duckdb`
- **Purpose:** Trading signals, positions, execution history, quotes
- **Access Method:** DuckDB bridge pattern (C++23 module safe)
- **Performance:** <5ms for typical queries
- **Zero Setup:** Embedded database, auto-creates schema

#### Bridge Pattern (C++23 Module Compatibility)
The DuckDB bridge isolates incomplete DuckDB types from C++23 modules:

**Files:**
- `src/schwab_api/duckdb_bridge.hpp` - Opaque handle declarations
- `src/schwab_api/duckdb_bridge.cpp` - Hidden DuckDB implementations

**Handles:**
- `DatabaseHandle` - Represents open DuckDB database
- `ConnectionHandle` - Represents database connection
- `QueryResultHandle` - Represents query result set
- `PreparedStatementHandle` - Represents prepared statement

**Usage Pattern (MANDATORY for C++23 modules):**
```cpp
module;
#include "schwab_api/duckdb_bridge.hpp"  // ✅ Bridge, not DuckDB

export module my_module;

using namespace bigbrother::duckdb_bridge;

// Code uses DatabaseHandle, ConnectionHandle, etc.
auto db = openDatabase("data/bigbrother.duckdb");
auto conn = createConnection(*db);
executeQuery(*conn, "SELECT * FROM trading_signals");
```

**Never:**
```cpp
module;
#include <duckdb.hpp>  // ❌ Causes C++23 instantiation errors
```

#### Database Schema

**Core Tables:**

1. **trading_signals**
   - Columns: id (PK), symbol, strategy, signal, confidence, created_at
   - Purpose: Store generated trading signals
   - Indexed: symbol, created_at (for fast queries)

2. **executions**
   - Columns: id (PK), signal_id (FK), order_id, status, executed_at
   - Purpose: Track which signals were executed and their status
   - Indexed: signal_id, order_id

3. **positions**
   - Columns: id (PK), symbol, quantity, entry_price, entry_time, strategy
   - Purpose: Active positions and portfolio state
   - Indexed: symbol, strategy

4. **quotes** (cached)
   - Columns: symbol, bid, ask, last_price, iv, timestamp
   - Purpose: Latest market quotes with caching
   - Indexed: symbol, timestamp

5. **news_articles** (news ingestion)
   - Columns: id (PK), title, url, source, sentiment, created_at
   - Purpose: Financial news with sentiment scores
   - Indexed: created_at, sentiment

#### Migration Status
- **Token Manager:** ✅ Migrated to bridge (token_manager.cpp)
- **Resilient Database:** ✅ Migrated to bridge (resilient_database.cppm)
- **Build:** ✅ 61/61 targets successful
- **Runtime:** ✅ No segmentation faults
- **Exception Handling:** Changed from duckdb::Exception to std::exception (bridge returns bool)

### ML/AI Stack

#### Price Prediction Model v3.0 (PROFITABLE)
- **Accuracy:** 56.3% (5-day), 56.6% (20-day) - Above 55% profitability threshold
- **Features:** 60 comprehensive features (time, treasury, Greeks, sentiment, momentum, volatility, interactions)
- **Architecture:** [256, 128, 64, 32] neurons with ReLU + dropout
- **Parameters:** 58,947 total
- **Training Data:** 24,300 samples from 20 symbols over 5 years
- **Loss Function:** DirectionalLoss (90% direction, 10% magnitude focus)
- **Framework:** PyTorch (training) + ONNX Runtime (inference)
- **GPU:** CUDA 13.0 with RTX 4070 (12GB VRAM)

**Feature Categories:**
1. **Identification (3):** symbol_encoded, sector_encoded, is_option
2. **Time (8):** hour, minute, day_of_week, day_of_month, month, quarter, day_of_year, is_market_open
3. **Treasury Rates (7):** fed_funds, 3mo, 2yr, 5yr, 10yr, slope, inversion
4. **Options Greeks (6):** delta, gamma, theta, vega, rho, implied_volatility
5. **Sentiment (2):** avg_sentiment, news_count
6. **Price (5):** close, open, high, low, volume
7. **Momentum (7):** return_1d/5d/20d, RSI, MACD, signal, volume_ratio
8. **Volatility (4):** ATR, BB_upper/lower, BB_position
9. **Interactions (10):** sentiment×momentum, volume×RSI, yield×volatility, delta×IV, etc.
10. **Directionality (8):** trend_strength, price_above_MA5/20, momentum_3d, win_rate, etc.

**Inference:**
- Input: Real-time market data + features
- Output: 3 price change forecasts (1-day, 5-day, 20-day)
- Confidence: Separate scores for each horizon
- Signals: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL (threshold-based)

**Files:**
- Model: `models/price_predictor.onnx` + `.onnx.data` (15KB + 230KB)
- Metadata: `models/price_predictor_info.json`
- Training script: `scripts/ml/train_model.py` (680+ lines)

#### FRED Risk-Free Rates Integration
- **Source:** Federal Reserve Economic Data API
- **Series:** 3-Month T-Bill, 2Y/5Y/10Y/30Y T-Notes, Federal Funds Rate
- **Update Frequency:** 1 hour with auto-refresh background thread
- **Performance:** 4x speedup with AVX2 SIMD JSON parsing
- **Module:** `src/market_intelligence/fred_rates.cppm` (455 lines)
- **Provider:** `src/market_intelligence/fred_rate_provider.cppm` (280 lines)
- **Python Bindings:** `fred_rates_py` module (364KB shared library)

#### News Ingestion System
- **Source:** NewsAPI (100 requests/day limit)
- **Processing:** Keyword-based sentiment analysis (-1.0 to 1.0 scale)
- **Keywords:** 60+ positive and negative financial keywords
- **Deduplication:** Hash-based article_id from URL
- **Module:** `src/market_intelligence/news_ingestion.cppm` (402 lines)
- **Sentiment Analyzer:** `src/market_intelligence/sentiment_analyzer.cppm` (260 lines)
- **Python Bindings:** `news_ingestion_py` module (236KB shared library)

#### Feature Extraction
- **Module:** `src/market_intelligence/feature_extractor.cppm` (620 lines)
- **Performance:** <0.5ms for 60 features (AVX2 SIMD optimized)
- **Normalization:** StandardScaler with AVX2 intrinsics (8x speedup)
- **History Buffers:** 30-day rolling windows for technical indicators
- **Fallback:** Approximations when history < 26 days

### GPU Acceleration

#### Hardware
- **GPU:** NVIDIA RTX 4070 (Ada Lovelace architecture)
- **VRAM:** 12GB
- **CUDA Cores:** 5,888
- **Tensor Cores:** 184 (4th gen - FP16/BF16/FP8/INT8)
- **Compute Capability:** 8.9

#### Software
- **CUDA Toolkit:** 13.0
- **Driver:** 581.80+
- **cuBLAS:** Available for linear algebra
- **cuDNN:** Available for deep learning
- **JAX:** Python GPU acceleration for dashboard (3.8x speedup)

#### Current Utilization
- **Active:** JAX GPU acceleration for dashboard and Greeks calculation
- **Available:** Native CUDA C++ kernels (post-training)
- **Performance:** 100-1000x speedup potential for feature extraction/inference

### Parallelization & Optimization

#### SIMD Optimization
- **AVX-512/AVX2:** Comprehensive intrinsics for numerical code
- **Monte Carlo:** 9.87M simulations/sec (AVX2, 6-7x speedup)
- **Correlation Analysis:** 6-8x speedup with horizontal reduction
- **StandardScaler:** 8x speedup for feature normalization
- **Compiler Flags:** `-O3 -march=native -mavx2 -mfma`

#### Multi-threading
- **OpenMP:** Multi-threaded options pricing and matrix operations
- **MPI:** Optional for distributed backtesting (future)
- **UPC++/GASNet-EX:** Available for ultra-low latency (future)
- **Thread-Local Storage:** Token refresh service, simdjson parser

#### JSON Performance
- **Library:** simdjson v4.2.1 (SIMD-accelerated)
- **Performance:** 3-32x faster parsing
- **Quote Parsing:** 32.2x (3449ns → 107ns)
- **NewsAPI:** 23.0x (8474ns → 369ns)
- **Account Data:** 28.4x (3383ns → 119ns)
- **Usage:** All hot paths (quotes, news, account updates)

### Code Quality Infrastructure

#### C++23 Modules
- **30 modules implemented** across market intelligence, Schwab API, account management
- **Compilation:** Clang 21 with module BMI caching
- **Build Time:** ~30 seconds for full rebuild (optimized with Ninja)
- **Standards:** 100% trailing return syntax, [[nodiscard]] attributes

#### Static Analysis
- **clang-tidy:** Comprehensive C++ Core Guidelines enforcement
- **Checks:** 50+ check categories (cppcoreguidelines, cert, concurrency, performance, etc.)
- **Integration:** Automatic before build and commit
- **Failures:** BLOCK build and commit (strict enforcement)

#### Memory Safety
- **Valgrind v3.24.0:** Memory leak detection and thread safety validation
- **Status:** Zero memory leaks detected
- **Test Coverage:** 23 unit tests + 8 benchmarks
- **Report:** `docs/VALGRIND_MEMORY_SAFETY_REPORT.md`

#### Error Handling
- **C++ Style:** `std::expected<T, E>` for error propagation
- **Logging:** spdlog for comprehensive logging (structured, format strings)
- **Exceptions:** Minimal - used only for critical failures
- **Graceful Degradation:** Fallbacks for unavailable services

---

## C++23 Modules Architecture

### Module Hierarchy

```
bigbrother.utils (foundation)
├── types                    [Common types, enums, constants]
├── logger                   [spdlog-based logging]
├── timer                    [Performance timing]
├── retry                    [Automatic retry logic]
├── simd                     [SIMD intrinsics wrappers]
└── resilient_database       [Database operations with retry]

bigbrother.market_intelligence
├── sentiment_analyzer       [Keyword-based sentiment]
├── news_ingestion          [NewsAPI integration]
├── fred_rates              [FRED API client]
├── fred_rate_provider      [Thread-safe singleton]
├── feature_extractor       [60-feature ML extraction]
└── price_predictor         [ONNX inference]

bigbrother.schwab_api (OAuth + Account Management)
├── account_types           [Account, Position, Order data structures]
├── duckdb_bridge           [Database abstraction (❌ NOT a module)]
├── token_manager           [OAuth token refresh via sockets]
├── oauth_config            [OAuth configuration]
├── account_client          [Lightweight fluent API]
└── account_manager         [Full account management]

bigbrother.core.trading
├── order_types             [Platform-agnostic Order, Position types]
├── platform_interface      [Abstract trading platform]
└── orders_manager          [Platform-agnostic order lifecycle]

bigbrother.risk_management
├── risk                    [Portfolio risk data structures]
├── position_sizer          [Position sizing algorithms]
├── monte_carlo             [Monte Carlo simulations]
└── risk_management         [VaR, Sharpe, stress testing]

bigbrother.options.pricing
├── black_scholes           [Option valuation]
├── implied_volatility      [IV calculation]
└── greeks                  [Delta, gamma, theta, vega, rho]

bigbrother.trading_decision
├── strategies              [52+ trading strategy implementations]
└── strategy_manager        [Signal aggregation and execution]
```

### Module Compilation

**Build Output:**
```
src/utils/types.cppm                → types.pcm      (20KB)
src/utils/logger.cppm               → logger.pcm     (15KB)
src/utils/retry.cppm                → retry.pcm      (8KB)
src/market_intelligence/*.cppm       → *.pcm          (150KB total)
src/schwab_api/*.cppm                → *.pcm          (200KB total)
src/options/pricing.cppm             → pricing.pcm    (30KB)
src/trading_decision/strategies.cppm → strategies.pcm (250KB)

Final Binary: bin/bigbrother (15MB with symbols)
```

### Module Import Chain

**Correct Order (verified by CMake):**
1. `types.cppm` (no dependencies)
2. `logger.cppm` (imports types)
3. `retry.cppm` (imports types)
4. `resilient_database.cppm` (imports types, logger, retry, uses bridge)
5. `market_intelligence/*.cppm` (import types, logger)
6. `schwab_api/*.cppm` (import types, logger, market_intelligence)
7. `core.trading/*.cppm` (import types)
8. `trading_decision/*.cppm` (import all above)

---

## Critical Patterns & Standards

### DuckDB Bridge Pattern (MANDATORY)

**When to Use:**
- Any C++23 module (`.cppm`) accessing DuckDB
- Any file in a C++23 module import chain
- Prevents incomplete type instantiation errors

**Pattern:**
```cpp
module;
#include "schwab_api/duckdb_bridge.hpp"

export module my_module;

using namespace bigbrother::duckdb_bridge;

class MyComponent {
    std::unique_ptr<DatabaseHandle> db_;
    std::unique_ptr<ConnectionHandle> conn_;
};
```

**Never:**
```cpp
#include <duckdb.hpp>  // In C++23 modules!
```

**Bridge API:** openDatabase, createConnection, executeQuery, prepareStatement, bindString/Int/Double, executeStatement, getValueAsString/Int/Double, etc.

**Documentation:** See `AGENT_CODING_GUIDE.md` for complete API reference with examples.

### Exception Handling

**Strategy:**
- Prefer `std::expected<T, std::string>` for recoverable errors
- Throw `std::exception` or subclasses only for critical failures
- Use error codes + logging for expected failures
- Bridge operations return `bool` (not exceptions)

**Exception Hierarchy:**
```
std::exception
├── std::runtime_error         [Critical failures - trading loop unstable]
├── std::invalid_argument      [Invalid configuration or input]
└── std::logic_error          [Programming logic error]
```

**Database Error Example:**
```cpp
// Before (Direct DuckDB - NOT allowed in modules)
try {
    auto result = conn_->query("SELECT ...");
} catch (duckdb::Exception& e) {  // ❌ Modules can't use
    // ...
}

// After (Using Bridge)
auto result = executeQueryWithResults(*conn, "SELECT ...");
if (!result || hasError(*result)) {
    logger_->error("Query failed: {}", getErrorMessage(*result));  // ✅ Correct
}
```

### Memory Management

**MANDATORY Rules:**
- `std::unique_ptr` for exclusive ownership
- `std::shared_ptr` only with pybind11 holder pattern
- **NEVER** raw `new`/`delete` in modern code
- RAII for all resources (sockets, handles, connections)

**Example:**
```cpp
// ✅ CORRECT - RAII with unique_ptr
auto db = openDatabase("data/bigbrother.duckdb");
auto conn = createConnection(*db);
// Automatically cleaned up when going out of scope

// ❌ WRONG - Manual memory management
auto db_ptr = new DatabaseHandle();  // Memory leak!
```

### Trailing Return Type Syntax (MANDATORY)

**ALWAYS use:**
```cpp
auto functionName(int param) -> ReturnType;
auto myFunction() -> void;
auto getData() -> std::vector<Trade>;
```

**NEVER use:**
```cpp
ReturnType functionName(int param);  // ❌ Clang-tidy rejects
void myFunction();                   // ❌ Rejected
```

### [[nodiscard]] Attribute (MANDATORY for Getters)

```cpp
[[nodiscard]] auto getSymbol() const -> std::string;
[[nodiscard]] auto calculateProfit() -> double;
[[nodiscard]] auto executeQuery(...) -> bool;

// Usage
auto symbol = getSymbol();  // ✅ Correctly assigned

calculateProfit();  // ⚠️ Clang-tidy warning (should handle result)
```

### Logging Pattern

```cpp
import bigbrother.utils.logger;

// Use spdlog formatted strings
logger_->info("Trade executed: {} @ ${}", symbol, price);
logger_->warn("Risk threshold breached: VaR={}", var_value);
logger_->error("Database error: {}", error_message);
logger_->debug("Feature extraction took {}ms", duration_ms);
```

**Log Levels:**
- `debug()` - Development troubleshooting
- `info()` - Important events (trades, signals, system state)
- `warn()` - Recoverable issues (rejected trades, warnings)
- `error()` - Critical failures (database down, risk halt)

---

## Trading Architecture

### Strategy System

**52+ Implemented Strategies:**
1. **Directional:** Buy call, buy put, long stock, short stock
2. **Spreads:** Bull call, bear call, bull put, bear put, iron condor
3. **Volatility:** Straddle, strangle, butterfly, calendar spread
4. **Arbitrage:** Box spread, conversion, reversal
5. **Custom:** ML-based price predictor, sentiment-based, news-triggered

**Signal Generation:**
```cpp
struct Signal {
    std::string strategy;
    std::string action;           // BUY, SELL, HOLD
    std::vector<Order> orders;
    double confidence;            // 0.0 to 1.0
    std::string explanation;      // Why generated
};
```

**Strategy Lifecycle:**
```
1. Fetch market data (quotes, Greeks, sentiment, news)
2. Extract features (60-feature ML extraction)
3. Generate signals (52+ strategies)
4. Aggregate signals (weighted voting by strategy reliability)
5. Check risk management (VaR, daily loss, position limits)
6. Execute orders (if risk approved)
7. Log execution (database audit trail)
```

### Risk Management

**Real-Time Monitoring:**
- **VaR (95% confidence):** Historical simulation, calculated every cycle (~5μs AVX2)
- **Sharpe Ratio:** Risk-adjusted returns (~8μs AVX2)
- **Position Limits:** $2,000 per trade (paper trading)
- **Daily Loss Limit:** $2,000 (automated halt)
- **Max Concurrent:** 2-3 positions

**Automated Circuit Breakers:**
- VaR < -3% → Stop all trading
- Daily loss > $900 → Stop all trading
- Low confidence signals → Reduce position size

### Order Execution

**Platform Abstraction:**
```cpp
class TradingPlatformInterface {
    virtual auto submitOrder(const Order&) -> OrderResult;
    virtual auto cancelOrder(const std::string& order_id) -> bool;
    virtual auto modifyOrder(const Order&) -> bool;
    virtual auto getOrders() -> std::vector<Order>;
    virtual auto getPositions() -> std::vector<Position>;
};
```

**Current Implementation:** Schwab API (OrderExecutor adapter)

**Platform-Agnostic:** OrdersManager depends only on interface (Dependency Inversion Principle)

---

## Data Ingestion Pipeline

### Quote Ingestion (Every 60 seconds)
```
Schwab API
  └─> Quote Parser (simdjson, 32x faster)
       └─> Greeks Calculator (Black-Scholes, AVX2)
            └─> Database Storage (DuckDB bridge)
                 └─> Dashboard Display (Flask/Streamlit)
```

**Performance:**
- Quote fetch: 100-200ms
- Parsing: 0.1ms (simdjson, 32x vs JSON)
- Greeks: 0.2ms per symbol
- Storage: <1ms (DuckDB)
- Total cycle: ~300ms

### News Ingestion (Every 10 minutes)
```
NewsAPI
  └─> Deduplication (URL hash)
       └─> Sentiment Analysis (60+ keywords)
            └─> Database Storage (DuckDB)
                 └─> Feature Engineering (for ML)
```

**Performance:**
- API fetch: 200-400ms
- Sentiment: <1ms per article
- Storage: <2ms per batch

### FRED Rates (Every 1 hour)
```
Federal Reserve API
  └─> JSON Parser (simdjson AVX2)
       └─> Cache (1-hour TTL)
            └─> Feature Engineering (for ML)
```

**Performance:**
- API fetch: 300-500ms
- Parsing: <1ms (AVX2)
- Caching: <1ms

---

## Testing & Validation

### Test Coverage

**Unit Tests:**
- 23 core modules
- 210-line integration test suite for risk analytics
- 100% passing (8/8 modules)

**Integration Tests:**
- 8 Phase 5 validation checks
- Paper trading validation
- Token refresh testing

**Performance Testing:**
- 8 benchmarks validating SIMD speedups
- Monte Carlo: 9.87M simulations/sec
- SIMD correlation: 8.5x speedup

**Regression Tests:**
```bash
./scripts/validate_code.sh          # clang-tidy + build
./scripts/test_*.sh                 # Unit test suites
./scripts/run_valgrind_tests.sh     # Memory leak detection
```

### Build Verification

**All Checks Automated:**
1. **clang-tidy:** 50+ checks (C++ Core Guidelines)
2. **CMake Build:** Full C++23 module compilation
3. **Binary Verification:** Symbol table inspection
4. **Runtime Tests:** Core functionality validation

**Success Criteria:**
```
✅ 61/61 build targets
✅ 0 clang-tidy errors (acceptable warnings allowed)
✅ 100% module import chain valid
✅ Zero runtime segmentation faults
✅ <500ms cold startup
```

---

## Development Workflow

### Required Tools

**Mandatory:**
- Clang 21.1.5+
- Ninja build system
- CMake 4.1.2+
- Git 2.37+
- uv (Python package manager)

**Optional but Recommended:**
- clang-format (code formatting)
- Valgrind (memory profiling)
- perf (performance profiling)
- gdb (debugging)

### Build Commands

```bash
# Configure (first time)
cmake -G Ninja -B build

# Build (incremental)
ninja -C build

# Full rebuild with validation
./scripts/validate_code.sh && cd build && ninja

# Run tests
./scripts/test_*.sh

# Memory check
./scripts/run_valgrind_tests.sh
```

### Code Changes Workflow

```
1. Edit code
   ↓
2. Run validation
   ./scripts/validate_code.sh
   (clang-tidy + build checks)
   ↓
3. Build
   cd build && ninja
   (CMake auto-runs clang-tidy)
   ↓
4. Test
   ./scripts/test_*.sh
   ↓
5. Commit
   git add -A
   git commit -m "message

   Author: Olumuyiwa Oluwasanmi"
   (pre-commit hook runs clang-tidy)
```

---

## Performance Targets & Benchmarks

### Current Performance

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Quote parsing | <1ms | 0.1ms (32x) | ✅ 32x faster |
| News sentiment | <1ms | <0.5ms | ✅ Met |
| Feature extraction | <1ms | <0.5ms | ✅ Met |
| ONNX inference | <1ms | 0.8-1.2ms | ✅ Met |
| Batch 1000 | <50ms | 15ms | ✅ 3.3x faster |
| Monte Carlo 1M | <120ms | 25ms | ✅ 4.8x faster |
| Dashboard load | <5s | 1.2s | ✅ 3.8x faster |

### Profiling Output

```
Trading Cycle (1000 iterations):
┌─ Data ingestion:       23.5% (70.5ms)
├─ Feature extraction:    8.2% (24.6ms)
├─ ONNX inference:        7.1% (21.3ms)
├─ Strategy signals:     12.4% (37.2ms)
├─ Risk calculation:     14.8% (44.4ms)
├─ Order execution:      18.2% (54.6ms)
└─ Database logging:      5.8% (17.4ms)
───────────────────────────────────
Total:                   100%  (300ms per cycle)
```

---

## Key Files & Documentation

### Core Implementation Files
- **Token Manager:** `src/schwab_api/token_manager.cpp` (135 lines)
- **DuckDB Bridge:** `src/schwab_api/duckdb_bridge.{hpp,cpp}` (300+ lines)
- **Resilient Database:** `src/utils/resilient_database.cppm` (500+ lines)
- **Feature Extractor:** `src/market_intelligence/feature_extractor.cppm` (620 lines)
- **Price Predictor:** `src/market_intelligence/price_predictor.cppm` (525 lines)
- **Strategies:** `src/trading_decision/strategies.cppm` (1500+ lines)
- **Risk Management:** `src/risk_management/risk_management.cppm` (400+ lines)

### Documentation Files
- **C++23 Modules Guide:** `docs/CPP23_MODULES_GUIDE.md` (1000+ lines)
- **Coding Standards:** `docs/CODING_STANDARDS.md` (simdjson usage, patterns)
- **Trading Platform Architecture:** `docs/TRADING_PLATFORM_ARCHITECTURE.md` (590 lines)
- **ML Integration Guide:** `ML_INTEGRATION_DEPLOYMENT_GUIDE.md` (500+ lines)
- **News Ingestion System:** `docs/NEWS_INGESTION_SYSTEM.md` (620 lines)
- **Price Predictor System:** `docs/PRICE_PREDICTOR_SYSTEM.md` (800 lines)
- **VALGRIND Report:** `docs/VALGRIND_MEMORY_SAFETY_REPORT.md`

### Python Scripts
- **Setup:** `scripts/phase5_setup.py` (450 lines, auto-start dashboard + trading)
- **Shutdown:** `scripts/phase5_shutdown.py` (250 lines, cleanup + backup)
- **Token Refresh:** `scripts/token_refresh_service.py` (182 lines)
- **ML Training:** `scripts/ml/train_model.py` (680+ lines)
- **Data Collection:** `scripts/data_collection/historical_data.py` (320 lines)
- **Reporting:** `scripts/reporting/generate_daily_report.py` (750+ lines)

---

## Critical Constraints & Guidelines

### Must-Follow Rules

1. **DuckDB Bridge (MANDATORY for C++23)**
   - ALL C++23 modules accessing database MUST use bridge
   - Never include `<duckdb.hpp>` in modules
   - Exception: Python C++ bindings can use direct DuckDB

2. **Trailing Return Syntax (MANDATORY)**
   - Every function: `auto func() -> ReturnType`
   - Enforced by clang-tidy (build fails otherwise)

3. **No Raw Memory Management**
   - Only `std::unique_ptr` and `std::shared_ptr`
   - No raw `new`/`delete` in modern code

4. **Use `uv run python` for ALL Python**
   - Never use `python` directly
   - Environment isolation + proper dependencies

5. **RAII for All Resources**
   - Handles, connections, sockets, files
   - Automatic cleanup on scope exit

6. **Test Before Committing**
   - Run `./scripts/validate_code.sh`
   - Build must pass (clang-tidy + ninja)
   - Unit tests should pass

### Performance Priorities

1. **Trading Loop Speed** (CRITICAL)
   - Target: <300ms per cycle
   - Current: 300ms (quotes + ML + execution)

2. **Quote Parsing** (CRITICAL for latency)
   - Target: <1ms
   - Current: 0.1ms (32x faster with simdjson)

3. **ONNX Inference** (CRITICAL for accuracy)
   - Target: <1ms per prediction
   - Current: 0.8ms (CUDA accelerated)

4. **Memory Efficiency**
   - Target: <5GB resident (includes ML models)
   - Current: ~2GB (efficient SIMD code)

---

## Integration Checklist for New Modules

When creating a new C++23 module:

- [ ] Create `.cppm` file (not `.hpp`)
- [ ] Start with `module;` global fragment
- [ ] Include only `#include "schwab_api/duckdb_bridge.hpp"` (if DB access)
- [ ] Use `export module bigbrother.category.component;`
- [ ] Import dependencies: `import bigbrother.*.type;`
- [ ] Trailing return syntax: `auto func() -> Type`
- [ ] [[nodiscard]] on all getters
- [ ] Add to CMakeLists.txt (FILE_SET CXX_MODULES)
- [ ] Use proper logging: `import bigbrother.utils.logger;`
- [ ] Handle errors with `std::expected` or return codes
- [ ] Test with `./scripts/validate_code.sh`
- [ ] Run: `ninja -C build module_name`
- [ ] Commit with proper author attribution

---

## Support & References

### Quick Links
- **Bridge Documentation:** `ai/AGENT_CODING_GUIDE.md` (complete API reference)
- **C++23 Guide:** `docs/CPP23_MODULES_GUIDE.md` (module patterns)
- **Build System:** `CMakeLists.txt` (61 targets, module configuration)
- **Validation Script:** `scripts/validate_code.sh` (clang-tidy enforcement)

### Common Issues

**Q: Module won't compile - "incomplete type" error?**
A: You're using `#include <duckdb.hpp>` in a module. Switch to `#include "schwab_api/duckdb_bridge.hpp"`

**Q: Clang-tidy failing on every build?**
A: Check trailing return syntax. All functions must be: `auto func() -> Type`

**Q: Database operations segfaulting?**
A: Check result validity before access: `if (!result || hasError(*result)) { ... }`

**Q: Python module import failing?**
A: Set LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH`

---

**Last Updated:** November 12, 2025 | **Status:** ✅ Current (all files migrated, zero build errors)
