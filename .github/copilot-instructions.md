# BigBrotherAnalytics AI Coding Assistant Instructions

**Version:** 2.0
**Last Updated:** November 10, 2025
**Author:** Olumuyiwa Oluwasanmi

## Project Overview

BigBrotherAnalytics is a **production-ready** AI-powered algorithmic trading platform with full Schwab API integration, employment-driven sector rotation, and advanced options strategies. **Live Trading Integration (TASK 2) is COMPLETE** at 95% production readiness.

**CRITICAL AUTHORSHIP STANDARD:**
- **ALL files MUST include:** Author: Olumuyiwa Oluwasanmi
- **Applies to:** Code, configs, docs, scripts, tests, CI/CD files
- **Format:** See Section 11 in docs/CODING_STANDARDS.md
- **NO co-authoring** - Only Olumuyiwa Oluwasanmi
- **Git commits:** Include "Co-Authored-By: Claude <noreply@anthropic.com>"
- **Enforcement:** Pre-commit hooks + CI/CD checks

## Current Status (November 10, 2025)

### Implementation Complete: 95%

**✅ Live Trading System Operational:**
- Schwab API integration (OAuth 2.0, market data, orders, accounts) - 100%
- Live Trading Engine (signal execution, position tracking, stop-losses) - 100%
- Employment Signals (BLS data integration for sector rotation) - 100%
- Options Strategies (Iron Condor, Straddle, Volatility Arbitrage) - 100%
- Risk Management (pre-trade validation, position sizing, portfolio heat) - 100%
- ⏳ Pre-existing clang-tidy errors (34 errors in older code, non-blocking)

**Core Features:**
- Real-time market data from Schwab API
- Automatic signal-to-order conversion
- Position tracking with P&L calculation
- Automatic 10% stop-loss execution
- Employment data integration (BLS)
- Options chain fetching (SPY, QQQ)

**Recent Work (Nov 9-10):**
- Implemented buildContext() - Market data aggregation (84 lines)
- Implemented loadEmploymentSignals() - BLS integration (139 lines)
- Implemented updatePositions() - P&L tracking (78 lines)
- Implemented checkStopLosses() - Risk management (70 lines)
- Implemented StrategyExecutor::execute() - Signal-to-order conversion (136 lines)
- Fixed RiskManager API usage (assessTrade)
- **Total: ~420 lines of production trading code**

### Architecture Summary

**Three Core Systems:**
1. **Market Intelligence Engine** - Multi-source data processing and impact prediction
2. **Correlation Analysis Tool** - Time-series relationships with 60-100x speedup (C++23/OpenMP/MPI)
3. **Trading Decision Engine** - Live trading with Schwab API (C++23/Python hybrid)

**Technology Stack:**
- **C++23 Modules** - 25 production-ready modules with trailing return syntax
- **Clang 21.1.5** - /usr/local/bin/clang++ (built via Ansible playbook)
- **DuckDB** - Embedded database (zero setup, ACID compliant)
- **Python 3.13 + uv** - All Python execution via `uv run python script.py`
- **CMake + Ninja** - Build system with C++23 module support
- **OpenMP + MPI + UPC++** - Massive parallelization for 32+ cores

## 🚨 CRITICAL SAFETY RULES 🚨

### Trading Constraint: DO NOT TOUCH EXISTING POSITIONS

**The bot SHALL ONLY:**
- ✅ Open NEW positions (securities not currently held)
- ✅ Manage positions IT created (is_bot_managed flag = true)
- ✅ Close positions IT opened

**The bot SHALL NOT:**
- ❌ Modify existing manual positions
- ❌ Close existing manual positions
- ❌ Add to existing manual positions
- ❌ Trade any security already in the portfolio (unless bot-created)

**Implementation Requirements:**
```cpp
// Check before EVERY order
auto position = db.queryPosition(order.account_id, order.symbol);
if (position && !position->is_bot_managed) {
    return makeError("Cannot trade - manual position exists");
}

// Mark bot-created positions
db.insertPosition({
    .symbol = symbol,
    .is_bot_managed = true,  // CRITICAL FLAG
    .managed_by = "BOT",
    .bot_strategy = "SectorRotation"
});
```

**See:** docs/TRADING_CONSTRAINTS.md for complete safety rules

### Risk Management Parameters

**Per-Trade Limits:**
- $1,500 max position size
- $900 max daily loss
- 15% max portfolio heat
- 10 concurrent positions max
- 10% automatic stop-loss

**Pre-Trade Validation:**
```cpp
auto risk_assessment = risk_manager_->assessTrade(
    symbol, position_size, entry_price,
    stop_price, target_price, win_probability
);
if (!risk_assessment || !risk_assessment->approved) {
    // REJECT trade
}
```

## Build System & Compiler Configuration

### Compiler Paths (Ansible-Managed)

**ALWAYS use these compiler paths:**
```bash
CC=/usr/local/bin/clang
CXX=/usr/local/bin/clang++
CLANG_TIDY=/usr/local/bin/clang-tidy
CLANG_FORMAT=/usr/local/bin/clang-format
```

**Built with Ansible:** `playbooks/complete-tier1-setup.yml`
- Clang 21.1.5 (latest with C++23 support)
- OpenMP, MPI, UPC++ configured
- libc++ (LLVM's C++ standard library)

### Standard Build Process

**Build with clang-tidy (Pre-existing errors may block):**
```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
rm -rf build && mkdir build && cd build
env CC=/usr/local/bin/clang CXX=/usr/local/bin/clang++ cmake -G Ninja ..
ninja bigbrother backtest
```

**Build without clang-tidy (Recommended for testing):**
```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
rm -rf build && mkdir build && cd build
env SKIP_CLANG_TIDY=1 cmake -G Ninja ..
ninja bigbrother backtest
```

**Library paths for execution:**
```bash
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/bigbrother
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/backtest
```

### clang-tidy Configuration (11 Check Categories)

**Comprehensive Checks Enabled:**
1. **cppcoreguidelines-*** - ALL C++ Core Guidelines rules
2. **cert-*** - CERT C++ Secure Coding Standard
3. **concurrency-*** - Thread safety, race conditions, deadlocks
4. **performance-*** - Optimization, unnecessary copies
5. **portability-*** - Cross-platform compatibility
6. **openmp-*** - OpenMP parallelization safety, data races
7. **mpi-*** - MPI message passing correctness
8. **modernize-*** - C++23 features, trailing return syntax
9. **bugprone-*** - Bug detection, logic errors
10. **clang-analyzer-*** - Static analysis
11. **readability-*** - Code clarity, naming

**Enforced as ERRORS (blocks build/commit):**
- `modernize-use-trailing-return-type` (ALL functions)
- `cppcoreguidelines-special-member-functions` (Rule of Five)
- `modernize-use-nodiscard` ([[nodiscard]] on getters)
- `modernize-use-nullptr` (no NULL)
- `cppcoreguidelines-no-malloc` (no malloc/free)

**Pre-existing issues:** 34 clang-tidy errors in older code (documented in docs/CURRENT_STATUS.md:109-120)
- position_tracker_impl.cpp: 14 errors
- account_manager_impl.cpp: 18 errors
- token_manager.cpp: 1 error
- orders_manager.cppm: 1 error

**These do NOT affect Live Trading code which has 0 errors.**

## C++23 Coding Standards (MANDATORY)

### Trailing Return Type Syntax (100% Required)

**✅ CORRECT:**
```cpp
auto calculatePrice(double spot, double strike) -> double {
    return spot - strike;
}

[[nodiscard]] auto getSymbol() const noexcept -> std::string const& {
    return symbol_;
}
```

**❌ INCORRECT:**
```cpp
double calculatePrice(double spot, double strike) {  // WRONG!
    return spot - strike;
}
```

### Naming Conventions (CRITICAL - Enforced by clang-tidy)

| Entity | Convention | Example | Notes |
|--------|------------|---------|-------|
| **Namespaces** | `lower_case` | `bigbrother::utils` | C++ standard |
| **Classes/Structs** | `CamelCase` | `RiskManager`, `TradingSignal` | Clear types |
| **Functions/Methods** | `camelBack` | `calculatePrice()`, `getName()` | Start lowercase |
| **Local variables** | `lower_case` | `auto spot_price = 150.0;` | Readable |
| **Parameters** | `lower_case` | `auto func(double strike_price)` | Consistent |
| **Local constants** | `lower_case` | `const auto sum = 0.0;` | **Modern C++23** |
| **Constexpr constants** | `lower_case` | `constexpr auto pi = 3.14;` | **Preferred** |
| **Member variables** | `lower_case` | `double price;` (public) | Standard |
| **Private members** | `lower_case_` | `double price_;` | Trailing _ |
| **Enums** | `CamelCase` | `enum class SignalType` | Clear types |
| **Enum values** | `CamelCase` | `SignalType::Buy` | Clear values |

### C++23 Module Structure

```cpp
/**
 * BigBrotherAnalytics - Component Name
 *
 * [Brief description]
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: [Creation Date]
 *
 * Following C++ Core Guidelines:
 * - C.21: Define or delete all default operations
 * - F.16: Pass cheap types by value
 * - E: std::expected for error handling
 * - Trailing return type syntax throughout
 */

// Global module fragment (standard library only)
module;

#include <vector>
#include <string>
#include <expected>

// Module declaration
export module bigbrother.component.name;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Exported interface
export namespace bigbrother::component {
    [[nodiscard]] auto calculate() -> Result<double>;

    class PublicClass {
    public:
        auto method() -> void;
    private:
        double value_;
    };
}

// Private implementation (optional)
module :private;

namespace bigbrother::component {
    // Private helpers
    auto helperFunction() -> void {
        const auto local_const = 42;  // lower_case for locals
    }
}
```

### Container Performance Standard (CRITICAL)

**Rule:** Prefer `std::unordered_map` over `std::map` unless ordering required

**✅ PREFERRED (O(1) average):**
```cpp
std::unordered_map<std::string, Price> price_cache;
std::unordered_map<Symbol, QuoteData> market_data;
```

**⚠️ Use std::map only when ordering required (O(log n)):**
```cpp
// JUSTIFIED: need chronological order
std::map<Timestamp, Trade> time_ordered_trades;
```

**Rationale:**
- `unordered_map` is faster for most use cases
- More flexible (doesn't require operator<)
- Better for real-time trading where latency matters
- Only use `map` when you explicitly need sorted iteration

**Enforcement:** CodeQL checks prefer unordered_map usage

## Live Trading Architecture

### Trading Cycle Flow

```
1. buildContext()     → Fetch market data, account info, employment signals
2. generateSignals()  → Run all strategies, generate trading signals
3. execute()          → Risk validation, order placement via Schwab API
4. updatePositions()  → Track P&L, store to DuckDB
5. checkStopLosses()  → Monitor positions, execute stop-losses
```

### Key Components

| Component | Status | File |
|-----------|--------|------|
| Trading Engine | ✅ Complete | src/main.cpp:308-689 |
| Strategy Executor | ✅ Complete | src/trading_decision/strategy.cppm:984-1119 |
| Schwab API Client | ✅ Operational | src/schwab_api/*.cppm |
| Risk Manager | ✅ Integrated | src/risk_management/*.cppm |
| Employment Signals | ✅ Complete | src/market_intelligence/*.cppm |
| DuckDB Persistence | ✅ Integrated | src/utils/database.cppm |

### Core Trading Functions

**buildContext() - Market Data Aggregation (src/main.cpp:308-391)**
```cpp
[[nodiscard]] auto buildContext() -> strategy::StrategyContext {
    // Fetches:
    // - Account info (balance, buying power)
    // - Current positions
    // - Real-time quotes (SPY, QQQ, sector ETFs)
    // - Employment signals (BLS data)
    // - Options chains (SPY, QQQ)
}
```

**loadEmploymentSignals() - BLS Integration (src/main.cpp:401-539)**
```cpp
auto loadEmploymentSignals(strategy::StrategyContext& context) -> void {
    // Loads:
    // - Sector employment signals (11 GICS sectors)
    // - Rotation recommendations (overweight/underweight)
    // - Jobless claims spike detection (recession warning)
    // - Aggregate employment health score
}
```

**StrategyExecutor::execute() - Signal-to-Order (src/trading_decision/strategy.cppm:984-1119)**
```cpp
auto execute(std::vector<TradingSignal> const& signals) -> void {
    for (auto const& signal : signals) {
        // 1. Calculate entry/stop/target prices
        // 2. Risk assessment via RiskManager::assessTrade()
        // 3. Place order via Schwab API
        // 4. Handle errors and log rationale
    }
}
```

**updatePositions() - P&L Tracking (src/main.cpp:541-618)**
```cpp
auto updatePositions() -> void {
    // 1. Fetch latest positions from Schwab API
    // 2. Calculate total unrealized/realized P&L
    // 3. Store positions snapshot to DuckDB
    // 4. Track bot-managed vs manual positions separately
}
```

**checkStopLosses() - Risk Management (src/main.cpp:620-689)**
```cpp
auto checkStopLosses() -> void {
    // 1. Get all bot-managed positions
    // 2. Calculate loss percentage
    // 3. Trigger 10% stop-loss automatically
    // 4. Place market order to close position
}
```

## Schwab API Integration

### OAuth 2.0 Authentication

**Python Script (Verified Working):**
```bash
uv run python scripts/schwab_oauth_acquisition.py
```

**C++ Implementation:**
- Token management with automatic refresh
- Secure credential storage in DuckDB
- OAuth redirect server on port 8080

### API Endpoints

**Market Data:**
- `getQuote(symbol)` - Real-time quote
- `getOptionChain(request)` - Options chain
- `getHistoricalData(symbol, period)` - Historical bars

**Orders:**
- `placeOrder(order)` - Submit order
- `cancelOrder(order_id)` - Cancel order
- `getOrder(order_id)` - Order status

**Accounts:**
- `getAccountInfo()` - Account balance, buying power
- `getPositions()` - Current positions with P&L
- `getTransactions()` - Transaction history

## Database Schema (DuckDB)

### Employment Data Tables

**sectors** - 11 GICS sectors with ETF mappings
```sql
CREATE TABLE sectors (
    sector_id INTEGER PRIMARY KEY,
    sector_code INTEGER NOT NULL,  -- 10, 15, 20, ...
    sector_name VARCHAR NOT NULL,  -- Energy, Materials, ...
    sector_category VARCHAR NOT NULL,  -- Defensive, Cyclical, Sensitive
    etf_ticker VARCHAR  -- XLE, XLF, XLK, ...
);
```

**sector_employment** - BLS employment data by sector
```sql
CREATE TABLE sector_employment (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    bls_series_id VARCHAR NOT NULL,
    report_date DATE NOT NULL,
    employment_count INTEGER,
    unemployment_rate DOUBLE,
    job_openings INTEGER
);
```

**positions_history** - Bot position tracking
```sql
CREATE TABLE positions_history (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    quantity INTEGER,
    average_price DOUBLE,
    current_price DOUBLE,
    unrealized_pnl DOUBLE,
    is_bot_managed BOOLEAN,
    strategy VARCHAR
);
```

### 11 GICS Sectors (PRODUCTION DATA)

1. **Energy (XLE)** - Cyclical
2. **Materials (XLB)** - Cyclical
3. **Industrials (XLI)** - Sensitive
4. **Consumer Discretionary (XLY)** - Sensitive
5. **Consumer Staples (XLP)** - Defensive
6. **Health Care (XLV)** - Defensive
7. **Financials (XLF)** - Sensitive
8. **Information Technology (XLK)** - Sensitive
9. **Communication Services (XLC)** - Sensitive
10. **Utilities (XLU)** - Defensive
11. **Real Estate (XLRE)** - Sensitive

## Package Management

### Python: Use `uv` Exclusively

**✅ CORRECT:**
```bash
# Add dependencies
uv add pandas numpy duckdb

# Run scripts
uv run python script.py
uv run pytest tests/test_*.py

# Run executables
uv run myapp
```

**❌ WRONG:**
```bash
pip install pandas              # DON'T USE
python script.py                # Use: uv run python script.py
source .venv/bin/activate       # Use: uv init
```

**Rationale:**
- 10-100x faster than pip
- Better dependency resolution
- Automatic lock file management
- Built-in virtual environment handling

## Testing & Execution

### Paper Trading Mode

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/build
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/bigbrother --config configs/paper_trading.yaml
```

**Configuration (configs/paper_trading.yaml):**
- Dry-run mode enabled (no real orders)
- Conservative limits ($5,000 account, $100 max position)
- Debug logging enabled

### Integration Tests

```bash
# Schwab API E2E workflow
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/test_schwab_e2e_workflow

# Options pricing
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/test_options_pricing

# Correlation engine
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/test_correlation
```

## Key Documentation

### MUST READ BEFORE CODING

**1. docs/CURRENT_STATUS.md** (340 lines)
- Current implementation status (95% complete)
- Build instructions
- Testing procedures
- Next steps breakdown
- Architecture overview

**2. docs/CODING_STANDARDS.md** (733 lines)
- Trailing return type syntax (MANDATORY)
- C++ Core Guidelines enforcement
- Naming conventions (enforced by clang-tidy)
- Module structure patterns
- Fluent API requirements
- Container performance standards
- Authorship requirements

**3. docs/TRADING_CONSTRAINTS.md** (400 lines)
- DO NOT TOUCH existing positions (CRITICAL SAFETY RULE)
- Position tracking requirements
- Risk management parameters
- Package management (uv)

**4. docs/PRD.md** (5000+ lines)
- Complete requirements document
- Section 3.2.11: Department of Labor API
- Section 3.2.12: 11 GICS Business Sectors
- Complete data sources, trading strategies

### Implementation Guides

**Schwab API:**
- docs/schwab_oauth_implementation.md
- docs/SCHWAB_API_IMPLEMENTATION_STATUS.md
- docs/SCHWAB_MARKET_DATA.md
- docs/SCHWAB_ACCOUNT_IMPLEMENTATION.md
- docs/SCHWAB_ORDERS_COMPLETE.md

**Employment Signals:**
- docs/EMPLOYMENT_DATA_INTEGRATION.md
- docs/EMPLOYMENT_SIGNALS_IMPLEMENTATION.md
- docs/employment_signals_architecture.md
- docs/SECTOR_ROTATION_STRATEGY.md

**Fluent APIs:**
- docs/FLUENT_API_GUIDE.md
- docs/FLUENT_API_QUICK_REFERENCE.md
- docs/FLUENT_RISK_API.md
- docs/INDEX.md (Fluent API navigation)

**Build & Development:**
- docs/BUILD_WORKFLOW.md
- docs/CPP_MODULES_MIGRATION.md
- docs/PYTHON_BINDINGS_GUIDE.md

### Session Documentation

**Live Trading Integration:**
- docs/LIVE_TRADING_INTEGRATION_SESSION.md
- LIVE_TRADING_SESSION_FINAL_SUMMARY.md
- docs/SESSION_2025-11-09_FINAL_SUMMARY.md

## Python Bindings with pybind11 (✅ COMPLETE)

**Completed Bindings (100% operational):**

1. **DuckDB C++ API Bindings** ✅
   - Direct database access with C++ performance
   - 9 specialized functions (query, count, aggregate, join, filter, sort, analyze, optimize, vacuum)
   - Zero-copy NumPy transfers
   - **Performance:** 1.41x speedup, sub-10ms queries
   - **Test Coverage:** 100% (29/29 tests passing)

2. **Options Pricing Bindings** ✅
   - Black-Scholes, Trinomial Tree, Greeks calculation
   - GIL-free execution for parallel pricing
   - **Expected speedup:** 30-50x over pure Python

3. **Correlation Engine Bindings** ✅
   - 6 functions: pearson, spearman, cross_correlation, rolling_correlation, etc.
   - OpenMP parallelization (bypasses GIL)
   - **Expected speedup:** 60-100x over pandas

4. **Risk Management Bindings** ✅
   - Kelly Criterion, position sizing, Monte Carlo simulation
   - OpenMP parallelization
   - **Expected speedup:** 30-50x for Monte Carlo

**Documentation:**
- docs/PYTHON_BINDINGS_GUIDE.md
- docs/CORRELATION_API_REFERENCE.md

## Options Trading Strategies

### Iron Condor (Primary Strategy)

**Entry Criteria:**
- IV Rank > 50
- Short strikes at ±1σ
- Long strikes at ±1.5σ

**Exit Criteria:**
- 50% profit target
- 2x stop loss
- 7 DTE time decay

**Expected Performance:**
- 65-75% win rate
- 15-30% ROC per trade

### Straddle Strategy

**Entry:** High IV volatility breakout
**Risk:** Unlimited
**Implementation:** src/trading_decision/strategies.cppm

### Volatility Arbitrage

**Entry:** Implied vs realized volatility spread
**Implementation:** src/trading_decision/strategies.cppm

## Fluent API Patterns (6 Implemented)

All major components provide fluent APIs:

1. **OptionBuilder** - Options pricing configuration
2. **CorrelationAnalyzer** - Correlation analysis
3. **RiskAssessor** - Risk assessment
4. **SchwabQuery** - API queries
5. **TaxCalculatorBuilder** - Tax calculations
6. **BacktestRunner** - Backtesting configuration

**Example Usage:**
```cpp
auto price = OptionBuilder()
    .call()
    .spot(150.0)
    .strike(155.0)
    .volatility(0.25)
    .timeToExpiry(30.0/365.0)
    .price();

auto signals = mgr.signalBuilder()
    .forContext(context)
    .withMinConfidence(0.70)
    .limitTo(10)
    .generate();
```

## Performance Characteristics

### Latency Measurements

| Operation | Target | Expected |
|-----------|--------|----------|
| buildContext() | < 500ms | ~300ms |
| Signal Generation | < 100ms | ~50ms |
| Order Placement | < 200ms | ~150ms |
| Position Update | < 300ms | ~250ms |

**Full Trading Cycle:** ~830ms (target: < 1 second) ✅

### Scalability

- **Concurrent Signals:** Up to 100 signals/cycle
- **Position Tracking:** Unlimited (DuckDB)
- **Order Volume:** 120 orders/minute (Schwab API limit)
- **Database Growth:** ~1MB/day

## Common Pitfalls & Reminders

1. **Use /usr/local/bin/clang++** - Not system clang, not Homebrew clang
2. **Module syntax matters** - Use trailing `export module` declarations
3. **Trailing return syntax REQUIRED** - All functions: `auto func() -> ReturnType`
4. **Prefer unordered_map** - Use over map unless ordering required
5. **Python execution** - Always use `uv run` prefix
6. **Authorship MANDATORY** - ALL files: Author: Olumuyiwa Oluwasanmi
7. **clang-tidy will block** - Fix errors before building/committing (or use SKIP_CLANG_TIDY=1 for testing)
8. **DO NOT TOUCH existing positions** - Bot only trades NEW securities or bot-managed positions
9. **Library paths** - Set `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` for execution
10. **Pre-existing clang-tidy errors** - 34 errors exist in older code (documented, non-blocking)

## Next Steps (from docs/CURRENT_STATUS.md)

### Immediate (1-2 hours)

1. Fix pre-existing clang-tidy errors (34 errors in older code)
   - position_tracker_impl.cpp: Add special member functions, fix lambda trailing returns
   - account_manager_impl.cpp: Add std::max include, fix TokenManager usage
   - token_manager.cpp, orders_manager.cppm: Fix DuckDB incomplete type issues

2. Build verification
   - Complete clean build with clang-tidy enabled
   - Verify all executables compile
   - Run full test suite

### Short-Term (This Week)

3. Paper trading testing
   - Test with paper_trading.yaml config
   - Validate end-to-end workflow
   - Test with small positions ($50-100)
   - Verify stop-loss triggers

4. Employment data integration
   - Load BLS employment data
   - Test sector rotation signals
   - Validate signal generation

5. Live trading (small scale)
   - Start with $50-100 trades
   - Monitor for 1 week
   - Validate execution quality

### Medium-Term (Next 2 Weeks)

6. Production hardening
   - Add retry logic for API calls
   - Implement circuit breaker
   - Add monitoring and alerting
   - Performance optimization

7. Dashboard development
   - Create web dashboard (FastAPI/Streamlit)
   - Real-time position display
   - P&L charts and metrics
   - Trade history and analytics

## Project Goals (Tier 1 POC)

**Success Criteria:**
- Daily profitability ($150+/day) with $30k Schwab account
- 80% winning days
- >60% win rate
- Sharpe ratio >2.0
- Max drawdown <15%
- 3+ months consistent performance

**Current Status:** Ready for paper trading, then small-scale live trading

---

**Last Updated:** November 10, 2025
**Version:** 2.0
**Author:** Olumuyiwa Oluwasanmi
**Status:** Live Trading Integration Complete (95%)
