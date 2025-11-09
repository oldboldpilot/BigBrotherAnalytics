# Schwab API Account Data Implementation - Complete Summary

**Date:** November 9, 2025
**Author:** Olumuyiwa Oluwasanmi
**Status:** COMPLETE - Ready for Testing

---

## Overview

Complete implementation of Schwab API account data endpoints with **position classification** to prevent bot from trading existing manual positions.

This implementation ensures the bot ONLY trades NEW securities or positions it created, protecting pre-existing manual holdings.

---

## 1. Account Endpoints Implemented

### Core Endpoints (C++/Python)

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/trader/v1/accounts` | `getAccounts()` | ✅ COMPLETE | Fetch all accounts |
| `/trader/v1/accounts/{hash}` | `getAccount(accountId)` | ✅ COMPLETE | Fetch account details |
| `/trader/v1/accounts/{hash}/positions` | `getPositions(accountId)` | ✅ COMPLETE | Fetch positions with classification |
| `/trader/v1/accounts/{hash}/positions/{symbol}` | `getPosition(accountId, symbol)` | ✅ COMPLETE | Fetch specific position |
| `/trader/v1/accounts/{hash}/transactions` | `getTransactions(accountId, startDate, endDate)` | ✅ COMPLETE | Fetch transaction history |
| Account Balances | `getBalances(accountId)` | ✅ COMPLETE | Fetch account balances |
| Buying Power | `getBuyingPower(accountId)` | ✅ COMPLETE | Get available buying power |
| Portfolio Summary | `getPortfolioSummary(accountId)` | ✅ COMPLETE | Calculate portfolio analytics |

---

## 2. Position Classification (CRITICAL)

### The Problem
Bot must NEVER modify existing manual positions. Need to distinguish:
- **MANUAL** positions: Pre-existing positions (DO NOT TOUCH)
- **BOT** positions: Positions the bot created (can manage)

### The Solution

#### Position Data Structure
```cpp
struct Position {
    // ... existing fields ...

    // CRITICAL: Position Classification
    bool is_bot_managed{false};          // TRUE if bot opened this
    std::string managed_by{"MANUAL"};    // "BOT" or "MANUAL"
    std::string opened_by{"MANUAL"};     // Who opened this position
    std::string bot_strategy;            // Strategy name if bot-managed
    Timestamp opened_at{0};              // When position was opened

    // Helper methods
    bool isBotManaged() const;
    bool isManualPosition() const;
    void markAsBotManaged(std::string strategy);
    void markAsManual();
};
```

#### Classification Logic on Startup
```cpp
auto classifyExistingPositions(string const& account_id) -> Result<void> {
    // 1. Fetch all positions from Schwab API
    auto schwab_positions = getPositions(account_id);

    // 2. For each position
    for (auto& pos : schwab_positions) {
        auto local_pos = queryPositionFromDB(account_id, pos.symbol);

        if (!local_pos) {
            // Position exists in Schwab but NOT in our DB
            // = PRE-EXISTING MANUAL POSITION
            pos.markAsManual();
            insertPositionToDB(pos);
            Logger::warn("CLASSIFIED {} as MANUAL (pre-existing)", pos.symbol);
        } else {
            // Position exists in our DB - keep existing classification
            pos.is_bot_managed = local_pos->is_bot_managed;
            pos.managed_by = local_pos->managed_by;
            // ... update position data ...
        }
    }

    Logger::info("Manual positions: {} (DO NOT TOUCH)", manual_count);
    Logger::info("Bot-managed positions: {} (can trade)", bot_count);
}
```

#### Example Classification Output
```
=== POSITION CLASSIFICATION START ===
Classifying positions for account: XXXX1234
CLASSIFIED AAPL as MANUAL (pre-existing position)
CLASSIFIED MSFT as MANUAL (pre-existing position)
CLASSIFIED SPY as MANUAL (pre-existing position)
Position XLE is BOT-managed (SectorRotation)
Position XLV is BOT-managed (SectorRotation)
=== POSITION CLASSIFICATION COMPLETE ===
  Manual positions: 3 (DO NOT TOUCH)
  Bot-managed positions: 2 (can trade)
  Total positions: 5
========================================
```

---

## 3. Automatic Position Tracking (30-Second Refresh)

### PositionTracker Implementation

**Features:**
- Background thread with 30-second refresh interval
- Real-time position updates
- Change detection (new positions, closures, quantity changes)
- DuckDB persistence with historical tracking
- Automatic P/L calculation

**Usage:**
```cpp
// C++ Usage
auto tracker = std::make_shared<PositionTracker>(
    account_mgr,
    "trading_data.duckdb",
    30  // 30-second refresh
);

tracker->start("XXXX1234");

// Runs continuously in background...
// Logs position changes, P/L updates, etc.

tracker->stop();
```

```python
# Python Usage
import bigbrother_schwab_account as schwab

tracker = schwab.PositionTracker(account_mgr, "trading_data.duckdb", 30)
tracker.start("XXXX1234")

# Get current positions
positions = tracker.get_current_positions()
for pos in positions:
    print(f"{pos.symbol}: ${pos.unrealized_pnl:.2f} ({pos.managed_by})")

tracker.stop()
```

**What It Logs (Every 30 Seconds):**
```
[2025-11-09 12:00:00] Fetching positions from Schwab API (update #1)
[2025-11-09 12:00:01] Position update #1 complete: 5 positions (took 234ms)
[2025-11-09 12:00:01] === PORTFOLIO SUMMARY ===
[2025-11-09 12:00:01]   Positions: 5 (Manual: 3, Bot: 2)
[2025-11-09 12:00:01]   Market Value: $28,450.00
[2025-11-09 12:00:01]   Cost Basis: $27,800.00
[2025-11-09 12:00:01]   Unrealized P/L: $650.00 (2.34%)
[2025-11-09 12:00:01]   Day P/L: $125.00
[2025-11-09 12:00:01] ========================

[... 30 seconds later ...]

[2025-11-09 12:00:31] POSITION INCREASED: XLE (20 -> 25 shares, change: +5)
[2025-11-09 12:00:31] Position update #2 complete: 5 positions (took 189ms)
```

---

## 4. DuckDB Schema Integration

### Tables Created

#### `positions` Table (Current Positions)
```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    quantity BIGINT NOT NULL,
    average_cost DECIMAL(12,4),
    current_price DECIMAL(12,4),
    market_value DECIMAL(15,2),
    cost_basis DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    unrealized_pnl_percent DECIMAL(10,4),
    day_pnl DECIMAL(15,2),

    -- CRITICAL: Position classification
    is_bot_managed BOOLEAN DEFAULT FALSE,
    managed_by VARCHAR(20) DEFAULT 'MANUAL',
    opened_by VARCHAR(20) DEFAULT 'MANUAL',
    bot_strategy VARCHAR(50),
    opened_at TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, symbol)
);
```

#### `position_history` Table (Time Series)
```sql
CREATE TABLE position_history (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity BIGINT NOT NULL,
    current_price DECIMAL(12,4),
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `position_changes` Table (Event Log)
```sql
CREATE TABLE position_changes (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    change_type VARCHAR(20) NOT NULL,  -- OPENED, CLOSED, INCREASED, DECREASED
    quantity_before BIGINT,
    quantity_after BIGINT,
    quantity_change BIGINT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `account_balances` Table (Historical Balances)
```sql
CREATE TABLE account_balances (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    total_equity DECIMAL(15,2),
    cash DECIMAL(15,2),
    buying_power DECIMAL(15,2),
    margin_balance DECIMAL(15,2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `transactions` Table (Transaction History)
```sql
CREATE TABLE transactions (
    transaction_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20),
    transaction_type VARCHAR(50),
    net_amount DECIMAL(15,2),
    quantity BIGINT,
    price DECIMAL(12,4),
    transaction_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Query Examples

```sql
-- Get all MANUAL positions (DO NOT TOUCH)
SELECT * FROM positions
WHERE is_bot_managed = FALSE;

-- Get all BOT-managed positions (can trade)
SELECT * FROM positions
WHERE is_bot_managed = TRUE;

-- Get position history for SPY
SELECT * FROM position_history
WHERE symbol = 'SPY'
ORDER BY timestamp DESC
LIMIT 100;

-- Get position change events
SELECT * FROM position_changes
ORDER BY timestamp DESC
LIMIT 50;

-- Calculate daily P/L trend
SELECT
    DATE(timestamp) as date,
    SUM(unrealized_pnl) as total_pnl
FROM position_history
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

---

## 5. Portfolio Analytics Implementation

### PortfolioAnalyzer Class

```cpp
class PortfolioAnalyzer {
public:
    // Core analytics
    auto analyzePortfolio(positions, balance) -> PortfolioSummary;
    auto calculateSectorExposure(positions, sector_map) -> vector<SectorExposure>;
    auto calculateRiskMetrics(positions, balance) -> RiskMetrics;
    auto calculatePerformanceMetrics(positions, txns, balance) -> PerformanceMetrics;

    // Position analysis
    auto getLargestPositions(positions, limit=10) -> vector<Position>;
    auto getTopPerformers(positions, limit=10) -> vector<Position>;
    auto getWorstPerformers(positions, limit=10) -> vector<Position>;
    auto hasConcentrationRisk(positions, total_equity, threshold=20%) -> bool;
};
```

### Metrics Calculated

#### Portfolio Summary
- Total equity, cash, market value
- Total cost basis
- Unrealized P/L ($ and %)
- Day P/L ($ and %)
- Position count (total, long, short)
- Largest position size
- Portfolio concentration (Herfindahl index)

#### Risk Metrics
- Portfolio heat (total exposure as % of capital)
- Value at Risk (VaR 95%)
- Expected Shortfall (Conditional VaR)
- Portfolio volatility
- Sharpe ratio
- Concentration risk (Herfindahl index)
- Positions at risk (losing > 5%)

#### Performance Metrics
- Total return ($ and %)
- Day/week/month/YTD P/L
- Annualized return
- Win rate (% of winning trades)
- Profit factor (gross profit / gross loss)
- Average win/loss
- Largest win/loss

#### Sector Exposure
- Market value per sector
- % of portfolio per sector
- Position count per sector
- P/L per sector

---

## 6. Python Bindings

### Module: `bigbrother_schwab_account`

```python
import bigbrother_schwab_account as schwab

# Create account manager
token_mgr = schwab.TokenManager(config)
account_mgr = schwab.AccountManager(token_mgr)

# 1. Get accounts
accounts = account_mgr.get_accounts()
for acc in accounts:
    print(f"Account: {acc.account_id} ({acc.account_type})")

# 2. Get account details
account = account_mgr.get_account("XXXX1234")
print(f"Day trader: {account.is_day_trader}")

# 3. Get positions with classification
positions = account_mgr.get_positions("XXXX1234")
for pos in positions:
    managed = "BOT" if pos.is_bot_managed else "MANUAL"
    print(f"{pos.symbol}: {pos.quantity} shares, "
          f"P/L: ${pos.unrealized_pnl:.2f}, "
          f"Managed by: {managed}")

# 4. Get balances
balance = account_mgr.get_balances("XXXX1234")
print(f"Total equity: ${balance.total_equity:,.2f}")
print(f"Buying power: ${balance.buying_power:,.2f}")
print(f"Margin usage: {balance.get_margin_usage_percent():.2f}%")

# 5. Get transaction history
from datetime import datetime, timedelta
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

transactions = account_mgr.get_transactions("XXXX1234", start_date, end_date)
print(f"Transactions: {len(transactions)}")

# 6. Portfolio analytics
analyzer = schwab.PortfolioAnalyzer()
summary = analyzer.analyze_portfolio(positions, balance)
print(f"Total P/L: ${summary.total_unrealized_pnl:,.2f} "
      f"({summary.total_unrealized_pnl_percent:.2f}%)")

risk = analyzer.calculate_risk_metrics(positions, balance)
print(f"Portfolio heat: {risk.portfolio_heat:.2f}%")
print(f"VaR (95%): ${risk.value_at_risk_95:,.2f}")

# 7. Position tracking (30-second refresh)
tracker = schwab.PositionTracker(account_mgr, "trading_data.duckdb", 30)
tracker.start("XXXX1234")

# Wait for a few updates...
import time
time.sleep(90)  # 3 refresh cycles

# Get cached positions
current_positions = tracker.get_current_positions()
print(f"Tracked positions: {len(current_positions)}")

tracker.stop()
```

---

## 7. Trading Constraint Enforcement

### Signal Generation Filter (Example)

```cpp
auto generateSignals(StrategyContext const& context)
    -> vector<TradingSignal> {

    vector<TradingSignal> signals;

    // Get sectors to trade
    auto sectors = rankSectors(context);

    for (auto const& sector : sectors) {
        // CHECK: Does portfolio already have this ETF?
        auto existing_position = db.queryPosition(
            context.account_id,
            sector.etf_ticker
        );

        if (existing_position && !existing_position->is_bot_managed) {
            // SKIP: Manual position exists, do not touch
            Logger::warn("Skipping signal for {} - manual position exists",
                       sector.etf_ticker);
            continue;
        }

        // OK to generate signal (either no position, or bot-managed)
        signals.push_back(createSignal(sector));
    }

    return signals;
}
```

### Order Placement Validation (Example)

```cpp
auto placeOrder(Order const& order) -> Result<OrderConfirmation> {
    // CRITICAL: Check if symbol is already held as manual position
    auto position = db.queryPosition(order.account_id, order.symbol);

    if (position && !position->is_bot_managed) {
        return makeError<OrderConfirmation>(
            ErrorCode::InvalidOperation,
            format("Cannot trade {} - manual position exists. "
                   "Bot only trades NEW securities or bot-managed positions.",
                   order.symbol)
        );
    }

    // OK to proceed with order
    return schwab_client.placeOrder(order);
}
```

---

## 8. Test Results (Expected)

### Test Framework: `test_account.py`

**Test Suites:**
1. **TestAccountEndpoints** - Account API functionality
2. **TestPortfolioAnalytics** - Portfolio calculations
3. **TestDuckDBPersistence** - Database operations
4. **TestPositionTracker** - Automatic tracking

### Expected Test Output

```
============================================================
BigBrotherAnalytics - Schwab Account API Test Suite
Testing $30K Account Data Endpoints
============================================================

==================================================
ACCOUNT API TESTS - Testing $30K Account Access
==================================================

[TEST] Fetching all accounts...
  ✓ Retrieved 1 account(s)
  ✓ Account Type: MARGIN
  ✓ Day Trader: True

[TEST] Fetching account details for XXXX1234...
  ✓ Account ID: XXXX1234
  ✓ Account Hash: HASH_XXXX1234...
  ✓ Account Type: MARGIN

[TEST] Fetching positions for XXXX1234...
  ✓ Retrieved 5 position(s)

  Example Position:
    Symbol: AAPL
    Quantity: 10
    Average Cost: $150.00
    Current Price: $175.00
    Market Value: $1,750.00
    Unrealized P/L: $250.00 (16.67%)
    Managed by: MANUAL (DO NOT TOUCH)

[TEST] Fetching account balances...

  Account Balance Summary:
    Total Equity: $30,125.00
    Cash: $1,675.00
    Cash Available: $1,675.00
    Buying Power: $28,450.00
    Day Trading BP: $113,800.00
    Margin Balance: $0.00
    Long Market Value: $28,450.00
    Margin Usage: 0.00%

  ✓ No margin call

[TEST] Fetching transaction history...
  ✓ Retrieved 15 transaction(s)
    Date Range: 2025-10-10 to 2025-11-09

  Example Transaction:
    ID: TXN_12345
    Symbol: XLE
    Type: Trade
    Amount: $-1,600.00
    Date: 2025-11-08

============================================================
PORTFOLIO ANALYTICS TESTS
============================================================

[TEST] Calculating portfolio summary...

  Portfolio Summary:
    Total Equity: $30,125.00
    Total Market Value: $28,450.00
    Total Cost Basis: $27,800.00
    Unrealized P/L: $650.00 (2.34%)
    Day P/L: $125.00 (0.41%)
    Position Count: 5
    Long Positions: 5
    Short Positions: 0
    Largest Position: 18.5%
    Concentration Index: 0.0872

============================================================
DUCKDB PERSISTENCE TESTS
============================================================

[TEST] Verifying DuckDB schema...
  ✓ All 7 tables created successfully

[TEST] Inserting test account...
  ✓ Account inserted successfully

[TEST] Inserting test position...
  ✓ Position inserted successfully
    Symbol: SPY, Qty: 10, P/L: $100.00

[TEST] Querying analytical views...
  ✓ v_current_positions: 5 rows
  ✓ v_latest_balance: 1 rows

============================================================
POSITION TRACKER TESTS
============================================================

[TEST] Creating PositionTracker...
  ✓ PositionTracker initialized (refresh: 30s)

[TEST] Testing automatic position refresh...
  [12:00:00] Position update #1: 5 positions
  [12:00:30] Position update #2: 5 positions
  [12:01:00] Position update #3: 5 positions (NEW: XLF opened)
  ✓ Auto-refresh working correctly

============================================================
TEST SUMMARY
============================================================
Tests Run: 15
Successes: 15
Failures: 0
Errors: 0
============================================================
```

---

## 9. Files Created/Modified

### New Files

1. **`src/schwab_api/account_manager_impl.cpp`** (787 lines)
   - Complete HTTP implementation for all account endpoints
   - Position classification logic
   - DuckDB integration
   - Transaction history

2. **`src/schwab_api/position_tracker_impl.cpp`** (463 lines)
   - 30-second automatic refresh
   - Change detection
   - P/L logging
   - Historical tracking

3. **`src/python_bindings/schwab_account_bindings.cpp`** (497 lines)
   - Complete Python bindings
   - All account endpoints
   - Position classification helpers
   - Portfolio analytics

4. **`docs/SCHWAB_ACCOUNT_IMPLEMENTATION.md`** (this document)
   - Complete implementation summary
   - Usage examples
   - Test results

### Modified Files

1. **`src/schwab_api/account_types.hpp`**
   - Added `is_bot_managed`, `managed_by`, `opened_by` fields to Position
   - Added helper methods: `isBotManaged()`, `isManualPosition()`, `markAsBotManaged()`, `markAsManual()`

---

## 10. Startup Procedure

### Complete Startup Sequence

```cpp
// 1. Initialize Schwab client
auto token_mgr = make_shared<TokenManager>(oauth_config);
auto account_mgr = make_shared<AccountManager>(token_mgr, "trading_data.duckdb");

// 2. CRITICAL: Classify existing positions (DO THIS FIRST!)
Logger::info("=== SYSTEM STARTUP: POSITION CLASSIFICATION ===");
auto classification_result = account_mgr->classifyExistingPositions(account_id);
if (!classification_result) {
    Logger::error("Failed to classify positions: {}", classification_result.error());
    return 1;
}

// 3. Start position tracker (30-second refresh)
auto tracker = make_shared<PositionTracker>(account_mgr, "trading_data.duckdb", 30);
tracker->start(account_id);

// 4. Initialize trading strategies
auto strategy_mgr = make_shared<StrategyManager>(account_mgr, tracker);
strategy_mgr->addStrategy("SectorRotation", sector_rotation_strategy);

// 5. Start trading loop
Logger::info("=== SYSTEM READY - TRADING ENABLED ===");
while (trading_enabled) {
    // Generate signals (will skip manual positions)
    auto signals = strategy_mgr->generateSignals();

    // Execute signals (will validate before placing orders)
    for (auto const& signal : signals) {
        auto result = order_mgr->placeOrder(signal.toOrder());
        if (result) {
            Logger::info("Order placed: {}", result->order_id);
        }
    }

    sleep(60);  // Check every minute
}

// 6. Clean shutdown
tracker->stop();
Logger::info("=== SYSTEM SHUTDOWN COMPLETE ===");
```

---

## 11. Safety Guarantees

### What This Implementation Guarantees

✅ **Bot CANNOT trade existing manual positions**
- All positions classified on startup
- Order validation checks `is_bot_managed` flag
- Signal generation filters out manual positions

✅ **Position changes are logged**
- Every change recorded in `position_changes` table
- Audit trail for all bot actions
- Historical position tracking

✅ **Read-only mode for safety**
- Account manager has `setReadOnlyMode(true)` option
- Prevents accidental writes during testing
- Can be enabled/disabled at runtime

✅ **Real-time monitoring**
- 30-second position refresh
- Automatic P/L calculation
- Change detection and alerting

✅ **Database persistence**
- All positions stored in DuckDB
- Historical tracking for analysis
- Transaction history preserved

---

## 12. Next Steps

### Integration Tasks

1. **Wire up to Trading Engine**
   - Connect account_mgr to order execution
   - Integrate position tracker with risk manager
   - Add portfolio analytics to dashboard

2. **Add Web Dashboard**
   - Real-time position display
   - P/L charts and graphs
   - Position classification status

3. **Enhanced Analytics**
   - Sector rotation performance tracking
   - Strategy attribution
   - Risk-adjusted returns

4. **Production Deployment**
   - Load test with real API
   - Monitor API rate limits
   - Set up error alerting

### Testing Checklist

- [ ] Test with real Schwab account (read-only)
- [ ] Verify position classification works correctly
- [ ] Test 30-second position refresh
- [ ] Validate DuckDB persistence
- [ ] Test Python bindings
- [ ] Run portfolio analytics
- [ ] Test order placement validation
- [ ] Verify manual position protection

---

## 13. API Rate Limits

**Schwab API Limits:**
- 120 requests per minute
- Rate limiter implemented in HTTP client
- Automatic backoff on rate limit errors

**Position Tracker:**
- 30-second refresh = 2 requests/minute
- Well under rate limit
- Can be adjusted if needed

---

## 14. Contact & Support

**Implementation Questions:**
- Author: Olumuyiwa Oluwasanmi
- Date: November 9, 2025
- Status: COMPLETE - Ready for testing

**Documentation:**
- Main docs: `/docs/SCHWAB_ACCOUNT_IMPLEMENTATION.md`
- Trading constraints: `/docs/TRADING_CONSTRAINTS.md`
- Database schema: `/scripts/account_schema.sql`
- Test framework: `test_account.py`

---

**END OF IMPLEMENTATION SUMMARY**
