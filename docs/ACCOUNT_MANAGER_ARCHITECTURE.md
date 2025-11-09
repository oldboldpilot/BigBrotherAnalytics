# AccountManager Architecture

**Visual guide to the enhanced AccountManager implementation**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AccountManager                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Public API (Thread-Safe)                                │  │
│  │                                                          │  │
│  │  • initializeDatabase(db_path) -> Result<void>         │  │
│  │  • classifyExistingPositions() -> Result<void>         │  │
│  │  • getPositions() -> Result<vector<AccountPosition>>   │  │
│  │  • getManualPositions() -> Result<...>                 │  │
│  │  • getBotManagedPositions() -> Result<...>             │  │
│  │  • isSymbolBotManaged(symbol) -> bool                  │  │
│  │  • hasManualPosition(symbol) -> bool                   │  │
│  │  • markPositionAsBotManaged(symbol, strategy) -> void  │  │
│  │  • validateCanTrade(symbol) -> Result<void>            │  │
│  │  • getPositionStats() -> tuple<size_t, size_t, size_t> │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Internal Data Structures (Mutex-Protected)             │  │
│  │                                                          │  │
│  │  manual_positions_:                                     │  │
│  │    unordered_map<string, AccountPosition>              │  │
│  │    ↳ Pre-existing positions (DO NOT TOUCH)             │  │
│  │                                                          │  │
│  │  bot_managed_symbols_:                                  │  │
│  │    unordered_set<string>                               │  │
│  │    ↳ Bot-opened positions (can trade)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Database Integration (DuckDB)                          │  │
│  │                                                          │  │
│  │  • createPositionTables()                               │  │
│  │  • queryPositionFromDB(symbol)                          │  │
│  │  • persistManualPosition(pos)                           │  │
│  │  • updatePositionManagementInDB(...)                    │  │
│  │  • isBotManagedInDB(symbol)                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  External Dependencies                                   │  │
│  │                                                          │  │
│  │  TokenManager → OAuth2 token management                 │  │
│  │  HttpClient → Schwab API requests                       │  │
│  │  Logger → Audit logging                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Position Classification Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    System Startup                                │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  classifyExistingPositions()                                     │
│                                                                  │
│  1. Fetch positions from Schwab API                              │
│     GET /trader/v1/accounts/{accountHash}/positions              │
│                                                                  │
│  2. For each position in Schwab account:                         │
│     ┌────────────────────────────────────────────────┐          │
│     │ Query local DuckDB for position                │          │
│     └────────────────────────────────────────────────┘          │
│                        │                                         │
│           ┌────────────┴────────────┐                            │
│           ▼                         ▼                            │
│     ┌──────────┐              ┌──────────┐                       │
│     │ NOT FOUND│              │  FOUND   │                       │
│     │  in DB   │              │  in DB   │                       │
│     └──────────┘              └──────────┘                       │
│           │                         │                            │
│           │                ┌────────┴────────┐                   │
│           │                ▼                 ▼                   │
│           │         ┌─────────────┐   ┌─────────────┐           │
│           │         │is_bot_managed│   │is_bot_managed│          │
│           │         │   = TRUE    │   │   = FALSE   │           │
│           │         └─────────────┘   └─────────────┘           │
│           │                │                 │                   │
│           ▼                ▼                 ▼                   │
│     ┌──────────┐    ┌──────────┐     ┌──────────┐              │
│     │  MANUAL  │    │   BOT    │     │  MANUAL  │              │
│     │ (NEW)    │    │ (EXISTING)│    │ (EXISTING)│              │
│     └──────────┘    └──────────┘     └──────────┘              │
│           │                │                 │                   │
│           ▼                ▼                 ▼                   │
│  Add to manual_      Add to bot_       Add to manual_           │
│  positions_          managed_symbols_  positions_               │
│  map                 set               map                      │
│                                                                  │
│  DO NOT TOUCH        CAN TRADE         DO NOT TOUCH             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Trading Decision Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   Strategy Generates Signal                      │
│                   (e.g., "BUY XLE")                              │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Check if symbol has manual position                    │
│                                                                  │
│  if (account_mgr->hasManualPosition("XLE"))                      │
│      ↓ YES                                                       │
│      ABORT: "Manual position exists - DO NOT TRADE"             │
│      ↓ NO                                                        │
│      Continue to Step 2                                          │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: Validate can trade                                     │
│                                                                  │
│  auto validate = account_mgr->validateCanTrade("XLE")            │
│      ↓ ERROR                                                     │
│      ABORT: Return error                                         │
│      ↓ SUCCESS                                                   │
│      Continue to Step 3                                          │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Place order                                            │
│                                                                  │
│  auto result = schwab_client->placeOrder(order)                  │
│      ↓ FILLED                                                    │
│      Continue to Step 4                                          │
│      ↓ ERROR                                                     │
│      Return error                                                │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: Mark as bot-managed                                    │
│                                                                  │
│  account_mgr->markPositionAsBotManaged("XLE", "SectorRotation")  │
│                                                                  │
│  • Add to bot_managed_symbols_                                   │
│  • Persist to DuckDB (is_bot_managed = TRUE)                    │
│  • Log: "Marked XLE as BOT-MANAGED (SectorRotation)"           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS positions (
    -- Identity
    id INTEGER PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Position Details
    quantity INTEGER NOT NULL,
    avg_cost DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    market_value DECIMAL(10,2),
    unrealized_pnl DECIMAL(10,2),

    -- ┌────────────────────────────────────────────────┐
    -- │  CRITICAL FLAGS (TRADING_CONSTRAINTS.md)       │
    -- └────────────────────────────────────────────────┘
    is_bot_managed BOOLEAN DEFAULT FALSE,  -- Bot can trade?
    managed_by VARCHAR(20) DEFAULT 'MANUAL',  -- 'BOT' or 'MANUAL'
    bot_strategy VARCHAR(50),  -- Strategy that opened this

    -- Timestamps
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    opened_by VARCHAR(20) DEFAULT 'MANUAL',  -- Who opened
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(account_id, symbol)
);

-- Indexes for performance
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_bot_managed ON positions(is_bot_managed);
CREATE INDEX idx_positions_account_symbol ON positions(account_id, symbol);
```

---

## Data Flow Diagram

```
┌─────────────────┐
│  Schwab API     │
│  (positions)    │
└────────┬────────┘
         │
         │ HTTP GET
         │
         ▼
┌─────────────────────────────┐
│  AccountManager             │
│  getPositions()             │
│                             │
│  ┌────────────────────┐     │
│  │ In-Memory Cache    │     │
│  │                    │     │
│  │ manual_positions_  │◄────┼────────┐
│  │ bot_managed_       │     │        │
│  │ symbols_           │     │        │
│  └────────────────────┘     │        │
│           │                 │        │
│           ▼                 │        │
│  ┌────────────────────┐     │        │
│  │ DuckDB             │     │        │
│  │ positions table    │◄────┼────────┤
│  │                    │     │        │
│  │ is_bot_managed     │     │        │
│  │ managed_by         │     │        │
│  │ bot_strategy       │     │        │
│  └────────────────────┘     │        │
└─────────────────────────────┘        │
         │                             │
         │ Query Results               │ Persist
         │                             │
         ▼                             │
┌─────────────────────────────┐        │
│  Trading Strategy           │        │
│  • Filter manual positions  │        │
│  • Generate signals         │        │
└────────┬────────────────────┘        │
         │                             │
         │ Signals                     │
         │                             │
         ▼                             │
┌─────────────────────────────┐        │
│  Order Manager              │        │
│  • Validate can trade       │        │
│  • Place order              │        │
│  • Mark bot-managed         │────────┘
└─────────────────────────────┘
```

---

## Thread Safety Model

```
┌──────────────────────────────────────────────────────────────────┐
│                    AccountManager                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Mutex-Protected Operations                            │     │
│  │                                                         │     │
│  │  Thread 1                Thread 2                      │     │
│  │     │                        │                         │     │
│  │     ├──> getPositions()      │                         │     │
│  │     │    lock(mutex_)        │                         │     │
│  │     │    read positions      │                         │     │
│  │     │    unlock(mutex_)      │                         │     │
│  │     │                        │                         │     │
│  │     │                        ├──> isSymbolBotManaged() │     │
│  │     │                        │    lock(mutex_)         │     │
│  │     │                        │    read set             │     │
│  │     │                        │    unlock(mutex_)       │     │
│  │     │                        │                         │     │
│  │     ├──> markPositionAsBotManaged()                    │     │
│  │     │    lock(mutex_)        │                         │     │
│  │     │    write set           │                         │     │
│  │     │    unlock(mutex_)      │                         │     │
│  │     │                        │                         │     │
│  │                                                         │     │
│  │  ✅ All operations are serialized via mutex            │     │
│  │  ✅ No data races                                      │     │
│  │  ✅ Consistent state guaranteed                        │     │
│  └────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Flow

```
┌────────────────────────────────────────────────────────────┐
│  Method Call: getPositions()                               │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Get OAuth     │
         │ token         │
         └───────┬───────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │ SUCCESS │      │  ERROR   │
   └────┬────┘      └────┬─────┘
        │                │
        │                ▼
        │         Return std::unexpected(error)
        │
        ▼
   ┌────────────────┐
   │ Make HTTP      │
   │ request        │
   └────────┬───────┘
            │
   ┌────────┴────────┐
   │                 │
   ▼                 ▼
┌─────────┐    ┌──────────┐
│ SUCCESS │    │  ERROR   │
└────┬────┘    └────┬─────┘
     │              │
     │              ▼
     │       Return std::unexpected(error)
     │
     ▼
┌────────────────┐
│ Parse JSON     │
└────────┬───────┘
         │
┌────────┴────────┐
│                 │
▼                 ▼
┌─────────┐  ┌──────────┐
│ SUCCESS │  │  ERROR   │
└────┬────┘  └────┬─────┘
     │            │
     │            ▼
     │     Return std::unexpected(error)
     │
     ▼
Return Result<vector<AccountPosition>>
```

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                     BigBrotherAnalytics System                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                           │
│  │ Strategy Engine  │                                           │
│  │ • SectorRotation│                                           │
│  │ • MeanReversion │                                           │
│  │ • Momentum      │                                           │
│  └────────┬─────────┘                                           │
│           │ hasManualPosition()                                 │
│           │ isSymbolBotManaged()                                │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────┐                       │
│  │      AccountManager                  │◄───────────────┐      │
│  │  (Position Classification)           │                │      │
│  └──────────┬───────────────────────────┘                │      │
│             │                                             │      │
│             │ getPositions()                              │      │
│             │ validateCanTrade()                          │      │
│             │                                             │      │
│             ▼                                             │      │
│  ┌──────────────────┐         ┌──────────────────┐       │      │
│  │  Order Manager   │────────>│  Schwab Client   │       │      │
│  │ • Validate       │         │ • Place orders   │       │      │
│  │ • Execute        │         │ • Get positions  │       │      │
│  │ • Track fills    │         └──────────────────┘       │      │
│  └────────┬─────────┘                                     │      │
│           │                                               │      │
│           │ markPositionAsBotManaged()                    │      │
│           └───────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │ Position Tracker │                                           │
│  │ • 30s refresh    │                                           │
│  │ • DuckDB persist │                                           │
│  └────────┬─────────┘                                           │
│           │ getManualPositions()                                │
│           │ getBotManagedPositions()                            │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────┐                       │
│  │           DuckDB                     │                       │
│  │  • positions table                   │                       │
│  │  • position_history table            │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Layout

```
AccountManager object:
┌─────────────────────────────────────────────────────────┐
│  token_mgr_: shared_ptr<TokenManager>                   │  8 bytes
│  account_id_: string                                    │ 32 bytes
│  db_path_: string                                       │ 32 bytes
│  db_initialized_: bool                                  │  1 byte
│  mutex_: std::mutex                                     │ 40 bytes
│  manual_positions_: unordered_map<string, AccountPos>   │ 56 bytes
│  bot_managed_symbols_: unordered_set<string>            │ 56 bytes
└─────────────────────────────────────────────────────────┘
Total: ~225 bytes + dynamic allocations

Cached Data (manual_positions_):
┌─────────────────────────────────────────────────────────┐
│  symbol → AccountPosition                               │
│                                                         │
│  "AAPL" → { symbol: "AAPL",                            │
│             quantity: 10,                              │
│             average_price: 150.0,                      │
│             ... }                                      │
│                                                         │
│  "MSFT" → { symbol: "MSFT", ... }                      │
└─────────────────────────────────────────────────────────┘

Bot-Managed Symbols (bot_managed_symbols_):
┌─────────────────────────────────────────────────────────┐
│  { "XLE", "XLV", "XLK", ... }                           │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `isSymbolBotManaged()` | O(1) | Hash set lookup |
| `hasManualPosition()` | O(1) | Hash map lookup |
| `getPositions()` | O(n) | n = number of positions |
| `classifyExistingPositions()` | O(n) | n = number of positions |
| `markPositionAsBotManaged()` | O(1) | Hash set insert |
| `getPositionStats()` | O(1) | Size queries |

### Space Complexity

| Structure | Complexity | Notes |
|-----------|-----------|-------|
| `manual_positions_` | O(n) | n = manual positions |
| `bot_managed_symbols_` | O(m) | m = bot positions |
| Total | O(n + m) | Linear in position count |

### Concurrency

- **Thread-safe:** Yes (mutex-protected)
- **Lock granularity:** Coarse-grained (one mutex)
- **Contention:** Low (read-heavy workload)
- **Deadlock-free:** Yes (no lock ordering issues)

---

## Configuration Example

```cpp
// config.hpp
struct AccountManagerConfig {
    std::string db_path{"trading_data.duckdb"};
    std::string account_id;
    bool auto_classify{true};  // Auto-classify on startup
    int classification_interval_hours{24};  // Re-classify every 24h
};

// Usage
AccountManagerConfig config{
    .db_path = "/var/lib/trading/positions.duckdb",
    .account_id = "XXXX1234",
    .auto_classify = true,
    .classification_interval_hours = 12
};

auto account_mgr = std::make_shared<AccountManager>(
    token_mgr,
    config.account_id
);

account_mgr->initializeDatabase(config.db_path);

if (config.auto_classify) {
    account_mgr->classifyExistingPositions();
}
```

---

## Logging Example

```
[2025-11-09 10:30:00] [INFO] AccountManager database initialized: trading_data.duckdb
[2025-11-09 10:30:00] [INFO] Classifying existing positions for account: XXXX1234
[2025-11-09 10:30:01] [INFO] Classified AAPL as MANUAL position (pre-existing): 10 shares @ $150.00
[2025-11-09 10:30:01] [INFO] Classified MSFT as MANUAL position (pre-existing): 5 shares @ $300.00
[2025-11-09 10:30:01] [INFO] Classified XLE as BOT-MANAGED position: 20 shares
[2025-11-09 10:30:01] [INFO] Position Classification Summary:
[2025-11-09 10:30:01] [INFO]   Manual positions: 2 (DO NOT TOUCH)
[2025-11-09 10:30:01] [INFO]   Bot-managed positions: 1 (can trade)
[2025-11-09 10:30:05] [WARN] Skipping signal for SPY - manual position exists
[2025-11-09 10:30:10] [INFO] Marked XLV as BOT-MANAGED position (strategy: SectorRotation)
[2025-11-09 10:30:15] [DEBUG] Persisted manual position: AAPL
[2025-11-09 10:30:15] [DEBUG] Updated position management in DB: XLV -> bot_managed=true
```

---

## Summary

The `AccountManager` provides:

✅ **Thread-safe** position management
✅ **Type-safe** error handling (Result<T>)
✅ **RAII** resource management
✅ **O(1)** position lookups
✅ **DuckDB** persistence
✅ **TRADING_CONSTRAINTS.md** compliance
✅ **Trailing return** syntax throughout
✅ **Modern C++23** features

**Status:** Production-ready API surface, ready for HTTP/DB integration.

---

**Last Updated:** 2025-11-09
**Version:** 1.0.0
**Author:** Enhanced by Claude Code
