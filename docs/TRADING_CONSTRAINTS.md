# Trading System Constraints & Safety Rules

**Date:** November 9, 2025
**Author:** oldboldpilot
**Status:** CRITICAL - Must be enforced in all code

---

## 🔴 CRITICAL CONSTRAINT: Existing Positions Protection

### Rule: DO NOT TOUCH EXISTING SECURITIES

**The automated trading system SHALL ONLY:**
- ✅ Open NEW positions (securities not currently held)
- ✅ Manage positions IT created (bot-managed flag = true)
- ✅ Close positions IT opened

**The automated trading system SHALL NOT:**
- ❌ Modify existing manual positions
- ❌ Close existing manual positions
- ❌ Add to existing manual positions
- ❌ Trade any security already in the portfolio (unless it was bot-created)

### Rationale

**Separation of Concerns:**
- Human trader maintains full control over their manual positions
- Bot only manages its own positions
- Prevents conflicts between manual and automated trading
- Allows testing bot strategies without risking existing portfolio

**Safety:**
- Existing positions may have specific tax considerations
- Manual positions may be part of longer-term strategies
- Human judgment overrides bot decisions
- Clear accountability (human vs bot trades)

---

## Implementation Requirements

### 1. Position Tracking Schema

**DuckDB Table: `positions`**

```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    market_value DECIMAL(10,2),
    unrealized_pnl DECIMAL(10,2),

    -- CRITICAL FLAGS
    is_bot_managed BOOLEAN DEFAULT FALSE,  -- TRUE if bot opened this position
    managed_by VARCHAR(20),                -- 'BOT' or 'MANUAL'
    bot_strategy VARCHAR(50),              -- Strategy that opened this (if bot-managed)

    opened_at TIMESTAMP NOT NULL,
    opened_by VARCHAR(20) DEFAULT 'MANUAL', -- 'BOT' or 'MANUAL'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(account_id, symbol)
);
```

### 2. Position Classification on Startup

**When system starts, classify existing positions:**

```cpp
auto classifyExistingPositions(std::string const& account_id) -> void {
    // 1. Fetch all positions from Schwab API
    auto positions = schwab_client.getPositions(account_id);

    // 2. Check each position in local DuckDB
    for (auto const& position : positions) {
        auto local_pos = db.queryPosition(account_id, position.symbol);

        if (!local_pos) {
            // Position exists in Schwab but not in our DB
            // = Existing manual position, DO NOT TOUCH
            db.insertPosition({
                .symbol = position.symbol,
                .quantity = position.quantity,
                .avg_cost = position.cost_basis,
                .is_bot_managed = false,
                .managed_by = "MANUAL",
                .opened_by = "MANUAL"
            });

            Logger::info("Classified {} as MANUAL position (pre-existing)",
                        position.symbol);
        }
    }
}
```

### 3. Signal Generation Filter

**Strategies MUST filter out existing securities:**

```cpp
auto generateSignals(StrategyContext const& context)
    -> std::vector<TradingSignal> {

    std::vector<TradingSignal> signals;

    // Get all sectors to rotate into
    auto sectors = rankSectors(context);

    for (auto const& sector : sectors) {
        // CHECK: Does portfolio already have this ETF?
        auto existing_position = db.queryPosition(
            context.account_id,
            sector.etf_ticker
        );

        if (existing_position && !existing_position->is_bot_managed) {
            // SKIP: Manual position, do not touch
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

### 4. Order Placement Validation

**Pre-flight check before EVERY order:**

```cpp
auto placeOrder(Order const& order) -> Result<OrderConfirmation> {
    // CRITICAL: Check if symbol is already held as manual position
    auto position = db.queryPosition(order.account_id, order.symbol);

    if (position && !position->is_bot_managed) {
        return makeError<OrderConfirmation>(
            ErrorCode::InvalidOperation,
            fmt::format("Cannot trade {} - manual position exists. "
                       "Bot only trades NEW securities or bot-managed positions.",
                       order.symbol)
        );
    }

    // OK to proceed with order
    return schwab_client.placeOrder(order);
}
```

### 5. Position Opening

**When bot opens a new position:**

```cpp
auto onOrderFilled(OrderConfirmation const& confirmation) -> void {
    if (confirmation.side == "BUY" && confirmation.status == "FILLED") {
        // Bot just opened a new position
        db.insertPosition({
            .symbol = confirmation.symbol,
            .quantity = confirmation.filled_quantity,
            .avg_cost = confirmation.avg_fill_price,
            .is_bot_managed = true,         // CRITICAL FLAG
            .managed_by = "BOT",
            .bot_strategy = confirmation.strategy_name,
            .opened_at = std::time(nullptr),
            .opened_by = "BOT"
        });

        Logger::info("Bot opened new position: {} @ ${} ({})",
                    confirmation.symbol,
                    confirmation.avg_fill_price,
                    confirmation.strategy_name);
    }
}
```

### 6. Position Closing

**Bot can ONLY close its own positions:**

```cpp
auto closePosition(std::string const& symbol) -> Result<OrderConfirmation> {
    auto position = db.queryPosition(account_id, symbol);

    if (!position) {
        return makeError<OrderConfirmation>(
            ErrorCode::NotFound,
            "Position not found"
        );
    }

    if (!position->is_bot_managed) {
        return makeError<OrderConfirmation>(
            ErrorCode::InvalidOperation,
            fmt::format("Cannot close {} - manual position. "
                       "Only human can close manual positions.",
                       symbol)
        );
    }

    // OK to close - this is a bot-managed position
    return schwab_client.placeOrder(createSellOrder(position));
}
```

---

## Startup Procedure

**Every time the system starts:**

1. **Fetch Schwab Positions**
   ```cpp
   auto schwab_positions = schwab_client.getPositions(account_id);
   ```

2. **Compare with Local Database**
   ```cpp
   for (auto const& pos : schwab_positions) {
       auto local = db.queryPosition(account_id, pos.symbol);
       if (!local) {
           // New manual position - add as MANUAL
           db.insertManualPosition(pos);
       }
   }
   ```

3. **Log Position Status**
   ```cpp
   Logger::info("Position Summary:");
   Logger::info("  Manual positions: {} (DO NOT TOUCH)", manual_count);
   Logger::info("  Bot-managed positions: {} (can trade)", bot_count);
   ```

---

## Example Scenario

### Current Portfolio (Schwab Account)

| Symbol | Quantity | Cost Basis | Type | Can Bot Trade? |
|--------|----------|------------|------|----------------|
| AAPL | 10 | $150.00 | Manual | ❌ NO - Existing |
| MSFT | 5 | $300.00 | Manual | ❌ NO - Existing |
| SPY | 50 | $400.00 | Manual | ❌ NO - Existing |

### Bot Strategy Generates Signals

**Sector Rotation Strategy says:**
- BUY XLE (Energy) - ✅ **OK** (not in portfolio)
- BUY XLV (Health Care) - ✅ **OK** (not in portfolio)
- BUY SPY - ❌ **BLOCKED** (already in portfolio as manual position)

**Result:**
- Bot places orders for XLE and XLV only
- SPY signal is filtered out (manual position exists)
- Manual positions (AAPL, MSFT, SPY) are untouched

### After Bot Trades

| Symbol | Quantity | Cost Basis | Type | Can Bot Trade? |
|--------|----------|------------|------|----------------|
| AAPL | 10 | $150.00 | Manual | ❌ NO - Manual |
| MSFT | 5 | $300.00 | Manual | ❌ NO - Manual |
| SPY | 50 | $400.00 | Manual | ❌ NO - Manual |
| **XLE** | 20 | $80.00 | **Bot (SectorRotation)** | ✅ YES - Bot-managed |
| **XLV** | 15 | $120.00 | **Bot (SectorRotation)** | ✅ YES - Bot-managed |

---

## Dashboard Display

**Position Viewer:**
```
PORTFOLIO POSITIONS
═══════════════════════════════════════

Manual Positions (HANDS OFF):
  AAPL  10 shares @ $150.00  | P&L: +$500 | DO NOT TRADE
  MSFT   5 shares @ $300.00  | P&L: +$250 | DO NOT TRADE
  SPY   50 shares @ $400.00  | P&L: +$1000 | DO NOT TRADE

Bot-Managed Positions (Active):
  XLE   20 shares @ $80.00   | P&L: +$100 | Strategy: SectorRotation
  XLV   15 shares @ $120.00  | P&L: +$75  | Strategy: SectorRotation
```

---

## Testing

**Test Cases Required:**

1. ✅ **Test: Bot respects existing positions**
   - Add manual position to Schwab account
   - Run bot strategy
   - Verify bot does NOT trade that symbol

2. ✅ **Test: Bot can close its own positions**
   - Bot opens XLE position
   - Bot strategy generates SELL signal
   - Verify bot CAN close XLE

3. ✅ **Test: Bot cannot close manual positions**
   - Manual SPY position exists
   - Bot tries to close SPY
   - Verify order is REJECTED

4. ✅ **Test: Startup classification**
   - Start with mixed portfolio (manual + bot)
   - Restart bot
   - Verify correct classification on reload

---

## Package Management

### ✅ Use `uv` for Python Dependencies

**CORRECT Commands:**
```bash
# Initialize/activate environment
uv init

# Add dependencies
uv add pandas
uv add numpy
uv add duckdb
uv add pybind11

# Run Python scripts
uv run python script.py
uv run pytest tests/test_*.py

# Run specific packages
uv run myapp
```

**WRONG:**
```bash
pip install pandas              # ❌ DON'T USE
python script.py                # ❌ Use: uv run python script.py
source .venv/bin/activate       # ❌ Use: uv init
```

**Rationale:**
- `uv` is the project's package manager
- Faster than pip (10-100x)
- Better dependency resolution
- Automatic lock file management
- Built-in virtual environment handling

---

## Enforcement Checklist

### Code Review Checklist

Before merging ANY trading code:

- [ ] Does it check `is_bot_managed` before trading?
- [ ] Does it filter existing positions in signal generation?
- [ ] Does it validate orders against manual positions?
- [ ] Does it mark new positions as `is_bot_managed = true`?
- [ ] Does it log position type (manual vs bot)?
- [ ] Does it use `uv add` for dependencies (not pip)?
- [ ] Are manual positions clearly separated in logs?
- [ ] Are there tests for manual position protection?

---

## Summary

**Golden Rules:**

1. 🚫 **NEVER touch existing securities** (pre-existing positions)
2. ✅ **ONLY trade NEW securities** (not currently held)
3. ✅ **ONLY manage bot-created positions** (`is_bot_managed = true`)
4. 🔍 **ALWAYS check position type** before trading
5. 📦 **ALWAYS use `uv add`** for Python packages (not pip)

**This is a SAFETY CRITICAL requirement.**

**Non-compliance = System shutdown.**

---

**Last Updated:** November 9, 2025
**Enforcement:** MANDATORY in all code
