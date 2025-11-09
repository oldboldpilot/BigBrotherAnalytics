# Schwab Account API - Quick Start Guide

**5-Minute Guide to Account Data and Position Classification**

---

## 1. Setup (One-Time)

### Compile C++ Code
```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Initialize Database
```bash
# Run the account schema
duckdb trading_data.duckdb < scripts/account_schema.sql
```

---

## 2. Basic Python Usage

### Initialize Client
```python
import bigbrother_schwab_account as schwab

# OAuth configuration
config = schwab.OAuth2Config()
config.client_id = "YOUR_CLIENT_ID"
config.client_secret = "YOUR_CLIENT_SECRET"
config.refresh_token = "YOUR_REFRESH_TOKEN"

# Create account manager
token_mgr = schwab.TokenManager(config)
account_mgr = schwab.AccountManager(token_mgr, "trading_data.duckdb")
```

### Get Accounts
```python
# Fetch all accounts
accounts = account_mgr.get_accounts()
for account in accounts:
    print(f"Account: {account.account_id} ({account.account_type})")
    print(f"  Day Trader: {account.is_day_trader}")
    print(f"  Margin: {account.is_margin_account()}")
```

### Get Account Balance
```python
# Get balances
balance = account_mgr.get_balances("XXXX1234")
print(f"Total Equity: ${balance.total_equity:,.2f}")
print(f"Buying Power: ${balance.buying_power:,.2f}")
print(f"Margin Usage: {balance.get_margin_usage_percent():.2f}%")

# Check for margin call
if balance.has_margin_call():
    print(f"⚠ MARGIN CALL: ${balance.get_total_call_amount():,.2f}")
```

### Get Positions with Classification
```python
# Fetch positions
positions = account_mgr.get_positions("XXXX1234")

for pos in positions:
    # Position details
    print(f"{pos.symbol}:")
    print(f"  Quantity: {pos.quantity}")
    print(f"  Value: ${pos.market_value:,.2f}")
    print(f"  P/L: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_percent:.2f}%)")

    # CRITICAL: Position classification
    if pos.is_bot_managed():
        print(f"  ✓ BOT-MANAGED (Strategy: {pos.bot_strategy})")
    else:
        print(f"  ⚠ MANUAL POSITION - DO NOT TOUCH")
```

---

## 3. Position Classification on Startup

### CRITICAL: Classify Existing Positions
```python
# MUST run this on every startup!
# This marks pre-existing positions as MANUAL
result = account_mgr.classify_existing_positions("XXXX1234")

# Output:
# === POSITION CLASSIFICATION START ===
# CLASSIFIED AAPL as MANUAL (pre-existing position)
# CLASSIFIED MSFT as MANUAL (pre-existing position)
# CLASSIFIED SPY as MANUAL (pre-existing position)
# === POSITION CLASSIFICATION COMPLETE ===
#   Manual positions: 3 (DO NOT TOUCH)
#   Bot-managed positions: 0 (can trade)
```

### Mark New Bot Positions
```python
# When bot opens a new position
position = schwab.Position()
position.symbol = "XLE"
position.quantity = 20
position.average_cost = 80.00
position.mark_as_bot_managed("SectorRotation")

# Now position is classified as BOT-managed
assert position.is_bot_managed() == True
assert position.managed_by == "BOT"
assert position.bot_strategy == "SectorRotation"
```

---

## 4. Automatic Position Tracking (30-Second Refresh)

### Start Position Tracker
```python
# Create tracker (30-second refresh)
tracker = schwab.PositionTracker(
    account_mgr,
    "trading_data.duckdb",
    30  # refresh interval in seconds
)

# Start tracking
tracker.start("XXXX1234")

# Tracker now runs in background, logging every 30 seconds:
# [12:00:00] Position update #1: 5 positions
# [12:00:00] === PORTFOLIO SUMMARY ===
# [12:00:00]   Positions: 5 (Manual: 3, Bot: 2)
# [12:00:00]   Unrealized P/L: $650.00 (2.34%)
# [12:00:00] ========================
# [12:00:30] Position update #2: 5 positions
# [12:01:00] POSITION INCREASED: XLE (20 -> 25 shares)
```

### Query Tracked Positions
```python
# Get current positions from cache
positions = tracker.get_current_positions()
print(f"Tracking {len(positions)} positions")

# Get specific position
xle_position = tracker.get_position("XLE")
if xle_position:
    print(f"XLE: {xle_position.quantity} shares @ ${xle_position.current_price:.2f}")

# Check tracker status
print(f"Running: {tracker.is_running()}")
print(f"Last update: {tracker.get_last_update_time()}")
```

### Stop Tracker
```python
# Stop tracking when done
tracker.stop()
```

---

## 5. Portfolio Analytics

### Calculate Portfolio Summary
```python
analyzer = schwab.PortfolioAnalyzer()

# Get portfolio summary
summary = analyzer.analyze_portfolio(positions, balance)

print(f"Total Market Value: ${summary.total_market_value:,.2f}")
print(f"Total Cost Basis: ${summary.total_cost_basis:,.2f}")
print(f"Total P/L: ${summary.total_unrealized_pnl:,.2f} ({summary.total_unrealized_pnl_percent:.2f}%)")
print(f"Day P/L: ${summary.total_day_pnl:,.2f}")
print(f"Position Count: {summary.position_count}")
print(f"Largest Position: {summary.largest_position_percent:.2f}%")
print(f"Concentration: {summary.portfolio_concentration:.4f}")
```

### Calculate Risk Metrics
```python
risk = analyzer.calculate_risk_metrics(positions, balance)

print(f"Portfolio Heat: {risk.portfolio_heat:.2f}%")
print(f"VaR (95%): ${risk.value_at_risk_95:,.2f}")
print(f"Expected Shortfall: ${risk.expected_shortfall:,.2f}")
print(f"Sharpe Ratio: {risk.sharpe_ratio:.2f}")
print(f"Concentration Risk: {risk.concentration_risk:.4f}")
print(f"Positions at Risk: {risk.positions_at_risk}")
```

### Get Top/Worst Performers
```python
# Top 5 performers
top = analyzer.get_top_performers(positions, limit=5)
print("Top Performers:")
for pos in top:
    print(f"  {pos.symbol}: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_percent:.2f}%)")

# Worst 5 performers
worst = analyzer.get_worst_performers(positions, limit=5)
print("Worst Performers:")
for pos in worst:
    print(f"  {pos.symbol}: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_percent:.2f}%)")
```

---

## 6. Transaction History

### Get Transactions
```python
from datetime import datetime, timedelta

# Date range
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

# Fetch transactions
transactions = account_mgr.get_transactions("XXXX1234", start_date, end_date)

print(f"Found {len(transactions)} transactions")

for txn in transactions:
    print(f"{txn.transaction_date}: {txn.symbol}")
    print(f"  Type: {txn.type}")
    print(f"  Qty: {txn.quantity}")
    print(f"  Price: ${txn.price:.2f}")
    print(f"  Amount: ${txn.net_amount:.2f}")
    print(f"  Fees: ${txn.get_total_cost():.2f}")
```

---

## 7. Trading Constraints Enforcement

### Check Before Placing Order
```python
def can_trade_symbol(account_id: str, symbol: str) -> bool:
    """Check if bot can trade this symbol"""

    # Get current position (if exists)
    position_opt = account_mgr.get_position(account_id, symbol)

    if position_opt.has_value():
        position = position_opt.value()

        if position.is_manual_position():
            print(f"⚠ Cannot trade {symbol} - MANUAL position exists")
            print(f"  Bot only trades NEW securities or bot-managed positions")
            return False

    # Either no position, or bot-managed position
    print(f"✓ OK to trade {symbol}")
    return True

# Example usage
if can_trade_symbol("XXXX1234", "AAPL"):
    # Place order
    pass
else:
    # Skip this symbol
    pass
```

### Filter Signals by Position Type
```python
def filter_manual_positions(signals: list, account_id: str) -> list:
    """Remove signals for symbols with manual positions"""

    filtered = []

    for signal in signals:
        if can_trade_symbol(account_id, signal.symbol):
            filtered.append(signal)
        else:
            print(f"Filtered out {signal.symbol} - manual position")

    return filtered

# Example
raw_signals = strategy.generate_signals()
safe_signals = filter_manual_positions(raw_signals, "XXXX1234")

print(f"Signals: {len(raw_signals)} -> {len(safe_signals)} (after filtering)")
```

---

## 8. DuckDB Queries

### Query Positions Directly
```python
import duckdb

conn = duckdb.connect("trading_data.duckdb")

# Get all manual positions
manual = conn.execute("""
    SELECT symbol, quantity, market_value, unrealized_pnl
    FROM positions
    WHERE is_bot_managed = FALSE
    ORDER BY market_value DESC
""").fetchall()

print("Manual Positions (DO NOT TOUCH):")
for symbol, qty, value, pnl in manual:
    print(f"  {symbol}: {qty} shares, ${value:,.2f}, P/L: ${pnl:,.2f}")

# Get all bot positions
bot = conn.execute("""
    SELECT symbol, quantity, bot_strategy, unrealized_pnl
    FROM positions
    WHERE is_bot_managed = TRUE
    ORDER BY unrealized_pnl DESC
""").fetchall()

print("\nBot-Managed Positions:")
for symbol, qty, strategy, pnl in bot:
    print(f"  {symbol}: {qty} shares, Strategy: {strategy}, P/L: ${pnl:,.2f}")
```

### Query Position History
```python
# Get position value over time
history = conn.execute("""
    SELECT
        timestamp,
        SUM(market_value) as total_value,
        SUM(unrealized_pnl) as total_pnl
    FROM position_history
    WHERE symbol = 'XLE'
    GROUP BY timestamp
    ORDER BY timestamp DESC
    LIMIT 100
""").fetchall()

print("XLE Position History (last 100 updates):")
for ts, value, pnl in history:
    print(f"  {ts}: ${value:,.2f} (P/L: ${pnl:,.2f})")
```

### Query Position Changes
```python
# Get recent position changes
changes = conn.execute("""
    SELECT timestamp, symbol, change_type, quantity_change, price_at_change
    FROM position_changes
    ORDER BY timestamp DESC
    LIMIT 20
""").fetchall()

print("Recent Position Changes:")
for ts, symbol, change_type, qty_change, price in changes:
    print(f"  {ts}: {symbol} {change_type} ({qty_change:+d} shares @ ${price:.2f})")
```

---

## 9. Complete Example Script

```python
#!/usr/bin/env python3
"""
Complete example of Schwab account API usage
"""

import bigbrother_schwab_account as schwab
from datetime import datetime, timedelta
import time

def main():
    # 1. Initialize
    print("Initializing Schwab API client...")
    config = schwab.OAuth2Config()
    config.client_id = "YOUR_CLIENT_ID"
    config.client_secret = "YOUR_CLIENT_SECRET"
    config.refresh_token = "YOUR_REFRESH_TOKEN"

    token_mgr = schwab.TokenManager(config)
    account_mgr = schwab.AccountManager(token_mgr, "trading_data.duckdb")

    account_id = "XXXX1234"

    # 2. CRITICAL: Classify existing positions
    print("\n=== CLASSIFYING EXISTING POSITIONS ===")
    account_mgr.classify_existing_positions(account_id)

    # 3. Get account info
    print("\n=== ACCOUNT INFORMATION ===")
    balance = account_mgr.get_balances(account_id)
    print(f"Total Equity: ${balance.total_equity:,.2f}")
    print(f"Buying Power: ${balance.buying_power:,.2f}")

    # 4. Get positions
    print("\n=== CURRENT POSITIONS ===")
    positions = account_mgr.get_positions(account_id)
    for pos in positions:
        managed = "BOT" if pos.is_bot_managed() else "MANUAL"
        print(f"{pos.symbol}: {pos.quantity} shares, "
              f"P/L: ${pos.unrealized_pnl:,.2f}, "
              f"Managed: {managed}")

    # 5. Start position tracker
    print("\n=== STARTING POSITION TRACKER ===")
    tracker = schwab.PositionTracker(account_mgr, "trading_data.duckdb", 30)
    tracker.start(account_id)

    # 6. Let it run for a few cycles
    print("Tracking positions for 2 minutes...")
    time.sleep(120)

    # 7. Get analytics
    print("\n=== PORTFOLIO ANALYTICS ===")
    analyzer = schwab.PortfolioAnalyzer()
    summary = analyzer.analyze_portfolio(positions, balance)
    print(f"Total P/L: ${summary.total_unrealized_pnl:,.2f} "
          f"({summary.total_unrealized_pnl_percent:.2f}%)")

    risk = analyzer.calculate_risk_metrics(positions, balance)
    print(f"Portfolio Heat: {risk.portfolio_heat:.2f}%")
    print(f"VaR (95%): ${risk.value_at_risk_95:,.2f}")

    # 8. Clean up
    tracker.stop()
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
```

---

## 10. Important Notes

### Safety
- **ALWAYS** run `classify_existing_positions()` on startup
- **NEVER** modify positions where `is_bot_managed = False`
- Use `set_read_only_mode(True)` during testing

### Performance
- Position tracker refresh: 30 seconds (adjustable)
- API rate limit: 120 requests/minute (automatically enforced)
- DuckDB: All queries are fast (<1ms for cached data)

### Error Handling
```python
try:
    positions = account_mgr.get_positions(account_id)
except RuntimeError as e:
    print(f"Error fetching positions: {e}")
    # Handle error (retry, log, alert, etc.)
```

---

**For complete documentation, see:**
- `/docs/SCHWAB_ACCOUNT_IMPLEMENTATION.md` (full implementation details)
- `/docs/TRADING_CONSTRAINTS.md` (position classification rules)
- `test_account.py` (comprehensive test suite)
