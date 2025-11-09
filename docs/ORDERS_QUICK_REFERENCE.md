# Schwab Orders - Quick Reference Guide

**Quick access to common operations and safety checks**

---

## 🚀 Quick Start

```cpp
#include <bigbrother.schwab_api.orders>

// 1. Initialize (dry-run mode)
OrdersManager orders{"data/trading.duckdb", true};

// 2. Classify existing positions (CRITICAL)
orders.classifyExistingPositions(account_id, schwab_positions);

// 3. Check summary
auto summary = orders.getPositionSummary(account_id);
```

---

## 🔐 Safety Checks

### Check Before Trading
```cpp
// Query position
auto pos = orders.getPosition(account_id, "AAPL");

// Check if bot can trade
if (pos && !pos->is_bot_managed) {
    // STOP! Manual position - DO NOT TRADE
}
```

### Safety Flags
```cpp
pos->is_bot_managed   // true = bot can trade, false = hands off
pos->managed_by       // "BOT" or "MANUAL"
pos->bot_strategy     // Strategy name (if bot-managed)
```

---

## 📈 Place Orders

### Market Order
```cpp
Order order;
order.symbol = "SPY";
order.side = OrderSide::Buy;
order.quantity = 10;
order.type = OrderType::Market;

auto result = orders.placeOrder(order);
```

### Limit Order
```cpp
Order order;
order.symbol = "SPY";
order.side = OrderSide::Buy;
order.quantity = 10;
order.type = OrderType::Limit;
order.limit_price = 580.00;

auto result = orders.placeOrder(order);
```

### Stop Order
```cpp
Order order;
order.symbol = "SPY";
order.side = OrderSide::Sell;
order.quantity = 10;
order.type = OrderType::Stop;
order.stop_price = 570.00;

auto result = orders.placeOrder(order);
```

### Bracket Order
```cpp
Order entry;
entry.symbol = "SPY";
entry.side = OrderSide::Buy;
entry.quantity = 10;
entry.type = OrderType::Limit;
entry.limit_price = 580.00;

BracketOrder bracket;
bracket.entry_order = entry;
bracket.profit_target = 590.00;  // +$10
bracket.stop_loss = 575.00;      // -$5

auto result = orders.placeBracketOrder(bracket);
```

---

## 🔧 Order Management

### Cancel Order
```cpp
auto result = orders.cancelOrder(account_id, order_id);
```

### Modify Order
```cpp
Order modifications;
modifications.limit_price = 582.00;

auto result = orders.modifyOrder(order_id, modifications);
```

### Get Order Status
```cpp
auto status = orders.getOrderStatus(order_id);
```

### Get All Orders
```cpp
auto all_orders = orders.getOrders(account_id);
```

---

## 📊 Position Management

### Get All Positions
```cpp
auto positions = orders.getPositions(account_id);

for (auto const& pos : *positions) {
    std::cout << pos.symbol << ": "
              << (pos.is_bot_managed ? "BOT" : "MANUAL")
              << "\n";
}
```

### Get Single Position
```cpp
auto pos = orders.getPosition(account_id, "SPY");

if (pos) {
    std::cout << "Symbol: " << pos->symbol << "\n";
    std::cout << "Quantity: " << pos->quantity << "\n";
    std::cout << "Bot-managed: " << pos->is_bot_managed << "\n";
}
```

### Get Position Summary
```cpp
auto summary = orders.getPositionSummary(account_id);

std::cout << "Manual positions: "
          << (*summary)["manual_positions"] << "\n";
std::cout << "Bot positions: "
          << (*summary)["bot_managed_positions"] << "\n";
```

### Close Position
```cpp
// Only works if bot-managed!
auto result = orders.closePosition(account_id, "XLE");

if (!result) {
    // REJECTED if manual position
    std::cerr << result.error().message << "\n";
}
```

---

## 🧪 Testing

### Enable Dry-Run
```cpp
orders.setDryRunMode(true);  // Safe testing

// All orders will be simulated
auto result = orders.placeOrder(order);
if (result->dry_run) {
    std::cout << "DRY-RUN: No real order placed\n";
}
```

### Disable Dry-Run (LIVE)
```cpp
orders.setDryRunMode(false);  // ⚠️ LIVE TRADING!
```

### Check Mode
```cpp
if (orders.isDryRunMode()) {
    std::cout << "Safe: Dry-run mode enabled\n";
} else {
    std::cout << "WARNING: Live trading enabled!\n";
}
```

---

## ❌ Error Handling

### Check Result
```cpp
auto result = orders.placeOrder(order);

if (!result) {
    // Error occurred
    std::cerr << "Error: " << result.error().message << "\n";

    // Check error type
    if (result.error().code == ErrorCode::InvalidOperation) {
        // Safety violation (manual position)
    }
}
```

### Common Errors
```cpp
ErrorCode::InvalidOperation    // Safety violation
ErrorCode::InvalidParameter    // Bad order parameters
ErrorCode::NotFound           // Position/order not found
ErrorCode::DatabaseError      // Database operation failed
```

---

## 📋 Database Queries

### Query Orders
```sql
-- Today's orders
SELECT * FROM todays_orders;

-- Active orders
SELECT * FROM active_orders;

-- Order history
SELECT * FROM orders WHERE symbol = 'SPY';

-- Audit trail
SELECT * FROM order_updates WHERE order_id = 'ORD_123';

-- Performance
SELECT * FROM order_performance WHERE symbol = 'SPY';
```

### Query Positions
```sql
-- All positions
SELECT * FROM positions WHERE account_id = 'ACC_123';

-- Manual positions (HANDS OFF)
SELECT * FROM positions WHERE is_bot_managed = FALSE;

-- Bot-managed positions (can trade)
SELECT * FROM positions WHERE is_bot_managed = TRUE;

-- Position summary
SELECT
    managed_by,
    COUNT(*) as count,
    SUM(market_value) as total_value
FROM positions
GROUP BY managed_by;
```

---

## ⚠️ Safety Reminders

### DO
✅ ALWAYS classify positions on startup
✅ ALWAYS check `is_bot_managed` before trading
✅ ALWAYS start in dry-run mode
✅ ALWAYS log orders for compliance
✅ ALWAYS handle errors appropriately

### DON'T
❌ NEVER bypass safety checks
❌ NEVER trade manual positions
❌ NEVER skip position classification
❌ NEVER disable dry-run without testing
❌ NEVER ignore error messages

---

## 📞 Quick Help

### Position is MANUAL - What to do?
```
Error: "Cannot trade AAPL - manual position exists"

Solution:
1. This is EXPECTED behavior (safety feature)
2. Bot CANNOT trade existing holdings
3. Choose a DIFFERENT symbol (one not in portfolio)
4. OR wait for position to be closed manually
```

### Order Rejected - What to do?
```
1. Check error message
2. Verify order parameters
3. Check position safety flags
4. Review dry-run mode setting
5. Check compliance logs
```

### Database Error - What to do?
```
1. Check database file exists
2. Verify schema is loaded
3. Check file permissions
4. Review database logs
5. Run test suite to validate
```

---

## 🔗 Additional Resources

- **Full Documentation:** `docs/SCHWAB_ORDERS_COMPLETE.md`
- **Trading Constraints:** `docs/TRADING_CONSTRAINTS.md`
- **Database Schema:** `scripts/database_schema_orders.sql`
- **Python Tests:** `test_orders.py`
- **C++ Tests:** `tests/test_orders_safety.cpp`

---

**Remember:** Safety first! When in doubt, use dry-run mode.
