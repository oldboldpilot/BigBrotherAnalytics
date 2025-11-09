# AccountManager Quick Start Guide

**Quick reference for using the enhanced AccountManager class**

---

## Initialization (Required on Startup)

```cpp
#include <schwab_api>

using namespace bigbrother::schwab;

// 1. Create token manager
OAuth2Config config{
    .client_id = "YOUR_CLIENT_ID",
    .client_secret = "YOUR_CLIENT_SECRET",
    .refresh_token = "YOUR_REFRESH_TOKEN"
};
auto token_mgr = std::make_shared<TokenManager>(config);

// 2. Create account manager
std::string account_id = "XXXX1234";
auto account_mgr = std::make_shared<AccountManager>(token_mgr, account_id);

// 3. Initialize database
auto db_result = account_mgr->initializeDatabase("trading_data.duckdb");
if (!db_result) {
    Logger::error("Failed to init DB: {}", db_result.error());
    return -1;
}

// 4. CRITICAL: Classify existing positions
auto classify_result = account_mgr->classifyExistingPositions();
if (!classify_result) {
    Logger::error("Failed to classify: {}", classify_result.error());
    return -1;
}

// 5. Log results
auto [total, manual, bot] = account_mgr->getPositionStats();
Logger::info("Position Summary:");
Logger::info("  Total: {}", total);
Logger::info("  Manual: {} (DO NOT TOUCH)", manual);
Logger::info("  Bot-managed: {} (can trade)", bot);
```

---

## Before Trading (CRITICAL)

**ALWAYS validate before placing orders:**

```cpp
auto placeOrder(std::string const& symbol, /* ... */) -> Result<OrderConfirmation> {
    // CRITICAL PRE-FLIGHT CHECK
    auto validate = account_mgr->validateCanTrade(symbol);
    if (!validate) {
        // ERROR: Manual position exists - DO NOT TRADE
        Logger::error("Cannot trade {}: {}", symbol, validate.error());
        return std::unexpected(validate.error());
    }

    // OK to proceed with order
    // ...
}
```

---

## In Signal Generation

**Filter out manual positions:**

```cpp
auto generateSignals(/* ... */) -> std::vector<TradingSignal> {
    std::vector<TradingSignal> signals;

    auto candidates = rankSymbols();

    for (auto const& symbol : candidates) {
        // CHECK: Manual position exists?
        if (account_mgr->hasManualPosition(symbol)) {
            Logger::warn("Skipping {} - manual position exists", symbol);
            continue;  // SKIP
        }

        // OK to generate signal
        signals.push_back(createSignal(symbol));
    }

    return signals;
}
```

---

## After Order Fill

**Mark new positions as bot-managed:**

```cpp
auto onOrderFilled(OrderConfirmation const& confirmation) -> void {
    if (confirmation.side == "BUY" && confirmation.status == "FILLED") {
        // Mark as bot-managed
        account_mgr->markPositionAsBotManaged(
            confirmation.symbol,
            "SectorRotation"  // Strategy name
        );

        Logger::info("Bot opened position: {} ({})",
                    confirmation.symbol,
                    "SectorRotation");
    }
}
```

---

## Query Positions

```cpp
// Get all positions
auto all_positions = account_mgr->getPositions();
if (all_positions) {
    for (auto const& pos : *all_positions) {
        Logger::info("{}: {} shares @ ${:.2f}",
                    pos.symbol, pos.quantity, pos.average_price);
    }
}

// Get manual positions only (DO NOT TOUCH)
auto manual = account_mgr->getManualPositions();

// Get bot-managed positions only (can trade)
auto bot = account_mgr->getBotManagedPositions();

// Get specific position
auto position = account_mgr->getPosition("AAPL");
if (position && position->has_value()) {
    auto const& pos = position->value();
    Logger::info("AAPL: {} shares", pos.quantity);
}

// Check if bot-managed
if (account_mgr->isSymbolBotManaged("XLE")) {
    Logger::info("XLE is bot-managed");
}

// Check if manual
if (account_mgr->hasManualPosition("MSFT")) {
    Logger::warn("MSFT is manual - DO NOT TRADE");
}
```

---

## Statistics

```cpp
auto [total, manual, bot] = account_mgr->getPositionStats();

std::cout << "Position Statistics:\n";
std::cout << "  Total positions: " << total << "\n";
std::cout << "  Manual (DO NOT TOUCH): " << manual << "\n";
std::cout << "  Bot-managed (can trade): " << bot << "\n";
```

---

## Error Handling

```cpp
// All methods return Result<T>
auto result = account_mgr->getPositions();
if (!result) {
    // Error occurred
    Logger::error("Failed: {}", result.error());
    return;
}

// Success - use result
auto const& positions = *result;
```

---

## Common Patterns

### Strategy Integration

```cpp
class MyStrategy {
public:
    explicit MyStrategy(std::shared_ptr<AccountManager> account_mgr)
        : account_mgr_{std::move(account_mgr)} {}

    auto generateSignals() -> std::vector<TradingSignal> {
        std::vector<TradingSignal> signals;

        auto candidates = analyzeMarket();

        for (auto const& symbol : candidates) {
            // FILTER: Check if we can trade this symbol
            if (account_mgr_->hasManualPosition(symbol)) {
                continue;  // Skip manual positions
            }

            signals.push_back(createSignal(symbol));
        }

        return signals;
    }

private:
    std::shared_ptr<AccountManager> account_mgr_;
};
```

### Order Placement Integration

```cpp
class OrderExecutor {
public:
    explicit OrderExecutor(
        std::shared_ptr<AccountManager> account_mgr,
        std::shared_ptr<SchwabClient> schwab_client
    ) : account_mgr_{std::move(account_mgr)},
        schwab_client_{std::move(schwab_client)} {}

    auto executeOrder(Order const& order) -> Result<OrderConfirmation> {
        // VALIDATE
        auto validate = account_mgr_->validateCanTrade(order.symbol);
        if (!validate) {
            return std::unexpected(validate.error());
        }

        // PLACE ORDER
        auto result = schwab_client_->placeOrder(order);
        if (!result) {
            return std::unexpected(result.error());
        }

        // MARK AS BOT-MANAGED
        if (result->status == "FILLED" && order.side == "BUY") {
            account_mgr_->markPositionAsBotManaged(
                order.symbol,
                order.strategy_name
            );
        }

        return result;
    }

private:
    std::shared_ptr<AccountManager> account_mgr_;
    std::shared_ptr<SchwabClient> schwab_client_;
};
```

---

## Key Rules (TRADING_CONSTRAINTS.md)

### ✅ Bot CAN:
- Trade NEW securities (not in portfolio)
- Close positions IT created (bot-managed)
- Modify positions IT created

### ❌ Bot CANNOT:
- Trade existing securities (manual positions)
- Close manual positions
- Modify manual positions
- Add to manual positions

### 🔍 ALWAYS Check:
```cpp
// Before EVERY trade
if (account_mgr->hasManualPosition(symbol)) {
    // ABORT - manual position exists
    return error;
}

// Or use validation
auto validate = account_mgr->validateCanTrade(symbol);
if (!validate) {
    return validate.error();
}
```

---

## Troubleshooting

### "Cannot trade X - manual position exists"

**Cause:** Symbol has pre-existing manual position

**Solution:** Bot should skip this symbol. Only human can trade manual positions.

```cpp
if (account_mgr->hasManualPosition(symbol)) {
    Logger::warn("Skipping {} - manual position", symbol);
    continue;  // Skip to next symbol
}
```

### "Database not initialized"

**Cause:** `initializeDatabase()` not called

**Solution:**
```cpp
account_mgr->initializeDatabase("trading_data.duckdb");
```

### Position not classified correctly

**Cause:** `classifyExistingPositions()` not called on startup

**Solution:**
```cpp
// MUST call on startup
account_mgr->classifyExistingPositions();
```

---

## Thread Safety

All methods are thread-safe:

```cpp
// Safe to call from multiple threads
std::thread t1([&]() {
    auto positions = account_mgr->getPositions();
});

std::thread t2([&]() {
    bool is_bot = account_mgr->isSymbolBotManaged("XLE");
});

t1.join();
t2.join();
```

---

## Complete Example

```cpp
#include <schwab_api>
#include <memory>

using namespace bigbrother::schwab;

auto main() -> int {
    // 1. Setup
    auto token_mgr = std::make_shared<TokenManager>(config);
    auto account_mgr = std::make_shared<AccountManager>(token_mgr, "XXXX1234");

    // 2. Initialize
    account_mgr->initializeDatabase("trading.duckdb");
    account_mgr->classifyExistingPositions();

    // 3. Check positions
    auto [total, manual, bot] = account_mgr->getPositionStats();
    Logger::info("Manual: {}, Bot: {}", manual, bot);

    // 4. Generate signals
    std::vector<TradingSignal> signals;
    auto candidates = {"AAPL", "XLE", "SPY"};

    for (auto const& symbol : candidates) {
        // FILTER manual positions
        if (account_mgr->hasManualPosition(symbol)) {
            Logger::warn("Skipping {} - manual", symbol);
            continue;
        }

        // OK to trade
        signals.push_back(createBuySignal(symbol));
    }

    // 5. Execute orders
    for (auto const& signal : signals) {
        // VALIDATE
        auto validate = account_mgr->validateCanTrade(signal.symbol);
        if (!validate) {
            Logger::error("Cannot trade {}: {}", signal.symbol, validate.error());
            continue;
        }

        // PLACE ORDER
        auto result = placeOrder(signal);
        if (result && result->status == "FILLED") {
            // MARK as bot-managed
            account_mgr->markPositionAsBotManaged(
                signal.symbol,
                "MyStrategy"
            );
        }
    }

    return 0;
}
```

---

## See Also

- **`docs/ACCOUNT_MANAGER_IMPLEMENTATION.md`** - Full implementation details
- **`docs/TRADING_CONSTRAINTS.md`** - Position classification rules
- **`src/schwab_api/account_types.hpp`** - Data structures
- **`src/schwab_api/account_manager.hpp`** - Full interface

---

**Last Updated:** 2025-11-09
**Status:** Production-ready API
