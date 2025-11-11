# Trading Platform Architecture - Loose Coupling Design

**Author**: Olumuyiwa Oluwasanmi
**Date**: 2025-11-11
**Phase**: 4 (Week 2 - Architecture Refactoring)

## Overview

This document describes the loose coupling architecture implemented for the trading system, enabling support for multiple trading platforms (Schwab, Interactive Brokers, TD Ameritrade, etc.) without modifying core trading logic.

## Architecture Principles

### 1. Dependency Inversion Principle (DIP)

The system follows the **Dependency Inversion Principle** from SOLID principles:

- **High-level modules** (OrdersManager) depend on **abstractions** (TradingPlatformInterface)
- **Low-level modules** (SchwabOrderExecutor, IBKRExecutor) implement the abstraction
- **Zero coupling** between OrdersManager and platform-specific implementations

### 2. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Layer 1: Platform-Agnostic Types              │
│                  (order_types.cppm)                     │
│  • Position, Order, OrderSide, OrderType, OrderStatus   │
│  • Common vocabulary for all trading platforms          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│        Layer 2: Abstract Interface (DIP)                │
│            (platform_interface.cppm)                    │
│  • TradingPlatformInterface (pure virtual)              │
│  • submitOrder(), cancelOrder(), getOrders()            │
│  • getPositions(), getPlatformName()                    │
└────────────────────────┬────────────────────────────────┘
                         │ depends on
┌────────────────────────▼────────────────────────────────┐
│      Layer 3: Platform-Agnostic Business Logic          │
│               (orders_manager.cppm)                     │
│  • OrdersManager (uses TradingPlatformInterface)        │
│  • Risk management, position tracking, order logging    │
│  • NEVER knows about concrete platform types            │
└─────────────────────────────────────────────────────────┘
                         ▲ injected at runtime
                         │
┌────────────────────────┴────────────────────────────────┐
│    Layer 4: Platform-Specific Implementations           │
│  ┌────────────────────┐  ┌────────────────────┐         │
│  │ SchwabOrderExecutor│  │  IBKROrderExecutor │  ...    │
│  │   (Adapter)        │  │     (Adapter)      │         │
│  │ • Converts types   │  │  • Converts types  │         │
│  │ • Delegates to API │  │  • Delegates to API│         │
│  └────────────────────┘  └────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Trading Library (`trading_core`)

**Location**: `src/core/trading/`

**Purpose**: Platform-agnostic trading infrastructure

**Modules**:
1. **order_types.cppm** - Common data types
   - `Position`, `Order`, `OrderSide`, `OrderType`, `OrderStatus`, `OrderDuration`
   - `OrderConfirmation`, `BracketOrder`
   - Safety flags for manual position protection

2. **platform_interface.cppm** - Abstract interface
   - `TradingPlatformInterface` base class
   - Pure virtual methods for order execution and position queries
   - Factory pattern support for dynamic platform selection

3. **orders_manager.cppm** - Generic order manager
   - `OrdersManager` - platform-agnostic order management
   - `PositionDatabase` - position tracking with DuckDB
   - `OrderDatabaseLogger` - compliance logging
   - Takes `std::unique_ptr<TradingPlatformInterface>` via constructor

### Platform-Specific Implementations

**Location**: `src/schwab_api/` (and similar for other platforms)

**Schwab Executor**:
- **schwab_order_executor.cppm**
  - `SchwabOrderExecutor : public TradingPlatformInterface`
  - Type conversion between `trading::Order` ↔ `schwab::Order`
  - Adapter pattern to Schwab OrderManager

## Key Design Patterns

### 1. Adapter Pattern

**Purpose**: Translate generic operations to platform-specific API calls

**Example** (SchwabOrderExecutor):

```cpp
class SchwabOrderExecutor : public TradingPlatformInterface {
public:
    // Type aliases for disambiguation
    using Order = trading::Order;
    using OrderStatus = trading::OrderStatus;

    [[nodiscard]] auto submitOrder(Order const& order)
        -> Result<std::string> override {

        // Convert generic Order to Schwab Order format
        auto schwab_order = convertToSchwabOrder(order);

        // Delegate to Schwab API
        auto result = schwab_order_manager_->placeOrder(schwab_order);

        return result;
    }

private:
    auto convertToSchwabOrder(trading::Order const& generic_order) const
        -> ::bigbrother::schwab::Order {
        // Type conversion logic
    }

    std::shared_ptr<OrderManager> schwab_order_manager_;
    std::string account_id_;
};
```

### 2. Dependency Injection

**Purpose**: Inject platform executor at runtime

**Example** (OrdersManager constructor):

```cpp
class OrdersManager {
public:
    explicit OrdersManager(
        std::string db_path,
        std::unique_ptr<TradingPlatformInterface> platform,  // Injected
        bool enable_dry_run = true)
        : position_db_{db_path},
          order_logger_{db_path},
          platform_{std::move(platform)},  // Store abstraction
          dry_run_mode_{enable_dry_run} {}

    [[nodiscard]] auto placeOrder(Order order) -> Result<OrderConfirmation> {
        // ... validation and safety checks ...

        // Submit via injected platform (could be Schwab, IBKR, etc.)
        auto submit_result = platform_->submitOrder(order);

        return confirmation;
    }

private:
    std::unique_ptr<TradingPlatformInterface> platform_;  // Abstraction only
};
```

### 3. Factory Pattern (Optional)

**Purpose**: Dynamic platform selection based on configuration

```cpp
class PlatformFactory {
public:
    [[nodiscard]] virtual auto createPlatform(
        PlatformType type,
        std::string const& config)
        -> Result<std::unique_ptr<TradingPlatformInterface>> = 0;
};
```

## Type Conversion Strategy

### Challenge: Different Order Structures

**Generic Order** (`trading::Order`):
- 16 fields: `order_id`, `account_id`, `symbol`, `side`, `quantity`, `filled_quantity`,
  `type`, `duration`, `limit_price`, `stop_price`, `trail_amount`, `avg_fill_price`,
  `status`, `strategy_name`, `dry_run`, `rejection_reason`, timestamps

**Schwab Order** (`schwab::Order`):
- 10 fields: `order_id`, `symbol`, `type`, `duration`, `quantity`, `limit_price`,
  `stop_price`, `status`, `created_at`, `updated_at`

### Solution: Adapter Type Conversion

1. **Generic → Platform**: Copy common fields, discard unsupported fields
2. **Platform → Generic**: Copy common fields, set defaults for missing fields
3. **Timestamp Conversion**: `std::chrono::time_point` ↔ `int64_t` milliseconds
4. **Enum Conversion**: Cast between namespaced enums with same underlying values

**Example**:

```cpp
auto convertToSchwabOrder(trading::Order const& generic_order) const
    -> ::bigbrother::schwab::Order {

    ::bigbrother::schwab::Order schwab_order;

    // Copy common fields
    schwab_order.order_id = generic_order.order_id;
    schwab_order.symbol = generic_order.symbol;
    schwab_order.quantity = generic_order.quantity;

    // Convert timestamps
    schwab_order.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        generic_order.created_at.time_since_epoch()).count();

    // Convert enums
    schwab_order.type = static_cast<::bigbrother::schwab::OrderType>(generic_order.type);

    return schwab_order;
}
```

## Adding a New Trading Platform

### Step-by-Step Guide

#### 1. Create Platform Executor

**File**: `src/ibkr_api/ibkr_order_executor.cppm`

```cpp
export module bigbrother.ibkr_api.order_executor;

import bigbrother.utils.types;
import bigbrother.trading.order_types;
import bigbrother.trading.platform_interface;
import bigbrother.ibkr_api;  // Platform-specific API

export namespace bigbrother::ibkr {

class IBKROrderExecutor : public trading::TradingPlatformInterface {
public:
    // Type aliases
    using Order = trading::Order;
    using OrderStatus = trading::OrderStatus;

    explicit IBKROrderExecutor(
        std::shared_ptr<IBKRClient> ibkr_client,
        std::string account_id);

    // Implement TradingPlatformInterface methods
    [[nodiscard]] auto submitOrder(Order const& order)
        -> Result<std::string> override;

    [[nodiscard]] auto cancelOrder(
        std::string const& account_id,
        std::string const& order_id)
        -> Result<void> override;

    // ... other interface methods ...

    [[nodiscard]] auto getPlatformName() const noexcept
        -> std::string override {
        return "Interactive Brokers";
    }

private:
    auto convertToIBKROrder(trading::Order const& generic_order) const
        -> ::bigbrother::ibkr::Order;

    auto convertFromIBKROrder(::bigbrother::ibkr::Order const& ibkr_order) const
        -> trading::Order;

    std::shared_ptr<IBKRClient> ibkr_client_;
    std::string account_id_;
};

}  // namespace bigbrother::ibkr
```

#### 2. Update CMakeLists.txt

Add IBKR library and link dependencies:

```cmake
# IBKR API library
add_library(ibkr_api SHARED)

target_sources(ibkr_api
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/ibkr_api/ibkr_client.cppm
            src/ibkr_api/ibkr_order_executor.cppm
)

target_link_libraries(ibkr_api
    PUBLIC
    utils
    trading_core  # Platform-agnostic order management
    IBKR::API     # IBKR-specific dependencies
)
```

#### 3. Inject Platform at Runtime

```cpp
// Create IBKR executor
auto ibkr_executor = std::make_unique<ibkr::IBKROrderExecutor>(
    ibkr_client,
    account_id
);

// Inject into OrdersManager
auto orders_manager = trading::OrdersManager(
    db_path,
    std::move(ibkr_executor),  // Platform-agnostic!
    enable_dry_run
);

// OrdersManager works identically regardless of platform
orders_manager.placeOrder(order);
```

#### 4. Zero Changes to OrdersManager

**Critical**: Adding IBKR requires ZERO modifications to `orders_manager.cppm`. The same OrdersManager code works with Schwab, IBKR, TD Ameritrade, Alpaca, or any other platform.

## Build Configuration

### CMake Targets

**trading_core**:
```cmake
add_library(trading_core SHARED)
target_sources(trading_core
    PUBLIC FILE_SET CXX_MODULES FILES
        src/core/trading/order_types.cppm
        src/core/trading/platform_interface.cppm
        src/core/trading/orders_manager.cppm
)
target_link_libraries(trading_core PUBLIC utils duckdb_bridge)
```

**schwab_api** (with executor):
```cmake
add_library(schwab_api SHARED)
target_sources(schwab_api
    PUBLIC FILE_SET CXX_MODULES FILES
        src/schwab_api/schwab_api.cppm
        src/schwab_api/schwab_order_executor.cppm  # Adapter
)
target_link_libraries(schwab_api PUBLIC trading_core CURL::libcurl)
```

## Benefits of This Architecture

### 1. Multi-Platform Support
- Add new brokers without modifying existing code
- Each platform is an independent module
- Mix and match platforms at runtime

### 2. Testability
- Mock platforms for unit testing
- Test OrdersManager in isolation
- Platform-specific tests remain separate

### 3. Maintainability
- Changes to Schwab API don't affect IBKR
- Clear separation of concerns
- Single Responsibility Principle

### 4. Extensibility
- New platforms implement interface
- Zero impact on existing platforms
- Factory pattern for dynamic selection

### 5. Safety
- Compile-time enforcement of interface
- Type-safe abstractions
- Clear contract between layers

## Usage Example

### Complete Workflow

```cpp
#include <memory>
import bigbrother.schwab_api.order_executor;
import bigbrother.trading.orders_manager;
import bigbrother.trading.order_types;

// 1. Create platform executor (Schwab)
auto schwab_executor = std::make_unique<schwab::SchwabOrderExecutor>(
    schwab_order_manager,
    account_id
);

// 2. Inject into OrdersManager
auto orders_manager = trading::OrdersManager(
    "trading.db",
    std::move(schwab_executor),  // Dependency injection
    true  // dry-run mode
);

// 3. Create order (platform-agnostic)
trading::Order order{
    .symbol = "AAPL",
    .side = trading::OrderSide::Buy,
    .quantity = 100,
    .type = trading::OrderType::Limit,
    .limit_price = 150.00,
    .duration = trading::OrderDuration::Day
};

// 4. Place order (works with any platform)
auto result = orders_manager.placeOrder(order);

if (result) {
    std::cout << "Order placed: " << result->order_id << std::endl;
} else {
    std::cerr << "Order failed: " << result.error().message << std::endl;
}
```

### Switching Platforms

To switch from Schwab to IBKR, change **only** step 1:

```cpp
// 1. Create platform executor (IBKR instead of Schwab)
auto ibkr_executor = std::make_unique<ibkr::IBKROrderExecutor>(
    ibkr_client,
    account_id
);

// 2-4. Identical to above!
auto orders_manager = trading::OrdersManager(
    "trading.db",
    std::move(ibkr_executor),  // Different platform, same interface
    true
);
```

## Testing Strategy

### Unit Tests

1. **Platform-Agnostic Tests** (OrdersManager)
   - Use mock TradingPlatformInterface
   - Test business logic in isolation
   - No platform dependencies

2. **Platform-Specific Tests** (Executors)
   - Test Schwab type conversions
   - Test IBKR API integration
   - Mock underlying API clients

3. **Integration Tests**
   - End-to-end with real platforms (sandbox)
   - Multi-platform scenarios
   - Failover testing

### Mock Platform Example

```cpp
class MockPlatform : public TradingPlatformInterface {
public:
    [[nodiscard]] auto submitOrder(Order const& order)
        -> Result<std::string> override {
        submitted_orders.push_back(order);
        return "MOCK-" + std::to_string(order_counter++);
    }

    std::vector<Order> submitted_orders;
    int order_counter = 0;
};

// Test OrdersManager with mock
TEST(OrdersManagerTest, PlaceOrderSucceeds) {
    auto mock_platform = std::make_unique<MockPlatform>();
    auto* mock_ptr = mock_platform.get();

    OrdersManager manager("test.db", std::move(mock_platform), true);

    Order order{/* ... */};
    auto result = manager.placeOrder(order);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(mock_ptr->submitted_orders.size(), 1);
}
```

## Future Enhancements

### 1. Platform Selection via Configuration

```yaml
trading:
  platform: "schwab"  # or "ibkr", "td_ameritrade", etc.
  account_id: "12345678"
  dry_run: true
```

### 2. Multi-Platform Support

Route orders to different platforms based on symbol, strategy, or load balancing:

```cpp
class MultiPlatformRouter : public TradingPlatformInterface {
public:
    auto submitOrder(Order const& order) -> Result<std::string> override {
        // Route to best platform based on symbol/strategy
        if (order.symbol.starts_with("SPX")) {
            return schwab_executor_->submitOrder(order);
        } else {
            return ibkr_executor_->submitOrder(order);
        }
    }
};
```

### 3. Platform Health Monitoring

```cpp
class HealthMonitoredPlatform : public TradingPlatformInterface {
public:
    auto submitOrder(Order const& order) -> Result<std::string> override {
        if (!health_monitor_.isHealthy()) {
            return failover_platform_->submitOrder(order);
        }
        return primary_platform_->submitOrder(order);
    }
};
```

## Conclusion

This loose coupling architecture provides a robust, extensible foundation for multi-platform trading. By following the Dependency Inversion Principle and Adapter Pattern, we achieve:

- **Zero coupling** between business logic and platform APIs
- **Easy addition** of new trading platforms
- **Testable** components at every layer
- **Maintainable** codebase with clear separation of concerns

The architecture is production-ready and follows C++23 best practices with modern module system integration.

---

**References**:
- [src/core/trading/order_types.cppm](../src/core/trading/order_types.cppm)
- [src/core/trading/platform_interface.cppm](../src/core/trading/platform_interface.cppm)
- [src/core/trading/orders_manager.cppm](../src/core/trading/orders_manager.cppm)
- [src/schwab_api/schwab_order_executor.cppm](../src/schwab_api/schwab_order_executor.cppm)
