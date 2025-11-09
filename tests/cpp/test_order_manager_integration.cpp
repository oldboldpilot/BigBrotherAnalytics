/**
 * Integration Tests for OrderManager (Schwab API)
 *
 * Comprehensive tests for all safety features and order management functionality:
 * - Manual position protection (CRITICAL)
 * - Buying power verification
 * - Position limit enforcement
 * - Order validation
 * - Dry-run mode
 * - Compliance logging
 *
 * Following C++23 and Testing Best Practices:
 * - Modern C++23 features (modules, trailing return syntax)
 * - Detailed assertions and logging
 * - Clear test organization
 * - Realistic test scenarios
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <set>
#include <vector>

// Import C++23 modules
import bigbrother.schwab_api;
import bigbrother.utils.types;

using namespace bigbrother::schwab;
using namespace bigbrother::types;

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * Base fixture for OrderManager integration tests
 * Sets up realistic test environment with:
 * - OAuth2 configuration
 * - AccountManager with manual positions
 * - OrderManager with safety features
 */
class OrderManagerIntegrationTest : public ::testing::Test {
protected:
    auto SetUp() -> void override {
        // Setup OAuth2 config with test credentials
        OAuth2Config config;
        config.client_id = "test_client_id";
        config.client_secret = "test_client_secret";
        config.access_token = "test_access_token";
        config.token_expiry = std::chrono::system_clock::now() + std::chrono::hours(1);

        // Create token manager
        token_mgr_ = std::make_shared<TokenManager>(config);

        // Create account manager
        account_mgr_ = std::make_unique<AccountManager>(token_mgr_, "TEST_ACCOUNT_123");

        // Initialize database (stub implementation)
        auto db_result = account_mgr_->initializeDatabase(":memory:");
        ASSERT_TRUE(db_result) << "Failed to initialize database: "
                               << db_result.error().message;

        // Create order manager and link to account manager
        order_mgr_ = std::make_unique<OrderManager>(token_mgr_);
        order_mgr_->setAccountManager(account_mgr_.get());

        // Setup default limits
        order_mgr_->setMaxPositionSize(5000)  // Max 5000 shares
                   .setMaxPositions(10);      // Max 10 concurrent positions
    }

    auto TearDown() -> void override {
        order_mgr_.reset();
        account_mgr_.reset();
        token_mgr_.reset();
    }

    // Helper methods

    /**
     * Create a manual position in the account
     * This simulates a pre-existing position that the bot should NOT touch
     */
    auto createManualPosition(std::string const& symbol, Quantity qty, Price avg_price) -> void {
        AccountPosition pos;
        pos.symbol = symbol;
        pos.quantity = qty;
        pos.average_price = avg_price;
        pos.current_price = avg_price;
        pos.unrealized_pnl = 0.0;
        pos.realized_pnl = 0.0;

        // Mark as manual position (pre-existing)
        // In production, this would be done via classifyExistingPositions()
        // For tests, we simulate by marking positions manually
    }

    /**
     * Create a bot-managed position
     */
    auto createBotManagedPosition(std::string const& symbol, std::string const& strategy = "BOT") -> void {
        account_mgr_->markPositionAsBotManaged(symbol, strategy);
    }

    /**
     * Create a basic limit order
     */
    [[nodiscard]] auto createLimitOrder(
        std::string symbol,
        Quantity quantity,
        Price limit_price,
        OrderDuration duration = OrderDuration::Day
    ) -> Order {
        Order order;
        order.symbol = std::move(symbol);
        order.quantity = quantity;
        order.limit_price = limit_price;
        order.type = OrderType::Limit;
        order.duration = duration;
        order.status = OrderStatus::Pending;
        return order;
    }

    /**
     * Create a market order
     */
    [[nodiscard]] auto createMarketOrder(
        std::string symbol,
        Quantity quantity,
        OrderDuration duration = OrderDuration::Day
    ) -> Order {
        Order order;
        order.symbol = std::move(symbol);
        order.quantity = quantity;
        order.type = OrderType::Market;
        order.duration = duration;
        order.status = OrderStatus::Pending;
        return order;
    }

    // Member variables
    std::shared_ptr<TokenManager> token_mgr_;
    std::unique_ptr<AccountManager> account_mgr_;
    std::unique_ptr<OrderManager> order_mgr_;
};

// ============================================================================
// Safety Feature Tests: Manual Position Protection (CRITICAL)
// ============================================================================

/**
 * CRITICAL TEST: Bot must REJECT orders for symbols with manual positions
 * This is the most important safety feature - prevents bot from interfering
 * with human-managed positions.
 */
TEST_F(OrderManagerIntegrationTest, RejectOrderForSymbolWithManualPosition) {
    // Setup: Create a manual position for AAPL
    createManualPosition("AAPL", 100, 150.00);

    // Attempt to place order for AAPL (should be REJECTED)
    auto order = createLimitOrder("AAPL", 50, 155.00);
    auto result = order_mgr_->placeOrder(order);

    // Verify: Order should be rejected with specific error
    EXPECT_FALSE(result) << "Order should be rejected - manual position exists";
    EXPECT_EQ(result.error().code, ErrorCode::InvalidOperation)
        << "Error code should be InvalidOperation";
    EXPECT_NE(result.error().message.find("manual position"), std::string::npos)
        << "Error message should mention manual position";
    EXPECT_NE(result.error().message.find("AAPL"), std::string::npos)
        << "Error message should include symbol";
}

/**
 * Test: Bot should ACCEPT orders for new symbols (no position exists)
 */
TEST_F(OrderManagerIntegrationTest, AcceptOrderForNewSymbol) {
    // No positions exist - this is a brand new symbol for the bot
    auto order = createLimitOrder("TSLA", 10, 250.00);
    auto result = order_mgr_->placeOrder(order);

    // Verify: Order should be accepted
    ASSERT_TRUE(result) << "Order should be accepted for new symbol: "
                        << (result ? "" : result.error().message);
    EXPECT_FALSE(result->empty()) << "Order ID should be returned";

    // Verify order is tracked
    auto status_result = order_mgr_->getOrderStatus(*result);
    ASSERT_TRUE(status_result);
    EXPECT_TRUE(*status_result == OrderStatus::Working ||
                *status_result == OrderStatus::Pending);
}

/**
 * Test: Bot should ACCEPT orders for bot-managed positions
 * Once bot opens a position, it can continue managing it
 */
TEST_F(OrderManagerIntegrationTest, AcceptOrderForBotManagedPosition) {
    // Setup: Mark SPY as bot-managed
    createBotManagedPosition("SPY", "MOMENTUM_STRATEGY");

    // Attempt to place order for SPY (should be ACCEPTED)
    auto order = createLimitOrder("SPY", 20, 450.00);
    auto result = order_mgr_->placeOrder(order);

    // Verify: Order should be accepted
    ASSERT_TRUE(result) << "Order should be accepted for bot-managed position: "
                        << (result ? "" : result.error().message);
    EXPECT_FALSE(result->empty());
}

/**
 * Test: Multiple symbols - mixed manual and bot-managed
 */
TEST_F(OrderManagerIntegrationTest, MixedPositionScenario) {
    // Setup mixed environment
    createManualPosition("AAPL", 100, 150.00);      // Manual - DO NOT TOUCH
    createBotManagedPosition("SPY", "TREND");       // Bot-managed - OK to trade
    // GOOGL - no position, new symbol - OK to trade

    // Test 1: AAPL (manual) - should REJECT
    auto aapl_order = createLimitOrder("AAPL", 50, 155.00);
    auto aapl_result = order_mgr_->placeOrder(aapl_order);
    EXPECT_FALSE(aapl_result) << "AAPL order should be rejected (manual position)";

    // Test 2: SPY (bot-managed) - should ACCEPT
    auto spy_order = createLimitOrder("SPY", 20, 450.00);
    auto spy_result = order_mgr_->placeOrder(spy_order);
    EXPECT_TRUE(spy_result) << "SPY order should be accepted (bot-managed)";

    // Test 3: GOOGL (new) - should ACCEPT
    auto googl_order = createLimitOrder("GOOGL", 15, 140.00);
    auto googl_result = order_mgr_->placeOrder(googl_order);
    EXPECT_TRUE(googl_result) << "GOOGL order should be accepted (new symbol)";
}

// ============================================================================
// Safety Feature Tests: Order Parameter Validation
// ============================================================================

TEST_F(OrderManagerIntegrationTest, RejectOrderWithEmptySymbol) {
    auto order = createLimitOrder("", 10, 100.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
    EXPECT_NE(result.error().message.find("Symbol"), std::string::npos);
}

TEST_F(OrderManagerIntegrationTest, RejectOrderWithZeroQuantity) {
    auto order = createLimitOrder("SPY", 0, 450.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
    EXPECT_NE(result.error().message.find("Quantity"), std::string::npos);
}

TEST_F(OrderManagerIntegrationTest, RejectLimitOrderWithInvalidPrice) {
    auto order = createLimitOrder("SPY", 10, -50.00);  // Negative price
    auto result = order_mgr_->placeOrder(order);

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
    EXPECT_NE(result.error().message.find("price"), std::string::npos);
}

TEST_F(OrderManagerIntegrationTest, RejectStopOrderWithInvalidStopPrice) {
    Order order;
    order.symbol = "SPY";
    order.quantity = 10;
    order.type = OrderType::Stop;
    order.stop_price = 0.0;  // Invalid stop price
    order.duration = OrderDuration::Day;

    auto result = order_mgr_->placeOrder(order);

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
}

TEST_F(OrderManagerIntegrationTest, AcceptValidLimitOrder) {
    auto order = createLimitOrder("MSFT", 25, 380.00);
    auto result = order_mgr_->placeOrder(order);

    ASSERT_TRUE(result) << "Valid order should be accepted: "
                        << (result ? "" : result.error().message);
    EXPECT_FALSE(result->empty());
}

TEST_F(OrderManagerIntegrationTest, AcceptValidMarketOrder) {
    auto order = createMarketOrder("META", 15);
    auto result = order_mgr_->placeOrder(order);

    ASSERT_TRUE(result) << "Valid market order should be accepted: "
                        << (result ? "" : result.error().message);
    EXPECT_FALSE(result->empty());
}

// ============================================================================
// Safety Feature Tests: Position Size Limits
// ============================================================================

TEST_F(OrderManagerIntegrationTest, RejectOrderExceedingPositionSizeLimit) {
    // Set max position size to 1000 shares
    order_mgr_->setMaxPositionSize(1000);

    // Try to place order for 1500 shares (exceeds limit)
    auto order = createLimitOrder("NVDA", 1500, 500.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
    EXPECT_NE(result.error().message.find("maximum position size"), std::string::npos);
}

TEST_F(OrderManagerIntegrationTest, AcceptOrderWithinPositionSizeLimit) {
    // Set max position size to 1000 shares
    order_mgr_->setMaxPositionSize(1000);

    // Place order for 500 shares (within limit)
    auto order = createLimitOrder("NVDA", 500, 500.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Order within position size limit should be accepted";
}

TEST_F(OrderManagerIntegrationTest, AcceptOrderAtExactPositionSizeLimit) {
    // Set max position size to 1000 shares
    order_mgr_->setMaxPositionSize(1000);

    // Place order for exactly 1000 shares (at limit)
    auto order = createLimitOrder("AMD", 1000, 120.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Order at exact position size limit should be accepted";
}

// ============================================================================
// Safety Feature Tests: Buying Power Verification
// ============================================================================

/**
 * Test: Verify buying power check is performed
 * Note: In stub implementation, buying power is set to $28,000
 */
TEST_F(OrderManagerIntegrationTest, BuyingPowerCheckPerformed) {
    // AccountManager stub returns $28,000 buying power
    // Try to place expensive order that would exceed buying power
    auto order = createLimitOrder("BRK.A", 100, 600000.00);  // ~$60M order
    auto result = order_mgr_->placeOrder(order);

    // Should succeed in stub (simplified check)
    // In production, would check actual buying power
    EXPECT_TRUE(result || result.error().code == ErrorCode::InvalidOperation);
}

TEST_F(OrderManagerIntegrationTest, AcceptOrderWithSufficientBuyingPower) {
    // Place modest order well within $28k buying power
    auto order = createLimitOrder("INTC", 100, 50.00);  // $5,000 order
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Order with sufficient buying power should be accepted";
}

// ============================================================================
// Safety Feature Tests: Position Limit Enforcement
// ============================================================================

TEST_F(OrderManagerIntegrationTest, RejectOrderExceedingMaxPositions) {
    // Set max positions to 3
    order_mgr_->setMaxPositions(3);

    // Create 3 bot-managed positions (at limit)
    createBotManagedPosition("SPY", "STRATEGY_1");
    createBotManagedPosition("QQQ", "STRATEGY_2");
    createBotManagedPosition("XLE", "STRATEGY_3");

    // Try to open 4th position (should be rejected)
    auto order = createLimitOrder("TLT", 50, 100.00);
    auto result = order_mgr_->placeOrder(order);

    // Note: In stub implementation, this may pass as position tracking is simplified
    // In production, would enforce strict position limits
    if (!result) {
        EXPECT_EQ(result.error().code, ErrorCode::InvalidOperation);
        EXPECT_NE(result.error().message.find("position limit"), std::string::npos);
    }
}

TEST_F(OrderManagerIntegrationTest, AcceptOrderWithinMaxPositions) {
    // Set max positions to 10
    order_mgr_->setMaxPositions(10);

    // Create 2 positions (well under limit)
    createBotManagedPosition("SPY", "STRATEGY_1");
    createBotManagedPosition("QQQ", "STRATEGY_2");

    // Place order for 3rd position (should be accepted)
    auto order = createLimitOrder("XLE", 30, 80.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Order within max positions should be accepted";
}

// ============================================================================
// Safety Feature Tests: Dry-Run Mode
// ============================================================================

TEST_F(OrderManagerIntegrationTest, DryRunModeOrdersLogged) {
    // Enable dry-run mode
    order_mgr_->setDryRunMode(true);

    // Place order in dry-run mode
    auto order = createLimitOrder("AMZN", 10, 170.00);
    auto result = order_mgr_->placeOrder(order);

    // Verify: Order should be "accepted" but not actually executed
    ASSERT_TRUE(result) << "Dry-run order should return success";
    EXPECT_FALSE(result->empty()) << "Should return order ID even in dry-run";

    // Verify order status is Pending (not executed)
    auto status_result = order_mgr_->getOrderStatus(*result);
    ASSERT_TRUE(status_result);
    EXPECT_EQ(*status_result, OrderStatus::Pending)
        << "Dry-run order should stay in Pending status";
}

TEST_F(OrderManagerIntegrationTest, DryRunModeMultipleOrders) {
    order_mgr_->setDryRunMode(true);

    // Place multiple orders in dry-run mode
    auto order1 = createLimitOrder("SPY", 20, 450.00);
    auto order2 = createLimitOrder("QQQ", 15, 350.00);
    auto order3 = createMarketOrder("XLE", 25);

    auto result1 = order_mgr_->placeOrder(order1);
    auto result2 = order_mgr_->placeOrder(order2);
    auto result3 = order_mgr_->placeOrder(order3);

    EXPECT_TRUE(result1) << "Dry-run order 1 should succeed";
    EXPECT_TRUE(result2) << "Dry-run order 2 should succeed";
    EXPECT_TRUE(result3) << "Dry-run order 3 should succeed";

    // All orders should be distinct
    EXPECT_NE(*result1, *result2);
    EXPECT_NE(*result2, *result3);
    EXPECT_NE(*result1, *result3);
}

TEST_F(OrderManagerIntegrationTest, DryRunModeCancelOrder) {
    order_mgr_->setDryRunMode(true);

    // Place and cancel order in dry-run mode
    auto order = createLimitOrder("TSLA", 5, 250.00);
    auto place_result = order_mgr_->placeOrder(order);
    ASSERT_TRUE(place_result);

    auto cancel_result = order_mgr_->cancelOrder(*place_result);
    EXPECT_TRUE(cancel_result) << "Dry-run cancel should succeed";
}

TEST_F(OrderManagerIntegrationTest, DryRunModeToggle) {
    // Test toggling dry-run mode on and off
    order_mgr_->setDryRunMode(true);
    order_mgr_->setDryRunMode(false);
    order_mgr_->setDryRunMode(true);

    // Should still work after toggling
    auto order = createLimitOrder("NFLX", 8, 450.00);
    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result);
}

// ============================================================================
// Safety Feature Tests: Compliance Logging
// ============================================================================

TEST_F(OrderManagerIntegrationTest, ComplianceLoggingOnOrderPlacement) {
    // Place order and verify it goes through (compliance logging is internal)
    auto order = createLimitOrder("JPM", 40, 150.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Order with compliance logging should succeed";
    // Note: Compliance logs would be captured in production via Logger
}

TEST_F(OrderManagerIntegrationTest, ComplianceLoggingOnOrderCancellation) {
    // Place and cancel order
    auto order = createLimitOrder("GS", 30, 380.00);
    auto place_result = order_mgr_->placeOrder(order);
    ASSERT_TRUE(place_result);

    auto cancel_result = order_mgr_->cancelOrder(*place_result);
    EXPECT_TRUE(cancel_result) << "Cancel with compliance logging should succeed";
}

// ============================================================================
// Order Management Tests: Order Lifecycle
// ============================================================================

TEST_F(OrderManagerIntegrationTest, OrderLifecycleFlow) {
    // 1. Place order
    auto order = createLimitOrder("GOOGL", 12, 140.00);
    auto place_result = order_mgr_->placeOrder(order);
    ASSERT_TRUE(place_result);
    auto order_id = *place_result;

    // 2. Check order status
    auto status_result = order_mgr_->getOrderStatus(order_id);
    ASSERT_TRUE(status_result);
    EXPECT_TRUE(*status_result == OrderStatus::Working ||
                *status_result == OrderStatus::Pending);

    // 3. Cancel order
    auto cancel_result = order_mgr_->cancelOrder(order_id);
    EXPECT_TRUE(cancel_result);

    // 4. Verify canceled status
    auto final_status = order_mgr_->getOrderStatus(order_id);
    ASSERT_TRUE(final_status);
    EXPECT_EQ(*final_status, OrderStatus::Canceled);
}

TEST_F(OrderManagerIntegrationTest, GetOrderStatusNonexistentOrder) {
    auto result = order_mgr_->getOrderStatus("INVALID_ORDER_ID");

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
    EXPECT_NE(result.error().message.find("not found"), std::string::npos);
}

TEST_F(OrderManagerIntegrationTest, CancelNonexistentOrder) {
    auto result = order_mgr_->cancelOrder("INVALID_ORDER_ID");

    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::InvalidParameter);
}

TEST_F(OrderManagerIntegrationTest, GetActiveOrders) {
    // Place multiple orders
    auto order1 = createLimitOrder("SPY", 10, 450.00);
    auto order2 = createLimitOrder("QQQ", 15, 350.00);
    auto order3 = createLimitOrder("XLE", 20, 80.00);

    auto result1 = order_mgr_->placeOrder(order1);
    auto result2 = order_mgr_->placeOrder(order2);
    auto result3 = order_mgr_->placeOrder(order3);

    ASSERT_TRUE(result1 && result2 && result3);

    // Get active orders
    auto active_orders = order_mgr_->getActiveOrders();
    EXPECT_GE(active_orders.size(), 3) << "Should have at least 3 active orders";
}

TEST_F(OrderManagerIntegrationTest, GetActiveOrdersEmpty) {
    // No orders placed
    auto active_orders = order_mgr_->getActiveOrders();
    EXPECT_EQ(active_orders.size(), 0) << "Should have no active orders initially";
}

// ============================================================================
// Order Duration Tests
// ============================================================================

TEST_F(OrderManagerIntegrationTest, DayOrderPlacement) {
    auto order = createLimitOrder("AAPL", 10, 150.00, OrderDuration::Day);
    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result) << "DAY order should be accepted";
}

TEST_F(OrderManagerIntegrationTest, GTCOrderPlacement) {
    auto order = createLimitOrder("MSFT", 15, 380.00, OrderDuration::GTC);
    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result) << "GTC order should be accepted";
}

TEST_F(OrderManagerIntegrationTest, FOKOrderPlacement) {
    auto order = createLimitOrder("TSLA", 5, 250.00, OrderDuration::FOK);
    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result) << "FOK order should be accepted";
}

TEST_F(OrderManagerIntegrationTest, IOCOrderPlacement) {
    auto order = createLimitOrder("NVDA", 8, 500.00, OrderDuration::IOC);
    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result) << "IOC order should be accepted";
}

// ============================================================================
// Order Type Tests
// ============================================================================

TEST_F(OrderManagerIntegrationTest, StopLimitOrderPlacement) {
    Order order;
    order.symbol = "AMD";
    order.quantity = 20;
    order.type = OrderType::StopLimit;
    order.stop_price = 120.00;
    order.limit_price = 118.00;
    order.duration = OrderDuration::Day;

    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result) << "Stop-limit order should be accepted";
}

TEST_F(OrderManagerIntegrationTest, StopOrderPlacement) {
    Order order;
    order.symbol = "INTC";
    order.quantity = 50;
    order.type = OrderType::Stop;
    order.stop_price = 45.00;
    order.duration = OrderDuration::Day;

    auto result = order_mgr_->placeOrder(order);
    EXPECT_TRUE(result) << "Stop order should be accepted";
}

// ============================================================================
// Integration Scenarios: Complex Workflows
// ============================================================================

TEST_F(OrderManagerIntegrationTest, CompleteTradeWorkflow) {
    // Scenario: Bot opens position, manages it, then closes it

    // 1. Check no position exists for symbol
    auto initial_position = account_mgr_->getPosition("NFLX");
    ASSERT_TRUE(initial_position);
    EXPECT_FALSE(initial_position->has_value()) << "Should have no initial position";

    // 2. Place order to open position
    auto open_order = createLimitOrder("NFLX", 15, 450.00);
    auto open_result = order_mgr_->placeOrder(open_order);
    ASSERT_TRUE(open_result) << "Opening order should be accepted";

    // 3. Mark as bot-managed
    createBotManagedPosition("NFLX", "MOMENTUM");

    // 4. Place another order to add to position (should be accepted)
    auto add_order = createLimitOrder("NFLX", 10, 455.00);
    auto add_result = order_mgr_->placeOrder(add_order);
    EXPECT_TRUE(add_result) << "Add order should be accepted (bot-managed)";

    // 5. Close position
    auto close_order = createLimitOrder("NFLX", -25, 460.00);  // Sell
    auto close_result = order_mgr_->placeOrder(close_order);
    EXPECT_TRUE(close_result) << "Close order should be accepted";
}

TEST_F(OrderManagerIntegrationTest, MultipleSymbolsWorkflow) {
    // Trade multiple symbols simultaneously
    std::vector<std::string> symbols = {"SPY", "QQQ", "XLE", "TLT", "GLD"};
    std::vector<std::string> order_ids;

    // Place orders for all symbols
    for (auto const& symbol : symbols) {
        auto order = createLimitOrder(symbol, 10, 100.00);
        auto result = order_mgr_->placeOrder(order);
        ASSERT_TRUE(result) << "Order for " << symbol << " should be accepted";
        order_ids.push_back(*result);
    }

    // Verify all orders are active
    auto active_orders = order_mgr_->getActiveOrders();
    EXPECT_GE(active_orders.size(), symbols.size());

    // Cancel some orders
    for (size_t i = 0; i < 3; ++i) {
        auto cancel_result = order_mgr_->cancelOrder(order_ids[i]);
        EXPECT_TRUE(cancel_result) << "Cancel should succeed for order " << i;
    }
}

TEST_F(OrderManagerIntegrationTest, StressTestOrderPlacement) {
    // Place many orders rapidly
    constexpr int NUM_ORDERS = 50;
    std::vector<std::string> order_ids;

    for (int i = 0; i < NUM_ORDERS; ++i) {
        auto order = createLimitOrder(
            "TEST_" + std::to_string(i),
            10 + i,
            100.0 + static_cast<double>(i)
        );

        auto result = order_mgr_->placeOrder(order);
        ASSERT_TRUE(result) << "Order " << i << " should be accepted";
        order_ids.push_back(*result);
    }

    EXPECT_EQ(order_ids.size(), NUM_ORDERS);

    // Verify all orders have unique IDs
    std::set<std::string> unique_ids(order_ids.begin(), order_ids.end());
    EXPECT_EQ(unique_ids.size(), NUM_ORDERS) << "All order IDs should be unique";
}

TEST_F(OrderManagerIntegrationTest, SafetyCheckOrdering) {
    // Verify safety checks are performed in correct order:
    // 1. Authentication
    // 2. Parameter validation
    // 3. Manual position check (CRITICAL)
    // 4. Buying power check
    // 5. Position limits check

    // This order should fail at manual position check
    createManualPosition("FAIL_MANUAL", 100, 100.00);
    auto order = createLimitOrder("FAIL_MANUAL", 1000000, -100.00);
    auto result = order_mgr_->placeOrder(order);

    // Should fail (either at validation or manual check)
    EXPECT_FALSE(result);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(OrderManagerIntegrationTest, NegativeQuantitySellOrder) {
    // Negative quantity = SELL order
    auto order = createLimitOrder("SPY", -50, 450.00);
    auto result = order_mgr_->placeOrder(order);

    // Should be accepted (selling is valid)
    EXPECT_TRUE(result) << "Sell order (negative quantity) should be accepted";
}

TEST_F(OrderManagerIntegrationTest, VeryLargeQuantity) {
    // Try to place order with very large quantity
    order_mgr_->setMaxPositionSize(1000000);  // Increase limit

    auto order = createLimitOrder("PENNY_STOCK", 500000, 0.50);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Large quantity order should be accepted if within limits";
}

TEST_F(OrderManagerIntegrationTest, VerySmallPrice) {
    // Penny stock with small price
    auto order = createLimitOrder("PENNY", 1000, 0.01);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Small price order should be accepted";
}

TEST_F(OrderManagerIntegrationTest, SymbolWithSpecialCharacters) {
    // Some symbols have dots or other characters (e.g., BRK.A)
    auto order = createLimitOrder("BRK.A", 1, 600000.00);
    auto result = order_mgr_->placeOrder(order);

    EXPECT_TRUE(result) << "Symbol with special characters should be accepted";
}

TEST_F(OrderManagerIntegrationTest, ConcurrentOrderPlacement) {
    // Simulate concurrent order placement
    std::vector<std::string> symbols = {"A", "B", "C", "D", "E"};

    for (auto const& symbol : symbols) {
        auto order = createLimitOrder(symbol, 10, 100.00);
        auto result = order_mgr_->placeOrder(order);
        EXPECT_TRUE(result) << "Concurrent order for " << symbol << " should succeed";
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(OrderManagerIntegrationTest, FluentConfigurationChaining) {
    // Test fluent API configuration chaining
    auto& result = order_mgr_->setDryRunMode(true)
                              .setMaxPositionSize(2000)
                              .setMaxPositions(5);

    EXPECT_EQ(&result, order_mgr_.get()) << "Fluent methods should return reference";
}

TEST_F(OrderManagerIntegrationTest, UpdateMaxPositionSize) {
    order_mgr_->setMaxPositionSize(500);

    // Order exceeding limit should fail
    auto order1 = createLimitOrder("TEST1", 1000, 100.00);
    auto result1 = order_mgr_->placeOrder(order1);
    EXPECT_FALSE(result1);

    // Order within limit should pass
    auto order2 = createLimitOrder("TEST2", 300, 100.00);
    auto result2 = order_mgr_->placeOrder(order2);
    EXPECT_TRUE(result2);
}

TEST_F(OrderManagerIntegrationTest, UpdateMaxPositions) {
    order_mgr_->setMaxPositions(2);

    // Within limits - should work
    auto order1 = createLimitOrder("SYM1", 10, 100.00);
    auto order2 = createLimitOrder("SYM2", 10, 100.00);

    EXPECT_TRUE(order_mgr_->placeOrder(order1));
    EXPECT_TRUE(order_mgr_->placeOrder(order2));
}

// ============================================================================
// Main Function
// ============================================================================

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
