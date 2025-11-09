/**
 * BigBrotherAnalytics - End-to-End Trading Workflow Test (C++23)
 *
 * Complete workflow validation with safety constraints:
 * - Manual position protection (NEVER touch existing positions)
 * - Bot-managed position tracking
 * - Order validation and compliance
 * - Risk management integration
 * - Dry-run mode testing
 *
 * Test Scenarios:
 * 1. Bot tries to trade AAPL (manual position) → REJECT
 * 2. Bot tries to trade XLE (new symbol) → ACCEPT
 * 3. Bot tries to close own XLE position → ACCEPT
 * 4. Bot tries to trade SPY (bot-managed) → ACCEPT
 * 5. Dry-run mode test → Log but don't execute
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <fmt/core.h>

// Module imports
import bigbrother.risk_management;
import bigbrother.schwab_api;
import bigbrother.utils.types;
import bigbrother.utils.logger;

using namespace bigbrother::risk;
using namespace bigbrother::schwab;
using bigbrother::types::ErrorCode;
using bigbrother::types::Result;
using bigbrother::types::makeError;
using namespace bigbrother::utils;

// ============================================================================
// Test-specific types (to avoid conflicts with module types)
// ============================================================================

namespace test {

/**
 * Position with safety flags for manual position protection
 * Note: This is a test-specific struct since the main Position type in
 * bigbrother::types doesn't have the safety tracking fields we need.
 */
struct Position {
    std::string account_id;
    std::string symbol;
    int quantity{0};
    double avg_cost{0.0};
    double current_price{0.0};
    double market_value{0.0};
    double unrealized_pnl{0.0};

    // CRITICAL SAFETY FLAGS
    bool is_bot_managed{false};        // TRUE if bot opened this position
    std::string managed_by{"MANUAL"};  // "BOT" or "MANUAL"
    std::string bot_strategy{};        // Strategy that opened this (if bot-managed)

    std::chrono::system_clock::time_point opened_at;
    std::string opened_by{"MANUAL"};   // "BOT" or "MANUAL"
    std::chrono::system_clock::time_point updated_at;

    [[nodiscard]] auto canBotTrade() const noexcept -> bool {
        return is_bot_managed;
    }
};

/**
 * Order types for testing
 */
enum class OrderSide {
    Buy,
    Sell,
    SellShort,
    BuyToCover
};

enum class OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop
};

enum class OrderStatus {
    Pending,
    Working,
    Filled,
    PartiallyFilled,
    Canceled,
    Rejected
};

enum class OrderDuration {
    Day,
    GTC,  // Good Till Canceled
    GTD,  // Good Till Date
    FOK,  // Fill Or Kill
    IOC   // Immediate Or Cancel
};

/**
 * Order structure for testing
 */
struct Order {
    std::string order_id;
    std::string account_id;
    std::string symbol;
    OrderSide side{OrderSide::Buy};
    int quantity{0};
    int filled_quantity{0};
    OrderType type{OrderType::Market};
    OrderDuration duration{OrderDuration::Day};

    double limit_price{0.0};
    double stop_price{0.0};
    double trail_amount{0.0};
    double avg_fill_price{0.0};

    OrderStatus status{OrderStatus::Pending};
    std::string strategy_name;
    bool dry_run{true};  // Default to dry-run for safety

    std::string rejection_reason;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::chrono::system_clock::time_point filled_at;

    [[nodiscard]] auto estimatedCost() const noexcept -> double {
        double price = (type == OrderType::Market) ? limit_price : limit_price;
        return static_cast<double>(quantity) * price;
    }

    [[nodiscard]] auto isActive() const noexcept -> bool {
        return status == OrderStatus::Pending ||
               status == OrderStatus::Working ||
               status == OrderStatus::PartiallyFilled;
    }
};

/**
 * Order confirmation response
 */
struct OrderConfirmation {
    std::string order_id;
    std::string symbol;
    OrderSide side;
    int quantity{0};
    int filled_quantity{0};
    double avg_fill_price{0.0};
    OrderStatus status;
    std::string strategy_name;
    bool dry_run{false};
    std::chrono::system_clock::time_point timestamp;
};

} // namespace test

// ============================================================================
// Mock Account Manager
// ============================================================================

/**
 * Mock AccountManager for testing
 * Simulates Schwab account state with positions
 */
class MockAccountManager {
public:
    explicit MockAccountManager(std::string account_id)
        : account_id_{std::move(account_id)} {}

    /**
     * Add a manual position (existing before bot started)
     */
    auto addManualPosition(std::string symbol, int quantity, double avg_cost) -> void {
        test::Position pos{};
        pos.account_id = account_id_;
        pos.symbol = std::move(symbol);
        pos.quantity = quantity;
        pos.avg_cost = avg_cost;
        pos.current_price = avg_cost;
        pos.market_value = static_cast<double>(quantity) * avg_cost;
        pos.is_bot_managed = false;
        pos.managed_by = "MANUAL";
        pos.opened_by = "MANUAL";
        pos.opened_at = std::chrono::system_clock::now();

        positions_.push_back(pos);

        Logger::getInstance().info(
            "MockAccountManager: Added manual position {} (qty: {}, cost: ${:.2f})",
            pos.symbol, quantity, avg_cost
        );
    }

    /**
     * Add a bot-managed position (opened by bot)
     */
    auto addBotPosition(std::string symbol, int quantity, double avg_cost,
                       std::string strategy_name) -> void {
        test::Position pos{};
        pos.account_id = account_id_;
        pos.symbol = std::move(symbol);
        pos.quantity = quantity;
        pos.avg_cost = avg_cost;
        pos.current_price = avg_cost;
        pos.market_value = static_cast<double>(quantity) * avg_cost;
        pos.is_bot_managed = true;
        pos.managed_by = "BOT";
        pos.bot_strategy = std::move(strategy_name);
        pos.opened_by = "BOT";
        pos.opened_at = std::chrono::system_clock::now();

        positions_.push_back(pos);

        Logger::getInstance().info(
            "MockAccountManager: Added bot position {} (qty: {}, cost: ${:.2f}, strategy: {})",
            pos.symbol, quantity, avg_cost, pos.bot_strategy
        );
    }

    /**
     * Get position by symbol
     */
    [[nodiscard]] auto getPosition(std::string const& symbol) const
        -> std::optional<test::Position> {

        for (auto const& pos : positions_) {
            if (pos.symbol == symbol) {
                return pos;
            }
        }
        return std::nullopt;
    }

    /**
     * Get all positions
     */
    [[nodiscard]] auto getAllPositions() const -> std::vector<test::Position> const& {
        return positions_;
    }

    /**
     * Check if symbol exists in portfolio
     */
    [[nodiscard]] auto hasPosition(std::string const& symbol) const -> bool {
        return getPosition(symbol).has_value();
    }

    /**
     * Remove position (when closed)
     */
    auto removePosition(std::string const& symbol) -> void {
        positions_.erase(
            std::remove_if(positions_.begin(), positions_.end(),
                          [&symbol](auto const& p) { return p.symbol == symbol; }),
            positions_.end()
        );

        Logger::getInstance().info("MockAccountManager: Removed position {}", symbol);
    }

private:
    std::string account_id_;
    std::vector<test::Position> positions_;
};

// ============================================================================
// Mock Order Manager
// ============================================================================

/**
 * Mock OrderManager with safety constraints
 * Enforces manual position protection rules
 */
class MockOrderManager {
public:
    explicit MockOrderManager(std::shared_ptr<MockAccountManager> account_mgr)
        : account_mgr_{std::move(account_mgr)},
          dry_run_mode_{true} {}

    /**
     * Place order with CRITICAL safety checks
     */
    [[nodiscard]] auto placeOrder(test::Order const& order) -> Result<test::OrderConfirmation> {
        // CRITICAL CHECK 1: Is this symbol a manual position?
        auto existing_pos = account_mgr_->getPosition(order.symbol);

        if (existing_pos && !existing_pos->is_bot_managed) {
            // REJECT: Manual position exists, cannot trade
            std::string error_msg = fmt::format(
                "SAFETY VIOLATION: Cannot trade {} - manual position exists. "
                "Bot only trades NEW securities or bot-managed positions.",
                order.symbol
            );

            Logger::getInstance().error(error_msg);

            // Log to compliance file
            logComplianceViolation(order, "MANUAL_POSITION_PROTECTION", error_msg);

            return makeError<test::OrderConfirmation>(
                ErrorCode::InvalidParameter,
                error_msg
            );
        }

        // CRITICAL CHECK 2: If closing, verify we own the position
        if (order.side == test::OrderSide::Sell) {
            if (!existing_pos) {
                return makeError<test::OrderConfirmation>(
                    ErrorCode::InvalidParameter,
                    fmt::format("Cannot close {} - no position found", order.symbol)
                );
            }

            if (!existing_pos->is_bot_managed) {
                std::string error_msg = fmt::format(
                    "SAFETY VIOLATION: Cannot close {} - manual position. "
                    "Only human can close manual positions.",
                    order.symbol
                );

                Logger::getInstance().error(error_msg);
                logComplianceViolation(order, "MANUAL_POSITION_CLOSE_BLOCKED", error_msg);

                return makeError<test::OrderConfirmation>(
                    ErrorCode::InvalidParameter,
                    error_msg
                );
            }
        }

        // Order passed safety checks
        test::OrderConfirmation confirmation{};
        confirmation.order_id = generateOrderId();
        confirmation.symbol = order.symbol;
        confirmation.side = order.side;
        confirmation.quantity = order.quantity;
        confirmation.filled_quantity = order.dry_run ? 0 : order.quantity;
        confirmation.avg_fill_price = order.limit_price;
        confirmation.status = order.dry_run ? test::OrderStatus::Pending : test::OrderStatus::Filled;
        confirmation.strategy_name = order.strategy_name;
        confirmation.dry_run = order.dry_run;
        confirmation.timestamp = std::chrono::system_clock::now();

        if (dry_run_mode_ || order.dry_run) {
            Logger::getInstance().info(
                "[DRY-RUN] Order placed: {} {} shares of {} @ ${:.2f} (strategy: {})",
                order.side == test::OrderSide::Buy ? "BUY" : "SELL",
                order.quantity, order.symbol, order.limit_price, order.strategy_name
            );
        } else {
            Logger::getInstance().info(
                "Order EXECUTED: {} {} shares of {} @ ${:.2f} (strategy: {})",
                order.side == test::OrderSide::Buy ? "BUY" : "SELL",
                order.quantity, order.symbol, order.limit_price, order.strategy_name
            );

            // Update positions
            if (order.side == test::OrderSide::Buy) {
                // Opening position
                account_mgr_->addBotPosition(
                    order.symbol,
                    order.quantity,
                    order.limit_price,
                    order.strategy_name
                );
            } else if (order.side == test::OrderSide::Sell) {
                // Closing position
                account_mgr_->removePosition(order.symbol);
            }
        }

        // Log compliance
        logComplianceSuccess(order, confirmation);

        orders_.push_back(order);
        confirmations_.push_back(confirmation);

        return confirmation;
    }

    /**
     * Set dry-run mode
     */
    auto setDryRunMode(bool enabled) -> void {
        dry_run_mode_ = enabled;
        Logger::getInstance().info("Dry-run mode: {}", enabled ? "ENABLED" : "DISABLED");
    }

    /**
     * Get all orders
     */
    [[nodiscard]] auto getAllOrders() const -> std::vector<test::Order> const& {
        return orders_;
    }

    /**
     * Get all confirmations
     */
    [[nodiscard]] auto getAllConfirmations() const -> std::vector<test::OrderConfirmation> const& {
        return confirmations_;
    }

    /**
     * Get compliance log
     */
    [[nodiscard]] auto getComplianceLog() const -> std::vector<std::string> const& {
        return compliance_log_;
    }

private:
    auto generateOrderId() -> std::string {
        static int counter = 1000;
        return fmt::format("ORDER_{}", counter++);
    }

    auto logComplianceViolation(test::Order const& order, std::string const& violation_type,
                               std::string const& reason) -> void {
        std::string log_entry = fmt::format(
            "[COMPLIANCE_VIOLATION] Type: {} | Symbol: {} | Side: {} | Reason: {}",
            violation_type, order.symbol,
            order.side == test::OrderSide::Buy ? "BUY" : "SELL",
            reason
        );

        compliance_log_.push_back(log_entry);
        Logger::getInstance().warn(log_entry);
    }

    auto logComplianceSuccess(test::Order const& order, test::OrderConfirmation const& confirmation) -> void {
        std::string log_entry = fmt::format(
            "[COMPLIANCE_OK] Order: {} | Symbol: {} | Side: {} | Qty: {} | Strategy: {} | Dry-run: {}",
            confirmation.order_id, order.symbol,
            order.side == test::OrderSide::Buy ? "BUY" : "SELL",
            order.quantity, order.strategy_name,
            order.dry_run ? "YES" : "NO"
        );

        compliance_log_.push_back(log_entry);
        Logger::getInstance().info(log_entry);
    }

    std::shared_ptr<MockAccountManager> account_mgr_;
    bool dry_run_mode_;
    std::vector<test::Order> orders_;
    std::vector<test::OrderConfirmation> confirmations_;
    std::vector<std::string> compliance_log_;
};

// ============================================================================
// Test Fixture
// ============================================================================

class SchwabE2EWorkflowTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::getInstance().info("=== Setting up E2E Workflow Test ===");

        // Initialize account manager with test account
        account_mgr_ = std::make_shared<MockAccountManager>("XXXX1234");

        // Add existing manual positions (pre-existing before bot started)
        account_mgr_->addManualPosition("AAPL", 10, 150.00);  // Manual: Apple
        account_mgr_->addManualPosition("MSFT", 5, 300.00);   // Manual: Microsoft

        // Initialize order manager with safety constraints
        order_mgr_ = std::make_unique<MockOrderManager>(account_mgr_);

        // Initialize risk manager with $30K account
        risk_mgr_ = std::make_unique<RiskManager>(RiskLimits::forThirtyKAccount());

        Logger::getInstance().info("Test setup complete");
        Logger::getInstance().info("Manual positions: AAPL (10 shares), MSFT (5 shares)");
    }

    void TearDown() override {
        Logger::getInstance().info("=== Test teardown complete ===\n");
    }

    std::shared_ptr<MockAccountManager> account_mgr_;
    std::unique_ptr<MockOrderManager> order_mgr_;
    std::unique_ptr<RiskManager> risk_mgr_;
};

// ============================================================================
// SCENARIO 1: Bot tries to trade AAPL (manual position) → REJECT
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, Scenario1_RejectManualPosition) {
    Logger::getInstance().info("\n=== SCENARIO 1: Bot tries to trade AAPL (manual) ===");

    // Bot generates signal for AAPL (which is a manual position)
    test::Order order{};
    order.account_id = "XXXX1234";
    order.symbol = "AAPL";
    order.side = test::OrderSide::Buy;
    order.quantity = 5;
    order.limit_price = 155.00;
    order.type = test::OrderType::Limit;
    order.strategy_name = "SectorRotation";
    order.dry_run = false;

    // Attempt to place order
    auto result = order_mgr_->placeOrder(order);

    // ASSERT: Order should be REJECTED
    ASSERT_FALSE(result.has_value()) << "Order for manual position should be rejected";
    EXPECT_TRUE(result.error().message.find("manual position exists") != std::string::npos)
        << "Error message should mention manual position";

    // Verify compliance logging
    auto const& compliance_log = order_mgr_->getComplianceLog();
    EXPECT_GT(compliance_log.size(), 0) << "Compliance violation should be logged";

    bool found_violation = false;
    for (auto const& entry : compliance_log) {
        if (entry.find("COMPLIANCE_VIOLATION") != std::string::npos &&
            entry.find("AAPL") != std::string::npos) {
            found_violation = true;
            Logger::getInstance().info("Compliance log: {}", entry);
        }
    }

    EXPECT_TRUE(found_violation) << "AAPL violation should be in compliance log";

    Logger::getInstance().info("✅ SCENARIO 1 PASSED: Manual position protected");
}

// ============================================================================
// SCENARIO 2: Bot tries to trade XLE (new symbol) → ACCEPT
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, Scenario2_AcceptNewSymbol) {
    Logger::getInstance().info("\n=== SCENARIO 2: Bot tries to trade XLE (new) ===");

    // Verify XLE is NOT in portfolio
    ASSERT_FALSE(account_mgr_->hasPosition("XLE"))
        << "XLE should not exist in portfolio initially";

    // Assess trade risk
    double position_size = 20 * 80.00;  // quantity * price
    auto risk_assessment = risk_mgr_->assessTrade(
        "XLE",
        position_size,
        80.00,   // entry price
        75.00,   // stop price
        90.00,   // target price
        0.70     // win probability
    );

    ASSERT_TRUE(risk_assessment.has_value()) << "Risk assessment should succeed";
    EXPECT_TRUE(risk_assessment->approved) << "Trade should be approved by risk manager";

    Logger::getInstance().info("Risk assessment: Approved = {}, Expected value = ${:.2f}",
                              risk_assessment->approved, risk_assessment->expected_value);

    // Create order for XLE
    test::Order order{};
    order.account_id = "XXXX1234";
    order.symbol = "XLE";
    order.side = test::OrderSide::Buy;
    order.quantity = 20;
    order.limit_price = 80.00;
    order.type = test::OrderType::Limit;
    order.strategy_name = "SectorRotation";
    order.dry_run = false;

    // Place order
    order_mgr_->setDryRunMode(false);  // Execute real order
    auto result = order_mgr_->placeOrder(order);

    // ASSERT: Order should be ACCEPTED
    ASSERT_TRUE(result.has_value()) << "Order for new symbol should be accepted";
    EXPECT_EQ(result->symbol, "XLE");
    EXPECT_EQ(result->quantity, 20);
    EXPECT_EQ(result->strategy_name, "SectorRotation");

    Logger::getInstance().info("Order confirmation: {} | Status: {}",
                              result->order_id,
                              result->status == test::OrderStatus::Filled ? "FILLED" : "PENDING");

    // Verify position was created
    auto xle_position = account_mgr_->getPosition("XLE");
    ASSERT_TRUE(xle_position.has_value()) << "XLE position should exist after order";
    EXPECT_TRUE(xle_position->is_bot_managed) << "XLE should be marked as bot-managed";
    EXPECT_EQ(xle_position->managed_by, "BOT");
    EXPECT_EQ(xle_position->bot_strategy, "SectorRotation");
    EXPECT_EQ(xle_position->quantity, 20);

    Logger::getInstance().info("✅ SCENARIO 2 PASSED: New symbol accepted and position created");
}

// ============================================================================
// SCENARIO 3: Bot tries to close own XLE position → ACCEPT
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, Scenario3_CloseOwnPosition) {
    Logger::getInstance().info("\n=== SCENARIO 3: Bot closes own XLE position ===");

    // Setup: Add bot-managed XLE position
    account_mgr_->addBotPosition("XLE", 20, 80.00, "SectorRotation");

    // Verify XLE exists and is bot-managed
    auto xle_position = account_mgr_->getPosition("XLE");
    ASSERT_TRUE(xle_position.has_value());
    ASSERT_TRUE(xle_position->is_bot_managed) << "XLE should be bot-managed";

    Logger::getInstance().info("Existing XLE position: {} shares, bot-managed = {}",
                              xle_position->quantity, xle_position->is_bot_managed);

    // Create SELL order to close position
    test::Order order{};
    order.account_id = "XXXX1234";
    order.symbol = "XLE";
    order.side = test::OrderSide::Sell;
    order.quantity = 20;
    order.limit_price = 85.00;
    order.type = test::OrderType::Limit;
    order.strategy_name = "SectorRotation";
    order.dry_run = false;

    // Place order
    order_mgr_->setDryRunMode(false);
    auto result = order_mgr_->placeOrder(order);

    // ASSERT: Order should be ACCEPTED
    ASSERT_TRUE(result.has_value()) << "Closing bot-managed position should succeed";
    EXPECT_EQ(result->symbol, "XLE");
    EXPECT_EQ(result->side, test::OrderSide::Sell);
    EXPECT_EQ(result->quantity, 20);

    Logger::getInstance().info("Sell order confirmation: {} | Price: ${:.2f}",
                              result->order_id, result->avg_fill_price);

    // Verify position was removed
    auto closed_position = account_mgr_->getPosition("XLE");
    EXPECT_FALSE(closed_position.has_value()) << "XLE position should be closed";

    Logger::getInstance().info("✅ SCENARIO 3 PASSED: Bot successfully closed own position");
}

// ============================================================================
// SCENARIO 4: Bot tries to trade SPY (bot-managed) → ACCEPT
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, Scenario4_TradeBotManagedPosition) {
    Logger::getInstance().info("\n=== SCENARIO 4: Bot trades SPY (bot-managed) ===");

    // Setup: Add bot-managed SPY position
    account_mgr_->addBotPosition("SPY", 10, 450.00, "MomentumStrategy");

    // Verify SPY is bot-managed
    auto spy_position = account_mgr_->getPosition("SPY");
    ASSERT_TRUE(spy_position.has_value());
    ASSERT_TRUE(spy_position->is_bot_managed);

    Logger::getInstance().info("Existing SPY position: {} shares (bot-managed)",
                              spy_position->quantity);

    // Bot decides to add to SPY position
    test::Order order{};
    order.account_id = "XXXX1234";
    order.symbol = "SPY";
    order.side = test::OrderSide::Buy;
    order.quantity = 5;
    order.limit_price = 455.00;
    order.type = test::OrderType::Limit;
    order.strategy_name = "MomentumStrategy";
    order.dry_run = false;

    // Place order
    order_mgr_->setDryRunMode(false);
    auto result = order_mgr_->placeOrder(order);

    // ASSERT: Order should be ACCEPTED (bot can trade its own positions)
    ASSERT_TRUE(result.has_value()) << "Trading bot-managed position should succeed";
    EXPECT_EQ(result->symbol, "SPY");
    EXPECT_EQ(result->quantity, 5);

    Logger::getInstance().info("✅ SCENARIO 4 PASSED: Bot can trade its own positions");
}

// ============================================================================
// SCENARIO 5: Dry-run mode test → Log but don't execute
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, Scenario5_DryRunMode) {
    Logger::getInstance().info("\n=== SCENARIO 5: Dry-run mode test ===");

    // Enable dry-run mode
    order_mgr_->setDryRunMode(true);

    // Create order for new symbol
    test::Order order{};
    order.account_id = "XXXX1234";
    order.symbol = "QQQ";
    order.side = test::OrderSide::Buy;
    order.quantity = 15;
    order.limit_price = 350.00;
    order.type = test::OrderType::Limit;
    order.strategy_name = "TechRotation";
    order.dry_run = true;

    // Place order in dry-run mode
    auto result = order_mgr_->placeOrder(order);

    // ASSERT: Order should be accepted but not executed
    ASSERT_TRUE(result.has_value()) << "Dry-run order should be accepted";
    EXPECT_EQ(result->symbol, "QQQ");
    EXPECT_TRUE(result->dry_run) << "Order should be marked as dry-run";
    EXPECT_EQ(result->status, test::OrderStatus::Pending) << "Status should be Pending (not Filled)";
    EXPECT_EQ(result->filled_quantity, 0) << "No shares should be filled in dry-run";

    Logger::getInstance().info("Dry-run order: {} | Symbol: {} | Filled: {}",
                              result->order_id, result->symbol, result->filled_quantity);

    // Verify position was NOT created
    auto qqq_position = account_mgr_->getPosition("QQQ");
    EXPECT_FALSE(qqq_position.has_value())
        << "Position should NOT be created in dry-run mode";

    // Verify compliance logging
    auto const& compliance_log = order_mgr_->getComplianceLog();
    bool found_dryrun_log = false;
    for (auto const& entry : compliance_log) {
        if (entry.find("QQQ") != std::string::npos &&
            entry.find("Dry-run: YES") != std::string::npos) {
            found_dryrun_log = true;
            Logger::getInstance().info("Compliance: {}", entry);
        }
    }

    EXPECT_TRUE(found_dryrun_log) << "Dry-run should be logged";

    Logger::getInstance().info("✅ SCENARIO 5 PASSED: Dry-run mode works correctly");
}

// ============================================================================
// INTEGRATION TEST: Complete workflow with multiple positions
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, CompleteWorkflow) {
    Logger::getInstance().info("\n=== COMPLETE WORKFLOW TEST ===");

    // Starting state
    Logger::getInstance().info("Initial portfolio:");
    Logger::getInstance().info("  Manual: AAPL (10 shares), MSFT (5 shares)");
    Logger::getInstance().info("  Bot-managed: None");

    // Step 1: Generate signals (should filter out manual positions)
    std::vector<std::string> candidate_symbols = {"AAPL", "XLE", "XLV", "MSFT", "GLD"};
    std::vector<std::string> valid_signals;

    for (auto const& symbol : candidate_symbols) {
        auto pos = account_mgr_->getPosition(symbol);
        if (!pos || pos->is_bot_managed) {
            valid_signals.push_back(symbol);
            Logger::getInstance().info("Signal generated: {} (OK to trade)", symbol);
        } else {
            Logger::getInstance().info("Signal filtered: {} (manual position)", symbol);
        }
    }

    // ASSERT: Manual positions should be filtered
    EXPECT_EQ(valid_signals.size(), 3) << "Should have 3 valid signals (XLE, XLV, GLD)";
    EXPECT_TRUE(std::find(valid_signals.begin(), valid_signals.end(), "AAPL") == valid_signals.end())
        << "AAPL should be filtered (manual)";
    EXPECT_TRUE(std::find(valid_signals.begin(), valid_signals.end(), "MSFT") == valid_signals.end())
        << "MSFT should be filtered (manual)";

    // Step 2: Place orders for valid signals
    order_mgr_->setDryRunMode(false);

    for (auto const& symbol : valid_signals) {
        test::Order order{};
        order.account_id = "XXXX1234";
        order.symbol = symbol;
        order.side = test::OrderSide::Buy;
        order.quantity = 20;
        order.limit_price = (symbol == "XLE") ? 80.00 : ((symbol == "XLV") ? 120.00 : 200.00);
        order.type = test::OrderType::Limit;
        order.strategy_name = "SectorRotation";
        order.dry_run = false;

        auto result = order_mgr_->placeOrder(order);
        ASSERT_TRUE(result.has_value()) << "Order for " << symbol << " should succeed";

        Logger::getInstance().info("Executed: {} - {} shares @ ${:.2f}",
                                  symbol, order.quantity, order.limit_price);
    }

    // Step 3: Verify portfolio state
    auto all_positions = account_mgr_->getAllPositions();

    int manual_count = 0;
    int bot_count = 0;

    Logger::getInstance().info("\nFinal portfolio:");
    for (auto const& pos : all_positions) {
        if (pos.is_bot_managed) {
            bot_count++;
            Logger::getInstance().info("  Bot: {} ({} shares, strategy: {})",
                                      pos.symbol, pos.quantity, pos.bot_strategy);
        } else {
            manual_count++;
            Logger::getInstance().info("  Manual: {} ({} shares) [PROTECTED]",
                                      pos.symbol, pos.quantity);
        }
    }

    EXPECT_EQ(manual_count, 2) << "Should still have 2 manual positions";
    EXPECT_EQ(bot_count, 3) << "Should have 3 bot-managed positions";

    // Step 4: Close one bot position
    test::Order close_order{};
    close_order.account_id = "XXXX1234";
    close_order.symbol = "XLE";
    close_order.side = test::OrderSide::Sell;
    close_order.quantity = 20;
    close_order.limit_price = 85.00;
    close_order.type = test::OrderType::Limit;
    close_order.strategy_name = "SectorRotation";
    close_order.dry_run = false;

    auto close_result = order_mgr_->placeOrder(close_order);
    ASSERT_TRUE(close_result.has_value()) << "Closing bot position should succeed";

    Logger::getInstance().info("Closed XLE position @ ${:.2f}", close_order.limit_price);

    // Step 5: Attempt to close manual position (should fail)
    test::Order invalid_close{};
    invalid_close.account_id = "XXXX1234";
    invalid_close.symbol = "AAPL";
    invalid_close.side = test::OrderSide::Sell;
    invalid_close.quantity = 10;
    invalid_close.limit_price = 160.00;
    invalid_close.type = test::OrderType::Limit;
    invalid_close.strategy_name = "SectorRotation";
    invalid_close.dry_run = false;

    auto invalid_result = order_mgr_->placeOrder(invalid_close);
    EXPECT_FALSE(invalid_result.has_value())
        << "Closing manual position should be rejected";

    Logger::getInstance().info("Attempted to close AAPL (manual) → REJECTED ✅");

    // Verify compliance log
    auto const& compliance_log = order_mgr_->getComplianceLog();
    EXPECT_GT(compliance_log.size(), 0) << "Should have compliance entries";

    Logger::getInstance().info("\n✅ COMPLETE WORKFLOW PASSED");
}

// ============================================================================
// Risk Manager Integration Test
// ============================================================================

TEST_F(SchwabE2EWorkflowTest, RiskManagerIntegration) {
    Logger::getInstance().info("\n=== RISK MANAGER INTEGRATION TEST ===");

    // Configure risk manager with custom limits
    RiskLimits custom_limits = RiskLimits::forThirtyKAccount();
    custom_limits.max_daily_loss = 900.0;
    custom_limits.max_position_size = 1500.0;
    custom_limits.max_portfolio_heat = 0.15;
    custom_limits.require_stop_loss = true;
    risk_mgr_->withLimits(custom_limits);

    // Test trade assessment for multiple symbols
    std::vector<std::string> symbols = {"XLE", "XLV", "XLI"};

    for (auto const& symbol : symbols) {
        double position_size = 20 * 80.00;
        auto assessment = risk_mgr_->assessTrade(
            symbol,
            position_size,
            80.00,   // entry price
            75.00,   // stop price
            90.00,   // target price
            0.65     // win probability
        );

        ASSERT_TRUE(assessment.has_value()) << "Assessment for " << symbol << " should succeed";

        Logger::getInstance().info(
            "Risk assessment for {}: Approved = {}, Position size = ${:.2f}, "
            "Max loss = ${:.2f}, Expected value = ${:.2f}",
            symbol, assessment->approved, assessment->position_size,
            assessment->max_loss, assessment->expected_value
        );

        EXPECT_TRUE(assessment->approved) << symbol << " should be approved";
        EXPECT_GT(assessment->expected_value, 0.0) << symbol << " should have positive EV";
    }

    Logger::getInstance().info("✅ RISK MANAGER INTEGRATION PASSED");
}

// ============================================================================
// Main
// ============================================================================

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);

    Logger::getInstance().info("╔════════════════════════════════════════════════════╗");
    Logger::getInstance().info("║  Schwab E2E Trading Workflow Test Suite          ║");
    Logger::getInstance().info("║  Testing safety constraints & compliance          ║");
    Logger::getInstance().info("╚════════════════════════════════════════════════════╝\n");

    return RUN_ALL_TESTS();
}
