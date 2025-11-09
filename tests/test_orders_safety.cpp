/**
 * BigBrotherAnalytics - Order Safety Tests (C++23)
 *
 * Test suite for CRITICAL manual position protection
 *
 * Tests:
 * 1. Manual position protection (bot CANNOT trade manual positions)
 * 2. Bot position management (bot CAN trade bot-managed positions)
 * 3. Position classification on startup
 * 4. Dry-run mode functionality
 * 5. Order validation
 * 6. Bracket orders
 * 7. Compliance logging
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <filesystem>

import bigbrother.schwab_api.orders;
import bigbrother.utils.logger;

using namespace bigbrother::schwab;
using namespace bigbrother::utils;
namespace fs = std::filesystem;

// Test counter
int tests_passed = 0;
int tests_failed = 0;

// Helper macros
#define TEST_START(name) \
    std::cout << "\n[TEST] " << name << std::endl; \
    std::cout << std::string(60, '-') << std::endl;

#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "FAILED: " << message << std::endl; \
        tests_failed++; \
        return false; \
    } \
    tests_passed++;

#define TEST_END() \
    std::cout << "PASSED" << std::endl; \
    return true;

// Test database path
constexpr auto TEST_DB_PATH = "data/test_orders_safety.duckdb";
constexpr auto TEST_ACCOUNT = "TEST_ACCOUNT_123";

/**
 * Test 1: Manual Position Protection
 * Bot MUST NOT trade existing manual positions
 */
auto test_manual_position_protection() -> bool {
    TEST_START("Manual Position Protection");

    // Create orders manager
    OrdersManager orders_mgr{TEST_DB_PATH, true};  // dry-run enabled
    PositionDatabase pos_db{TEST_DB_PATH};

    // Simulate existing manual position (AAPL)
    Position manual_pos;
    manual_pos.account_id = TEST_ACCOUNT;
    manual_pos.symbol = "AAPL";
    manual_pos.quantity = 10;
    manual_pos.avg_cost = 150.00;
    manual_pos.market_value = 1500.00;
    manual_pos.is_bot_managed = false;  // MANUAL
    manual_pos.managed_by = "MANUAL";
    manual_pos.opened_by = "MANUAL";
    manual_pos.opened_at = std::chrono::system_clock::now();

    auto insert_result = pos_db.insertPosition(manual_pos);
    TEST_ASSERT(insert_result.has_value(), "Failed to insert manual position");

    // Try to place order for AAPL (should be REJECTED)
    Order order;
    order.account_id = TEST_ACCOUNT;
    order.symbol = "AAPL";
    order.side = OrderSide::Buy;
    order.quantity = 5;
    order.type = OrderType::Market;

    auto order_result = orders_mgr.placeOrder(order);
    TEST_ASSERT(!order_result.has_value(), "Order should be REJECTED for manual position");
    TEST_ASSERT(order_result.error().code == ErrorCode::InvalidOperation,
               "Should be InvalidOperation error");

    std::cout << "✓ Bot correctly rejected order for manual position (AAPL)" << std::endl;

    TEST_END();
}

/**
 * Test 2: Bot Position Management
 * Bot CAN trade bot-managed positions
 */
auto test_bot_position_management() -> bool {
    TEST_START("Bot Position Management");

    OrdersManager orders_mgr{TEST_DB_PATH, true};
    PositionDatabase pos_db{TEST_DB_PATH};

    // Create bot-managed position (XLE)
    Position bot_pos;
    bot_pos.account_id = TEST_ACCOUNT;
    bot_pos.symbol = "XLE";
    bot_pos.quantity = 20;
    bot_pos.avg_cost = 80.00;
    bot_pos.market_value = 1600.00;
    bot_pos.is_bot_managed = true;  // BOT-MANAGED
    bot_pos.managed_by = "BOT";
    bot_pos.bot_strategy = "SectorRotation";
    bot_pos.opened_by = "BOT";
    bot_pos.opened_at = std::chrono::system_clock::now();

    auto insert_result = pos_db.insertPosition(bot_pos);
    TEST_ASSERT(insert_result.has_value(), "Failed to insert bot position");

    // Try to place order for XLE (should be ACCEPTED)
    Order order;
    order.account_id = TEST_ACCOUNT;
    order.symbol = "XLE";
    order.side = OrderSide::Sell;
    order.quantity = 10;
    order.type = OrderType::Market;

    auto order_result = orders_mgr.placeOrder(order);
    TEST_ASSERT(order_result.has_value(), "Order should be ACCEPTED for bot position");
    TEST_ASSERT(order_result->dry_run == true, "Should be in dry-run mode");

    std::cout << "✓ Bot correctly accepted order for bot-managed position (XLE)" << std::endl;

    TEST_END();
}

/**
 * Test 3: New Security Trading
 * Bot CAN trade NEW securities (not in portfolio)
 */
auto test_new_security_trading() -> bool {
    TEST_START("New Security Trading");

    OrdersManager orders_mgr{TEST_DB_PATH, true};
    PositionDatabase pos_db{TEST_DB_PATH};

    // Try to place order for NEW security (SPY - not in portfolio)
    Order order;
    order.account_id = TEST_ACCOUNT;
    order.symbol = "SPY";
    order.side = OrderSide::Buy;
    order.quantity = 10;
    order.type = OrderType::Limit;
    order.limit_price = 580.00;

    auto order_result = orders_mgr.placeOrder(order);
    TEST_ASSERT(order_result.has_value(), "Order should be ACCEPTED for new security");
    TEST_ASSERT(order_result->symbol == "SPY", "Symbol should match");
    TEST_ASSERT(order_result->quantity == 10, "Quantity should match");

    std::cout << "✓ Bot correctly accepted order for new security (SPY)" << std::endl;

    TEST_END();
}

/**
 * Test 4: Position Classification
 * On startup, classify existing positions as MANUAL
 */
auto test_position_classification() -> bool {
    TEST_START("Position Classification on Startup");

    OrdersManager orders_mgr{TEST_DB_PATH, true};

    // Simulate positions from Schwab API (pre-existing holdings)
    std::vector<Position> schwab_positions;

    Position msft;
    msft.account_id = TEST_ACCOUNT;
    msft.symbol = "MSFT";
    msft.quantity = 5;
    msft.avg_cost = 300.00;
    schwab_positions.push_back(msft);

    Position googl;
    googl.account_id = TEST_ACCOUNT;
    googl.symbol = "GOOGL";
    googl.quantity = 3;
    googl.avg_cost = 140.00;
    schwab_positions.push_back(googl);

    // Classify positions
    auto classify_result = orders_mgr.classifyExistingPositions(
        TEST_ACCOUNT,
        schwab_positions
    );
    TEST_ASSERT(classify_result.has_value(), "Position classification failed");

    // Verify positions are marked as MANUAL
    auto msft_pos = orders_mgr.getPosition(TEST_ACCOUNT, "MSFT");
    TEST_ASSERT(msft_pos.has_value(), "MSFT position not found");
    TEST_ASSERT(msft_pos->is_bot_managed == false, "MSFT should be MANUAL");
    TEST_ASSERT(msft_pos->managed_by == "MANUAL", "MSFT should be managed by MANUAL");

    auto googl_pos = orders_mgr.getPosition(TEST_ACCOUNT, "GOOGL");
    TEST_ASSERT(googl_pos.has_value(), "GOOGL position not found");
    TEST_ASSERT(googl_pos->is_bot_managed == false, "GOOGL should be MANUAL");

    std::cout << "✓ Positions correctly classified as MANUAL on startup" << std::endl;

    TEST_END();
}

/**
 * Test 5: Dry-Run Mode
 * Verify dry-run mode prevents real orders
 */
auto test_dry_run_mode() -> bool {
    TEST_START("Dry-Run Mode");

    OrdersManager orders_mgr{TEST_DB_PATH, true};  // dry-run ENABLED

    TEST_ASSERT(orders_mgr.isDryRunMode(), "Dry-run should be enabled");

    Order order;
    order.account_id = TEST_ACCOUNT;
    order.symbol = "QQQ";
    order.side = OrderSide::Buy;
    order.quantity = 15;
    order.type = OrderType::Market;

    auto order_result = orders_mgr.placeOrder(order);
    TEST_ASSERT(order_result.has_value(), "Order should succeed in dry-run");
    TEST_ASSERT(order_result->dry_run == true, "Order should be marked as dry-run");
    TEST_ASSERT(order_result->status == OrderStatus::Pending, "Status should be Pending");

    std::cout << "✓ Dry-run mode correctly prevents real order submission" << std::endl;

    // Disable dry-run
    orders_mgr.setDryRunMode(false);
    TEST_ASSERT(!orders_mgr.isDryRunMode(), "Dry-run should be disabled");

    std::cout << "✓ Dry-run mode can be toggled" << std::endl;

    TEST_END();
}

/**
 * Test 6: Bracket Order
 * Entry + Profit Target + Stop Loss
 */
auto test_bracket_order() -> bool {
    TEST_START("Bracket Order");

    OrdersManager orders_mgr{TEST_DB_PATH, true};

    // Create bracket order
    Order entry;
    entry.account_id = TEST_ACCOUNT;
    entry.symbol = "IWM";
    entry.side = OrderSide::Buy;
    entry.quantity = 10;
    entry.type = OrderType::Limit;
    entry.limit_price = 200.00;

    BracketOrder bracket;
    bracket.entry_order = entry;
    bracket.profit_target = 210.00;  // +$10 profit
    bracket.stop_loss = 195.00;      // -$5 stop

    auto bracket_result = orders_mgr.placeBracketOrder(bracket);
    TEST_ASSERT(bracket_result.has_value(), "Bracket order should succeed");
    TEST_ASSERT(bracket_result->size() == 3, "Bracket should have 3 orders");

    std::cout << "✓ Bracket order created: entry + profit + stop" << std::endl;

    TEST_END();
}

/**
 * Test 7: Close Position Safety
 * Bot can only close bot-managed positions
 */
auto test_close_position_safety() -> bool {
    TEST_START("Close Position Safety");

    OrdersManager orders_mgr{TEST_DB_PATH, true};
    PositionDatabase pos_db{TEST_DB_PATH};

    // Try to close manual position (AAPL) - should FAIL
    auto close_manual = orders_mgr.closePosition(TEST_ACCOUNT, "AAPL");
    TEST_ASSERT(!close_manual.has_value(), "Closing manual position should be REJECTED");
    TEST_ASSERT(close_manual.error().code == ErrorCode::InvalidOperation,
               "Should be InvalidOperation error");

    std::cout << "✓ Bot correctly rejected closing manual position (AAPL)" << std::endl;

    // Try to close bot position (XLE) - should SUCCEED
    auto close_bot = orders_mgr.closePosition(TEST_ACCOUNT, "XLE");
    TEST_ASSERT(close_bot.has_value(), "Closing bot position should be ACCEPTED");
    TEST_ASSERT(close_bot->symbol == "XLE", "Symbol should match");

    std::cout << "✓ Bot correctly accepted closing bot-managed position (XLE)" << std::endl;

    TEST_END();
}

/**
 * Test 8: Order Validation
 * Validate order parameters
 */
auto test_order_validation() -> bool {
    TEST_START("Order Validation");

    OrdersManager orders_mgr{TEST_DB_PATH, true};

    // Test 1: Invalid quantity (negative)
    Order invalid_qty;
    invalid_qty.account_id = TEST_ACCOUNT;
    invalid_qty.symbol = "TEST";
    invalid_qty.quantity = -5;  // INVALID

    auto result1 = orders_mgr.placeOrder(invalid_qty);
    TEST_ASSERT(!result1.has_value(), "Negative quantity should be rejected");

    // Test 2: Invalid limit price
    Order invalid_price;
    invalid_price.account_id = TEST_ACCOUNT;
    invalid_price.symbol = "TEST";
    invalid_price.quantity = 10;
    invalid_price.type = OrderType::Limit;
    invalid_price.limit_price = 0.0;  // INVALID

    auto result2 = orders_mgr.placeOrder(invalid_price);
    TEST_ASSERT(!result2.has_value(), "Zero limit price should be rejected");

    std::cout << "✓ Order validation correctly rejects invalid orders" << std::endl;

    TEST_END();
}

/**
 * Test 9: Position Summary
 * Get summary of manual vs bot positions
 */
auto test_position_summary() -> bool {
    TEST_START("Position Summary");

    OrdersManager orders_mgr{TEST_DB_PATH, true};

    auto summary_result = orders_mgr.getPositionSummary(TEST_ACCOUNT);
    TEST_ASSERT(summary_result.has_value(), "Position summary should succeed");

    auto summary = *summary_result;
    TEST_ASSERT(summary.contains("manual_positions"), "Should have manual_positions");
    TEST_ASSERT(summary.contains("bot_managed_positions"), "Should have bot_managed_positions");

    int manual_count = summary["manual_positions"];
    int bot_count = summary["bot_managed_positions"];

    std::cout << "✓ Position Summary:" << std::endl;
    std::cout << "  Manual positions: " << manual_count << " (HANDS OFF)" << std::endl;
    std::cout << "  Bot-managed positions: " << bot_count << " (can trade)" << std::endl;

    TEST_END();
}

/**
 * Main test runner
 */
int main() {
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "SCHWAB API ORDERS - SAFETY TEST SUITE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode: DRY-RUN (No real orders)" << std::endl;
    std::cout << "Focus: Manual Position Protection" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Setup: Create test database directory
    fs::create_directories("data");

    // Remove old test database
    if (fs::exists(TEST_DB_PATH)) {
        fs::remove(TEST_DB_PATH);
        std::cout << "Removed old test database\n" << std::endl;
    }

    // Initialize logger
    Logger::getInstance().setLevel(Logger::Level::Info);

    // Run tests
    std::vector<std::pair<std::string, bool(*)()>> tests = {
        {"Manual Position Protection", test_manual_position_protection},
        {"Bot Position Management", test_bot_position_management},
        {"New Security Trading", test_new_security_trading},
        {"Position Classification", test_position_classification},
        {"Dry-Run Mode", test_dry_run_mode},
        {"Bracket Order", test_bracket_order},
        {"Close Position Safety", test_close_position_safety},
        {"Order Validation", test_order_validation},
        {"Position Summary", test_position_summary}
    };

    int total_tests = tests.size();

    for (auto const& [name, test_fn] : tests) {
        try {
            if (!test_fn()) {
                std::cerr << "TEST FAILED: " << name << std::endl;
            }
        } catch (std::exception const& e) {
            std::cerr << "TEST EXCEPTION: " << name << " - " << e.what() << std::endl;
            tests_failed++;
        }
    }

    // Print summary
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "TEST RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total Tests: " << total_tests << std::endl;
    std::cout << "Assertions Passed: " << tests_passed << std::endl;
    std::cout << "Assertions Failed: " << tests_failed << std::endl;

    if (tests_failed == 0) {
        std::cout << "\n✓ ALL TESTS PASSED" << std::endl;
        std::cout << "========================================\n" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
        std::cout << "========================================\n" << std::endl;
        return 1;
    }
}
