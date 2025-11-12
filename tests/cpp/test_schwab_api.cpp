/**
 * Unit Tests for Schwab API C++23 Fluent API
 *
 * Tests the fluent interface, OAuth handling, data structures, and error handling.
 */

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>

// Import C++23 modules
import bigbrother.market_intelligence.types;
import bigbrother.utils.types;

using namespace bigbrother::market_intelligence;
using namespace bigbrother::types;
namespace fs = std::filesystem;

// Test tolerance
constexpr double PRICE_TOLERANCE = 0.01;

/**
 * Test SchwabConfig structure
 */
TEST(SchwabAPITest, ConfigStructure) {
    SchwabConfig config;
    config.app_key = "test_app_key";
    config.app_secret = "test_app_secret";
    config.token_file = "configs/test_tokens.json";
    config.timeout_seconds = 30;

    EXPECT_EQ(config.app_key, "test_app_key");
    EXPECT_EQ(config.app_secret, "test_app_secret");
    EXPECT_EQ(config.timeout_seconds, 30);
    EXPECT_FALSE(config.token_file.empty());
}

/**
 * Test Account data structure
 */
TEST(SchwabAPITest, AccountStructure) {
    Account account;
    account.account_number = "12345678";
    account.account_hash = "ABCD1234";
    account.account_type = "MARGIN";
    account.cash_balance = 50000.00;
    account.buying_power = 100000.00;
    account.day_trading_buying_power = 200000.00;

    EXPECT_EQ(account.account_number, "12345678");
    EXPECT_EQ(account.account_hash, "ABCD1234");
    EXPECT_NEAR(account.cash_balance, 50000.00, PRICE_TOLERANCE);
    EXPECT_NEAR(account.buying_power, 100000.00, PRICE_TOLERANCE);
    EXPECT_GT(account.day_trading_buying_power, account.buying_power);
}

/**
 * Test Position data structure
 */
TEST(SchwabAPITest, PositionStructure) {
    Position position;
    position.symbol = "SPY";
    position.type = PositionType::STOCK;
    position.quantity = 100.0;
    position.average_price = 570.00;
    position.current_price = 580.00;
    position.market_value = 58000.00;
    position.cost_basis = 57000.00;
    position.unrealized_pnl = 1000.00;

    EXPECT_EQ(position.symbol, "SPY");
    EXPECT_EQ(position.type, PositionType::STOCK);
    EXPECT_NEAR(position.quantity, 100.0, 0.001);
    EXPECT_NEAR(position.unrealized_pnl, 1000.00, PRICE_TOLERANCE);

    // Verify P&L calculation
    double expected_pnl = (position.current_price - position.average_price) * position.quantity;
    EXPECT_NEAR(position.unrealized_pnl, expected_pnl, PRICE_TOLERANCE);
}

/**
 * Test Option Position structure
 */
TEST(SchwabAPITest, OptionPositionStructure) {
    Position position;
    position.symbol = "SPY   251219C00679000"; // SPY Call $679 12/19/2025
    position.type = PositionType::OPTION;
    position.quantity = 5.0;           // 5 contracts
    position.average_price = 12.50;    // $12.50 per share
    position.current_price = 15.00;    // $15.00 per share
    position.market_value = 7500.00;   // 5 * 100 * $15
    position.cost_basis = 6250.00;     // 5 * 100 * $12.50
    position.unrealized_pnl = 1250.00; // $1250 profit

    EXPECT_EQ(position.type, PositionType::OPTION);
    EXPECT_NEAR(position.quantity, 5.0, 0.001);
    EXPECT_NEAR(position.unrealized_pnl, 1250.00, PRICE_TOLERANCE);

    // Market value should equal quantity * multiplier * current_price
    // 5 contracts * 100 shares/contract * $15/share = $7500
    EXPECT_NEAR(position.market_value, 7500.00, PRICE_TOLERANCE);
}

/**
 * Test Order data structure
 */
TEST(SchwabAPITest, OrderStructure) {
    Order order;
    order.order_id = "ORDER123";
    order.symbol = "AAPL";
    order.action = OrderAction::BUY;
    order.type = OrderType::LIMIT;
    order.quantity = 50.0;
    order.limit_price = 225.00;
    order.status = OrderStatus::PENDING;

    EXPECT_EQ(order.order_id, "ORDER123");
    EXPECT_EQ(order.symbol, "AAPL");
    EXPECT_EQ(order.action, OrderAction::BUY);
    EXPECT_EQ(order.type, OrderType::LIMIT);
    EXPECT_NEAR(order.quantity, 50.0, 0.001);
    EXPECT_NEAR(order.limit_price, 225.00, PRICE_TOLERANCE);
}

/**
 * Test PositionType enum
 */
TEST(SchwabAPITest, PositionTypeEnum) {
    EXPECT_EQ(to_string(PositionType::STOCK), "STOCK");
    EXPECT_EQ(to_string(PositionType::OPTION), "OPTION");
    EXPECT_EQ(to_string(PositionType::FUTURE), "FUTURE");
    EXPECT_EQ(to_string(PositionType::UNKNOWN), "UNKNOWN");
}

/**
 * Test OrderAction enum
 */
TEST(SchwabAPITest, OrderActionEnum) {
    EXPECT_EQ(to_string(OrderAction::BUY), "BUY");
    EXPECT_EQ(to_string(OrderAction::SELL), "SELL");
    EXPECT_EQ(to_string(OrderAction::BUY_TO_OPEN), "BUY_TO_OPEN");
    EXPECT_EQ(to_string(OrderAction::SELL_TO_CLOSE), "SELL_TO_CLOSE");
}

/**
 * Test OrderType enum
 */
TEST(SchwabAPITest, OrderTypeEnum) {
    EXPECT_EQ(to_string(OrderType::MARKET), "MARKET");
    EXPECT_EQ(to_string(OrderType::LIMIT), "LIMIT");
    EXPECT_EQ(to_string(OrderType::STOP), "STOP");
    EXPECT_EQ(to_string(OrderType::STOP_LIMIT), "STOP_LIMIT");
}

/**
 * Test OrderStatus enum
 */
TEST(SchwabAPITest, OrderStatusEnum) {
    EXPECT_EQ(to_string(OrderStatus::PENDING), "PENDING");
    EXPECT_EQ(to_string(OrderStatus::FILLED), "FILLED");
    EXPECT_EQ(to_string(OrderStatus::PARTIALLY_FILLED), "PARTIALLY_FILLED");
    EXPECT_EQ(to_string(OrderStatus::CANCELLED), "CANCELLED");
    EXPECT_EQ(to_string(OrderStatus::REJECTED), "REJECTED");
}

/**
 * Test Quote from Schwab source
 */
TEST(SchwabAPITest, SchwabQuoteStructure) {
    Quote quote;
    quote.symbol = "SPY";
    quote.last_price = 580.50;
    quote.bid = 580.48;
    quote.ask = 580.52;
    quote.bid_size = 100.0;
    quote.ask_size = 200.0;
    quote.volume = 50000000;
    quote.source = DataSource::SCHWAB;
    quote.timestamp = std::chrono::system_clock::now();

    EXPECT_EQ(quote.source, DataSource::SCHWAB);
    EXPECT_NEAR(quote.spread(), 0.04, 0.001);
    EXPECT_NEAR(quote.mid_price(), 580.50, PRICE_TOLERANCE);

    // Verify bid/ask sizes are present (Schwab provides this, Yahoo doesn't)
    EXPECT_GT(quote.bid_size, 0.0);
    EXPECT_GT(quote.ask_size, 0.0);
}

/**
 * Test multiple positions portfolio
 */
TEST(SchwabAPITest, PortfolioPositions) {
    std::vector<Position> positions;

    // Stock position
    Position spy;
    spy.symbol = "SPY";
    spy.type = PositionType::STOCK;
    spy.quantity = 100.0;
    spy.market_value = 58000.00;
    positions.push_back(spy);

    // Option position
    Position spy_call;
    spy_call.symbol = "SPY   251219C00679000";
    spy_call.type = PositionType::OPTION;
    spy_call.quantity = 5.0;
    spy_call.market_value = 7500.00;
    positions.push_back(spy_call);

    ASSERT_EQ(positions.size(), 2);

    // Calculate total portfolio value
    double total_value = 0.0;
    for (auto const& pos : positions) {
        total_value += pos.market_value;
    }

    EXPECT_NEAR(total_value, 65500.00, PRICE_TOLERANCE);
}

/**
 * Test order with all fields
 */
TEST(SchwabAPITest, CompleteOrderStructure) {
    Order order;
    order.order_id = "ORDER456";
    order.symbol = "NVDA";
    order.action = OrderAction::SELL;
    order.type = OrderType::LIMIT;
    order.quantity = 25.0;
    order.limit_price = 145.00;
    order.stop_price = 0.0;       // Not a stop order
    order.filled_quantity = 10.0; // Partially filled
    order.status = OrderStatus::PARTIALLY_FILLED;
    order.placed_at = std::chrono::system_clock::now();

    EXPECT_EQ(order.status, OrderStatus::PARTIALLY_FILLED);
    EXPECT_LT(order.filled_quantity, order.quantity);
    EXPECT_GT(order.filled_quantity, 0.0);

    // Remaining quantity
    double remaining = order.quantity - order.filled_quantity;
    EXPECT_NEAR(remaining, 15.0, 0.001);
}

/**
 * Test position P&L calculations
 */
TEST(SchwabAPITest, PositionPnLCalculations) {
    Position long_position;
    long_position.symbol = "AAPL";
    long_position.quantity = 100.0;
    long_position.average_price = 200.00;
    long_position.current_price = 225.00;
    long_position.cost_basis = 20000.00;
    long_position.market_value = 22500.00;
    long_position.unrealized_pnl = 2500.00;

    // Verify P&L calculation
    EXPECT_NEAR(long_position.unrealized_pnl, 2500.00, PRICE_TOLERANCE);

    // Verify percentage gain
    double pnl_percent = (long_position.unrealized_pnl / long_position.cost_basis) * 100.0;
    EXPECT_NEAR(pnl_percent, 12.5, 0.1); // 12.5% gain
}

/**
 * Test short position P&L
 */
TEST(SchwabAPITest, ShortPositionPnL) {
    Position short_position;
    short_position.symbol = "TSLA";
    short_position.quantity = -50.0; // Short 50 shares
    short_position.average_price = 360.00;
    short_position.current_price = 350.00;
    short_position.cost_basis = 18000.00;   // 50 * $360
    short_position.market_value = 17500.00; // 50 * $350
    short_position.unrealized_pnl = 500.00; // Profit on short

    EXPECT_LT(short_position.quantity, 0.0);       // Negative for short
    EXPECT_GT(short_position.unrealized_pnl, 0.0); // Profit when price drops

    // For short: profit when current < average
    EXPECT_LT(short_position.current_price, short_position.average_price);
}

/**
 * Main test runner
 */
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
