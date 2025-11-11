/**
 * BigBrotherAnalytics - Schwab API Orders Manager (C++23)
 *
 * CRITICAL SAFETY: Manual Position Protection
 * - Bot ONLY trades NEW securities (not in portfolio)
 * - Bot ONLY manages positions IT created (is_bot_managed = true)
 * - Pre-flight checks BEFORE every order
 * - Dry-run mode for testing (default: enabled)
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

// Global module fragment
module;

#include <chrono>
#include <expected>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

// DuckDB bridge - isolates DuckDB incomplete types from C++23 modules
#include "schwab_api/duckdb_bridge.hpp"

// Module declaration
export module bigbrother.schwab_api.orders;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Forward declaration for OrderManager from parent module
export namespace bigbrother::schwab {
class OrderManager;
}

export namespace bigbrother::schwab {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using json = nlohmann::json;

// ============================================================================
// Position Safety Types
// ============================================================================

/**
 * Position with safety flags for manual position protection
 */
struct Position {
    std::string account_id;
    std::string symbol;
    Quantity quantity{0.0};
    double avg_cost{0.0};
    double current_price{0.0};
    double market_value{0.0};
    double unrealized_pnl{0.0};

    // CRITICAL SAFETY FLAGS
    bool is_bot_managed{false};       // TRUE if bot opened this position
    std::string managed_by{"MANUAL"}; // "BOT" or "MANUAL"
    std::string bot_strategy{};       // Strategy that opened this (if bot-managed)

    std::chrono::system_clock::time_point opened_at;
    std::string opened_by{"MANUAL"}; // "BOT" or "MANUAL"
    std::chrono::system_clock::time_point updated_at;

    [[nodiscard]] auto canBotTrade() const noexcept -> bool { return is_bot_managed; }
};

/**
 * Order types
 */
enum class OrderSide { Buy, Sell, SellShort, BuyToCover };

enum class OrderType { Market, Limit, Stop, StopLimit, TrailingStop };

enum class OrderStatus { Pending, Working, Filled, PartiallyFilled, Canceled, Rejected };

enum class OrderDuration {
    Day,
    GTC, // Good Till Canceled
    GTD, // Good Till Date
    FOK, // Fill Or Kill
    IOC  // Immediate Or Cancel
};

/**
 * Order structure
 */
struct Order {
    std::string order_id;
    std::string account_id;
    std::string symbol;
    OrderSide side{OrderSide::Buy};
    Quantity quantity{0.0};
    Quantity filled_quantity{0.0};
    OrderType type{OrderType::Market};
    OrderDuration duration{OrderDuration::Day};

    double limit_price{0.0};
    double stop_price{0.0};
    double trail_amount{0.0};
    double avg_fill_price{0.0};

    OrderStatus status{OrderStatus::Pending};
    std::string strategy_name;
    bool dry_run{true}; // Default to dry-run for safety

    std::string rejection_reason;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::chrono::system_clock::time_point filled_at;

    [[nodiscard]] auto estimatedCost() const noexcept -> double {
        double price = (type == OrderType::Market) ? limit_price : limit_price;
        return quantity * price;
    }

    [[nodiscard]] auto isActive() const noexcept -> bool {
        return status == OrderStatus::Pending || status == OrderStatus::Working ||
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
    Quantity quantity{0.0};
    Quantity filled_quantity{0.0};
    double avg_fill_price{0.0};
    OrderStatus status;
    std::string strategy_name;
    bool dry_run{false};
    std::chrono::system_clock::time_point timestamp;
};

/**
 * Bracket order (Entry + Profit + Stop)
 */
struct BracketOrder {
    Order entry_order;
    double profit_target{0.0};
    double stop_loss{0.0};
};

// ============================================================================
// Position Database Manager
// ============================================================================

/**
 * Position database for tracking and safety checks
 */
class PositionDatabase {
  public:
    explicit PositionDatabase(std::string db_path) : db_path_{std::move(db_path)} {

        // Open DuckDB connection using bridge library
        db_ = duckdb_bridge::openDatabase(db_path_);
        conn_ = duckdb_bridge::createConnection(*db_);

        // Create schema
        createSchema();
    }

    // Rule of Five - non-copyable due to mutex, move operations defined separately
    PositionDatabase(PositionDatabase const&) = delete;
    auto operator=(PositionDatabase const&) -> PositionDatabase& = delete;
    PositionDatabase(PositionDatabase&&) noexcept = delete;
    auto operator=(PositionDatabase&&) noexcept -> PositionDatabase& = delete;
    ~PositionDatabase();

    /**
     * Query position by symbol (CRITICAL for safety checks)
     */
    [[nodiscard]] auto queryPosition(std::string const& account_id, std::string const& symbol)
        -> std::optional<Position> {

        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query =
                "SELECT account_id, symbol, quantity, avg_cost, current_price, "
                "market_value, unrealized_pnl, is_bot_managed, managed_by, "
                "bot_strategy, opened_at, opened_by, updated_at "
                "FROM positions "
                "WHERE account_id = '" +
                account_id + "' AND symbol = '" + symbol + "'";

            auto result = duckdb_bridge::executeQueryWithResults(*conn_, query);

            if (!result || duckdb_bridge::getRowCount(*result) == 0) {
                return std::nullopt;
            }

            Position pos;
            pos.account_id = duckdb_bridge::getValueAsString(*result, 0, 0);
            pos.symbol = duckdb_bridge::getValueAsString(*result, 1, 0);
            pos.quantity = duckdb_bridge::getValueAsDouble(*result, 2, 0);
            pos.avg_cost = duckdb_bridge::getValueAsDouble(*result, 3, 0);
            pos.current_price = duckdb_bridge::getValueAsDouble(*result, 4, 0);
            pos.market_value = duckdb_bridge::getValueAsDouble(*result, 5, 0);
            pos.unrealized_pnl = duckdb_bridge::getValueAsDouble(*result, 6, 0);
            pos.is_bot_managed = duckdb_bridge::getValueAsBool(*result, 7, 0);
            pos.managed_by = duckdb_bridge::getValueAsString(*result, 8, 0);
            pos.bot_strategy = duckdb_bridge::getValueAsString(*result, 9, 0);
            // Note: timestamps would need proper parsing

            return pos;

        } catch (std::exception const& e) {
            Logger::getInstance().error("Failed to query position: {}", e.what());
            return std::nullopt;
        }
    }

    /**
     * Insert new position (marked as bot-managed)
     */
    [[nodiscard]] auto insertPosition(Position const& pos) -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query =
                "INSERT INTO positions ("
                "account_id, symbol, quantity, avg_cost, current_price, "
                "market_value, unrealized_pnl, is_bot_managed, managed_by, "
                "bot_strategy, opened_at, opened_by, updated_at"
                ") VALUES ('" +
                pos.account_id + "', '" + pos.symbol + "', " + std::to_string(pos.quantity) + ", " +
                std::to_string(pos.avg_cost) + ", " + std::to_string(pos.current_price) + ", " +
                std::to_string(pos.market_value) + ", " + std::to_string(pos.unrealized_pnl) +
                ", " + std::to_string(pos.is_bot_managed) + ", '" + pos.managed_by + "', '" +
                pos.bot_strategy + "', CURRENT_TIMESTAMP, '" + pos.opened_by +
                "', CURRENT_TIMESTAMP)";

            duckdb_bridge::executeQuery(*conn_, query);

            Logger::getInstance().info("Inserted position: {} ({} managed)", pos.symbol,
                                       pos.managed_by);
            return {};

        } catch (std::exception const& e) {
            return makeError<void>(ErrorCode::DatabaseError,
                                   std::string("Failed to insert position: ") + e.what());
        }
    }

    /**
     * Update existing position
     */
    [[nodiscard]] auto updatePosition(Position const& pos) -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query =
                "UPDATE positions SET "
                "quantity = " +
                std::to_string(pos.quantity) + ", avg_cost = " + std::to_string(pos.avg_cost) +
                ", current_price = " + std::to_string(pos.current_price) + ", market_value = " +
                std::to_string(pos.market_value) + ", unrealized_pnl = " +
                std::to_string(pos.unrealized_pnl) +
                ", updated_at = CURRENT_TIMESTAMP "
                "WHERE account_id = '" +
                pos.account_id + "' AND symbol = '" + pos.symbol + "'";

            duckdb_bridge::executeQuery(*conn_, query);

            return {};

        } catch (std::exception const& e) {
            return makeError<void>(ErrorCode::DatabaseError,
                                   std::string("Failed to update position: ") + e.what());
        }
    }

    /**
     * Delete position
     */
    [[nodiscard]] auto deletePosition(std::string const& account_id, std::string const& symbol)
        -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query = "DELETE FROM positions WHERE account_id = '" + account_id +
                                "' AND symbol = '" + symbol + "'";

            duckdb_bridge::executeQuery(*conn_, query);

            return {};

        } catch (std::exception const& e) {
            return makeError<void>(ErrorCode::DatabaseError,
                                   std::string("Failed to delete position: ") + e.what());
        }
    }

    /**
     * Get all positions for account
     */
    [[nodiscard]] auto getAllPositions(std::string const& account_id)
        -> Result<std::vector<Position>> {

        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query =
                "SELECT account_id, symbol, quantity, avg_cost, current_price, "
                "market_value, unrealized_pnl, is_bot_managed, managed_by, "
                "bot_strategy "
                "FROM positions "
                "WHERE account_id = '" +
                account_id + "'";

            auto result = duckdb_bridge::executeQueryWithResults(*conn_, query);

            if (!result) {
                return makeError<std::vector<Position>>(ErrorCode::DatabaseError,
                                                         "Failed to execute query");
            }

            std::vector<Position> positions;
            size_t row_count = duckdb_bridge::getRowCount(*result);
            for (size_t i = 0; i < row_count; ++i) {
                Position pos;
                pos.account_id = duckdb_bridge::getValueAsString(*result, 0, i);
                pos.symbol = duckdb_bridge::getValueAsString(*result, 1, i);
                pos.quantity = duckdb_bridge::getValueAsDouble(*result, 2, i);
                pos.avg_cost = duckdb_bridge::getValueAsDouble(*result, 3, i);
                pos.current_price = duckdb_bridge::getValueAsDouble(*result, 4, i);
                pos.market_value = duckdb_bridge::getValueAsDouble(*result, 5, i);
                pos.unrealized_pnl = duckdb_bridge::getValueAsDouble(*result, 6, i);
                pos.is_bot_managed = duckdb_bridge::getValueAsBool(*result, 7, i);
                pos.managed_by = duckdb_bridge::getValueAsString(*result, 8, i);
                pos.bot_strategy = duckdb_bridge::getValueAsString(*result, 9, i);

                positions.push_back(pos);
            }

            return positions;

        } catch (std::exception const& e) {
            return makeError<std::vector<Position>>(
                ErrorCode::DatabaseError, std::string("Failed to get positions: ") + e.what());
        }
    }

  private:
    auto createSchema() -> void {
        try {
            std::string query =
                "CREATE TABLE IF NOT EXISTS positions ("
                "id INTEGER PRIMARY KEY, "
                "account_id VARCHAR(50) NOT NULL, "
                "symbol VARCHAR(20) NOT NULL, "
                "quantity DOUBLE NOT NULL, "
                "avg_cost DECIMAL(10,2) NOT NULL, "
                "current_price DECIMAL(10,2), "
                "market_value DECIMAL(10,2), "
                "unrealized_pnl DECIMAL(10,2), "
                "is_bot_managed BOOLEAN DEFAULT FALSE, "
                "managed_by VARCHAR(20) DEFAULT 'MANUAL', "
                "bot_strategy VARCHAR(50), "
                "opened_at TIMESTAMP NOT NULL, "
                "opened_by VARCHAR(20) DEFAULT 'MANUAL', "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE(account_id, symbol)"
                ")";

            duckdb_bridge::executeQuery(*conn_, query);

            Logger::getInstance().info("Position database schema created");

        } catch (std::exception const& e) {
            Logger::getInstance().error("Failed to create schema: {}", e.what());
        }
    }

    std::string db_path_;
    std::unique_ptr<duckdb_bridge::DatabaseHandle> db_;
    std::unique_ptr<duckdb_bridge::ConnectionHandle> conn_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Order Database Logger
// ============================================================================

/**
 * Order logging for compliance and audit trail
 */
class OrderDatabaseLogger {
  public:
    explicit OrderDatabaseLogger(std::string db_path) : db_path_{std::move(db_path)} {

        db_ = duckdb_bridge::openDatabase(db_path_);
        conn_ = duckdb_bridge::createConnection(*db_);

        // Schema is created by SQL script
    }

    OrderDatabaseLogger(OrderDatabaseLogger const&) = delete;
    auto operator=(OrderDatabaseLogger const&) -> OrderDatabaseLogger& = delete;
    OrderDatabaseLogger(OrderDatabaseLogger&&) noexcept = delete;
    auto operator=(OrderDatabaseLogger&&) noexcept -> OrderDatabaseLogger& = delete;
    ~OrderDatabaseLogger();

    /**
     * Log order to database (COMPLIANCE REQUIREMENT)
     */
    [[nodiscard]] auto logOrder(Order const& order) -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query =
                "INSERT INTO orders ("
                "order_id, account_id, symbol, side, quantity, "
                "filled_quantity, order_type, limit_price, stop_price, "
                "trail_amount, avg_fill_price, status, duration, "
                "dry_run, created_at"
                ") VALUES ('" +
                order.order_id + "', '" + order.account_id + "', '" + order.symbol + "', " +
                std::to_string(static_cast<int>(order.side)) + ", " +
                std::to_string(order.quantity) + ", " + std::to_string(order.filled_quantity) +
                ", " + std::to_string(static_cast<int>(order.type)) + ", " +
                std::to_string(order.limit_price) + ", " + std::to_string(order.stop_price) +
                ", " + std::to_string(order.trail_amount) + ", " +
                std::to_string(order.avg_fill_price) + ", " +
                std::to_string(static_cast<int>(order.status)) + ", " +
                std::to_string(static_cast<int>(order.duration)) + ", " +
                std::to_string(order.dry_run) + ", CURRENT_TIMESTAMP)";

            duckdb_bridge::executeQuery(*conn_, query);

            Logger::getInstance().info("Order logged: {} ({} mode)", order.order_id,
                                       order.dry_run ? "DRY-RUN" : "LIVE");
            return {};

        } catch (std::exception const& e) {
            return makeError<void>(ErrorCode::DatabaseError,
                                   std::string("Failed to log order: ") + e.what());
        }
    }

    /**
     * Update order status
     */
    [[nodiscard]] auto updateOrderStatus(std::string const& order_id, OrderStatus new_status)
        -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            std::string query = "UPDATE orders SET status = " +
                                std::to_string(static_cast<int>(new_status)) +
                                ", updated_at = CURRENT_TIMESTAMP WHERE order_id = '" + order_id +
                                "'";

            duckdb_bridge::executeQuery(*conn_, query);

            return {};

        } catch (std::exception const& e) {
            return makeError<void>(ErrorCode::DatabaseError,
                                   std::string("Failed to update order status: ") + e.what());
        }
    }

  private:
    std::string db_path_;
    std::unique_ptr<duckdb_bridge::DatabaseHandle> db_;
    std::unique_ptr<duckdb_bridge::ConnectionHandle> conn_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Orders Manager (Main Class)
// ============================================================================

/**
 * Schwab API Orders Manager with CRITICAL safety features
 */
class OrdersManager {
  public:
    explicit OrdersManager(std::string db_path, bool enable_dry_run = true)
        : position_db_{db_path}, order_logger_{db_path}, dry_run_mode_{enable_dry_run},
          order_counter_{0} {

        Logger::getInstance().info("OrdersManager initialized (dry-run: {})", dry_run_mode_);
    }

    OrdersManager(OrdersManager const&) = delete;
    auto operator=(OrdersManager const&) -> OrdersManager& = delete;
    OrdersManager(OrdersManager&&) noexcept = delete;
    auto operator=(OrdersManager&&) noexcept -> OrdersManager& = delete;
    ~OrdersManager();

    // ========================================================================
    // Configuration
    // ========================================================================

    /**
     * Enable/disable dry-run mode
     */
    auto setDryRunMode(bool enabled) -> void {
        dry_run_mode_ = enabled;
        Logger::getInstance().info("Dry-run mode: {}", enabled ? "ENABLED" : "DISABLED");
    }

    [[nodiscard]] auto isDryRunMode() const noexcept -> bool { return dry_run_mode_; }

    // ========================================================================
    // Position Classification (CRITICAL - Run on startup)
    // ========================================================================

    /**
     * Classify existing positions as MANUAL on startup
     * This prevents the bot from trading existing holdings
     */
    [[nodiscard]] auto classifyExistingPositions(std::string const& account_id,
                                                 std::vector<Position> const& schwab_positions)
        -> Result<void> {

        Logger::getInstance().info("Classifying existing positions for safety...");

        int manual_count = 0;
        int bot_count = 0;

        for (auto const& schwab_pos : schwab_positions) {
            auto local_pos = position_db_.queryPosition(account_id, schwab_pos.symbol);

            if (!local_pos) {
                // Position exists in Schwab but not in our DB
                // = Existing manual position, DO NOT TOUCH
                Position manual_pos = schwab_pos;
                manual_pos.is_bot_managed = false;
                manual_pos.managed_by = "MANUAL";
                manual_pos.opened_by = "MANUAL";

                auto insert_result = position_db_.insertPosition(manual_pos);
                if (insert_result) {
                    manual_count++;
                    Logger::getInstance().info("Classified {} as MANUAL position (DO NOT TOUCH)",
                                               schwab_pos.symbol);
                }
            } else {
                // Position already in DB
                if (local_pos->is_bot_managed) {
                    bot_count++;
                } else {
                    manual_count++;
                }
            }
        }

        Logger::getInstance().info("Position classification complete:");
        Logger::getInstance().info("  Manual positions: {} (HANDS OFF)", manual_count);
        Logger::getInstance().info("  Bot-managed positions: {} (can trade)", bot_count);

        return {};
    }

    // ========================================================================
    // Order Placement (with PRE-FLIGHT SAFETY CHECKS)
    // ========================================================================

    /**
     * Place order with CRITICAL safety checks
     */
    [[nodiscard]] auto placeOrder(Order order) -> Result<OrderConfirmation> {

        // CRITICAL: Check if symbol is already held as manual position
        auto position = position_db_.queryPosition(order.account_id, order.symbol);

        if (position && !position->is_bot_managed) {
            return makeError<OrderConfirmation>(
                ErrorCode::RuntimeError,
                std::format("SAFETY VIOLATION: Cannot trade {} - manual position exists. "
                            "Bot only trades NEW securities or bot-managed positions.",
                            order.symbol));
        }

        // Generate order ID
        order.order_id = "ORD_" + std::to_string(++order_counter_);
        order.created_at = std::chrono::system_clock::now();
        order.dry_run = dry_run_mode_;

        // Validate order parameters
        auto validation_result = validateOrder(order);
        if (!validation_result) {
            return std::unexpected(validation_result.error());
        }

        // Log order for compliance (BEFORE submission)
        auto log_result = order_logger_.logOrder(order);
        if (!log_result) {
            Logger::getInstance().warn("Failed to log order: {}", log_result.error().message);
        }

        // Submit order (or simulate in dry-run mode)
        OrderConfirmation confirmation;

        if (dry_run_mode_) {
            // DRY-RUN: Simulate order placement
            Logger::getInstance().info(
                "DRY-RUN: Would place order for {} {} {} @ {}", order.quantity, order.symbol,
                order.side == OrderSide::Buy ? "BUY" : "SELL", order.limit_price);

            confirmation = OrderConfirmation{.order_id = order.order_id,
                                             .symbol = order.symbol,
                                             .side = order.side,
                                             .quantity = order.quantity,
                                             .filled_quantity = 0.0,
                                             .avg_fill_price = 0.0,
                                             .status = OrderStatus::Pending,
                                             .strategy_name = order.strategy_name,
                                             .dry_run = true,
                                             .timestamp = order.created_at};
        } else {
            // LIVE: Submit to Schwab API
            // TODO: Implement actual Schwab API call
            Logger::getInstance().info("LIVE: Placing order for {} {} {}", order.quantity,
                                       order.symbol, order.side == OrderSide::Buy ? "BUY" : "SELL");

            confirmation = OrderConfirmation{.order_id = order.order_id,
                                             .symbol = order.symbol,
                                             .side = order.side,
                                             .quantity = order.quantity,
                                             .filled_quantity = 0.0,
                                             .avg_fill_price = 0.0,
                                             .status = OrderStatus::Working,
                                             .strategy_name = order.strategy_name,
                                             .dry_run = false,
                                             .timestamp = order.created_at};
        }

        return confirmation;
    }

    /**
     * Place bracket order (Entry + Profit Target + Stop Loss)
     */
    [[nodiscard]] auto placeBracketOrder(BracketOrder const& bracket)
        -> Result<std::vector<OrderConfirmation>> {

        // Check manual position protection
        auto position =
            position_db_.queryPosition(bracket.entry_order.account_id, bracket.entry_order.symbol);

        if (position && !position->is_bot_managed) {
            return makeError<std::vector<OrderConfirmation>>(
                ErrorCode::RuntimeError, "Cannot place bracket order - manual position exists");
        }

        std::vector<OrderConfirmation> confirmations;

        // Place entry order
        auto entry_result = placeOrder(bracket.entry_order);
        if (!entry_result) {
            return std::unexpected(entry_result.error());
        }
        confirmations.push_back(*entry_result);

        // Place profit target order (linked to entry)
        Order profit_order = bracket.entry_order;
        profit_order.side =
            (bracket.entry_order.side == OrderSide::Buy) ? OrderSide::Sell : OrderSide::BuyToCover;
        profit_order.type = OrderType::Limit;
        profit_order.limit_price = bracket.profit_target;

        auto profit_result = placeOrder(profit_order);
        if (profit_result) {
            confirmations.push_back(*profit_result);
        }

        // Place stop-loss order (linked to entry)
        Order stop_order = bracket.entry_order;
        stop_order.side =
            (bracket.entry_order.side == OrderSide::Buy) ? OrderSide::Sell : OrderSide::BuyToCover;
        stop_order.type = OrderType::Stop;
        stop_order.stop_price = bracket.stop_loss;

        auto stop_result = placeOrder(stop_order);
        if (stop_result) {
            confirmations.push_back(*stop_result);
        }

        Logger::getInstance().info("Bracket order placed: {} (entry + profit + stop)",
                                   bracket.entry_order.symbol);

        return confirmations;
    }

    // ========================================================================
    // Position Management
    // ========================================================================

    /**
     * Handle order fill - Create/update position
     */
    [[nodiscard]] auto onOrderFilled(OrderConfirmation const& confirmation) -> Result<void> {

        if (confirmation.side == OrderSide::Buy) {
            // Opening or adding to position
            auto existing = position_db_.queryPosition("", // account_id from confirmation
                                                       confirmation.symbol);

            if (!existing) {
                // New position - mark as bot-managed
                Position new_pos;
                new_pos.symbol = confirmation.symbol;
                new_pos.quantity = confirmation.filled_quantity;
                new_pos.avg_cost = confirmation.avg_fill_price;
                new_pos.market_value = confirmation.filled_quantity * confirmation.avg_fill_price;
                new_pos.is_bot_managed = true;
                new_pos.managed_by = "BOT";
                new_pos.bot_strategy = confirmation.strategy_name;
                new_pos.opened_by = "BOT";
                new_pos.opened_at = confirmation.timestamp;

                auto insert_result = position_db_.insertPosition(new_pos);
                if (!insert_result) {
                    return std::unexpected(insert_result.error());
                }

                Logger::getInstance().info("Bot opened new position: {} @ ${} ({})",
                                           confirmation.symbol, confirmation.avg_fill_price,
                                           confirmation.strategy_name);
            }
        } else {
            // Closing or reducing position
            // Update or delete position
        }

        return {};
    }

    /**
     * Close position (ONLY if bot-managed)
     */
    [[nodiscard]] auto closePosition(std::string const& account_id, std::string const& symbol)
        -> Result<OrderConfirmation> {

        auto position = position_db_.queryPosition(account_id, symbol);

        if (!position) {
            return makeError<OrderConfirmation>(ErrorCode::DatabaseError, "Position not found");
        }

        if (!position->is_bot_managed) {
            return makeError<OrderConfirmation>(
                ErrorCode::RuntimeError,
                std::format("SAFETY VIOLATION: Cannot close {} - manual position. "
                            "Only human can close manual positions.",
                            symbol));
        }

        // OK to close - this is a bot-managed position
        Order sell_order;
        sell_order.account_id = account_id;
        sell_order.symbol = symbol;
        sell_order.side = OrderSide::Sell;
        sell_order.quantity = position->quantity;
        sell_order.type = OrderType::Market;

        return placeOrder(sell_order);
    }

    // ========================================================================
    // Order Management APIs
    // ========================================================================

    /**
     * Modify existing order
     */
    [[nodiscard]] auto modifyOrder(std::string const& order_id, Order const& modifications)
        -> Result<OrderConfirmation> {

        Logger::getInstance().info("Modifying order: {}", order_id);

        if (dry_run_mode_) {
            Logger::getInstance().info("DRY-RUN: Would modify order {}", order_id);

            OrderConfirmation confirmation;
            confirmation.order_id = order_id;
            confirmation.symbol = modifications.symbol;
            confirmation.status = OrderStatus::Working;
            confirmation.dry_run = true;
            confirmation.timestamp = std::chrono::system_clock::now();

            return confirmation;
        }

        // TODO: Implement actual Schwab API call
        return makeError<OrderConfirmation>(ErrorCode::RuntimeError,
                                            "Modify order not yet implemented for live trading");
    }

    /**
     * Cancel order
     */
    [[nodiscard]] auto cancelOrder(std::string const& account_id, std::string const& order_id)
        -> Result<void> {

        Logger::getInstance().info("Canceling order: {}", order_id);

        if (dry_run_mode_) {
            Logger::getInstance().info("DRY-RUN: Would cancel order {}", order_id);

            // Update status in database
            auto update_result = order_logger_.updateOrderStatus(order_id, OrderStatus::Canceled);

            return update_result;
        }

        // TODO: Implement actual Schwab API call
        return makeError<void>(ErrorCode::RuntimeError,
                               "Cancel order not yet implemented for live trading");
    }

    /**
     * Get all orders (with optional filtering)
     */
    [[nodiscard]] auto getOrders(std::string const& account_id,
                                 std::optional<std::string> symbol = std::nullopt,
                                 std::optional<OrderStatus> status = std::nullopt)
        -> Result<std::vector<Order>> {

        Logger::getInstance().info("Getting orders for account: {}", account_id);

        if (dry_run_mode_) {
            Logger::getInstance().info("DRY-RUN: Would fetch orders from Schwab API");
            return std::vector<Order>{};
        }

        // TODO: Implement actual Schwab API call
        return std::vector<Order>{};
    }

    /**
     * Get single order details
     */
    [[nodiscard]] auto getOrder(std::string const& account_id, std::string const& order_id)
        -> Result<Order> {

        Logger::getInstance().info("Getting order: {}", order_id);

        if (dry_run_mode_) {
            Logger::getInstance().info("DRY-RUN: Would fetch order from Schwab API");
        }

        // TODO: Implement actual Schwab API call
        return makeError<Order>(ErrorCode::RuntimeError, "Get order not yet implemented");
    }

    /**
     * Get order status
     */
    [[nodiscard]] auto getOrderStatus(std::string const& order_id) -> Result<OrderStatus> {

        // Query from database first
        // TODO: Optionally refresh from Schwab API

        if (dry_run_mode_) {
            return OrderStatus::Pending;
        }

        return OrderStatus::Working;
    }

    // ========================================================================
    // Position Queries
    // ========================================================================

    /**
     * Get all positions
     */
    [[nodiscard]] auto getPositions(std::string const& account_id)
        -> Result<std::vector<Position>> {

        return position_db_.getAllPositions(account_id);
    }

    /**
     * Get single position
     */
    [[nodiscard]] auto getPosition(std::string const& account_id, std::string const& symbol)
        -> std::optional<Position> {

        return position_db_.queryPosition(account_id, symbol);
    }

    /**
     * Get position summary (manual vs bot-managed)
     */
    [[nodiscard]] auto getPositionSummary(std::string const& account_id) -> Result<json> {

        auto positions_result = position_db_.getAllPositions(account_id);
        if (!positions_result) {
            return std::unexpected(positions_result.error());
        }

        int manual_count = 0;
        int bot_count = 0;
        double manual_value = 0.0;
        double bot_value = 0.0;

        for (auto const& pos : *positions_result) {
            if (pos.is_bot_managed) {
                bot_count++;
                bot_value += pos.market_value;
            } else {
                manual_count++;
                manual_value += pos.market_value;
            }
        }

        json summary;
        summary["total_positions"] = positions_result->size();
        summary["manual_positions"] = manual_count;
        summary["bot_managed_positions"] = bot_count;
        summary["manual_value"] = manual_value;
        summary["bot_managed_value"] = bot_value;

        return summary;
    }

  private:
    /**
     * Validate order parameters
     */
    [[nodiscard]] auto validateOrder(Order const& order) -> Result<void> {

        if (order.symbol.empty()) {
            return makeError<void>(ErrorCode::InvalidParameter, "Symbol is required");
        }

        if (order.quantity <= quantity_epsilon) {
            return makeError<void>(ErrorCode::InvalidParameter, "Quantity must be positive");
        }

        if (order.type == OrderType::Limit && order.limit_price <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Limit price must be positive for limit orders");
        }

        if (order.type == OrderType::Stop && order.stop_price <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Stop price must be positive for stop orders");
        }

        return {};
    }

    PositionDatabase position_db_;
    OrderDatabaseLogger order_logger_;
    bool dry_run_mode_;
    std::atomic<int> order_counter_;
};

} // namespace bigbrother::schwab

// ============================================================================
// Module Private Implementation (Move Operations)
// ============================================================================
module :private;

namespace bigbrother::schwab {

// Destructor implementations
PositionDatabase::~PositionDatabase() = default;
OrderDatabaseLogger::~OrderDatabaseLogger() = default;
OrdersManager::~OrdersManager() = default;

} // namespace bigbrother::schwab
