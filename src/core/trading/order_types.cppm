/**
 * BigBrotherAnalytics - Common Trading Types (C++23)
 *
 * ARCHITECTURE: Platform-Agnostic Data Types
 *
 * These types are shared across:
 * - OrdersManager (business logic)
 * - TradingPlatformInterface (abstraction)
 * - Platform executors (Schwab, IBKR, etc.)
 *
 * DESIGN PRINCIPLE: Common vocabulary for all trading platforms
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 */

// Global module fragment
module;

#include <chrono>
#include <string>

// Module declaration
export module bigbrother.trading.order_types;

// Import dependencies
import bigbrother.utils.types;

export namespace bigbrother::trading {

using namespace bigbrother::types;

// ============================================================================
// Position Types
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

// ============================================================================
// Order Types
// ============================================================================

/**
 * Order side (Buy/Sell)
 */
enum class OrderSide { Buy, Sell, SellShort, BuyToCover };

/**
 * Order type
 */
enum class OrderType { Market, Limit, Stop, StopLimit, TrailingStop };

/**
 * Order status
 */
enum class OrderStatus { Pending, Working, Filled, PartiallyFilled, Canceled, Rejected };

/**
 * Order duration
 */
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

} // namespace bigbrother::trading
