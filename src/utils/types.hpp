#pragma once

#include <cstdint>
#include <string>
#include <chrono>
#include <expected>
#include <optional>

namespace bigbrother::types {

// Time types
using Timestamp = int64_t;  // Microseconds since epoch
using Duration = int64_t;   // Duration in microseconds

// Price and quantity types
using Price = double;
using Quantity = int64_t;
using Volume = int64_t;

// Option types
enum class OptionType : uint8_t {
    Call,
    Put
};

enum class OptionStyle : uint8_t {
    American,  // Can exercise anytime before expiration
    European   // Can only exercise at expiration
};

// Order types
enum class OrderType : uint8_t {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop
};

enum class OrderSide : uint8_t {
    Buy,
    Sell
};

enum class OrderStatus : uint8_t {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired
};

// Time in force
enum class TimeInForce : uint8_t {
    Day,        // Good for day
    GTC,        // Good til cancelled
    IOC,        // Immediate or cancel
    FOK,        // Fill or kill
    GTD,        // Good til date
    MOC,        // Market on close
    MOO         // Market on open
};

// Greeks
struct Greeks {
    double delta;    // ∂V/∂S - Price sensitivity
    double gamma;    // ∂²V/∂S² - Delta sensitivity
    double theta;    // ∂V/∂t - Time decay
    double vega;     // ∂V/∂σ - Volatility sensitivity
    double rho;      // ∂V/∂r - Interest rate sensitivity

    [[nodiscard]] constexpr auto isValid() const noexcept -> bool {
        return !std::isnan(delta) && !std::isnan(gamma) &&
               !std::isnan(theta) && !std::isnan(vega) &&
               !std::isnan(rho);
    }
};

// Option contract
struct OptionContract {
    std::string symbol;
    std::string underlying;
    OptionType type;
    OptionStyle style;
    Price strike;
    Timestamp expiration;
    double implied_volatility;

    [[nodiscard]] auto isCall() const noexcept -> bool {
        return type == OptionType::Call;
    }

    [[nodiscard]] auto isPut() const noexcept -> bool {
        return type == OptionType::Put;
    }

    [[nodiscard]] auto isAmerican() const noexcept -> bool {
        return style == OptionStyle::American;
    }

    [[nodiscard]] auto isEuropean() const noexcept -> bool {
        return style == OptionStyle::European;
    }

    [[nodiscard]] auto isExpired(Timestamp current_time) const noexcept -> bool {
        return current_time >= expiration;
    }

    [[nodiscard]] auto daysToExpiration(Timestamp current_time) const noexcept -> double {
        return static_cast<double>(expiration - current_time) / (1'000'000.0 * 86400.0);
    }

    [[nodiscard]] auto yearsToExpiration(Timestamp current_time) const noexcept -> double {
        return daysToExpiration(current_time) / 365.0;
    }
};

// Market data quote
struct Quote {
    std::string symbol;
    Timestamp timestamp;
    Price bid;
    Price ask;
    Quantity bid_size;
    Quantity ask_size;
    Price last;
    Volume volume;

    [[nodiscard]] auto midPrice() const noexcept -> Price {
        return (bid + ask) / 2.0;
    }

    [[nodiscard]] auto spread() const noexcept -> Price {
        return ask - bid;
    }

    [[nodiscard]] auto spreadBps() const noexcept -> double {
        return (spread() / midPrice()) * 10000.0;
    }
};

// OHLCV bar
struct Bar {
    std::string symbol;
    Timestamp timestamp;
    Price open;
    Price high;
    Price low;
    Price close;
    Volume volume;

    [[nodiscard]] auto range() const noexcept -> Price {
        return high - low;
    }

    [[nodiscard]] auto bodySize() const noexcept -> Price {
        return std::abs(close - open);
    }

    [[nodiscard]] auto isBullish() const noexcept -> bool {
        return close > open;
    }

    [[nodiscard]] auto isBearish() const noexcept -> bool {
        return close < open;
    }
};

// Trade
struct Trade {
    std::string symbol;
    Timestamp timestamp;
    Price price;
    Quantity quantity;
    OrderSide side;

    [[nodiscard]] auto notional() const noexcept -> double {
        return price * static_cast<double>(quantity);
    }
};

// Position
struct Position {
    std::string symbol;
    Quantity quantity;
    Price average_price;
    Price current_price;
    Timestamp entry_time;

    [[nodiscard]] auto isLong() const noexcept -> bool {
        return quantity > 0;
    }

    [[nodiscard]] auto isShort() const noexcept -> bool {
        return quantity < 0;
    }

    [[nodiscard]] auto isFlat() const noexcept -> bool {
        return quantity == 0;
    }

    [[nodiscard]] auto marketValue() const noexcept -> double {
        return current_price * static_cast<double>(std::abs(quantity));
    }

    [[nodiscard]] auto unrealizedPnL() const noexcept -> double {
        return (current_price - average_price) * static_cast<double>(quantity);
    }

    [[nodiscard]] auto unrealizedPnLPercent() const noexcept -> double {
        if (average_price == 0.0) return 0.0;
        return ((current_price - average_price) / average_price) * 100.0;
    }
};

// Order
struct Order {
    std::string order_id;
    std::string symbol;
    OrderType type;
    OrderSide side;
    Quantity quantity;
    Quantity filled_quantity;
    Price limit_price;   // For limit orders
    Price stop_price;    // For stop orders
    OrderStatus status;
    TimeInForce tif;
    Timestamp submit_time;
    Timestamp fill_time;
    std::string reject_reason;

    [[nodiscard]] auto isFilled() const noexcept -> bool {
        return status == OrderStatus::Filled;
    }

    [[nodiscard]] auto isActive() const noexcept -> bool {
        return status == OrderStatus::Pending ||
               status == OrderStatus::Submitted ||
               status == OrderStatus::PartiallyFilled;
    }

    [[nodiscard]] auto remainingQuantity() const noexcept -> Quantity {
        return quantity - filled_quantity;
    }

    [[nodiscard]] auto fillRate() const noexcept -> double {
        if (quantity == 0) return 0.0;
        return static_cast<double>(filled_quantity) / static_cast<double>(quantity);
    }
};

// Strategy signal
enum class SignalType : uint8_t {
    Buy,
    Sell,
    Hold,
    CloseAll
};

struct Signal {
    std::string strategy_name;
    std::string symbol;
    SignalType type;
    double confidence;  // 0.0 to 1.0
    Timestamp timestamp;
    std::string rationale;

    [[nodiscard]] auto isActionable() const noexcept -> bool {
        return type != SignalType::Hold && confidence > 0.5;
    }
};

// Performance metrics
struct PerformanceMetrics {
    double total_return;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double win_rate;
    int64_t total_trades;
    int64_t winning_trades;
    int64_t losing_trades;
    double avg_win;
    double avg_loss;
    double profit_factor;
    double expectancy;

    [[nodiscard]] auto isprofitable() const noexcept -> bool {
        return total_return > 0.0;
    }

    [[nodiscard]] auto isGoodSharpe() const noexcept -> bool {
        return sharpe_ratio >= 2.0;
    }
};

// Error types using std::expected (C++23)
enum class ErrorCode {
    Success = 0,
    NetworkError,
    AuthenticationError,
    InvalidParameter,
    InsufficientFunds,
    OrderRejected,
    MarketClosed,
    DataNotAvailable,
    DatabaseError,
    TimeoutError,
    UnknownError
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string details;

    [[nodiscard]] static auto success() -> Error {
        return {ErrorCode::Success, "", ""};
    }

    [[nodiscard]] auto isSuccess() const noexcept -> bool {
        return code == ErrorCode::Success;
    }
};

// Result type aliases using std::expected
template<typename T>
using Result = std::expected<T, Error>;

// Helper function to create error results
template<typename T>
[[nodiscard]] inline auto makeError(ErrorCode code,
                                     std::string message,
                                     std::string details = "") -> Result<T> {
    return std::unexpected(Error{code, std::move(message), std::move(details)});
}

// Helper function to create success results
template<typename T>
[[nodiscard]] inline auto makeSuccess(T value) -> Result<T> {
    return std::expected<T, Error>(std::move(value));
}

} // namespace bigbrother::types
