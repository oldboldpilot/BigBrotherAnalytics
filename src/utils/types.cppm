/**
 * BigBrotherAnalytics - Types Module (C++23)
 *
 * Core type definitions following C++ Core Guidelines:
 * - C.1: Use struct for passive data
 * - F.4: Use constexpr for compile-time evaluation
 * - F.6: Use noexcept where applicable
 * - F.16: Pass cheap types by value
 * - F.20: Prefer return values to output parameters
 * - P.4: Type safety over primitives
 */

// Global module fragment
module;

#include <chrono>
#include <cmath>
#include <cstdint>
#include <expected>
#include <optional>
#include <source_location>
#include <string>

// Module declaration
export module bigbrother.utils.types;

export namespace bigbrother::types {

// ============================================================================
// Time Types (C++ Core Guidelines: Use strong types, not primitives)
// ============================================================================

using Timestamp = int64_t; // Microseconds since epoch
using Duration = int64_t;  // Duration in microseconds

// ============================================================================
// Financial Types (P.4: Type safety)
// ============================================================================

using Price = double;                   // Stock/option price in dollars
using Quantity = double;                // Number of shares/contracts (supports fractional)
constexpr auto quantity_epsilon = 1e-6; // Tolerance for quantity comparisons
using Volume = int64_t;                 // Trading volume

// ============================================================================
// Option Types
// ============================================================================

/**
 * Option Type (Call or Put)
 * C.1: Use enum class for type safety
 */
enum class OptionType : uint8_t { Call, Put };

/**
 * Greeks Structure
 *
 * Following C++ Core Guidelines:
 * - C.1: Struct for passive data (no invariants)
 * - F.4: constexpr member functions
 * - F.6: noexcept where no exceptions possible
 */
struct Greeks {
    double delta{0.0}; // ∂V/∂S - Price sensitivity
    double gamma{0.0}; // ∂²V/∂S² - Delta sensitivity
    double theta{0.0}; // ∂V/∂t - Time decay
    double vega{0.0};  // ∂V/∂σ - Volatility sensitivity
    double rho{0.0};   // ∂V/∂r - Interest rate sensitivity

    /**
     * Validate Greeks values
     * F.4: constexpr for compile-time evaluation when possible
     * F.6: noexcept - no exceptions
     */
    [[nodiscard]] constexpr auto isValid() const noexcept -> bool {
        return !std::isnan(delta) && !std::isnan(gamma) && !std::isnan(theta) &&
               !std::isnan(vega) && !std::isnan(rho);
    }

    /**
     * Check if portfolio is delta-neutral
     * F.16: Pass double by value (cheap to copy)
     */
    [[nodiscard]] constexpr auto isDeltaNeutral(double tolerance = 0.1) const noexcept -> bool {
        return std::abs(delta) <= tolerance;
    }
};

// ============================================================================
// Error Handling (Following I.10, E: Use exceptions or std::expected)
// ============================================================================

/**
 * Error Codes
 * C.1: enum class for type safety
 */
enum class ErrorCode : uint32_t {
    Success = 0,
    InvalidParameter,
    OutOfRange,
    FileNotFound,
    DatabaseError,
    NetworkError,
    AuthenticationFailed,
    InsufficientFunds,
    OrderRejected,
    InitializationError,
    RuntimeError,
    APIError,
    ParseError,
    CircuitBreakerOpen,
    UnknownError
};

/**
 * Error Information
 *
 * C.1: Struct for passive data
 * Following error handling guidelines
 */
struct Error {
    ErrorCode code{ErrorCode::Success};
    std::string message;
    std::source_location location;

    /**
     * Create error with automatic source location
     * F.20: Return value, not output parameter
     */
    [[nodiscard]] static auto make(ErrorCode code, std::string message,
                                   std::source_location location = std::source_location::current())
        -> Error {
        return Error{code, std::move(message), location};
    }

    /**
     * Check if error represents success
     * F.6: noexcept - no exceptions possible
     */
    [[nodiscard]] constexpr auto isSuccess() const noexcept -> bool {
        return code == ErrorCode::Success;
    }
};

/**
 * Result Type (std::expected wrapper)
 *
 * Following E: Error handling guidelines
 * Use std::expected for functions that can fail
 */
template <typename T>
using Result = std::expected<T, Error>;

/**
 * Helper to create error Result
 * F.1: Meaningfully named function
 * F.20: Return value
 */
template <typename T>
[[nodiscard]] inline auto makeError(ErrorCode code, std::string message,
                                    std::source_location location = std::source_location::current())
    -> Result<T> {
    return std::unexpected(Error::make(code, std::move(message), location));
}

// ============================================================================
// Market Data Types
// ============================================================================

/**
 * Option Pricing Parameters
 *
 * C.1: Struct for passive data
 * C.47: Define and initialize member variables in order of declaration
 */
struct PricingParams {
    Price spot_price{0.0};
    Price strike_price{0.0};
    double risk_free_rate{0.0};
    double time_to_expiration{0.0}; // Years
    double volatility{0.0};         // Annual volatility
    double dividend_yield{0.0};
    OptionType option_type{OptionType::Call};

    /**
     * Validate pricing parameters
     *
     * F.6: noexcept - validation doesn't throw
     * F.20: Return Result<void> for validation
     */
    [[nodiscard]] constexpr auto validate() const noexcept -> Result<void> {
        if (spot_price <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Spot price must be positive");
        }

        if (strike_price <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Strike price must be positive");
        }

        if (time_to_expiration < 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Time to expiration cannot be negative");
        }

        if (volatility < 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Volatility cannot be negative");
        }

        return {}; // Success
    }
};

/**
 * Trade Signal
 *
 * C.1: Struct for passive data
 * Represents a trading signal from a strategy
 */
struct TradingSignal {
    std::string symbol;
    OptionType option_type;
    double strike_price;
    double target_price;
    double stop_loss;
    double confidence; // 0.0 to 1.0
    std::string strategy_name;

    /**
     * Check if signal is valid
     * F.4: constexpr
     * F.6: noexcept
     */
    [[nodiscard]] constexpr auto isValid() const noexcept -> bool {
        return !symbol.empty() && strike_price > 0.0 && confidence >= 0.0 && confidence <= 1.0;
    }
};

/**
 * Portfolio Position
 *
 * C.2: Private data with public interface when invariants exist
 * Represents an active trading position
 */
class Position {
  public:
    /**
     * Constructor
     * C.41: Constructor establishes class invariant
     * F.16: Pass string by value (will be moved)
     */
    explicit Position(std::string symbol, Quantity quantity, Price entry_price) noexcept
        : symbol_{std::move(symbol)}, quantity_{quantity}, entry_price_{entry_price},
          current_price_{entry_price} {}

    // C.21: Define or delete default operations as group
    Position(Position const&) = default;
    auto operator=(Position const&) -> Position& = default;
    Position(Position&&) noexcept = default;
    auto operator=(Position&&) noexcept -> Position& = default;
    ~Position() = default;

    // Accessors (F.20: Return by value for cheap types)
    [[nodiscard]] auto symbol() const noexcept -> std::string const& { return symbol_; }
    [[nodiscard]] constexpr auto quantity() const noexcept -> Quantity { return quantity_; }
    [[nodiscard]] constexpr auto entryPrice() const noexcept -> Price { return entry_price_; }
    [[nodiscard]] constexpr auto currentPrice() const noexcept -> Price { return current_price_; }

    // Mutators
    auto updatePrice(Price new_price) noexcept -> void { current_price_ = new_price; }

    /**
     * Calculate unrealized P/L
     * F.4: constexpr
     * F.6: noexcept
     */
    [[nodiscard]] constexpr auto unrealizedPnL() const noexcept -> double {
        return (current_price_ - entry_price_) * static_cast<double>(quantity_);
    }

  private:
    std::string symbol_;
    Quantity quantity_;
    Price entry_price_;
    Price current_price_;
};

} // namespace bigbrother::types
