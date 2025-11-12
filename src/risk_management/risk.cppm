/**
 * BigBrotherAnalytics - Risk Management Module (C++23)
 *
 * Comprehensive risk management for algorithmic trading.
 * Protects capital through position sizing, stop losses, and portfolio constraints.
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - Fluent API for risk assessment
 * - Thread-safe operations
 * - std::expected for error handling
 * - Modern C++23 features
 *
 * Risk Management for $30k Account (per PRD):
 * - Max daily loss: $900 (3%)
 * - Max position size: $1,500 (5%)
 * - Max concurrent positions: 10
 * - Mandatory stop losses
 */

// Global module fragment
module;

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>

// Module declaration
export module bigbrother.risk;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.options.pricing;

export namespace bigbrother::risk {

using namespace bigbrother::types;

// ============================================================================
// Risk Configuration
// ============================================================================

/**
 * Risk Limits Configuration
 * C.1: Struct for passive data with validation
 */
struct RiskLimits {
    double account_value{30'000.0};           // Total account value
    double max_daily_loss{900.0};             // Maximum daily loss (3%)
    double max_position_size{1'500.0};        // Maximum per position (5%)
    int max_concurrent_positions{10};         // Maximum positions
    double max_portfolio_heat{0.15};          // Maximum portfolio risk (15%)
    double max_correlation_exposure{0.30};    // Maximum correlated exposure (30%)
    bool require_stop_loss{true};             // Require stop loss on all trades

    /**
     * Create default limits for $30k account (per PRD)
     */
    [[nodiscard]] static constexpr auto forThirtyKAccount() noexcept -> RiskLimits {
        return RiskLimits{};  // Already defaults to $30k settings
    }

    [[nodiscard]] auto validate() const noexcept -> Result<void>;
};

/**
 * Position Risk Metrics
 */
struct PositionRisk {
    std::string symbol;
    double position_value{0.0};
    double unrealized_pnl{0.0};
    double max_loss{0.0};
    double probability_of_loss{0.0};
    double expected_value{0.0};
    double var_95{0.0};

    [[nodiscard]] auto isHighRisk() const noexcept -> bool {
        return max_loss / position_value > 0.20 || var_95 / position_value > 0.15;
    }
};

/**
 * Portfolio Risk Summary
 */
struct PortfolioRisk {
    double total_value{0.0};
    double daily_pnl{0.0};
    double daily_loss_remaining{0.0};
    int active_positions{0};
    double portfolio_heat{0.0};
    double max_drawdown{0.0};
    double var_95{0.0};           // Value at Risk (95% confidence) - Real-time
    double sharpe_ratio{0.0};     // Sharpe ratio (risk-adjusted returns) - Real-time

    [[nodiscard]] auto canOpenNewPosition() const noexcept -> bool {
        return active_positions < 10 && daily_loss_remaining > 0.0;
    }

    [[nodiscard]] auto getRiskLevel() const noexcept -> std::string {
        if (portfolio_heat > 0.15) return "HIGH";
        if (portfolio_heat > 0.10) return "MEDIUM";
        return "LOW";
    }
};

/**
 * Trade Risk Assessment
 */
struct TradeRisk {
    double position_size{0.0};
    double max_loss{0.0};
    double expected_return{0.0};
    double win_probability{0.0};
    double risk_reward_ratio{0.0};
    bool approved{false};
    std::string rejection_reason;

    [[nodiscard]] auto isApproved() const noexcept -> bool {
        return approved;
    }
};

// ============================================================================
// Risk Manager - Core Risk Management System
// ============================================================================

/**
 * Risk Manager
 *
 * Central risk management with thread-safe operations.
 * Uses pImpl pattern for ABI stability.
 */
class RiskManager {
public:
    explicit RiskManager(RiskLimits limits = RiskLimits::forThirtyKAccount());
    ~RiskManager();

    // C.21: Delete copy, allow move
    RiskManager(RiskManager const&) = delete;
    auto operator=(RiskManager const&) -> RiskManager& = delete;
    RiskManager(RiskManager&&) noexcept;
    auto operator=(RiskManager&&) noexcept -> RiskManager&;

    /**
     * Assess risk for a proposed trade
     */
    [[nodiscard]] auto assessTrade(
        std::string symbol,
        double position_size,
        Price entry_price,
        Price stop_price,
        Price target_price,
        double win_probability
    ) noexcept -> Result<TradeRisk>;

    /**
     * Assess risk for options trade
     */
    [[nodiscard]] auto assessOptionsTrade(
        std::string symbol,
        options::PricingParams const& params,
        double position_size,
        double win_probability,
        double target_profit
    ) noexcept -> Result<TradeRisk>;

    /**
     * Register new position
     */
    auto registerPosition(Position const& position, Price stop_price) -> void;

    /**
     * Update position price
     */
    auto updatePosition(std::string const& symbol, Price current_price) -> void;

    /**
     * Close position
     */
    auto closePosition(std::string const& symbol, Price exit_price) -> void;

    /**
     * Get portfolio risk
     */
    [[nodiscard]] auto getPortfolioRisk() const noexcept -> PortfolioRisk;

    /**
     * Check if daily loss limit reached
     */
    [[nodiscard]] auto isDailyLossLimitReached() const noexcept -> bool;

    /**
     * Reset daily P&L
     */
    auto resetDailyPnL() -> void;

    /**
     * Emergency stop all positions
     */
    auto emergencyStopAll() -> void;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

// ============================================================================
// Fluent API for Risk Assessment
// ============================================================================

/**
 * Risk Assessor - Fluent API
 *
 * Example Usage:
 *
 *   // Assess stock trade
 *   auto risk = RiskAssessor()
 *       .symbol("AAPL")
 *       .positionSize(1000.0)
 *       .entryPrice(150.0)
 *       .stopPrice(145.0)
 *       .targetPrice(160.0)
 *       .winProbability(0.65)
 *       .assess();
 *
 *   if (risk->isApproved()) {
 *       // Execute trade
 *   }
 *
 *   // Assess options trade
 *   auto options_risk = RiskAssessor()
 *       .symbol("SPY")
 *       .optionsPosition(params, 500.0)
 *       .winProbability(0.70)
 *       .targetProfit(100.0)
 *       .useKellyCriterion()
 *       .assess();
 */
class RiskAssessor {
public:
    RiskAssessor() = default;

    [[nodiscard]] auto symbol(std::string sym) -> RiskAssessor& {
        symbol_ = std::move(sym);
        return *this;
    }

    [[nodiscard]] auto positionSize(double size) noexcept -> RiskAssessor& {
        position_size_ = size;
        return *this;
    }

    [[nodiscard]] auto entryPrice(Price price) noexcept -> RiskAssessor& {
        entry_price_ = price;
        return *this;
    }

    [[nodiscard]] auto stopPrice(Price price) noexcept -> RiskAssessor& {
        stop_price_ = price;
        return *this;
    }

    [[nodiscard]] auto targetPrice(Price price) noexcept -> RiskAssessor& {
        target_price_ = price;
        return *this;
    }

    [[nodiscard]] auto winProbability(double prob) noexcept -> RiskAssessor& {
        win_probability_ = prob;
        return *this;
    }

    [[nodiscard]] auto optionsPosition(
        options::PricingParams params,
        double size
    ) noexcept -> RiskAssessor& {
        options_params_ = std::move(params);
        position_size_ = size;
        is_options_ = true;
        return *this;
    }

    [[nodiscard]] auto targetProfit(double profit) noexcept -> RiskAssessor& {
        target_profit_ = profit;
        return *this;
    }

    [[nodiscard]] auto useKellyCriterion() noexcept -> RiskAssessor& {
        use_kelly_ = true;
        return *this;
    }

    /**
     * Assess the trade (terminal operation)
     */
    [[nodiscard]] auto assess() -> Result<TradeRisk> {
        // Stub implementation - will integrate with RiskManager
        TradeRisk risk{};
        risk.position_size = position_size_;
        risk.max_loss = std::abs(entry_price_ - stop_price_) * 100.0;
        risk.expected_return = std::abs(target_price_ - entry_price_) * 100.0;
        risk.win_probability = win_probability_;
        risk.risk_reward_ratio = risk.expected_return / risk.max_loss;
        risk.approved = risk.win_probability > 0.60 && risk.risk_reward_ratio > 1.5;

        return risk;
    }

private:
    std::string symbol_;
    double position_size_{0.0};
    Price entry_price_{0.0};
    Price stop_price_{0.0};
    Price target_price_{0.0};
    double win_probability_{0.0};
    bool is_options_{false};
    options::PricingParams options_params_{};
    double target_profit_{0.0};
    bool use_kelly_{false};
};

/**
 * Position Sizer - Kelly Criterion and Variants
 */
class PositionSizer {
public:
    enum class Method {
        FixedDollar,
        FixedPercent,
        KellyCriterion,
        KellyHalf,
        VolatilityAdjusted
    };

    /**
     * Calculate position size
     */
    [[nodiscard]] static auto calculateSize(
        Method method,
        double account_value,
        double win_probability,
        double win_amount,
        double loss_amount,
        double max_position = std::numeric_limits<double>::infinity()
    ) noexcept -> Result<double>;

    /**
     * Calculate Kelly Criterion fraction
     * f* = (p * b - q) / b
     */
    [[nodiscard]] static constexpr auto kellyFraction(
        double win_probability,
        double win_amount,
        double loss_amount
    ) noexcept -> double {
        if (loss_amount == 0.0) return 0.0;

        double const q = 1.0 - win_probability;
        double const b = win_amount / loss_amount;

        double const kelly = (win_probability * b - q) / b;

        return std::max(0.0, kelly);  // Never go negative
    }
};

} // export namespace bigbrother::risk

// ============================================================================
// Implementation Section (module-private)
// ============================================================================

module :private;

namespace bigbrother::risk {

// RiskLimits validation
auto RiskLimits::validate() const noexcept -> Result<void> {
    if (account_value <= 0.0) {
        return makeError<void>(ErrorCode::InvalidParameter, "Account value must be positive");
    }
    if (max_daily_loss <= 0.0 || max_daily_loss > account_value) {
        return makeError<void>(ErrorCode::InvalidParameter, "Invalid daily loss limit");
    }
    if (max_position_size <= 0.0 || max_position_size > account_value) {
        return makeError<void>(ErrorCode::InvalidParameter, "Invalid position size limit");
    }
    return {};
}

// Position Sizer implementation
auto PositionSizer::calculateSize(
    Method method,
    double account_value,
    double win_probability,
    double win_amount,
    double loss_amount,
    double max_position
) noexcept -> Result<double> {

    double size = 0.0;

    switch (method) {
        case Method::FixedDollar:
            size = 1000.0;  // Fixed $1000
            break;

        case Method::FixedPercent:
            size = account_value * 0.05;  // 5% of account
            break;

        case Method::KellyCriterion: {
            double const kelly = kellyFraction(win_probability, win_amount, loss_amount);
            size = account_value * kelly;
            break;
        }

        case Method::KellyHalf: {
            double const kelly = kellyFraction(win_probability, win_amount, loss_amount);
            size = account_value * kelly * 0.5;  // Conservative: half Kelly
            break;
        }

        case Method::VolatilityAdjusted:
            size = account_value * 0.05;  // Stub
            break;
    }

    // Apply maximum limit
    size = std::min(size, max_position);

    return size;
}

} // namespace bigbrother::risk
