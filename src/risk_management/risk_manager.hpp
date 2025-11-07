#pragma once

#include "../utils/types.hpp"
#include "../correlation_engine/options_pricing.hpp"
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>

namespace bigbrother::risk {

using namespace types;

/**
 * Risk Management System
 *
 * Comprehensive risk management for algorithmic trading.
 * Protects capital through position sizing, stop losses, and portfolio constraints.
 *
 * Key Features:
 * - Position sizing (Kelly Criterion, fixed fractional)
 * - Stop loss management (hard stops, trailing stops)
 * - Daily loss limits (3% max per PRD)
 * - Position limits (5% max per position per PRD)
 * - Portfolio-level risk monitoring
 * - Monte Carlo simulation for trade validation
 * - Real-time P&L tracking
 *
 * Thread-Safe: All operations are thread-safe for concurrent trading
 */

/**
 * Risk Limits Configuration
 *
 * Per PRD requirements for $30k account
 */
struct RiskLimits {
    double account_value;              // Total account value ($30,000)
    double max_daily_loss;             // Maximum daily loss ($900 = 3%)
    double max_position_size;          // Maximum per position ($1,500 = 5%)
    int max_concurrent_positions;      // Maximum positions (10)
    double max_portfolio_heat;         // Maximum portfolio risk (0.15 = 15%)
    double max_correlation_exposure;   // Maximum correlated exposure (0.30 = 30%)
    bool require_stop_loss;            // Require stop loss on all trades (true)

    /**
     * Create default limits for $30k account (per PRD)
     */
    [[nodiscard]] static constexpr auto forThirtyKAccount() noexcept -> RiskLimits {
        return {
            .account_value = 30'000.0,
            .max_daily_loss = 900.0,           // 3% of capital
            .max_position_size = 1'500.0,       // 5% of capital
            .max_concurrent_positions = 10,
            .max_portfolio_heat = 0.15,         // 15% total risk
            .max_correlation_exposure = 0.30,   // 30% in correlated positions
            .require_stop_loss = true
        };
    }

    [[nodiscard]] auto validate() const noexcept -> Result<void>;
};

/**
 * Position Risk Metrics
 *
 * Detailed risk analysis for a single position
 */
struct PositionRisk {
    std::string symbol;
    double position_value;          // Market value of position
    double unrealized_pnl;          // Unrealized profit/loss
    double realized_pnl;            // Realized profit/loss (closed)
    double max_loss;                // Maximum possible loss (stop loss)
    double probability_of_loss;     // Estimated probability of loss
    double expected_value;          // Expected value of position
    double sharpe_ratio;            // Position Sharpe ratio
    double var_95;                  // Value at Risk (95% confidence)
    double delta_exposure;          // Delta-adjusted exposure
    double vega_exposure;           // Vega exposure (volatility risk)
    double theta_decay;             // Theta decay per day
    bool has_stop_loss;             // Has stop loss configured

    [[nodiscard]] auto isHighRisk() const noexcept -> bool {
        return max_loss / position_value > 0.20 || var_95 / position_value > 0.15;
    }
};

/**
 * Portfolio Risk Summary
 */
struct PortfolioRisk {
    double total_value;
    double total_exposure;
    double net_delta;
    double net_vega;
    double net_theta;
    double daily_pnl;
    double daily_loss_remaining;    // Remaining loss allowance today
    int active_positions;
    double portfolio_heat;          // Total portfolio risk
    double max_drawdown;
    double sharpe_ratio;
    std::vector<PositionRisk> positions;

    [[nodiscard]] auto canOpenNewPosition() const noexcept -> bool;
    [[nodiscard]] auto getRiskLevel() const noexcept -> std::string;
};

/**
 * Trade Risk Assessment
 *
 * Risk analysis before trade execution
 */
struct TradeRisk {
    double position_size;           // Recommended position size
    double max_loss;                // Maximum loss with stop
    double expected_return;         // Expected return
    double expected_value;          // Expected value
    double win_probability;         // Probability of profit
    double kelly_fraction;          // Kelly Criterion fraction
    double risk_reward_ratio;       // Reward/Risk ratio
    double var_95;                  // Value at Risk
    bool approved;                  // Risk approval
    std::string rejection_reason;   // Why rejected (if applicable)

    [[nodiscard]] auto isApproved() const noexcept -> bool {
        return approved;
    }
};

/**
 * Position Sizer
 *
 * Calculate optimal position sizes using various methods
 */
class PositionSizer {
public:
    enum class Method {
        FixedDollar,        // Fixed dollar amount
        FixedPercent,       // Fixed percentage of capital
        KellyCriterion,     // Kelly Criterion (optimal growth)
        KellyHalf,          // 50% of Kelly (conservative)
        VolatilityAdjusted, // Adjusted for volatility
        RiskParity          // Equal risk contribution
    };

    /**
     * Calculate position size
     *
     * @param method Sizing method
     * @param account_value Current account value
     * @param win_probability Probability of winning trade
     * @param win_amount Expected win amount
     * @param loss_amount Expected loss amount
     * @param max_position Maximum position size limit
     * @return Position size in dollars
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
     *
     * f* = (p * b - q) / b
     * where p = win probability, q = 1-p, b = win/loss ratio
     */
    [[nodiscard]] static constexpr auto kellyFraction(
        double win_probability,
        double win_amount,
        double loss_amount
    ) noexcept -> double;

    /**
     * Calculate position size for options trade
     */
    [[nodiscard]] static auto calculateOptionsSize(
        double account_value,
        options::PricingParams const& params,
        double win_probability,
        double target_profit,
        double max_position
    ) noexcept -> Result<double>;
};

/**
 * Stop Loss Manager
 *
 * Manages stop losses for all positions
 */
class StopLossManager {
public:
    enum class StopType {
        Hard,           // Fixed price stop
        Trailing,       // Trailing stop (follows price)
        TimeStop,       // Exit after time limit
        VolatilityStop, // Stop based on volatility
        Greeks          // Stop based on Greeks (delta, vega)
    };

    struct Stop {
        std::string position_id;
        StopType type;
        Price trigger_price;
        Price initial_price;
        double trail_amount;    // For trailing stops
        Timestamp expiration;   // For time stops
        bool triggered;

        [[nodiscard]] auto isTriggered(Price current_price) const noexcept -> bool;
    };

    StopLossManager() = default;

    /**
     * Add stop loss for position
     */
    auto addStop(
        std::string position_id,
        StopType type,
        Price trigger_price,
        Price initial_price = 0.0,
        double trail_amount = 0.0
    ) -> void;

    /**
     * Update stop losses based on current prices
     * @return List of triggered stops
     */
    [[nodiscard]] auto update(
        std::map<std::string, Price> const& current_prices
    ) -> std::vector<std::string>;

    /**
     * Remove stop loss
     */
    auto removeStop(std::string const& position_id) -> void;

    /**
     * Get all active stops
     */
    [[nodiscard]] auto getActiveStops() const noexcept
        -> std::vector<Stop> const&;

private:
    std::vector<Stop> stops_;
    mutable std::mutex mutex_;
};

/**
 * Monte Carlo Simulator
 *
 * Simulate trade outcomes before execution
 */
class MonteCarloSimulator {
public:
    struct SimulationResult {
        double mean_pnl;
        double median_pnl;
        double std_pnl;
        double max_pnl;
        double min_pnl;
        double var_95;              // Value at Risk (95%)
        double cvar_95;             // Conditional VaR (Expected Shortfall)
        double win_probability;
        double expected_value;
        int num_simulations;

        [[nodiscard]] auto isPositiveExpectedValue() const noexcept -> bool {
            return expected_value > 0.0;
        }

        [[nodiscard]] auto meetsThreshold(double min_ev = 50.0) const noexcept -> bool {
            return expected_value >= min_ev;
        }
    };

    /**
     * Simulate option trade outcomes
     *
     * @param params Option pricing parameters
     * @param position_size Position size in dollars
     * @param num_simulations Number of Monte Carlo paths
     * @param num_steps Time steps per simulation
     * @return Simulation results
     */
    [[nodiscard]] static auto simulateOptionTrade(
        options::PricingParams const& params,
        double position_size,
        int num_simulations = 10'000,
        int num_steps = 100
    ) noexcept -> Result<SimulationResult>;

    /**
     * Simulate stock trade outcomes
     */
    [[nodiscard]] static auto simulateStockTrade(
        Price entry_price,
        Price target_price,
        Price stop_price,
        double volatility,
        int num_simulations = 10'000
    ) noexcept -> Result<SimulationResult>;
};

/**
 * Risk Manager
 *
 * Central risk management system
 *
 * Thread-safe operations for concurrent trading
 */
class RiskManager {
public:
    explicit RiskManager(RiskLimits limits = RiskLimits::forThirtyKAccount());
    ~RiskManager();

    // Delete copy, allow move
    RiskManager(RiskManager const&) = delete;
    auto operator=(RiskManager const&) = delete;
    RiskManager(RiskManager&&) noexcept;
    auto operator=(RiskManager&&) noexcept -> RiskManager&;

    /**
     * Assess risk for a proposed trade
     *
     * @param symbol Symbol to trade
     * @param position_size Proposed position size
     * @param entry_price Entry price
     * @param stop_price Stop loss price
     * @param target_price Target profit price
     * @param win_probability Estimated win probability
     * @return Trade risk assessment
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
     * Check if trade is approved
     *
     * Validates against all risk limits
     */
    [[nodiscard]] auto approveTrade(TradeRisk const& trade_risk) const noexcept
        -> Result<bool>;

    /**
     * Register new position
     */
    auto registerPosition(Position const& position, Price stop_price) -> void;

    /**
     * Update position (price change, P&L update)
     */
    auto updatePosition(std::string const& symbol, Price current_price) -> void;

    /**
     * Close position
     */
    auto closePosition(std::string const& symbol, Price exit_price) -> void;

    /**
     * Get current portfolio risk
     */
    [[nodiscard]] auto getPortfolioRisk() const noexcept -> PortfolioRisk;

    /**
     * Get position risk
     */
    [[nodiscard]] auto getPositionRisk(std::string const& symbol) const noexcept
        -> Result<PositionRisk>;

    /**
     * Check daily loss limit
     */
    [[nodiscard]] auto isDailyLossLimitReached() const noexcept -> bool;

    /**
     * Reset daily P&L (call at start of trading day)
     */
    auto resetDailyPnL() -> void;

    /**
     * Get risk limits
     */
    [[nodiscard]] auto getRiskLimits() const noexcept -> RiskLimits const&;

    /**
     * Update risk limits
     */
    auto updateRiskLimits(RiskLimits limits) -> void;

    /**
     * Emergency kill switch - close all positions
     */
    auto emergencyStopAll() -> void;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace bigbrother::risk
