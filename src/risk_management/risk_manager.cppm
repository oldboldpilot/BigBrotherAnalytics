/**
 * BigBrotherAnalytics - Risk Manager Module (C++23)
 *
 * Fluent API for comprehensive risk management of algorithmic trading.
 * Thread-safe portfolio risk tracking with position management.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - Pimpl pattern for implementation hiding
 * - Thread-safe operations with mutex
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

// Module declaration
export module bigbrother.risk.manager;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.options.pricing;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using namespace bigbrother::options;
using bigbrother::utils::Logger;

// Type aliases for convenience
using PricingParams = bigbrother::options::ExtendedPricingParams;

// ============================================================================
// Risk Configuration
// ============================================================================

struct RiskLimits {
    double account_value{30'000.0};
    double max_daily_loss{900.0};
    double max_position_size{2'000.0};
    int max_concurrent_positions{15};
    double max_portfolio_heat{0.15};       // 15% max exposure
    double max_sector_concentration{0.40}; // 40% max in one sector
    bool allow_overnight{true};
    bool paper_trading{true};

    [[nodiscard]] auto validate() const noexcept -> Result<void> {
        if (account_value <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Account value must be positive");
        }
        if (max_daily_loss <= 0.0 || max_daily_loss > account_value) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid daily loss limit");
        }
        if (max_position_size <= 0.0 || max_position_size > account_value) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid position size limit");
        }
        if (max_concurrent_positions <= 0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Invalid concurrent positions limit");
        }
        return {};
    }
};

// ============================================================================
// Trade Risk Assessment
// ============================================================================

struct TradeRisk {
    std::string symbol;
    double position_size{0.0};
    double max_loss{0.0};
    double expected_return{0.0};
    double win_probability{0.0};
    double expected_value{0.0};
    double risk_reward_ratio{0.0};
    bool approved{false};
    std::string rejection_reason;

    [[nodiscard]] auto isApproved() const noexcept -> bool { return approved; }

    [[nodiscard]] auto getRiskScore() const noexcept -> double {
        // Higher score = better trade (considering EV and risk/reward)
        return expected_value * risk_reward_ratio;
    }
};

// ============================================================================
// Position Risk
// ============================================================================

struct PositionRisk {
    std::string symbol;
    double position_value{0.0};
    double unrealized_pnl{0.0};
    double unrealized_pnl_percent{0.0};
    double delta_exposure{0.0};
    double stop_distance{0.0};
    double time_in_position{0.0}; // Hours

    [[nodiscard]] auto isAtRisk() const noexcept -> bool {
        return unrealized_pnl_percent < -0.05; // Down 5%+
    }
};

// ============================================================================
// Portfolio Risk
// ============================================================================

struct PortfolioRisk {
    double total_value{0.0};
    double daily_pnl{0.0};
    double daily_loss_remaining{0.0};
    int active_positions{0};
    double portfolio_heat{0.0};
    double var_95{0.0};
    double sharpe_ratio{0.0};

    [[nodiscard]] auto canOpenNewPosition() const noexcept -> bool {
        return active_positions < 10 && daily_loss_remaining > 0.0;
    }

    [[nodiscard]] auto getRiskLevel() const noexcept -> std::string {
        if (portfolio_heat > 0.15)
            return "HIGH";
        if (portfolio_heat > 0.10)
            return "MEDIUM";
        return "LOW";
    }

    [[nodiscard]] auto isDailyLossLimitNear() const noexcept -> bool {
        return daily_loss_remaining < 300.0; // Within $300 of limit
    }
};

// ============================================================================
// Risk Manager - Fluent API
// ============================================================================

class RiskManager {
  public:
    // Factory method
    [[nodiscard]] static auto create(RiskLimits limits) noexcept -> Result<RiskManager> {
        if (auto validation = limits.validate(); !validation) {
            return makeError<RiskManager>(ErrorCode::InvalidParameter, "Invalid risk limits");
        }
        return RiskManager{limits};
    }

    // Destructor and move operations
    ~RiskManager();
    RiskManager(RiskManager&&) noexcept;
    auto operator=(RiskManager&&) noexcept -> RiskManager&;

    // Delete copy operations
    RiskManager(RiskManager const&) = delete;
    auto operator=(RiskManager const&) = delete;

    // Fluent API - Update configuration
    [[nodiscard]] auto withLimits(RiskLimits limits) noexcept -> RiskManager& {
        updateRiskLimits(limits);
        return *this;
    }

    [[nodiscard]] auto withAccountValue(double value) noexcept -> RiskManager& {
        auto limits = getRiskLimits();
        limits.account_value = value;
        updateRiskLimits(limits);
        return *this;
    }

    [[nodiscard]] auto withDailyLossLimit(double limit) noexcept -> RiskManager& {
        auto limits = getRiskLimits();
        limits.max_daily_loss = limit;
        updateRiskLimits(limits);
        return *this;
    }

    [[nodiscard]] auto withPositionSizeLimit(double limit) noexcept -> RiskManager& {
        auto limits = getRiskLimits();
        limits.max_position_size = limit;
        updateRiskLimits(limits);
        return *this;
    }

    // Trade assessment
    [[nodiscard]] auto assessTrade(std::string symbol, double position_size, Price entry_price,
                                   Price stop_price, Price target_price,
                                   double win_probability) noexcept -> Result<TradeRisk>;

    [[nodiscard]] auto assessOptionsTrade(std::string symbol, PricingParams const& params,
                                          double position_size, double win_probability,
                                          double target_profit) noexcept -> Result<TradeRisk>;

    [[nodiscard]] auto approveTrade(TradeRisk const& trade_risk) const noexcept -> Result<bool>;

    // Position management (chainable)
    [[nodiscard]] auto registerPosition(Position const& position, Price stop_price) noexcept
        -> RiskManager& {
        registerPositionInternal(position, stop_price);
        return *this;
    }

    [[nodiscard]] auto updatePosition(std::string const& symbol, Price current_price) noexcept
        -> RiskManager& {
        updatePositionInternal(symbol, current_price);
        return *this;
    }

    [[nodiscard]] auto closePosition(std::string const& symbol, Price exit_price) noexcept
        -> RiskManager& {
        closePositionInternal(symbol, exit_price);
        return *this;
    }

    // Query methods
    [[nodiscard]] auto getPortfolioRisk() const noexcept -> PortfolioRisk;

    [[nodiscard]] auto getPositionRisk(std::string const& symbol) const noexcept
        -> Result<PositionRisk>;

    [[nodiscard]] auto isDailyLossLimitReached() const noexcept -> bool;

    [[nodiscard]] auto getActivePositionCount() const noexcept -> int {
        return getPortfolioRisk().active_positions;
    }

    [[nodiscard]] auto getRiskLimits() const noexcept -> RiskLimits const&;

    // Daily operations (chainable)
    [[nodiscard]] auto resetDailyPnL() noexcept -> RiskManager& {
        resetDailyPnLInternal();
        return *this;
    }

    [[nodiscard]] auto emergencyStopAll() noexcept -> RiskManager& {
        emergencyStopAllInternal();
        return *this;
    }

  private:
    explicit RiskManager(RiskLimits limits);

    // Internal non-chainable methods
    auto registerPositionInternal(Position const& position, Price stop_price) -> void;
    auto updatePositionInternal(std::string const& symbol, Price current_price) -> void;
    auto closePositionInternal(std::string const& symbol, Price exit_price) -> void;
    auto resetDailyPnLInternal() -> void;
    auto updateRiskLimits(RiskLimits limits) -> void;
    auto emergencyStopAllInternal() -> void;

    // Pimpl pattern
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

// ============================================================================
// RiskManager::Impl (Implementation)
// ============================================================================

class RiskManager::Impl {
  public:
    explicit Impl(RiskLimits limits)
        : limits_{limits}, daily_pnl_{0.0}, daily_loss_remaining_{limits.max_daily_loss},
          positions_{} {
        Logger::getInstance().info(
            "RiskManager initialized: Account=${:.2f}, MaxLoss=${:.2f}, MaxPos=${:.2f}",
            limits.account_value, limits.max_daily_loss, limits.max_position_size);
    }

    [[nodiscard]] auto assessTrade(std::string symbol, double position_size, Price entry_price,
                                   Price stop_price, Price target_price,
                                   double win_probability) noexcept -> Result<TradeRisk> {
        std::lock_guard lock{mutex_};

        TradeRisk risk{};
        risk.symbol = symbol;
        risk.position_size = std::min(position_size, limits_.max_position_size);
        risk.max_loss = std::abs(entry_price - stop_price) * (position_size / entry_price);
        risk.expected_return = (target_price - entry_price) * (position_size / entry_price);
        risk.win_probability = win_probability;
        risk.expected_value =
            win_probability * risk.expected_return - (1.0 - win_probability) * risk.max_loss;
        risk.risk_reward_ratio = (risk.max_loss > 0.0) ? risk.expected_return / risk.max_loss : 0.0;
        risk.approved = true;

        // Check daily loss limit
        if (daily_loss_remaining_ < risk.max_loss) {
            risk.approved = false;
            risk.rejection_reason = "Daily loss limit would be exceeded";
        }

        // Check position size limit
        if (risk.position_size > limits_.max_position_size) {
            risk.approved = false;
            risk.rejection_reason = "Position size exceeds maximum";
        }

        // Check concurrent positions
        if (static_cast<int>(positions_.size()) >= limits_.max_concurrent_positions) {
            risk.approved = false;
            risk.rejection_reason = "Max concurrent positions reached";
        }

        // Check risk/reward ratio
        if (risk.risk_reward_ratio < 1.5) {
            risk.approved = false;
            risk.rejection_reason = "Risk/reward ratio below minimum 1.5";
        }

        if (risk.approved) {
            Logger::getInstance().info("Trade approved: {} ${:.2f} - EV: ${:.2f}, R/R: {:.2f}",
                                       symbol, risk.position_size, risk.expected_value,
                                       risk.risk_reward_ratio);
        } else {
            Logger::getInstance().warn("Trade rejected: {} - {}", symbol, risk.rejection_reason);
        }

        return risk;
    }

    [[nodiscard]] auto assessOptionsTrade(std::string symbol, PricingParams const& params,
                                          double position_size, double win_probability,
                                          double target_profit) noexcept -> Result<TradeRisk> {
        std::lock_guard lock{mutex_};

        TradeRisk risk{};
        risk.symbol = symbol;
        risk.position_size = std::min(position_size, limits_.max_position_size);
        risk.max_loss = position_size * 0.5; // Conservative: 50% loss
        risk.expected_return = target_profit;
        risk.win_probability = win_probability;
        risk.expected_value =
            win_probability * target_profit - (1.0 - win_probability) * risk.max_loss;
        risk.risk_reward_ratio = (risk.max_loss > 0.0) ? target_profit / risk.max_loss : 0.0;
        risk.approved = true;

        if (daily_loss_remaining_ < risk.max_loss) {
            risk.approved = false;
            risk.rejection_reason = "Daily loss limit";
        }

        if (static_cast<int>(positions_.size()) >= limits_.max_concurrent_positions) {
            risk.approved = false;
            risk.rejection_reason = "Max positions reached";
        }

        return risk;
    }

    [[nodiscard]] auto approveTrade(TradeRisk const& trade_risk) const noexcept -> Result<bool> {
        return trade_risk.approved;
    }

    auto registerPosition(Position const& position, Price stop_price) -> void {
        std::lock_guard lock{mutex_};
        std::string key = "POSITION_" + std::to_string(positions_.size());
        positions_.insert({key, position});

        Logger::getInstance().info("Position registered: {} (total: {})", key, positions_.size());
    }

    auto updatePosition(std::string const& symbol, Price current_price) -> void {
        std::lock_guard lock{mutex_};
        // Update position price tracking
        // Full implementation would update P&L, Greeks, etc.
    }

    auto closePosition(std::string const& symbol, Price exit_price) -> void {
        std::lock_guard lock{mutex_};
        if (auto it = positions_.find(symbol); it != positions_.end()) {
            positions_.erase(it);

            Logger::getInstance().info("Position closed: {} @ ${:.2f} (remaining: {})", symbol,
                                       exit_price, positions_.size());
        }
    }

    [[nodiscard]] auto getPortfolioRisk() const noexcept -> PortfolioRisk {
        std::lock_guard lock{mutex_};

        PortfolioRisk portfolio{};
        portfolio.total_value = limits_.account_value;
        portfolio.daily_pnl = daily_pnl_;
        portfolio.daily_loss_remaining = daily_loss_remaining_;
        portfolio.active_positions = static_cast<int>(positions_.size());
        portfolio.portfolio_heat = static_cast<double>(positions_.size()) /
                                   static_cast<double>(limits_.max_concurrent_positions);

        return portfolio;
    }

    [[nodiscard]] auto getPositionRisk(std::string const& symbol) const noexcept
        -> Result<PositionRisk> {
        std::lock_guard lock{mutex_};

        if (auto it = positions_.find(symbol); it != positions_.end()) {
            PositionRisk risk{};
            risk.symbol = symbol;
            risk.position_value = 1000.0; // Stub - would calculate from position
            risk.unrealized_pnl = 0.0;    // Stub
            return risk;
        }

        return makeError<PositionRisk>(ErrorCode::InvalidParameter,
                                       "Position not found: " + symbol);
    }

    [[nodiscard]] auto isDailyLossLimitReached() const noexcept -> bool {
        std::lock_guard lock{mutex_};
        return daily_pnl_ <= -limits_.max_daily_loss;
    }

    auto resetDailyPnL() -> void {
        std::lock_guard lock{mutex_};
        daily_pnl_ = 0.0;
        daily_loss_remaining_ = limits_.max_daily_loss;

        Logger::getInstance().info("Daily P&L reset");
    }

    [[nodiscard]] auto getRiskLimits() const noexcept -> RiskLimits const& { return limits_; }

    auto updateRiskLimits(RiskLimits limits) -> void {
        std::lock_guard lock{mutex_};
        limits_ = limits;

        Logger::getInstance().info("Risk limits updated: Account=${:.2f}, MaxLoss=${:.2f}",
                                   limits.account_value, limits.max_daily_loss);
    }

    auto emergencyStopAll() -> void {
        std::lock_guard lock{mutex_};
        positions_.clear();

        Logger::getInstance().error("EMERGENCY STOP: All positions cleared");
    }

  private:
    RiskLimits limits_;
    double daily_pnl_;
    mutable double daily_loss_remaining_;
    std::unordered_map<std::string, Position> positions_;
    mutable std::mutex mutex_;
};

// ============================================================================
// RiskManager Public Interface Implementation
// ============================================================================

RiskManager::RiskManager(RiskLimits limits) : pImpl_{std::make_unique<Impl>(limits)} {}

RiskManager::~RiskManager() = default;

RiskManager::RiskManager(RiskManager&&) noexcept = default;

auto RiskManager::operator=(RiskManager&&) noexcept -> RiskManager& = default;

auto RiskManager::assessTrade(std::string symbol, double position_size, Price entry_price,
                              Price stop_price, Price target_price, double win_probability) noexcept
    -> Result<TradeRisk> {
    return pImpl_->assessTrade(symbol, position_size, entry_price, stop_price, target_price,
                               win_probability);
}

auto RiskManager::assessOptionsTrade(std::string symbol, PricingParams const& params,
                                     double position_size, double win_probability,
                                     double target_profit) noexcept -> Result<TradeRisk> {
    return pImpl_->assessOptionsTrade(symbol, params, position_size, win_probability,
                                      target_profit);
}

auto RiskManager::approveTrade(TradeRisk const& trade_risk) const noexcept -> Result<bool> {
    return pImpl_->approveTrade(trade_risk);
}

auto RiskManager::registerPositionInternal(Position const& position, Price stop_price) -> void {
    pImpl_->registerPosition(position, stop_price);
}

auto RiskManager::updatePositionInternal(std::string const& symbol, Price current_price) -> void {
    pImpl_->updatePosition(symbol, current_price);
}

auto RiskManager::closePositionInternal(std::string const& symbol, Price exit_price) -> void {
    pImpl_->closePosition(symbol, exit_price);
}

auto RiskManager::getPortfolioRisk() const noexcept -> PortfolioRisk {
    return pImpl_->getPortfolioRisk();
}

auto RiskManager::getPositionRisk(std::string const& symbol) const noexcept
    -> Result<PositionRisk> {
    return pImpl_->getPositionRisk(symbol);
}

auto RiskManager::isDailyLossLimitReached() const noexcept -> bool {
    return pImpl_->isDailyLossLimitReached();
}

auto RiskManager::resetDailyPnLInternal() -> void {
    pImpl_->resetDailyPnL();
}

auto RiskManager::getRiskLimits() const noexcept -> RiskLimits const& {
    return pImpl_->getRiskLimits();
}

auto RiskManager::updateRiskLimits(RiskLimits limits) -> void {
    pImpl_->updateRiskLimits(limits);
}

auto RiskManager::emergencyStopAllInternal() -> void {
    pImpl_->emergencyStopAll();
}

} // namespace bigbrother::risk
