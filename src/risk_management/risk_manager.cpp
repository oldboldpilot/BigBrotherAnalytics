/**
 * Risk Manager Implementation
 * C++23 module implementation file
 * Thread-safe risk management for algorithmic trading.
 */

import bigbrother.risk_management;
import bigbrother.utils.logger;

#include <algorithm>

namespace bigbrother::risk {

// ============================================================================
// RiskLimits Implementation
// ============================================================================

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
    if (max_concurrent_positions <= 0) {
        return makeError<void>(ErrorCode::InvalidParameter, "Invalid concurrent positions limit");
    }
    return {};
}

// ============================================================================
// PortfolioRisk Implementation
// ============================================================================

auto PortfolioRisk::canOpenNewPosition() const noexcept -> bool {
    return active_positions < 10 && daily_loss_remaining > 0.0;
}

auto PortfolioRisk::getRiskLevel() const noexcept -> std::string {
    if (portfolio_heat > 0.15) return "HIGH";
    if (portfolio_heat > 0.10) return "MEDIUM";
    return "LOW";
}

// ============================================================================
// RiskManager::Impl
// ============================================================================

class RiskManager::Impl {
public:
    explicit Impl(RiskLimits limits)
        : limits_{limits},
          daily_pnl_{0.0},
          positions_{} {}

    auto assessTrade(
        std::string symbol,
        double position_size,
        Price entry_price,
        Price stop_price,
        Price target_price,
        double win_probability
    ) noexcept -> Result<TradeRisk> {

        TradeRisk risk{};
        risk.position_size = std::min(position_size, limits_.max_position_size);
        risk.max_loss = std::abs(entry_price - stop_price) * (position_size / entry_price);
        risk.expected_return = (target_price - entry_price) * (position_size / entry_price);
        risk.win_probability = win_probability;
        risk.expected_value = win_probability * risk.expected_return -
                              (1.0 - win_probability) * risk.max_loss;
        risk.risk_reward_ratio = risk.expected_return / risk.max_loss;
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

        return risk;
    }

    auto assessOptionsTrade(
        std::string symbol,
        options::PricingParams const& params,
        double position_size,
        double win_probability,
        double target_profit
    ) noexcept -> Result<TradeRisk> {

        TradeRisk risk{};
        risk.position_size = std::min(position_size, limits_.max_position_size);
        risk.max_loss = position_size * 0.5;  // Conservative estimate
        risk.expected_return = target_profit;
        risk.win_probability = win_probability;
        risk.expected_value = win_probability * target_profit -
                              (1.0 - win_probability) * risk.max_loss;
        risk.approved = true;

        if (daily_loss_remaining_ < risk.max_loss) {
            risk.approved = false;
            risk.rejection_reason = "Daily loss limit";
        }

        return risk;
    }

    auto approveTrade(TradeRisk const& trade_risk) const noexcept -> Result<bool> {
        return trade_risk.approved;
    }

    auto registerPosition(Position const& position, Price stop_price) -> void {
        std::lock_guard lock{mutex_};
        // Position class has symbol() method in types.cppm module
        // For now use string directly
        positions_["STUB_" + std::to_string(positions_.size())] = position;
    }

    auto updatePosition(std::string const& symbol, Price current_price) -> void {
        std::lock_guard lock{mutex_};
        // Stub - positions_ uses simple Position type
        // Will enhance when we have full position tracking
    }

    auto closePosition(std::string const& symbol, Price exit_price) -> void {
        std::lock_guard lock{mutex_};
        if (auto it = positions_.find(symbol); it != positions_.end()) {
            // Simple P&L calculation
            positions_.erase(it);
        }
    }

    auto getPortfolioRisk() const noexcept -> PortfolioRisk {
        std::lock_guard lock{mutex_};

        PortfolioRisk portfolio{};
        portfolio.total_value = limits_.account_value;
        portfolio.daily_pnl = daily_pnl_;
        portfolio.daily_loss_remaining = limits_.max_daily_loss - std::abs(std::min(0.0, daily_pnl_));
        portfolio.active_positions = static_cast<int>(positions_.size());
        portfolio.portfolio_heat = 0.05;  // Stub

        return portfolio;
    }

    auto getPositionRisk(std::string const& symbol) const noexcept -> Result<PositionRisk> {
        std::lock_guard lock{mutex_};

        if (auto it = positions_.find(symbol); it != positions_.end()) {
            PositionRisk risk{};
            risk.symbol = symbol;
            risk.position_value = 1000.0;  // Stub
            risk.unrealized_pnl = 0.0;     // Stub
            return risk;
        }

        return makeError<PositionRisk>(ErrorCode::InvalidParameter, "Position not found");
    }

    auto isDailyLossLimitReached() const noexcept -> bool {
        std::lock_guard lock{mutex_};
        return daily_pnl_ <= -limits_.max_daily_loss;
    }

    auto resetDailyPnL() -> void {
        std::lock_guard lock{mutex_};
        daily_pnl_ = 0.0;
    }

    auto getRiskLimits() const noexcept -> RiskLimits const& {
        return limits_;
    }

    auto updateRiskLimits(RiskLimits limits) -> void {
        std::lock_guard lock{mutex_};
        limits_ = limits;
    }

    auto emergencyStopAll() -> void {
        std::lock_guard lock{mutex_};
        positions_.clear();
    }

private:
    RiskLimits limits_;
    double daily_pnl_;
    mutable double daily_loss_remaining_;
    std::unordered_map<std::string, Position> positions_;
    mutable std::mutex mutex_;
};

// ============================================================================
// RiskManager Public Interface
// ============================================================================

RiskManager::RiskManager(RiskLimits limits)
    : pImpl_{std::make_unique<Impl>(limits)} {}

RiskManager::~RiskManager() = default;

RiskManager::RiskManager(RiskManager&&) noexcept = default;

auto RiskManager::operator=(RiskManager&&) noexcept -> RiskManager& = default;

auto RiskManager::assessTrade(
    std::string symbol,
    double position_size,
    Price entry_price,
    Price stop_price,
    Price target_price,
    double win_probability
) noexcept -> Result<TradeRisk> {
    return pImpl_->assessTrade(symbol, position_size, entry_price,
                               stop_price, target_price, win_probability);
}

auto RiskManager::assessOptionsTrade(
    std::string symbol,
    options::PricingParams const& params,
    double position_size,
    double win_probability,
    double target_profit
) noexcept -> Result<TradeRisk> {
    return pImpl_->assessOptionsTrade(symbol, params, position_size,
                                     win_probability, target_profit);
}

auto RiskManager::approveTrade(TradeRisk const& trade_risk) const noexcept -> Result<bool> {
    return pImpl_->approveTrade(trade_risk);
}

auto RiskManager::registerPosition(Position const& position, Price stop_price) -> void {
    pImpl_->registerPosition(position, stop_price);
}

auto RiskManager::updatePosition(std::string const& symbol, Price current_price) -> void {
    pImpl_->updatePosition(symbol, current_price);
}

auto RiskManager::closePosition(std::string const& symbol, Price exit_price) -> void {
    pImpl_->closePosition(symbol, exit_price);
}

auto RiskManager::getPortfolioRisk() const noexcept -> PortfolioRisk {
    return pImpl_->getPortfolioRisk();
}

auto RiskManager::getPositionRisk(std::string const& symbol) const noexcept -> Result<PositionRisk> {
    return pImpl_->getPositionRisk(symbol);
}

auto RiskManager::isDailyLossLimitReached() const noexcept -> bool {
    return pImpl_->isDailyLossLimitReached();
}

auto RiskManager::resetDailyPnL() -> void {
    pImpl_->resetDailyPnL();
}

auto RiskManager::getRiskLimits() const noexcept -> RiskLimits const& {
    return pImpl_->getRiskLimits();
}

auto RiskManager::updateRiskLimits(RiskLimits limits) -> void {
    pImpl_->updateRiskLimits(limits);
}

auto RiskManager::emergencyStopAll() -> void {
    pImpl_->emergencyStopAll();
}

} // namespace bigbrother::risk
