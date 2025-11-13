/**
 * BigBrotherAnalytics - Position Sizer Module (C++23)
 *
 * Fluent API for intelligent position sizing using multiple algorithms.
 * Maximizes returns while managing risk through Kelly Criterion and variants.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-12
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - constexpr for compile-time computation
 * - noexcept for performance-critical paths
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.risk.position_sizer;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.timer;
import bigbrother.options.pricing;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using namespace bigbrother::options;
using bigbrother::utils::Logger;

// Type aliases for convenience
using PricingParams = bigbrother::options::ExtendedPricingParams;

// ============================================================================
// Position Sizing Methods
// ============================================================================

enum class SizingMethod {
    FixedDollar,           // Fixed dollar amount per trade
    FixedPercent,          // Fixed percentage of capital
    KellyCriterion,        // Full Kelly (aggressive)
    KellyHalf,             // Half Kelly (recommended)
    KellyQuarter,          // Quarter Kelly (conservative)
    VolatilityAdjusted,    // Adjust for volatility
    RiskParity,            // Equal risk contribution
    DeltaAdjusted,         // Size based on delta exposure
    MaxDrawdown            // Based on maximum drawdown tolerance
};

// ============================================================================
// Position Size Result
// ============================================================================

struct PositionSize {
    double dollar_amount{0.0};
    int num_contracts{0};      // For options
    double kelly_fraction{0.0};
    double risk_percent{0.0};
    SizingMethod method_used{SizingMethod::KellyHalf};
    std::string reasoning;

    [[nodiscard]] auto isValid() const noexcept -> bool {
        return dollar_amount > 0.0 && num_contracts > 0;
    }
};

// ============================================================================
// Position Sizer - Fluent API
// ============================================================================

class PositionSizer {
public:
    // Factory method
    [[nodiscard]] static auto create() noexcept -> PositionSizer {
        return PositionSizer{};
    }

    // Fluent configuration
    [[nodiscard]] auto withMethod(SizingMethod method) noexcept -> PositionSizer& {
        method_ = method;
        return *this;
    }

    [[nodiscard]] auto withAccountValue(double value) noexcept -> PositionSizer& {
        account_value_ = value;
        return *this;
    }

    [[nodiscard]] auto withWinProbability(double prob) noexcept -> PositionSizer& {
        win_probability_ = prob;
        return *this;
    }

    [[nodiscard]] auto withExpectedGain(double gain) noexcept -> PositionSizer& {
        expected_gain_ = gain;
        return *this;
    }

    [[nodiscard]] auto withExpectedLoss(double loss) noexcept -> PositionSizer& {
        expected_loss_ = loss;
        return *this;
    }

    [[nodiscard]] auto withMaxPosition(double max) noexcept -> PositionSizer& {
        max_position_ = max;
        return *this;
    }

    [[nodiscard]] auto withVolatility(double vol) noexcept -> PositionSizer& {
        volatility_ = vol;
        return *this;
    }

    [[nodiscard]] auto withDelta(double delta) noexcept -> PositionSizer& {
        delta_ = delta;
        return *this;
    }

    [[nodiscard]] auto withMaxDrawdown(double dd) noexcept -> PositionSizer& {
        max_drawdown_ = dd;
        return *this;
    }

    // Calculate position size
    [[nodiscard]] auto calculate() const noexcept -> Result<PositionSize>;

    // Calculate for options
    [[nodiscard]] auto calculateForOptions(
        PricingParams const& params,
        double target_profit
    ) const noexcept -> Result<PositionSize>;

    // Kelly Criterion (static utility)
    [[nodiscard]] static constexpr auto kellyFraction(
        double win_prob,
        double win_amount,
        double loss_amount
    ) noexcept -> double {
        if (win_prob <= 0.0 || win_prob >= 1.0 || loss_amount <= 0.0) {
            return 0.0;
        }

        double const p = win_prob;
        double const q = 1.0 - p;
        double const b = win_amount / loss_amount;

        // Kelly fraction: f* = (p * b - q) / b
        double const kelly = (p * b - q) / b;

        return std::clamp(kelly, 0.0, 1.0);
    }

private:
    PositionSizer() = default;

    // Configuration
    SizingMethod method_{SizingMethod::KellyHalf};
    double account_value_{30'000.0};
    double win_probability_{0.55};
    double expected_gain_{0.0};
    double expected_loss_{0.0};
    double max_position_{2'000.0};
    double volatility_{0.20};
    double delta_{0.0};
    double max_drawdown_{0.15};

    // Internal calculation methods
    [[nodiscard]] auto calculateFixedDollar() const noexcept -> double {
        return std::min(1'000.0, max_position_);
    }

    [[nodiscard]] auto calculateFixedPercent() const noexcept -> double {
        return account_value_ * 0.02;  // 2% of capital
    }

    [[nodiscard]] auto calculateKelly(double fraction_multiplier) const noexcept -> double {
        double const kelly = kellyFraction(win_probability_, expected_gain_, expected_loss_);
        return account_value_ * kelly * fraction_multiplier;
    }

    [[nodiscard]] auto calculateVolatilityAdjusted() const noexcept -> double {
        // Higher volatility = smaller position
        double const vol_factor = std::min(1.0, 0.20 / volatility_);
        return account_value_ * 0.02 * vol_factor;
    }

    [[nodiscard]] auto calculateRiskParity() const noexcept -> double {
        double const target_risk = account_value_ * 0.02;  // 2% risk
        if (expected_loss_ > 0.0) {
            return target_risk / (expected_loss_ / 100.0);
        }
        return 0.0;
    }

    [[nodiscard]] auto calculateDeltaAdjusted() const noexcept -> double {
        // Size based on delta exposure
        if (std::abs(delta_) < 0.01) {
            return 0.0;
        }
        double const target_delta = account_value_ * 0.01;  // 1% delta exposure
        return target_delta / std::abs(delta_);
    }

    [[nodiscard]] auto calculateMaxDrawdownBased() const noexcept -> double {
        // Size to limit maximum drawdown
        double const acceptable_loss = account_value_ * max_drawdown_;
        if (expected_loss_ > 0.0) {
            return acceptable_loss / expected_loss_;
        }
        return 0.0;
    }
};

// ============================================================================
// Implementation
// ============================================================================

auto PositionSizer::calculate() const noexcept -> Result<PositionSize> {
    // Validate inputs
    if (account_value_ <= 0.0) {
        return makeError<PositionSize>(
            ErrorCode::InvalidParameter,
            "Account value must be positive"
        );
    }

    if (win_probability_ < 0.0 || win_probability_ > 1.0) {
        return makeError<PositionSize>(
            ErrorCode::InvalidParameter,
            "Win probability must be between 0 and 1"
        );
    }

    double position_size = 0.0;
    std::string reasoning;

    switch (method_) {
        case SizingMethod::FixedDollar:
            position_size = calculateFixedDollar();
            reasoning = "Fixed $1,000 per trade";
            break;

        case SizingMethod::FixedPercent:
            position_size = calculateFixedPercent();
            reasoning = "Fixed 2% of capital";
            break;

        case SizingMethod::KellyCriterion:
            position_size = calculateKelly(1.0);
            reasoning = "Full Kelly criterion (aggressive)";
            break;

        case SizingMethod::KellyHalf:
            position_size = calculateKelly(0.5);
            reasoning = "Half Kelly criterion (recommended)";
            break;

        case SizingMethod::KellyQuarter:
            position_size = calculateKelly(0.25);
            reasoning = "Quarter Kelly criterion (conservative)";
            break;

        case SizingMethod::VolatilityAdjusted:
            position_size = calculateVolatilityAdjusted();
            reasoning = "Volatility-adjusted";
            break;

        case SizingMethod::RiskParity:
            position_size = calculateRiskParity();
            reasoning = "Risk parity (equal risk contribution)";
            break;

        case SizingMethod::DeltaAdjusted:
            position_size = calculateDeltaAdjusted();
            reasoning = "Delta-adjusted";
            break;

        case SizingMethod::MaxDrawdown:
            position_size = calculateMaxDrawdownBased();
            reasoning = "Max drawdown limited";
            break;
    }

    // Apply maximum position limit
    position_size = std::min(position_size, max_position_);
    position_size = std::max(0.0, position_size);

    // Calculate Kelly fraction for reference
    double const kelly = kellyFraction(win_probability_, expected_gain_, expected_loss_);

    PositionSize result;
    result.dollar_amount = position_size;
    result.num_contracts = 1;  // Will be calculated for options
    result.kelly_fraction = kelly;
    result.risk_percent = (expected_loss_ / account_value_) * 100.0;
    result.method_used = method_;
    result.reasoning = reasoning;

    Logger::getInstance().debug(
        "Position size calculated: ${:.2f} ({}) - Kelly: {:.3f}",
        position_size,
        reasoning,
        kelly
    );

    return result;
}

auto PositionSizer::calculateForOptions(
    PricingParams const& params,
    double target_profit
) const noexcept -> Result<PositionSize> {
    // Calculate option price
    auto price_result = OptionsPricer::price(params, OptionsPricer::Model::Auto);
    if (!price_result) {
        return std::unexpected(price_result.error());
    }

    double const option_price = price_result->option_price;

    // Calculate Greeks for risk assessment
    auto greeks_result = OptionsPricer::greeks(params, OptionsPricer::Model::Auto);
    if (!greeks_result) {
        return std::unexpected(greeks_result.error());
    }

    auto const& greeks = *greeks_result;

    // Maximum loss for option = premium paid (100% loss)
    double const max_loss = option_price;

    // Use configured method (default: Half-Kelly for options)
    auto temp_sizer = *this;
    temp_sizer.expected_gain_ = target_profit;
    temp_sizer.expected_loss_ = max_loss;
    temp_sizer.delta_ = greeks.delta;

    auto size_result = temp_sizer.calculate();
    if (!size_result) {
        return std::unexpected(size_result.error());
    }

    auto result = *size_result;

    // Calculate number of contracts (each contract = 100 shares)
    result.num_contracts = static_cast<int>(result.dollar_amount / (option_price * 100.0));
    result.num_contracts = std::max(1, result.num_contracts);

    // Update total position value
    result.dollar_amount = result.num_contracts * option_price * 100.0;

    Logger::getInstance().info(
        "Options position: {} contracts @ ${:.2f}, Total: ${:.2f}, Delta: {:.3f}",
        result.num_contracts,
        option_price,
        result.dollar_amount,
        greeks.delta
    );

    return result;
}

} // namespace bigbrother::risk
