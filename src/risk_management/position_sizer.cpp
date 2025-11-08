/**
 * Position Sizer Implementation
 * C++23 module implementation unit
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-08
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>

// Module implementation unit declaration
module bigbrother.risk_management;

import bigbrother.utils.logger;
import bigbrother.utils.timer;

namespace bigbrother::risk {

/**
 * Kelly Criterion Implementation
 *
 * The Kelly Criterion maximizes long-term growth rate.
 * f* = (p * b - q) / b
 *
 * where:
 *   f* = fraction of capital to wager
 *   p = probability of winning
 *   q = probability of losing (1 - p)
 *   b = ratio of win amount to loss amount
 */
[[nodiscard]] constexpr auto PositionSizer::kellyFraction(
    double win_probability,
    double win_amount,
    double loss_amount
) noexcept -> double {

    if (win_probability <= 0.0 || win_probability >= 1.0) {
        return 0.0;  // Invalid probability
    }

    if (loss_amount <= 0.0) {
        return 0.0;  // Can't risk nothing
    }

    double const p = win_probability;
    double const q = 1.0 - p;
    double const b = win_amount / loss_amount;

    // Kelly fraction: f* = (p * b - q) / b
    double const kelly = (p * b - q) / b;

    // Kelly can be negative (don't take the trade)
    // or > 1 (leverage, which we avoid)
    return std::clamp(kelly, 0.0, 1.0);
}

[[nodiscard]] auto PositionSizer::calculateSize(
    Method method,
    double account_value,
    double win_probability,
    double win_amount,
    double loss_amount,
    double max_position
) noexcept -> Result<double> {

    PROFILE_SCOPE("PositionSizer::calculateSize");

    // Validate inputs
    if (account_value <= 0.0) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Account value must be positive"
        );
    }

    if (win_probability < 0.0 || win_probability > 1.0) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Win probability must be between 0 and 1"
        );
    }

    double position_size = 0.0;

    switch (method) {
        case Method::FixedDollar: {
            // Simple fixed dollar amount
            position_size = std::min(1000.0, max_position);
            break;
        }

        case Method::FixedPercent: {
            // Fixed 2% of capital per trade
            position_size = account_value * 0.02;
            break;
        }

        case Method::KellyCriterion: {
            // Full Kelly (aggressive)
            double const kelly = kellyFraction(win_probability, win_amount, loss_amount);
            position_size = account_value * kelly;

            LOG_DEBUG("Kelly fraction: {:.4f}, Position: ${:.2f}",
                     kelly, position_size);
            break;
        }

        case Method::KellyHalf: {
            // Half Kelly (conservative, recommended per PRD)
            double const kelly = kellyFraction(win_probability, win_amount, loss_amount);
            double const half_kelly = kelly * 0.5;
            position_size = account_value * half_kelly;

            LOG_DEBUG("Half-Kelly fraction: {:.4f}, Position: ${:.2f}",
                     half_kelly, position_size);
            break;
        }

        case Method::VolatilityAdjusted: {
            // Adjust for volatility (higher vol = smaller position)
            // Assume loss_amount represents volatility-adjusted risk
            double const vol_factor = std::min(1.0, 1000.0 / loss_amount);
            position_size = account_value * 0.02 * vol_factor;
            break;
        }

        case Method::RiskParity: {
            // Equal risk contribution
            // Size = (Target Risk) / (Position Risk)
            double const target_risk = account_value * 0.02;  // 2% risk
            if (loss_amount > 0.0) {
                position_size = target_risk / (loss_amount / 100.0);
            }
            break;
        }

        default:
            return makeError<double>(
                ErrorCode::InvalidParameter,
                "Unknown position sizing method"
            );
    }

    // Apply maximum position limit
    position_size = std::min(position_size, max_position);

    // Ensure non-negative
    position_size = std::max(0.0, position_size);

    return position_size;
}

[[nodiscard]] auto PositionSizer::calculateOptionsSize(
    double account_value,
    options::PricingParams const& params,
    double win_probability,
    double target_profit,
    double max_position
) noexcept -> Result<double> {

    PROFILE_SCOPE("PositionSizer::calculateOptionsSize");

    // Calculate option price first
    auto price_result = options::OptionsPricer::price(
        params,
        options::OptionsPricer::Model::Auto
    );

    if (!price_result) {
        return std::unexpected(price_result.error());
    }

    double const option_price = price_result->option_price;

    // Calculate Greeks for risk assessment
    auto greeks_result = options::OptionsPricer::greeks(
        params,
        options::OptionsPricer::Model::Auto
    );

    if (!greeks_result) {
        return std::unexpected(greeks_result.error());
    }

    auto const& greeks = *greeks_result;

    // Maximum loss for option = premium paid (100% loss)
    double const max_loss = option_price;

    // Use Half-Kelly for options (conservative)
    auto size_result = calculateSize(
        Method::KellyHalf,
        account_value,
        win_probability,
        target_profit,
        max_loss,
        max_position
    );

    if (!size_result) {
        return std::unexpected(size_result.error());
    }

    // Calculate number of contracts (each contract = 100 shares)
    double const position_size = *size_result;
    int const num_contracts = static_cast<int>(position_size / (option_price * 100.0));

    // Ensure at least 1 contract if position_size allows
    int const final_contracts = std::max(1, num_contracts);

    // Total position value
    double const total_position = final_contracts * option_price * 100.0;

    LOG_INFO("Options position: {} contracts, Total: ${:.2f}",
             final_contracts, total_position);

    return total_position;
}

} // namespace bigbrother::risk
