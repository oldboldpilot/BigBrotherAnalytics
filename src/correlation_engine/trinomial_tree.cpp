#include "options_pricing.hpp"
#include "../utils/logger.hpp"
#include "../utils/timer.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

namespace bigbrother::options {

/**
 * Trinomial Tree Model Implementation
 *
 * Default pricing method per PRD/Architecture.
 * Better convergence than binomial, handles American options optimally.
 *
 * Advantages over Binomial:
 * - More stable for small time steps
 * - Better approximation of continuous process
 * - Smoother Greeks calculation
 * - Natural handling of dividends
 */

[[nodiscard]] auto TrinomialTreeModel::price(
    PricingParams const& params,
    int num_steps
) noexcept -> Result<Price> {

    PROFILE_SCOPE("TrinomialTree::price");

    // Validate parameters
    if (auto validation = params.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    if (num_steps < 1) {
        return makeError<Price>(
            ErrorCode::InvalidParameter,
            "Number of steps must be at least 1"
        );
    }

    // Handle edge case: expired option
    if (params.time_to_expiration <= 0.0) {
        if (params.option_type == OptionType::Call) {
            return std::max(0.0, params.spot_price - params.strike_price);
        } else {
            return std::max(0.0, params.strike_price - params.spot_price);
        }
    }

    // Time step size
    double const dt = params.time_to_expiration / static_cast<double>(num_steps);

    // Risk-neutral drift
    double const drift = params.risk_free_rate - params.dividend_yield;

    // Volatility-adjusted parameters
    double const sigma_sqrt_dt = params.volatility * std::sqrt(dt);

    // Trinomial tree parameters (standard parametrization)
    double const lambda = std::sqrt(3.0);
    double const dx = sigma_sqrt_dt * lambda;

    // Risk-neutral probabilities
    double const nu = drift * dt / dx;

    double const p_up = 0.5 * (1.0 / 3.0 + nu * nu + nu);
    double const p_mid = 2.0 / 3.0 - nu * nu;
    double const p_down = 0.5 * (1.0 / 3.0 + nu * nu - nu);

    // Discount factor per step
    double const discount = std::exp(-params.risk_free_rate * dt);

    // Tree dimensions: at step i, we have (2*i + 1) nodes
    // Maximum number of nodes at final step
    int const max_nodes = 2 * num_steps + 1;

    // Asset prices at final nodes (index from bottom to top)
    std::vector<Price> asset_prices(max_nodes);
    std::vector<Price> option_values(max_nodes);

    // Initialize asset prices at final step
    // Node j at step i: S * exp((j - i) * dx)
    // At final step i = num_steps, j ranges from 0 to 2*num_steps
    for (int j = 0; j < max_nodes; ++j) {
        int const state = j - num_steps;  // Range from -num_steps to +num_steps
        asset_prices[j] = params.spot_price * std::exp(static_cast<double>(state) * dx);
    }

    // Calculate option values at final nodes (payoff)
    for (int j = 0; j < max_nodes; ++j) {
        if (params.option_type == OptionType::Call) {
            option_values[j] = std::max(0.0, asset_prices[j] - params.strike_price);
        } else {
            option_values[j] = std::max(0.0, params.strike_price - asset_prices[j]);
        }
    }

    // Backward induction through the tree
    for (int step = num_steps - 1; step >= 0; --step) {
        int const nodes_at_step = 2 * step + 1;

        for (int j = 0; j < nodes_at_step; ++j) {
            // Calculate expected value (risk-neutral valuation)
            double expected_value = p_up * option_values[j + 2] +
                                   p_mid * option_values[j + 1] +
                                   p_down * option_values[j];

            // Discount back one step
            expected_value *= discount;

            // For American options, check early exercise
            if (params.option_style == OptionStyle::American) {
                int const state = j - step;
                Price const spot_at_node = params.spot_price *
                                          std::exp(static_cast<double>(state) * dx);

                double intrinsic_value = 0.0;
                if (params.option_type == OptionType::Call) {
                    intrinsic_value = std::max(0.0, spot_at_node - params.strike_price);
                } else {
                    intrinsic_value = std::max(0.0, params.strike_price - spot_at_node);
                }

                // Exercise if intrinsic value is higher
                option_values[j] = std::max(expected_value, intrinsic_value);
            } else {
                // European option - just use expected value
                option_values[j] = expected_value;
            }
        }
    }

    // Option value at root node
    Price const option_price = option_values[0];

    return option_price;
}

[[nodiscard]] auto TrinomialTreeModel::greeks(
    PricingParams const& params,
    int num_steps
) noexcept -> Result<Greeks> {

    PROFILE_SCOPE("TrinomialTree::greeks");

    // Use finite differences to calculate Greeks
    // This is more accurate than analytical formulas for American options

    auto base_price = price(params, num_steps);
    if (!base_price) {
        return std::unexpected(base_price.error());
    }

    Greeks result;

    // Delta: ∂V/∂S (bump spot by 1%)
    {
        double const bump = params.spot_price * 0.01;
        auto params_up = params;
        params_up.spot_price += bump;

        auto price_up = price(params_up, num_steps);
        if (!price_up) {
            return std::unexpected(price_up.error());
        }

        result.delta = (*price_up - *base_price) / bump;
    }

    // Gamma: ∂²V/∂S² (second derivative)
    {
        double const bump = params.spot_price * 0.01;

        auto params_up = params;
        params_up.spot_price += bump;

        auto params_down = params;
        params_down.spot_price -= bump;

        auto price_up = price(params_up, num_steps);
        auto price_down = price(params_down, num_steps);

        if (!price_up || !price_down) {
            return makeError<Greeks>(
                ErrorCode::UnknownError,
                "Failed to calculate Gamma"
            );
        }

        result.gamma = (*price_up - 2.0 * (*base_price) + *price_down) / (bump * bump);
    }

    // Theta: ∂V/∂t (bump time by 1 day)
    {
        double const one_day = 1.0 / 365.0;

        if (params.time_to_expiration > one_day) {
            auto params_minus_day = params;
            params_minus_day.time_to_expiration -= one_day;

            auto price_minus_day = price(params_minus_day, num_steps);
            if (!price_minus_day) {
                return std::unexpected(price_minus_day.error());
            }

            result.theta = (*price_minus_day - *base_price);  // Per day
        } else {
            result.theta = -*base_price;  // Expires tomorrow
        }
    }

    // Vega: ∂V/∂σ (bump volatility by 1%)
    {
        double const bump = 0.01;  // 1% absolute change

        auto params_vega = params;
        params_vega.volatility += bump;

        auto price_vega = price(params_vega, num_steps);
        if (!price_vega) {
            return std::unexpected(price_vega.error());
        }

        result.vega = (*price_vega - *base_price) / bump;
    }

    // Rho: ∂V/∂r (bump rate by 1%)
    {
        double const bump = 0.01;  // 1% absolute change

        auto params_rho = params;
        params_rho.risk_free_rate += bump;

        auto price_rho = price(params_rho, num_steps);
        if (!price_rho) {
            return std::unexpected(price_rho.error());
        }

        result.rho = (*price_rho - *base_price) / bump;
    }

    return result;
}

} // namespace bigbrother::options
