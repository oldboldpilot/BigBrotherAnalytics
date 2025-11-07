#include "options_pricing.hpp"
#include "../utils/logger.hpp"
#include "../utils/timer.hpp"

#include <cmath>
#include <numbers>
#include <algorithm>

namespace bigbrother::options {

using std::numbers::pi;
using std::numbers::inv_sqrtpi;
using std::numbers::sqrt2;

// Standard normal cumulative distribution function
// Uses Abramowitz and Stegun approximation for speed
[[nodiscard]] constexpr auto BlackScholesModel::cumulative_normal(double x)
    noexcept -> double {

    constexpr double a1 = 0.254829592;
    constexpr double a2 = -0.284496736;
    constexpr double a3 = 1.421413741;
    constexpr double a4 = -1.453152027;
    constexpr double a5 = 1.061405429;
    constexpr double p = 0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0.0) {
        sign = -1;
    }
    double const abs_x = std::abs(x);

    // A&S formula 7.1.26
    double const t = 1.0 / (1.0 + p * abs_x);
    double const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
                     * std::exp(-abs_x * abs_x / 2.0) * inv_sqrtpi / sqrt2;

    return 0.5 * (1.0 + sign * y);
}

// Standard normal probability density function
[[nodiscard]] constexpr auto BlackScholesModel::probability_density(double x)
    noexcept -> double {
    return std::exp(-0.5 * x * x) * inv_sqrtpi / sqrt2;
}

// Calculate d1 parameter
[[nodiscard]] constexpr auto BlackScholesModel::calculate_d1(
    Price S, Price K, double T, double r, double sigma, double q
) noexcept -> double {

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    return (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
           (sigma * std::sqrt(T));
}

// Calculate d2 parameter
[[nodiscard]] constexpr auto BlackScholesModel::calculate_d2(
    double d1, double sigma, double T
) noexcept -> double {
    return d1 - sigma * std::sqrt(T);
}

// Black-Scholes option pricing
[[nodiscard]] constexpr auto BlackScholesModel::price(PricingParams const& params)
    noexcept -> Result<Price> {

    PROFILE_SCOPE("BlackScholes::price");

    // Validate parameters
    if (auto validation = params.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    // Handle edge cases
    if (params.time_to_expiration <= 0.0) {
        // Option has expired - intrinsic value only
        if (params.option_type == OptionType::Call) {
            return std::max(0.0, params.spot_price - params.strike_price);
        } else {
            return std::max(0.0, params.strike_price - params.spot_price);
        }
    }

    if (params.volatility <= 0.0) {
        // Zero volatility - forward value
        double const forward = params.spot_price *
                               std::exp((params.risk_free_rate - params.dividend_yield) *
                                       params.time_to_expiration);

        if (params.option_type == OptionType::Call) {
            return std::max(0.0, forward - params.strike_price) *
                   std::exp(-params.risk_free_rate * params.time_to_expiration);
        } else {
            return std::max(0.0, params.strike_price - forward) *
                   std::exp(-params.risk_free_rate * params.time_to_expiration);
        }
    }

    // Calculate d1 and d2
    double const d1 = calculate_d1(
        params.spot_price,
        params.strike_price,
        params.time_to_expiration,
        params.risk_free_rate,
        params.volatility,
        params.dividend_yield
    );

    double const d2 = calculate_d2(
        d1,
        params.volatility,
        params.time_to_expiration
    );

    // Calculate option price
    double const discount_factor = std::exp(-params.risk_free_rate *
                                           params.time_to_expiration);
    double const dividend_discount = std::exp(-params.dividend_yield *
                                              params.time_to_expiration);

    Price option_price = 0.0;

    if (params.option_type == OptionType::Call) {
        // Call option: S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
        option_price = params.spot_price * dividend_discount * cumulative_normal(d1) -
                      params.strike_price * discount_factor * cumulative_normal(d2);
    } else {
        // Put option: K*e^(-r*T)*N(-d2) - S*e^(-q*T)*N(-d1)
        option_price = params.strike_price * discount_factor * cumulative_normal(-d2) -
                      params.spot_price * dividend_discount * cumulative_normal(-d1);
    }

    // Ensure non-negative price
    option_price = std::max(0.0, option_price);

    return option_price;
}

// Calculate Greeks
[[nodiscard]] constexpr auto BlackScholesModel::greeks(PricingParams const& params)
    noexcept -> Result<Greeks> {

    PROFILE_SCOPE("BlackScholes::greeks");

    // Validate parameters
    if (auto validation = params.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    // Handle edge cases
    if (params.time_to_expiration <= 0.0 || params.volatility <= 0.0) {
        return Greeks{0.0, 0.0, 0.0, 0.0, 0.0};
    }

    // Calculate d1 and d2
    double const d1 = calculate_d1(
        params.spot_price,
        params.strike_price,
        params.time_to_expiration,
        params.risk_free_rate,
        params.volatility,
        params.dividend_yield
    );

    double const d2 = calculate_d2(
        d1,
        params.volatility,
        params.time_to_expiration
    );

    double const sqrt_T = std::sqrt(params.time_to_expiration);
    double const discount_factor = std::exp(-params.risk_free_rate *
                                           params.time_to_expiration);
    double const dividend_discount = std::exp(-params.dividend_yield *
                                              params.time_to_expiration);

    double const Nd1 = cumulative_normal(d1);
    double const Nd2 = cumulative_normal(d2);
    double const nd1 = probability_density(d1);

    Greeks result;

    // Delta: ∂V/∂S
    if (params.option_type == OptionType::Call) {
        result.delta = dividend_discount * Nd1;
    } else {
        result.delta = dividend_discount * (Nd1 - 1.0);
    }

    // Gamma: ∂²V/∂S²
    result.gamma = dividend_discount * nd1 /
                   (params.spot_price * params.volatility * sqrt_T);

    // Vega: ∂V/∂σ (per 1% change in volatility)
    result.vega = params.spot_price * dividend_discount * nd1 * sqrt_T / 100.0;

    // Theta: ∂V/∂t (per day)
    double const theta_term1 = -params.spot_price * dividend_discount * nd1 *
                                params.volatility / (2.0 * sqrt_T);

    if (params.option_type == OptionType::Call) {
        result.theta = (theta_term1 -
                       params.risk_free_rate * params.strike_price * discount_factor * Nd2 +
                       params.dividend_yield * params.spot_price * dividend_discount * Nd1) / 365.0;
    } else {
        result.theta = (theta_term1 +
                       params.risk_free_rate * params.strike_price * discount_factor *
                       cumulative_normal(-d2) -
                       params.dividend_yield * params.spot_price * dividend_discount *
                       cumulative_normal(-d1)) / 365.0;
    }

    // Rho: ∂V/∂r (per 1% change in interest rate)
    if (params.option_type == OptionType::Call) {
        result.rho = params.strike_price * params.time_to_expiration *
                    discount_factor * Nd2 / 100.0;
    } else {
        result.rho = -params.strike_price * params.time_to_expiration *
                     discount_factor * cumulative_normal(-d2) / 100.0;
    }

    return result;
}

// Implied volatility using Newton-Raphson
[[nodiscard]] auto BlackScholesModel::implied_volatility(
    Price market_price,
    PricingParams params,
    int max_iterations,
    double tolerance
) noexcept -> Result<double> {

    PROFILE_SCOPE("BlackScholes::implied_volatility");

    // Validate market price
    if (market_price <= 0.0) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Market price must be positive"
        );
    }

    // Validate other parameters
    if (auto validation = params.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    // Check intrinsic value
    double intrinsic_value = 0.0;
    if (params.option_type == OptionType::Call) {
        intrinsic_value = std::max(0.0, params.spot_price - params.strike_price);
    } else {
        intrinsic_value = std::max(0.0, params.strike_price - params.spot_price);
    }

    if (market_price < intrinsic_value) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Market price below intrinsic value"
        );
    }

    // Initial guess (at-the-money approximation)
    double sigma = std::sqrt(2.0 * pi / params.time_to_expiration) *
                   (market_price / params.spot_price);

    // Ensure reasonable initial guess
    sigma = std::clamp(sigma, 0.01, 5.0);

    // Newton-Raphson iteration
    for (int iter = 0; iter < max_iterations; ++iter) {
        params.volatility = sigma;

        // Calculate price and vega
        auto price_result = price(params);
        auto greeks_result = greeks(params);

        if (!price_result || !greeks_result) {
            return makeError<double>(
                ErrorCode::UnknownError,
                "Failed to calculate price or Greeks"
            );
        }

        double const calculated_price = *price_result;
        double const vega = greeks_result->vega;

        // Check for convergence
        double const price_diff = calculated_price - market_price;
        if (std::abs(price_diff) < tolerance) {
            return sigma;
        }

        // Check for zero vega (no sensitivity)
        if (std::abs(vega) < 1e-10) {
            return makeError<double>(
                ErrorCode::UnknownError,
                "Vega too small for convergence"
            );
        }

        // Newton-Raphson update (vega is per 1%, so multiply by 100)
        double const delta_sigma = -price_diff / (vega * 100.0);

        // Update with damping for stability
        sigma += std::clamp(delta_sigma, -0.5, 0.5);

        // Keep sigma in reasonable range
        sigma = std::clamp(sigma, 0.001, 10.0);
    }

    // Failed to converge
    return makeError<double>(
        ErrorCode::UnknownError,
        "Implied volatility did not converge"
    );
}

} // namespace bigbrother::options
