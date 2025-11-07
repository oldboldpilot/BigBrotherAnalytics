#pragma once

#include "options_pricing.hpp"
#include <optional>

namespace bigbrother::options {

/**
 * Fluent API Builder for Options Pricing
 *
 * Provides intuitive, chainable interface for option pricing.
 *
 * Example Usage:
 *
 *   // Price a call option
 *   auto result = OptionBuilder()
 *       .call()
 *       .american()
 *       .spot(150.0)
 *       .strike(155.0)
 *       .daysToExpiration(30)
 *       .volatility(0.25)
 *       .riskFreeRate(0.05)
 *       .price();
 *
 *   // Calculate Greeks
 *   auto greeks = OptionBuilder()
 *       .put()
 *       .european()
 *       .spot(100.0)
 *       .strike(100.0)
 *       .yearsToExpiration(0.25)
 *       .volatility(0.30)
 *       .greeks();
 *
 *   // Calculate implied volatility
 *   auto iv = OptionBuilder()
 *       .call()
 *       .spot(150.0)
 *       .strike(155.0)
 *       .daysToExpiration(30)
 *       .marketPrice(5.25)
 *       .impliedVolatility();
 */
class OptionBuilder {
public:
    OptionBuilder()
        : params_{
            .spot_price = 0.0,
            .strike_price = 0.0,
            .time_to_expiration = 0.0,
            .risk_free_rate = 0.0,
            .volatility = 0.0,
            .dividend_yield = 0.0,
            .option_type = OptionType::Call,
            .option_style = OptionStyle::American
        },
          model_{OptionsPricer::Model::Auto},
          tree_steps_{100},
          market_price_{std::nullopt} {}

    // Option type
    [[nodiscard]] auto call() noexcept -> OptionBuilder& {
        params_.option_type = OptionType::Call;
        return *this;
    }

    [[nodiscard]] auto put() noexcept -> OptionBuilder& {
        params_.option_type = OptionType::Put;
        return *this;
    }

    // Option style
    [[nodiscard]] auto american() noexcept -> OptionBuilder& {
        params_.option_style = OptionStyle::American;
        return *this;
    }

    [[nodiscard]] auto european() noexcept -> OptionBuilder& {
        params_.option_style = OptionStyle::European;
        return *this;
    }

    // Pricing parameters
    [[nodiscard]] auto spot(Price price) noexcept -> OptionBuilder& {
        params_.spot_price = price;
        return *this;
    }

    [[nodiscard]] auto strike(Price price) noexcept -> OptionBuilder& {
        params_.strike_price = price;
        return *this;
    }

    [[nodiscard]] auto yearsToExpiration(double years) noexcept -> OptionBuilder& {
        params_.time_to_expiration = years;
        return *this;
    }

    [[nodiscard]] auto daysToExpiration(int days) noexcept -> OptionBuilder& {
        params_.time_to_expiration = static_cast<double>(days) / 365.0;
        return *this;
    }

    [[nodiscard]] auto expiration(Timestamp timestamp, Timestamp current_time) noexcept
        -> OptionBuilder& {
        // Convert microseconds to years
        double const seconds = static_cast<double>(timestamp - current_time) / 1'000'000.0;
        params_.time_to_expiration = seconds / (365.0 * 86400.0);
        return *this;
    }

    [[nodiscard]] auto volatility(double vol) noexcept -> OptionBuilder& {
        params_.volatility = vol;
        return *this;
    }

    [[nodiscard]] auto riskFreeRate(double rate) noexcept -> OptionBuilder& {
        params_.risk_free_rate = rate;
        return *this;
    }

    [[nodiscard]] auto dividendYield(double yield) noexcept -> OptionBuilder& {
        params_.dividend_yield = yield;
        return *this;
    }

    // Pricing model selection
    [[nodiscard]] auto useBlackScholes() noexcept -> OptionBuilder& {
        model_ = OptionsPricer::Model::BlackScholes;
        return *this;
    }

    [[nodiscard]] auto useBinomial(int steps = 100) noexcept -> OptionBuilder& {
        model_ = OptionsPricer::Model::Binomial;
        tree_steps_ = steps;
        return *this;
    }

    [[nodiscard]] auto useTrinomial(int steps = 100) noexcept -> OptionBuilder& {
        model_ = OptionsPricer::Model::Trinomial;
        tree_steps_ = steps;
        return *this;
    }

    [[nodiscard]] auto useMonteCarlo(int simulations = 10000) noexcept -> OptionBuilder& {
        model_ = OptionsPricer::Model::MonteCarlo;
        tree_steps_ = simulations;
        return *this;
    }

    [[nodiscard]] auto autoSelectModel() noexcept -> OptionBuilder& {
        model_ = OptionsPricer::Model::Auto;
        return *this;
    }

    // Market price for IV calculation
    [[nodiscard]] auto marketPrice(Price price) noexcept -> OptionBuilder& {
        market_price_ = price;
        return *this;
    }

    // Terminal operations

    /**
     * Calculate option price
     */
    [[nodiscard]] auto price() const noexcept -> Result<Price> {
        auto result = OptionsPricer::price(params_, model_);
        if (!result) {
            return std::unexpected(result.error());
        }
        return result->option_price;
    }

    /**
     * Calculate full pricing result (price + Greeks)
     */
    [[nodiscard]] auto priceWithGreeks() const noexcept -> Result<PricingResult> {
        return OptionsPricer::price(params_, model_);
    }

    /**
     * Calculate Greeks only
     */
    [[nodiscard]] auto greeks() const noexcept -> Result<Greeks> {
        return OptionsPricer::greeks(params_, model_);
    }

    /**
     * Calculate implied volatility
     */
    [[nodiscard]] auto impliedVolatility() const noexcept -> Result<double> {
        if (!market_price_) {
            return makeError<double>(
                ErrorCode::InvalidParameter,
                "Market price not specified. Use marketPrice() method."
            );
        }
        return OptionsPricer::implied_volatility(*market_price_, params_, model_);
    }

    /**
     * Get the pricing parameters
     */
    [[nodiscard]] auto getParams() const noexcept -> PricingParams const& {
        return params_;
    }

    /**
     * Validate parameters before pricing
     */
    [[nodiscard]] auto validate() const noexcept -> Result<void> {
        return params_.validate();
    }

private:
    PricingParams params_;
    OptionsPricer::Model model_;
    int tree_steps_;
    std::optional<Price> market_price_;
};

/**
 * Fluent API for Option Contract
 *
 * Build and price options from contract data.
 *
 * Example:
 *   auto result = ContractPricer(contract)
 *       .spotPrice(150.0)
 *       .riskFreeRate(0.05)
 *       .price();
 */
class ContractPricer {
public:
    explicit ContractPricer(OptionContract const& contract)
        : contract_{contract},
          builder_{} {

        // Initialize builder from contract
        builder_.strike(contract.strike);

        if (contract.type == OptionType::Call) {
            builder_.call();
        } else {
            builder_.put();
        }

        if (contract.style == OptionStyle::American) {
            builder_.american();
        } else {
            builder_.european();
        }

        builder_.volatility(contract.implied_volatility);
    }

    // Required parameters
    [[nodiscard]] auto spotPrice(Price price) noexcept -> ContractPricer& {
        builder_.spot(price);
        return *this;
    }

    [[nodiscard]] auto currentTime(Timestamp time) noexcept -> ContractPricer& {
        builder_.expiration(contract_.expiration, time);
        return *this;
    }

    [[nodiscard]] auto riskFreeRate(double rate) noexcept -> ContractPricer& {
        builder_.riskFreeRate(rate);
        return *this;
    }

    [[nodiscard]] auto dividendYield(double yield) noexcept -> ContractPricer& {
        builder_.dividendYield(yield);
        return *this;
    }

    // Model selection
    [[nodiscard]] auto useBlackScholes() noexcept -> ContractPricer& {
        builder_.useBlackScholes();
        return *this;
    }

    [[nodiscard]] auto useTrinomial(int steps = 100) noexcept -> ContractPricer& {
        builder_.useTrinomial(steps);
        return *this;
    }

    // Terminal operations
    [[nodiscard]] auto price() const noexcept -> Result<Price> {
        return builder_.price();
    }

    [[nodiscard]] auto priceWithGreeks() const noexcept -> Result<PricingResult> {
        return builder_.priceWithGreeks();
    }

    [[nodiscard]] auto greeks() const noexcept -> Result<Greeks> {
        return builder_.greeks();
    }

private:
    OptionContract const& contract_;
    OptionBuilder builder_;
};

/**
 * Convenience functions for quick pricing
 */

// Price a call option using default trinomial model
[[nodiscard]] inline auto priceCall(
    Price spot,
    Price strike,
    double years_to_expiration,
    double volatility,
    double risk_free_rate = 0.05,
    double dividend_yield = 0.0
) noexcept -> Result<Price> {

    return OptionBuilder()
        .call()
        .american()  // Default to American
        .spot(spot)
        .strike(strike)
        .yearsToExpiration(years_to_expiration)
        .volatility(volatility)
        .riskFreeRate(risk_free_rate)
        .dividendYield(dividend_yield)
        .price();
}

// Price a put option using default trinomial model
[[nodiscard]] inline auto pricePut(
    Price spot,
    Price strike,
    double years_to_expiration,
    double volatility,
    double risk_free_rate = 0.05,
    double dividend_yield = 0.0
) noexcept -> Result<Price> {

    return OptionBuilder()
        .put()
        .american()  // Default to American
        .spot(spot)
        .strike(strike)
        .yearsToExpiration(years_to_expiration)
        .volatility(volatility)
        .riskFreeRate(risk_free_rate)
        .dividendYield(dividend_yield)
        .price();
}

// Calculate Greeks for a call option
[[nodiscard]] inline auto callGreeks(
    Price spot,
    Price strike,
    double years_to_expiration,
    double volatility,
    double risk_free_rate = 0.05,
    double dividend_yield = 0.0
) noexcept -> Result<Greeks> {

    return OptionBuilder()
        .call()
        .american()
        .spot(spot)
        .strike(strike)
        .yearsToExpiration(years_to_expiration)
        .volatility(volatility)
        .riskFreeRate(risk_free_rate)
        .dividendYield(dividend_yield)
        .greeks();
}

// Calculate implied volatility for a call option
[[nodiscard]] inline auto callImpliedVolatility(
    Price spot,
    Price strike,
    double years_to_expiration,
    Price market_price,
    double risk_free_rate = 0.05,
    double dividend_yield = 0.0
) noexcept -> Result<double> {

    return OptionBuilder()
        .call()
        .american()
        .spot(spot)
        .strike(strike)
        .yearsToExpiration(years_to_expiration)
        .marketPrice(market_price)
        .riskFreeRate(risk_free_rate)
        .dividendYield(dividend_yield)
        .impliedVolatility();
}

} // namespace bigbrother::options
