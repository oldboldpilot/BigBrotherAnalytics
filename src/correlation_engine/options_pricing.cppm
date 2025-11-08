/**
 * BigBrotherAnalytics - Options Pricing Module (C++23)
 *
 * High-performance options pricing library with fluent API.
 * Implements multiple pricing models with microsecond-level latency.
 *
 * Models:
 * - Black-Scholes-Merton (European options, closed-form)
 * - Binomial Tree (Cox-Ross-Rubinstein, American options)
 * - Trinomial Tree (American options, better convergence)
 * - Monte Carlo (path-dependent options)
 *
 * Performance Targets:
 * - Black-Scholes: < 1 microsecond
 * - Binomial/Trinomial: < 100 microseconds (100 steps)
 * - Greeks calculation: < 0.5 microseconds additional
 *
 * Following C++ Core Guidelines and Clang 21 best practices.
 */

// Global module fragment
module;

#include <span>
#include <expected>
#include <numbers>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// Module declaration
export module bigbrother.options.pricing;

// Import dependencies
import bigbrother.utils.types;

export namespace bigbrother::options {

using namespace bigbrother::types;

/**
 * Option Style
 */
enum class OptionStyle : uint8_t {
    American,  // Can exercise any time before expiration
    European   // Can only exercise at expiration
};

/**
 * Pricing Result with Error Handling
 * C.1: Struct for passive data
 */
struct PricingResult {
    Price option_price{0.0};
    Greeks greeks{};
    double implied_volatility{0.0};
    std::string model_used;

    [[nodiscard]] constexpr auto isValid() const noexcept -> bool {
        return !std::isnan(option_price) && option_price >= 0.0;
    }
};

/**
 * Extended Pricing Parameters
 * C.47: Initialize members in declaration order
 */
struct ExtendedPricingParams {
    Price spot_price{0.0};              // Current underlying price
    Price strike_price{0.0};            // Strike price
    double time_to_expiration{0.0};     // Time to expiration in years
    double risk_free_rate{0.0};         // Risk-free rate (annual)
    double volatility{0.0};             // Implied volatility (annual)
    double dividend_yield{0.0};         // Continuous dividend yield
    OptionType option_type{OptionType::Call};        // Call or Put
    OptionStyle option_style{OptionStyle::American}; // American or European

    /**
     * Validate parameters
     * F.6: noexcept for validation
     */
    [[nodiscard]] constexpr auto validate() const noexcept -> Result<void> {
        if (spot_price <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Spot price must be positive");
        }
        if (strike_price <= 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Strike price must be positive");
        }
        if (time_to_expiration < 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Time to expiration cannot be negative");
        }
        if (volatility < 0.0) {
            return makeError<void>(ErrorCode::InvalidParameter,
                                   "Volatility cannot be negative");
        }

        return {};
    }
};

/**
 * Black-Scholes-Merton Model
 *
 * Closed-form solution for European options.
 * Fastest pricing method (< 1 microsecond).
 */
class BlackScholesModel {
public:
    /**
     * Calculate option price
     * F.20: Return Result, not output parameter
     */
    [[nodiscard]] static constexpr auto price(ExtendedPricingParams const& params) noexcept
        -> Result<Price>;

    /**
     * Calculate Greeks
     */
    [[nodiscard]] static constexpr auto greeks(ExtendedPricingParams const& params) noexcept
        -> Result<Greeks>;

    /**
     * Calculate implied volatility using Newton-Raphson
     * @param market_price Observed market price
     * @param params Pricing parameters (volatility will be solved)
     * @param max_iterations Maximum iterations (default 100)
     * @param tolerance Convergence tolerance (default 1e-6)
     */
    [[nodiscard]] static auto implied_volatility(
        Price market_price,
        ExtendedPricingParams params,
        int max_iterations = 100,
        double tolerance = 1e-6
    ) noexcept -> Result<double>;

private:
    [[nodiscard]] static constexpr auto calculate_d1(
        Price S, Price K, double T, double r, double sigma, double q
    ) noexcept -> double;

    [[nodiscard]] static constexpr auto calculate_d2(
        double d1, double sigma, double T
    ) noexcept -> double;

    [[nodiscard]] static constexpr auto cumulative_normal(double x) noexcept -> double;

    [[nodiscard]] static constexpr auto probability_density(double x) noexcept -> double;
};

/**
 * Binomial Tree Model (Cox-Ross-Rubinstein)
 *
 * Lattice-based pricing for American options.
 * Handles early exercise optimally.
 */
class BinomialTreeModel {
public:
    /**
     * Calculate option price using binomial tree
     * @param params Pricing parameters
     * @param num_steps Number of time steps (more steps = more accuracy, slower)
     */
    [[nodiscard]] static auto price(
        ExtendedPricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Price>;

    /**
     * Calculate Greeks using finite differences
     */
    [[nodiscard]] static auto greeks(
        ExtendedPricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Greeks>;

private:
    struct TreeNode {
        Price stock_price;
        Price option_value;
        bool exercised;
    };

    [[nodiscard]] static auto build_tree(
        ExtendedPricingParams const& params,
        int num_steps
    ) noexcept -> Result<Price>;
};

/**
 * Trinomial Tree Model
 *
 * More stable than binomial for small time steps.
 * Better convergence properties.
 */
class TrinomialTreeModel {
public:
    [[nodiscard]] static auto price(
        ExtendedPricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Price>;

    [[nodiscard]] static auto greeks(
        ExtendedPricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Greeks>;
};

/**
 * Monte Carlo Simulation
 *
 * For path-dependent and exotic options.
 * Uses OpenMP for parallelization.
 */
class MonteCarloModel {
public:
    /**
     * Price option using Monte Carlo simulation
     * @param params Pricing parameters
     * @param num_simulations Number of simulation paths
     * @param num_steps Number of time steps per path
     * @param use_antithetic Use antithetic variance reduction
     */
    [[nodiscard]] static auto price(
        ExtendedPricingParams const& params,
        int num_simulations = 10000,
        int num_steps = 100,
        bool use_antithetic = true
    ) noexcept -> Result<Price>;

    /**
     * Calculate standard error of estimate
     */
    [[nodiscard]] static auto standard_error(
        ExtendedPricingParams const& params,
        int num_simulations = 10000
    ) noexcept -> double;
};

/**
 * Unified Options Pricer
 *
 * Automatically selects best pricing model based on option characteristics.
 */
class OptionsPricer {
public:
    enum class Model {
        Auto,          // Automatically select best model
        BlackScholes,  // Force Black-Scholes (European only)
        Binomial,      // Force Binomial tree
        Trinomial,     // Force Trinomial tree
        MonteCarlo     // Force Monte Carlo
    };

    /**
     * Price an option using optimal model
     */
    [[nodiscard]] static auto price(
        ExtendedPricingParams const& params,
        Model model = Model::Auto
    ) noexcept -> Result<PricingResult>;

    /**
     * Calculate Greeks for an option
     */
    [[nodiscard]] static auto greeks(
        ExtendedPricingParams const& params,
        Model model = Model::Auto
    ) noexcept -> Result<Greeks>;

    /**
     * Calculate implied volatility
     */
    [[nodiscard]] static auto implied_volatility(
        Price market_price,
        ExtendedPricingParams params,
        Model model = Model::Auto
    ) noexcept -> Result<double>;

    /**
     * Price multiple options in parallel (OpenMP)
     * @param params_vec Vector of pricing parameters
     * @return Vector of pricing results
     */
    [[nodiscard]] static auto price_batch(
        std::span<ExtendedPricingParams const> params_vec,
        Model model = Model::Auto
    ) noexcept -> std::vector<Result<PricingResult>>;

private:
    [[nodiscard]] static auto select_model(ExtendedPricingParams const& params) noexcept -> Model;
};

/**
 * Implied Volatility Surface
 *
 * Constructs and interpolates IV surface from market data.
 */
class IVSurface {
public:
    struct MarketQuote {
        Price strike;
        double time_to_expiration;
        Price option_price;
        OptionType type;
    };

    /**
     * Build IV surface from market quotes
     */
    [[nodiscard]] auto build(
        std::span<MarketQuote const> quotes,
        Price spot_price,
        double risk_free_rate
    ) noexcept -> Result<void>;

    /**
     * Get implied volatility for given strike and expiration
     * Uses bilinear interpolation
     */
    [[nodiscard]] auto get_iv(
        Price strike,
        double time_to_expiration
    ) const noexcept -> Result<double>;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Greeks Calculator
 *
 * Efficient Greeks calculation using analytical and numerical methods.
 */
class GreeksCalculator {
public:
    /**
     * Calculate all Greeks using finite differences
     * @param bump_size Size of parameter bump (default 0.01 for 1%)
     */
    [[nodiscard]] static auto calculate(
        ExtendedPricingParams const& params,
        double bump_size = 0.01
    ) noexcept -> Result<Greeks>;

    /**
     * Calculate Delta (∂V/∂S)
     */
    [[nodiscard]] static auto delta(ExtendedPricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Gamma (∂²V/∂S²)
     */
    [[nodiscard]] static auto gamma(ExtendedPricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Theta (∂V/∂t) - time decay per day
     */
    [[nodiscard]] static auto theta(ExtendedPricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Vega (∂V/∂σ) - sensitivity to 1% volatility change
     */
    [[nodiscard]] static auto vega(ExtendedPricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Rho (∂V/∂r) - sensitivity to 1% rate change
     */
    [[nodiscard]] static auto rho(ExtendedPricingParams const& params) noexcept
        -> Result<double>;
};

// ============================================================================
// Fluent API Builder for Options Pricing
// ============================================================================

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
 * Following Builder Pattern and Fluent API best practices
 */
class OptionBuilder {
public:
    /**
     * Constructor
     * C.41: Constructor establishes invariant
     */
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

    // Option type (fluent API)
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

    // Pricing parameters (fluent setters)
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

    // Pricing model selection (fluent)
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

    // Terminal operations (execute pricing)

    /**
     * Calculate option price
     * F.20: Return Result (not output parameter)
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
    [[nodiscard]] auto getParams() const noexcept -> ExtendedPricingParams const& {
        return params_;
    }

    /**
     * Validate parameters before pricing
     */
    [[nodiscard]] auto validate() const noexcept -> Result<void> {
        return params_.validate();
    }

private:
    ExtendedPricingParams params_;
    OptionsPricer::Model model_;
    int tree_steps_;
    std::optional<Price> market_price_;
};

// ============================================================================
// Convenience Functions for Quick Pricing
// ============================================================================

/**
 * Price a call option using default trinomial model
 * F.1: Meaningful function name
 */
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

/**
 * Price a put option using default trinomial model
 */
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

/**
 * Calculate Greeks for a call option
 */
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

/**
 * Calculate implied volatility for a call option
 */
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

} // export namespace bigbrother::options

// ============================================================================
// Implementation Section (module-private, stub implementations)
// ============================================================================

module :private;

namespace bigbrother::options {

// Stub implementations - will be filled from black_scholes.cppm, trinomial_tree.cppm, etc.

constexpr auto BlackScholesModel::price(ExtendedPricingParams const&) noexcept -> Result<Price> {
    return 0.0;  // Stub
}

constexpr auto BlackScholesModel::greeks(ExtendedPricingParams const&) noexcept -> Result<Greeks> {
    return Greeks{};  // Stub
}

auto BlackScholesModel::implied_volatility(Price, ExtendedPricingParams, int, double) noexcept -> Result<double> {
    return 0.0;  // Stub
}

constexpr auto BlackScholesModel::calculate_d1(Price, Price, double, double, double, double) noexcept -> double {
    return 0.0;  // Stub
}

constexpr auto BlackScholesModel::calculate_d2(double, double, double) noexcept -> double {
    return 0.0;  // Stub
}

constexpr auto BlackScholesModel::cumulative_normal(double) noexcept -> double {
    return 0.0;  // Stub
}

constexpr auto BlackScholesModel::probability_density(double) noexcept -> double {
    return 0.0;  // Stub
}

// More stub implementations...

auto BinomialTreeModel::price(ExtendedPricingParams const&, int) noexcept -> Result<Price> {
    return 0.0;  // Stub
}

auto BinomialTreeModel::greeks(ExtendedPricingParams const&, int) noexcept -> Result<Greeks> {
    return Greeks{};  // Stub
}

auto BinomialTreeModel::build_tree(ExtendedPricingParams const&, int) noexcept -> Result<Price> {
    return 0.0;  // Stub
}

auto TrinomialTreeModel::price(ExtendedPricingParams const&, int) noexcept -> Result<Price> {
    return 0.0;  // Stub
}

auto TrinomialTreeModel::greeks(ExtendedPricingParams const&, int) noexcept -> Result<Greeks> {
    return Greeks{};  // Stub
}

auto MonteCarloModel::price(ExtendedPricingParams const&, int, int, bool) noexcept -> Result<Price> {
    return 0.0;  // Stub
}

auto MonteCarloModel::standard_error(ExtendedPricingParams const&, int) noexcept -> double {
    return 0.0;  // Stub
}

auto OptionsPricer::price(ExtendedPricingParams const&, Model) noexcept -> Result<PricingResult> {
    return PricingResult{};  // Stub
}

auto OptionsPricer::greeks(ExtendedPricingParams const&, Model) noexcept -> Result<Greeks> {
    return Greeks{};  // Stub
}

auto OptionsPricer::implied_volatility(Price, ExtendedPricingParams, Model) noexcept -> Result<double> {
    return 0.0;  // Stub
}

auto OptionsPricer::price_batch(std::span<ExtendedPricingParams const>, Model) noexcept
    -> std::vector<Result<PricingResult>> {
    return {};  // Stub
}

auto OptionsPricer::select_model(ExtendedPricingParams const&) noexcept -> Model {
    return Model::Auto;  // Stub
}

auto IVSurface::build(std::span<MarketQuote const>, Price, double) noexcept -> Result<void> {
    return {};  // Stub
}

auto IVSurface::get_iv(Price, double) const noexcept -> Result<double> {
    return 0.0;  // Stub
}

auto GreeksCalculator::calculate(ExtendedPricingParams const&, double) noexcept -> Result<Greeks> {
    return Greeks{};  // Stub
}

auto GreeksCalculator::delta(ExtendedPricingParams const&) noexcept -> Result<double> {
    return 0.0;  // Stub
}

auto GreeksCalculator::gamma(ExtendedPricingParams const&) noexcept -> Result<double> {
    return 0.0;  // Stub
}

auto GreeksCalculator::theta(ExtendedPricingParams const&) noexcept -> Result<double> {
    return 0.0;  // Stub
}

auto GreeksCalculator::vega(ExtendedPricingParams const&) noexcept -> Result<double> {
    return 0.0;  // Stub
}

auto GreeksCalculator::rho(ExtendedPricingParams const&) noexcept -> Result<double> {
    return 0.0;  // Stub
}

} // namespace bigbrother::options
