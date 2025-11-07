#pragma once

#include "../utils/types.hpp"
#include <span>
#include <expected>
#include <numbers>

namespace bigbrother::options {

using namespace types;

/**
 * Options Pricing Engine
 *
 * High-performance options pricing library using C++23.
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
 * Thread Safety: All functions are thread-safe (no shared state)
 */

/**
 * Pricing result with error handling
 */
struct PricingResult {
    Price option_price;
    Greeks greeks;
    double implied_volatility;
    std::string model_used;

    [[nodiscard]] constexpr auto isValid() const noexcept -> bool {
        return !std::isnan(option_price) && option_price >= 0.0;
    }
};

/**
 * Pricing parameters
 */
struct PricingParams {
    Price spot_price;              // Current underlying price
    Price strike_price;            // Strike price
    double time_to_expiration;     // Time to expiration in years
    double risk_free_rate;         // Risk-free rate (annual)
    double volatility;             // Implied volatility (annual)
    double dividend_yield;         // Continuous dividend yield
    OptionType option_type;        // Call or Put
    OptionStyle option_style;      // American or European

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
     */
    [[nodiscard]] static constexpr auto price(PricingParams const& params) noexcept
        -> Result<Price>;

    /**
     * Calculate Greeks
     */
    [[nodiscard]] static constexpr auto greeks(PricingParams const& params) noexcept
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
        PricingParams params,
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
        PricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Price>;

    /**
     * Calculate Greeks using finite differences
     */
    [[nodiscard]] static auto greeks(
        PricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Greeks>;

private:
    struct TreeNode {
        Price stock_price;
        Price option_value;
        bool exercised;
    };

    [[nodiscard]] static auto build_tree(
        PricingParams const& params,
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
        PricingParams const& params,
        int num_steps = 100
    ) noexcept -> Result<Price>;

    [[nodiscard]] static auto greeks(
        PricingParams const& params,
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
        PricingParams const& params,
        int num_simulations = 10000,
        int num_steps = 100,
        bool use_antithetic = true
    ) noexcept -> Result<Price>;

    /**
     * Calculate standard error of estimate
     */
    [[nodiscard]] static auto standard_error(
        PricingParams const& params,
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
        PricingParams const& params,
        Model model = Model::Auto
    ) noexcept -> Result<PricingResult>;

    /**
     * Calculate Greeks for an option
     */
    [[nodiscard]] static auto greeks(
        PricingParams const& params,
        Model model = Model::Auto
    ) noexcept -> Result<Greeks>;

    /**
     * Calculate implied volatility
     */
    [[nodiscard]] static auto implied_volatility(
        Price market_price,
        PricingParams params,
        Model model = Model::Auto
    ) noexcept -> Result<double>;

    /**
     * Price multiple options in parallel (OpenMP)
     * @param params_vec Vector of pricing parameters
     * @return Vector of pricing results
     */
    [[nodiscard]] static auto price_batch(
        std::span<PricingParams const> params_vec,
        Model model = Model::Auto
    ) noexcept -> std::vector<Result<PricingResult>>;

private:
    [[nodiscard]] static auto select_model(PricingParams const& params) noexcept -> Model;
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

    /**
     * Get IV for specific option contract
     */
    [[nodiscard]] auto get_iv(OptionContract const& contract, Price spot_price)
        const noexcept -> Result<double>;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
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
        PricingParams const& params,
        double bump_size = 0.01
    ) noexcept -> Result<Greeks>;

    /**
     * Calculate Delta (∂V/∂S)
     */
    [[nodiscard]] static auto delta(PricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Gamma (∂²V/∂S²)
     */
    [[nodiscard]] static auto gamma(PricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Theta (∂V/∂t) - time decay per day
     */
    [[nodiscard]] static auto theta(PricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Vega (∂V/∂σ) - sensitivity to 1% volatility change
     */
    [[nodiscard]] static auto vega(PricingParams const& params) noexcept
        -> Result<double>;

    /**
     * Calculate Rho (∂V/∂r) - sensitivity to 1% rate change
     */
    [[nodiscard]] static auto rho(PricingParams const& params) noexcept
        -> Result<double>;
};

} // namespace bigbrother::options
