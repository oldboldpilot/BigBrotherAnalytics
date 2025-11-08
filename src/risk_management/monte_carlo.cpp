/**
 * Monte Carlo Simulation Implementation
 * C++23 module implementation unit
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-08
 */

// Global module fragment
module;

#include <random>
#include <algorithm>
#include <cmath>
#include <numbers>

#ifdef _OPENMP
#include <omp.h>
#endif

// Module implementation unit declaration
module bigbrother.risk_management;

import bigbrother.utils.logger;
import bigbrother.utils.timer;
import bigbrother.utils.math;

namespace bigbrother::risk {

using std::numbers::sqrt2;

/**
 * Monte Carlo Simulation for Trade Risk Assessment
 *
 * Simulates thousands of possible outcomes to estimate:
 * - Expected value
 * - Probability of profit
 * - Value at Risk (VaR)
 * - Maximum drawdown scenarios
 */

[[nodiscard]] auto MonteCarloSimulator::simulateOptionTrade(
    options::PricingParams const& params,
    double position_size,
    int num_simulations,
    int num_steps
) noexcept -> Result<SimulationResult> {

    PROFILE_SCOPE("MonteCarloSimulator::simulateOptionTrade");

    // Validate inputs
    if (auto validation = params.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    if (num_simulations < 100) {
        return makeError<SimulationResult>(
            ErrorCode::InvalidParameter,
            "Need at least 100 simulations"
        );
    }

    // Calculate initial option price
    auto price_result = options::OptionsPricer::price(
        params,
        options::OptionsPricer::Model::BlackScholes  // Fast for MC
    );

    if (!price_result) {
        return std::unexpected(price_result.error());
    }

    double const initial_option_price = price_result->option_price;
    double const time_step = params.time_to_expiration / static_cast<double>(num_steps);

    // Storage for simulation results
    std::vector<double> final_pnls(num_simulations);

    // Run simulations in parallel
    #pragma omp parallel
    {
        // Thread-local random number generator
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::normal_distribution<double> normal(0.0, 1.0);

        #pragma omp for
        for (int sim = 0; sim < num_simulations; ++sim) {
            // Simulate spot price path using Geometric Brownian Motion
            double spot = params.spot_price;

            for (int step = 0; step < num_steps; ++step) {
                // dS = μ*S*dt + σ*S*dW
                double const drift = (params.risk_free_rate - params.dividend_yield) * time_step;
                double const diffusion = params.volatility * std::sqrt(time_step) * normal(gen);

                spot *= std::exp(drift + diffusion);
            }

            // Calculate final option value
            double final_option_price = 0.0;

            // At expiration, option worth intrinsic value
            if (params.option_type == options::OptionType::Call) {
                final_option_price = std::max(0.0, spot - params.strike_price);
            } else {
                final_option_price = std::max(0.0, params.strike_price - spot);
            }

            // P&L for this simulation
            // Assume we bought the option (premium paid upfront)
            double const pnl = (final_option_price - initial_option_price) *
                              (position_size / initial_option_price);

            final_pnls[sim] = pnl;
        }
    }

    // Calculate statistics
    SimulationResult result;
    result.num_simulations = num_simulations;

    // Mean and standard deviation
    result.mean_pnl = utils::math::mean(final_pnls);
    result.std_pnl = utils::math::stddev(final_pnls);

    // Min and max
    auto const [min_it, max_it] = std::minmax_element(final_pnls.begin(), final_pnls.end());
    result.min_pnl = *min_it;
    result.max_pnl = *max_it;

    // Median
    result.median_pnl = utils::math::percentile(final_pnls, 0.50);

    // Value at Risk (95% confidence)
    result.var_95 = -utils::math::percentile(final_pnls, 0.05);  // Negative for loss

    // Conditional VaR (Expected Shortfall)
    // Average of worst 5% outcomes
    std::vector<double> sorted_pnls = final_pnls;
    std::sort(sorted_pnls.begin(), sorted_pnls.end());

    int const worst_5pct = static_cast<int>(num_simulations * 0.05);
    double const cvar_sum = std::accumulate(
        sorted_pnls.begin(),
        sorted_pnls.begin() + worst_5pct,
        0.0
    );
    result.cvar_95 = -cvar_sum / static_cast<double>(worst_5pct);

    // Win probability
    int const num_wins = std::count_if(
        final_pnls.begin(),
        final_pnls.end(),
        [](double pnl) { return pnl > 0.0; }
    );
    result.win_probability = static_cast<double>(num_wins) / static_cast<double>(num_simulations);

    // Expected value
    result.expected_value = result.mean_pnl;

    LOG_INFO("Monte Carlo simulation complete: {} paths, EV: ${:.2f}, Win%: {:.1f}%",
             num_simulations,
             result.expected_value,
             result.win_probability * 100.0);

    return result;
}

[[nodiscard]] auto MonteCarloSimulator::simulateStockTrade(
    Price entry_price,
    Price target_price,
    Price stop_price,
    double volatility,
    int num_simulations
) noexcept -> Result<SimulationResult> {

    PROFILE_SCOPE("MonteCarloSimulator::simulateStockTrade");

    // Validate inputs
    if (entry_price <= 0.0 || volatility <= 0.0) {
        return makeError<SimulationResult>(
            ErrorCode::InvalidParameter,
            "Entry price and volatility must be positive"
        );
    }

    // Determine if long or short
    bool const is_long = target_price > entry_price;

    // Calculate max profit and max loss
    double const max_profit = std::abs(target_price - entry_price);
    double const max_loss = std::abs(entry_price - stop_price);

    // Storage for results
    std::vector<double> pnls(num_simulations);

    // Run simulations
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::normal_distribution<double> normal(0.0, 1.0);

        #pragma omp for
        for (int sim = 0; sim < num_simulations; ++sim) {
            // Simulate price movement over time
            // Assume holding period of 1 day
            double const time_horizon = 1.0 / 252.0;  // 1 trading day

            // Random price change (GBM)
            double const z = normal(gen);
            double const price_change = volatility * std::sqrt(time_horizon) * z;

            double const final_price = entry_price * std::exp(price_change);

            // Determine outcome
            double pnl = 0.0;

            if (is_long) {
                // Long position
                if (final_price >= target_price) {
                    pnl = max_profit;  // Hit target
                } else if (final_price <= stop_price) {
                    pnl = -max_loss;  // Hit stop
                } else {
                    pnl = final_price - entry_price;  // Intermediate
                }
            } else {
                // Short position
                if (final_price <= target_price) {
                    pnl = max_profit;  // Hit target
                } else if (final_price >= stop_price) {
                    pnl = -max_loss;  // Hit stop
                } else {
                    pnl = entry_price - final_price;  // Intermediate
                }
            }

            pnls[sim] = pnl;
        }
    }

    // Calculate statistics (same as option simulation)
    SimulationResult result;
    result.num_simulations = num_simulations;

    result.mean_pnl = utils::math::mean(pnls);
    result.std_pnl = utils::math::stddev(pnls);

    auto const [min_it, max_it] = std::minmax_element(pnls.begin(), pnls.end());
    result.min_pnl = *min_it;
    result.max_pnl = *max_it;

    result.median_pnl = utils::math::percentile(pnls, 0.50);
    result.var_95 = -utils::math::percentile(pnls, 0.05);

    std::vector<double> sorted_pnls = pnls;
    std::sort(sorted_pnls.begin(), sorted_pnls.end());

    int const worst_5pct = static_cast<int>(num_simulations * 0.05);
    double const cvar_sum = std::accumulate(
        sorted_pnls.begin(),
        sorted_pnls.begin() + worst_5pct,
        0.0
    );
    result.cvar_95 = -cvar_sum / static_cast<double>(worst_5pct);

    int const num_wins = std::count_if(
        pnls.begin(),
        pnls.end(),
        [](double p) { return p > 0.0; }
    );
    result.win_probability = static_cast<double>(num_wins) / static_cast<double>(num_simulations);

    result.expected_value = result.mean_pnl;

    return result;
}

} // namespace bigbrother::risk
