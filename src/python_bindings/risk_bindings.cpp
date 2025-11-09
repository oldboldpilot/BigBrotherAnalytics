/**
 * BigBrotherAnalytics - Risk Management Python Bindings
 *
 * GIL-FREE risk calculations: Kelly Criterion, Monte Carlo, Position Sizing
 * 20x+ speedup over pure Python implementations
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 *
 * Tagged: PYTHON_BINDINGS
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>

// Import C++23 modules
import bigbrother.risk_management;
import bigbrother.utils.types;

namespace py = pybind11;

namespace bigbrother::risk {

using namespace bigbrother::types;

// Kelly Criterion calculation (GIL-free)
// Uses the C++ PositionSizer::calculate with KellyCriterion method
auto kelly_criterion(double win_probability, double win_loss_ratio) -> double {
    if (win_loss_ratio <= 0.0) return 0.0;

    // Calculate Kelly using PositionSizer with $1 account for fraction
    auto result = PositionSizer::calculate(
        SizingMethod::KellyCriterion,
        1.0,  // unit account value to get fraction
        win_probability,
        win_loss_ratio,  // win_amount (ratio)
        1.0,  // loss_amount (normalized)
        0.0   // volatility (not used)
    );

    if (!result) {
        return 0.0;  // Return 0 on error
    }

    // Result is already clamped in C++ implementation
    return *result;
}

// Position sizing (GIL-free)
// Applies Kelly fraction with risk limits
auto calculate_position_size(double account_value, double kelly_fraction,
                             double max_position_pct) -> double {
    // Calculate position size based on Kelly fraction
    auto kelly_size = account_value * kelly_fraction;

    // Apply maximum position size limit
    auto max_size = account_value * max_position_pct;

    // Return the smaller of Kelly size and max position size
    return std::min(kelly_size, max_size);
}

// Monte Carlo simulation (GIL-free, OpenMP parallel)
// Using the SimulationResult from risk_management module

auto monte_carlo_simulate(double spot, double vol, double drift,
                          int num_simulations) -> SimulationResult {
    // Create pricing params for Monte Carlo (field order matches PricingParams definition)
    // The drift parameter is passed as risk_free_rate in the PricingParams
    PricingParams params{
        .spot_price = spot,
        .strike_price = spot,  // ATM for simplicity
        .risk_free_rate = drift,  // Use provided drift parameter
        .time_to_expiration = 0.25,  // 3 months
        .volatility = vol,
        .dividend_yield = 0.0,
        .option_type = OptionType::Call
    };

    // Call the C++ MonteCarloSimulator (OpenMP parallel inside)
    auto result = MonteCarloSimulator::simulateOptionTrade(
        params,
        1.0,  // position_size (1 contract)
        num_simulations,
        100   // num_steps (for price path simulation)
    );

    if (!result) {
        // Return empty result on error with default values
        return SimulationResult{
            .expected_value = 0.0,
            .std_deviation = 0.0,
            .probability_of_profit = 0.0,
            .var_95 = 0.0,
            .var_99 = 0.0,
            .max_profit = 0.0,
            .max_loss = 0.0,
            .pnl_distribution = {}
        };
    }

    return *result;
}

} // namespace bigbrother::risk

// Tagged: PYTHON_BINDINGS
PYBIND11_MODULE(bigbrother_risk, m) {
    m.doc() = R"pbdoc(
        BigBrotherAnalytics Risk Management - GIL-Free Performance
        
        ALL FUNCTIONS RELEASE GIL: True multi-threading
        MONTE CARLO: OpenMP parallel (uses all CPU cores)
        
        Performance: 20x+ faster than pure Python
    )pbdoc";
    
    using namespace bigbrother::risk;
    
    // SimulationResult struct - Complete exposure of all fields
    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readwrite("expected_value", &SimulationResult::expected_value,
                       "Expected value (mean) of the PnL distribution")
        .def_readwrite("std_deviation", &SimulationResult::std_deviation,
                       "Standard deviation of the PnL distribution")
        .def_readwrite("probability_of_profit", &SimulationResult::probability_of_profit,
                       "Probability that the trade will be profitable (PnL > 0)")
        .def_readwrite("var_95", &SimulationResult::var_95,
                       "Value at Risk at 95% confidence level (5th percentile)")
        .def_readwrite("var_99", &SimulationResult::var_99,
                       "Value at Risk at 99% confidence level (1st percentile)")
        .def_readwrite("max_profit", &SimulationResult::max_profit,
                       "Maximum profit observed in simulations")
        .def_readwrite("max_loss", &SimulationResult::max_loss,
                       "Maximum loss observed in simulations")
        .def_readwrite("pnl_distribution", &SimulationResult::pnl_distribution,
                       "Full PnL distribution from all simulation paths")
        .def("__repr__", [](const SimulationResult& r) {
            return "SimulationResult(EV=" + std::to_string(r.expected_value) +
                   ", StdDev=" + std::to_string(r.std_deviation) +
                   ", P(profit)=" + std::to_string(r.probability_of_profit) +
                   ", VaR95=" + std::to_string(r.var_95) + ")";
        });
    
    // Kelly Criterion (GIL-FREE)
    m.def("kelly_criterion",
          [](double win_prob, double win_loss_ratio) {
              if (win_prob < 0.0 || win_prob > 1.0) {
                  throw std::invalid_argument("win_probability must be between 0 and 1");
              }
              if (win_loss_ratio <= 0.0) {
                  throw std::invalid_argument("win_loss_ratio must be positive");
              }
              py::gil_scoped_release release;  // GIL-FREE
              return kelly_criterion(win_prob, win_loss_ratio);
          },
          R"pbdoc(
              Calculate Kelly Criterion position sizing fraction (GIL-free).

              The Kelly Criterion determines the optimal position size to maximize
              long-term capital growth based on win probability and win/loss ratio.

              Parameters:
                  win_probability: Probability of winning (0.0 to 1.0)
                  win_loss_ratio: Ratio of average win to average loss

              Returns:
                  Optimal position size as fraction of account (0.0 to 0.25)

              Note: Result is automatically capped at 25% for risk management.
          )pbdoc",
          py::arg("win_probability"), py::arg("win_loss_ratio"));
    
    // Position sizing (GIL-FREE)
    m.def("position_size",
          [](double account_value, double kelly_fraction, double max_pct) {
              if (account_value <= 0.0) {
                  throw std::invalid_argument("account_value must be positive");
              }
              if (kelly_fraction < 0.0) {
                  throw std::invalid_argument("kelly_fraction must be non-negative");
              }
              if (max_pct <= 0.0 || max_pct > 1.0) {
                  throw std::invalid_argument("max_position_pct must be between 0 and 1");
              }
              py::gil_scoped_release release;  // GIL-FREE
              return calculate_position_size(account_value, kelly_fraction, max_pct);
          },
          R"pbdoc(
              Calculate position size with Kelly fraction and risk limits (GIL-free).

              Applies the Kelly fraction to determine position size, then caps it at
              the maximum position size limit for additional risk management.

              Parameters:
                  account_value: Total account value in dollars
                  kelly_fraction: Kelly Criterion fraction (from kelly_criterion())
                  max_position_pct: Maximum position size as fraction of account (default: 0.05)

              Returns:
                  Position size in dollars, capped by both Kelly and max limits

              Example:
                  kelly = kelly_criterion(0.65, 2.0)
                  size = position_size(30000, kelly, 0.05)
          )pbdoc",
          py::arg("account_value"), py::arg("kelly_fraction"),
          py::arg("max_position_pct") = 0.05);
    
    // Monte Carlo simulation (GIL-FREE + OpenMP parallel)
    m.def("monte_carlo",
          [](double spot, double vol, double drift, int sims) {
              if (spot <= 0.0) {
                  throw std::invalid_argument("spot_price must be positive");
              }
              if (vol < 0.0) {
                  throw std::invalid_argument("volatility must be non-negative");
              }
              if (sims < 100) {
                  throw std::invalid_argument("simulations must be at least 100");
              }
              if (sims > 1'000'000) {
                  throw std::invalid_argument("simulations cannot exceed 1,000,000");
              }
              py::gil_scoped_release release;  // GIL-FREE + OpenMP parallel
              return monte_carlo_simulate(spot, vol, drift, sims);
          },
          R"pbdoc(
              Run Monte Carlo simulation for option trade analysis (GIL-free, OpenMP parallel).

              Simulates multiple price paths using geometric Brownian motion and calculates
              comprehensive risk metrics including expected value, VaR, and probability of profit.
              This function releases the GIL and uses OpenMP for parallel execution across all
              CPU cores, providing significant performance improvements.

              Parameters:
                  spot_price: Current spot price of the underlying asset
                  volatility: Annualized volatility (e.g., 0.25 for 25%)
                  drift: Expected return/drift rate (default: 0.0, risk-neutral)
                  simulations: Number of simulation paths (default: 10000, min: 100, max: 1M)

              Returns:
                  SimulationResult object containing:
                      - expected_value: Mean PnL across all simulations
                      - std_deviation: Standard deviation of PnL
                      - probability_of_profit: Fraction of profitable outcomes
                      - var_95, var_99: Value at Risk at 95% and 99% confidence
                      - max_profit, max_loss: Extreme outcomes observed
                      - pnl_distribution: Complete distribution of all PnL values

              Performance:
                  - GIL-free: True multi-threaded execution
                  - OpenMP parallel: Utilizes all CPU cores
                  - Typical: 10,000 sims in <100ms on modern CPU

              Example:
                  result = monte_carlo(100, 0.25, 0.05, 50000)
                  print(f"Expected value: ${result.expected_value:.2f}")
                  print(f"Probability of profit: {result.probability_of_profit:.1%}")
                  print(f"95% VaR: ${result.var_95:.2f}")
          )pbdoc",
          py::arg("spot_price"), py::arg("volatility"),
          py::arg("drift") = 0.0, py::arg("simulations") = 10000);
}
