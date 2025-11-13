/**
 * BigBrotherAnalytics - Risk Management Python Bindings
 *
 * pybind11 bindings for all risk management modules:
 * - Position Sizer
 * - Stop Loss Manager
 * - Monte Carlo Simulator
 * - Risk Manager
 * - VaR Calculator
 * - Stress Testing Engine
 * - Performance Metrics Calculator
 * - Correlation Analyzer
 *
 * MEMORY MANAGEMENT APPROACH (CRITICAL):
 * =======================================
 * Several risk management classes contain std::mutex members for thread safety.
 * std::mutex is neither movable nor copyable, which creates challenges for pybind11.
 *
 * SOLUTION: std::shared_ptr holder with RAII
 * - All mutex-containing classes use std::shared_ptr<T> as pybind11 holder type
 * - Factory methods return std::make_shared<T>() - NO raw new/delete used
 * - std::shared_ptr provides automatic destruction via reference counting
 * - When Python reference count reaches zero, shared_ptr destructor runs automatically
 * - Fully RAII-compliant: no manual memory management, no leaks
 * - Follows C++ Core Guidelines R.20/R.21 (use smart pointers for ownership)
 *
 * Classes using shared_ptr holder (due to mutex):
 * - StopLossManager
 * - VaRCalculator
 * - StressTestingEngine
 * - PerformanceMetricsCalculator
 * - CorrelationAnalyzer
 *
 * Move semantics added to all mutex classes:
 * - Move constructor: moves data members, default-constructs new mutex
 * - Move assignment: moves data members, leaves mutex unchanged
 * - Copy operations: explicitly deleted (non-copyable due to mutex)
 *
 * This approach ensures:
 * 1. Zero memory leaks (shared_ptr RAII)
 * 2. No manual new/delete (violates coding standards)
 * 3. Thread-safe operations (mutex preserved)
 * 4. Python interoperability (pybind11 compatible)
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

// Import C++23 modules
import bigbrother.risk.position_sizer;
import bigbrother.risk.stop_loss;
import bigbrother.risk.monte_carlo;
import bigbrother.risk.manager;
import bigbrother.risk.var_calculator;
import bigbrother.risk.stress_testing;
import bigbrother.risk.performance_metrics;
import bigbrother.risk.correlation_analyzer;
import bigbrother.utils.types;

namespace py = pybind11;
using namespace bigbrother::risk;
using namespace bigbrother::types;

PYBIND11_MODULE(bigbrother_risk, m) {
    m.doc() = R"pbdoc(
        BigBrotherAnalytics Risk Management - Complete Framework

        Comprehensive risk management with 8 integrated modules:
        - Position Sizer: Kelly Criterion and position sizing strategies
        - Stop Loss Manager: Hard and trailing stops with real-time updates
        - Monte Carlo Simulator: OpenMP-parallel risk simulation
        - Risk Manager: Central risk assessment and portfolio management
        - VaR Calculator: Parametric, Historical, Monte Carlo, and Hybrid VaR
        - Stress Testing Engine: AVX2-accelerated stress scenarios
        - Performance Metrics: Sharpe, Sortino, Calmar, Omega ratios
        - Correlation Analyzer: MKL-accelerated correlation and diversification

        All modules use fluent API design with trailing return syntax.
        Performance: SIMD vectorization, OpenMP parallelization, MKL acceleration
    )pbdoc";

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<SizingMethod>(m, "SizingMethod")
        .value("FixedDollar", SizingMethod::FixedDollar)
        .value("FixedPercent", SizingMethod::FixedPercent)
        .value("KellyCriterion", SizingMethod::KellyCriterion)
        .value("KellyHalf", SizingMethod::KellyHalf)
        .value("KellyQuarter", SizingMethod::KellyQuarter)
        .value("VolatilityAdjusted", SizingMethod::VolatilityAdjusted)
        .value("RiskParity", SizingMethod::RiskParity)
        .value("DeltaAdjusted", SizingMethod::DeltaAdjusted)
        .value("MaxDrawdown", SizingMethod::MaxDrawdown)
        .export_values();

    py::enum_<VaRMethod>(m, "VaRMethod")
        .value("Parametric", VaRMethod::Parametric)
        .value("Historical", VaRMethod::Historical)
        .value("MonteCarlo", VaRMethod::MonteCarlo)
        .value("Hybrid", VaRMethod::Hybrid)
        .export_values();

    py::enum_<StressScenario>(m, "StressScenario")
        .value("MarketCrash", StressScenario::MarketCrash)
        .value("VolatilitySpike", StressScenario::VolatilitySpike)
        .value("SectorRotation", StressScenario::SectorRotation)
        .value("InterestRateShock", StressScenario::InterestRateShock)
        .value("CreditCrunch", StressScenario::CreditCrunch)
        .value("BlackSwan", StressScenario::BlackSwan)
        .value("FlashCrash", StressScenario::FlashCrash)
        .value("Custom", StressScenario::Custom)
        .export_values();

    py::enum_<PerformancePeriod>(m, "PerformancePeriod")
        .value("Daily", PerformancePeriod::Daily)
        .value("Weekly", PerformancePeriod::Weekly)
        .value("Monthly", PerformancePeriod::Monthly)
        .value("Quarterly", PerformancePeriod::Quarterly)
        .value("Annual", PerformancePeriod::Annual)
        .export_values();

    // ========================================================================
    // Position Sizer
    // ========================================================================

    py::class_<PositionSize>(m, "PositionSize")
        .def_readonly("dollar_amount", &PositionSize::dollar_amount)
        .def_readonly("num_contracts", &PositionSize::num_contracts)
        .def_readonly("kelly_fraction", &PositionSize::kelly_fraction)
        .def_readonly("risk_percent", &PositionSize::risk_percent)
        .def_readonly("method_used", &PositionSize::method_used);

    py::class_<PositionSizer>(m, "PositionSizer")
        .def_static("create", &PositionSizer::create)
        .def("with_method", &PositionSizer::withMethod,
             py::return_value_policy::reference_internal)
        .def("with_account_value", &PositionSizer::withAccountValue,
             py::return_value_policy::reference_internal)
        .def("with_win_probability", &PositionSizer::withWinProbability,
             py::return_value_policy::reference_internal)
        .def("with_expected_gain", &PositionSizer::withExpectedGain,
             py::return_value_policy::reference_internal)
        .def("with_expected_loss", &PositionSizer::withExpectedLoss,
             py::return_value_policy::reference_internal)
        .def("with_max_position", &PositionSizer::withMaxPosition,
             py::return_value_policy::reference_internal)
        .def("calculate", [](PositionSizer const& sizer) {
            auto result = sizer.calculate();
            if (!result) {
                throw std::runtime_error("Position sizing failed");
            }
            return *result;
        })
        .def_static("kelly_fraction", &PositionSizer::kellyFraction);

    // ========================================================================
    // Stop Loss Manager
    // ========================================================================

    py::class_<Stop>(m, "Stop")
        .def_readonly("position_id", &Stop::position_id)
        .def_readonly("type", &Stop::type)
        .def_readonly("trigger_price", &Stop::trigger_price)
        .def_readonly("initial_price", &Stop::initial_price)
        .def_readonly("trail_amount", &Stop::trail_amount)
        .def_readonly("expiration", &Stop::expiration)
        .def_readonly("triggered", &Stop::triggered);

    py::class_<StopLossManager, std::shared_ptr<StopLossManager>>(m, "StopLossManager")
        .def(py::init<>())
        .def_static("create", []() {
            return std::make_shared<StopLossManager>();
        })
        .def("add_hard_stop", &StopLossManager::addHardStop,
             py::return_value_policy::reference_internal)
        .def("add_trailing_stop", &StopLossManager::addTrailingStop,
             py::return_value_policy::reference_internal)
        .def("update", &StopLossManager::update)
        .def("remove_stop", &StopLossManager::removeStop,
             py::return_value_policy::reference_internal)
        .def("clear_all", &StopLossManager::clearAll,
             py::return_value_policy::reference_internal)
        .def("has_stop", &StopLossManager::hasStop)
        .def("get_stop_count", &StopLossManager::getStopCount);

    // ========================================================================
    // Monte Carlo Simulator
    // ========================================================================

    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readonly("num_simulations", &SimulationResult::num_simulations)
        .def_readonly("mean_pnl", &SimulationResult::mean_pnl)
        .def_readonly("std_pnl", &SimulationResult::std_pnl)
        .def_readonly("median_pnl", &SimulationResult::median_pnl)
        .def_readonly("min_pnl", &SimulationResult::min_pnl)
        .def_readonly("max_pnl", &SimulationResult::max_pnl)
        .def_readonly("var_95", &SimulationResult::var_95)
        .def_readonly("cvar_95", &SimulationResult::cvar_95)
        .def_readonly("win_probability", &SimulationResult::win_probability)
        .def_readonly("expected_value", &SimulationResult::expected_value);

    py::class_<MonteCarloSimulator>(m, "MonteCarloSimulator")
        .def_static("create", &MonteCarloSimulator::create)
        .def("with_simulations", &MonteCarloSimulator::withSimulations,
             py::return_value_policy::reference_internal)
        .def("with_parallel", &MonteCarloSimulator::withParallel,
             py::return_value_policy::reference_internal)
        .def("with_seed", &MonteCarloSimulator::withSeed,
             py::return_value_policy::reference_internal)
        .def("simulate_stock", [](MonteCarloSimulator const& sim,
                                  double entry, double target,
                                  double stop, double volatility) {
            auto result = sim.simulateStock(entry, target, stop, volatility);
            if (!result) {
                throw std::runtime_error("Monte Carlo simulation failed");
            }
            return *result;
        });

    // ========================================================================
    // Risk Manager
    // ========================================================================

    py::class_<RiskLimits>(m, "RiskLimits")
        .def(py::init<>())
        .def_readwrite("account_value", &RiskLimits::account_value)
        .def_readwrite("max_daily_loss", &RiskLimits::max_daily_loss)
        .def_readwrite("max_position_size", &RiskLimits::max_position_size)
        .def_readwrite("max_concurrent_positions", &RiskLimits::max_concurrent_positions);

    py::class_<TradeRisk>(m, "TradeRisk")
        .def_readonly("symbol", &TradeRisk::symbol)
        .def_readonly("position_size", &TradeRisk::position_size)
        .def_readonly("max_loss", &TradeRisk::max_loss)
        .def_readonly("expected_return", &TradeRisk::expected_return)
        .def_readonly("win_probability", &TradeRisk::win_probability)
        .def_readonly("expected_value", &TradeRisk::expected_value)
        .def_readonly("risk_reward_ratio", &TradeRisk::risk_reward_ratio)
        .def_readonly("approved", &TradeRisk::approved)
        .def_readonly("rejection_reason", &TradeRisk::rejection_reason);

    py::class_<PortfolioRisk>(m, "PortfolioRisk")
        .def_readonly("total_value", &PortfolioRisk::total_value)
        .def_readonly("daily_pnl", &PortfolioRisk::daily_pnl)
        .def_readonly("daily_loss_remaining", &PortfolioRisk::daily_loss_remaining)
        .def_readonly("active_positions", &PortfolioRisk::active_positions)
        .def_readonly("portfolio_heat", &PortfolioRisk::portfolio_heat)
        .def_readonly("var_95", &PortfolioRisk::var_95)
        .def_readonly("sharpe_ratio", &PortfolioRisk::sharpe_ratio)
        .def("can_open_new_position", &PortfolioRisk::canOpenNewPosition);

    py::class_<RiskManager>(m, "RiskManager")
        .def_static("create", [](RiskLimits const& limits) {
            auto result = RiskManager::create(limits);
            if (!result) {
                throw std::runtime_error("Risk manager creation failed");
            }
            return std::move(*result);
        })
        .def("with_account_value", &RiskManager::withAccountValue,
             py::return_value_policy::reference_internal)
        .def("with_daily_loss_limit", &RiskManager::withDailyLossLimit,
             py::return_value_policy::reference_internal)
        .def("with_position_size_limit", &RiskManager::withPositionSizeLimit,
             py::return_value_policy::reference_internal)
        .def("assess_trade", [](RiskManager& mgr, std::string const& symbol,
                               double position_size, double entry,
                               double stop, double target, double win_prob) {
            auto result = mgr.assessTrade(symbol, position_size, entry,
                                         stop, target, win_prob);
            if (!result) {
                throw std::runtime_error("Trade assessment failed");
            }
            return *result;
        })
        .def("get_portfolio_risk", &RiskManager::getPortfolioRisk)
        .def("get_risk_limits", &RiskManager::getRiskLimits);

    // ========================================================================
    // VaR Calculator
    // ========================================================================

    py::class_<VaRResult>(m, "VaRResult")
        .def_readonly("var_amount", &VaRResult::var_amount)
        .def_readonly("var_percentage", &VaRResult::var_percentage)
        .def_readonly("expected_shortfall", &VaRResult::expected_shortfall)
        .def_readonly("volatility", &VaRResult::volatility)
        .def_readonly("method_used", &VaRResult::method_used)
        .def_readonly("confidence_level", &VaRResult::confidence_level)
        .def_readonly("holding_period", &VaRResult::holding_period)
        .def("is_valid", &VaRResult::isValid)
        .def("get_risk_level", &VaRResult::getRiskLevel);

    py::class_<VaRCalculator, std::shared_ptr<VaRCalculator>>(m, "VaRCalculator")
        .def(py::init<>())
        .def_static("create", []() {
            return std::make_shared<VaRCalculator>();
        })
        .def("with_returns", &VaRCalculator::withReturns,
             py::return_value_policy::reference_internal)
        .def("with_confidence_level", &VaRCalculator::withConfidenceLevel,
             py::return_value_policy::reference_internal)
        .def("with_method", &VaRCalculator::withMethod,
             py::return_value_policy::reference_internal)
        .def("calculate", [](VaRCalculator const& calc, double portfolio_value) {
            auto result = calc.calculate(portfolio_value);
            if (!result) {
                throw std::runtime_error("VaR calculation failed");
            }
            return *result;
        }, py::arg("portfolio_value"));

    // ========================================================================
    // Stress Testing Engine
    // ========================================================================

    py::class_<StressPosition>(m, "StressPosition")
        .def(py::init<>())
        .def_readwrite("symbol", &StressPosition::symbol)
        .def_readwrite("quantity", &StressPosition::quantity)
        .def_readwrite("current_price", &StressPosition::current_price)
        .def_readwrite("beta", &StressPosition::beta)
        .def_readwrite("sector_exposure", &StressPosition::sector_exposure)
        .def_readwrite("duration", &StressPosition::duration)
        .def_readwrite("delta", &StressPosition::delta)
        .def_readwrite("vega", &StressPosition::vega);

    py::class_<StressTestResult>(m, "StressTestResult")
        .def_readonly("scenario", &StressTestResult::scenario)
        .def_readonly("initial_value", &StressTestResult::initial_value)
        .def_readonly("stressed_value", &StressTestResult::stressed_value)
        .def_readonly("pnl", &StressTestResult::pnl)
        .def_readonly("pnl_percentage", &StressTestResult::pnl_percentage)
        .def_readonly("position_impacts", &StressTestResult::position_impacts)
        .def("get_severity", &StressTestResult::getSeverity)
        .def("is_portfolio_viable", &StressTestResult::isPortfolioViable);

    py::class_<StressTestingEngine, std::shared_ptr<StressTestingEngine>>(m, "StressTestingEngine")
        .def(py::init<>())
        .def_static("create", []() {
            return std::make_shared<StressTestingEngine>();
        })
        .def("add_position", &StressTestingEngine::addPosition,
             py::return_value_policy::reference_internal)
        .def("clear_positions", &StressTestingEngine::clearPositions,
             py::return_value_policy::reference_internal)
        .def("run_stress_test", [](StressTestingEngine const& engine,
                                   StressScenario scenario) {
            auto result = engine.runStressTest(scenario);
            if (!result) {
                throw std::runtime_error("Stress test failed");
            }
            return *result;
        })
        .def("run_all_scenarios", &StressTestingEngine::runAllScenarios)
        .def("get_position_count", &StressTestingEngine::getPositionCount)
        .def("get_total_value", &StressTestingEngine::getTotalValue);

    // ========================================================================
    // Performance Metrics Calculator
    // ========================================================================

    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("total_return", &PerformanceMetrics::total_return)
        .def_readonly("annualized_return", &PerformanceMetrics::annualized_return)
        .def_readonly("average_return", &PerformanceMetrics::average_return)
        .def_readonly("volatility", &PerformanceMetrics::volatility)
        .def_readonly("downside_deviation", &PerformanceMetrics::downside_deviation)
        .def_readonly("max_drawdown", &PerformanceMetrics::max_drawdown)
        .def_readonly("sharpe_ratio", &PerformanceMetrics::sharpe_ratio)
        .def_readonly("sortino_ratio", &PerformanceMetrics::sortino_ratio)
        .def_readonly("calmar_ratio", &PerformanceMetrics::calmar_ratio)
        .def_readonly("omega_ratio", &PerformanceMetrics::omega_ratio)
        .def_readonly("win_rate", &PerformanceMetrics::win_rate)
        .def_readonly("profit_factor", &PerformanceMetrics::profit_factor)
        .def_readonly("expectancy", &PerformanceMetrics::expectancy)
        .def_readonly("skewness", &PerformanceMetrics::skewness)
        .def_readonly("kurtosis", &PerformanceMetrics::kurtosis)
        .def("is_healthy", &PerformanceMetrics::isHealthy)
        .def("get_rating", &PerformanceMetrics::getRating);

    py::class_<PerformanceMetricsCalculator, std::shared_ptr<PerformanceMetricsCalculator>>(m, "PerformanceMetricsCalculator")
        .def(py::init<>())
        .def_static("create", []() {
            return std::make_shared<PerformanceMetricsCalculator>();
        })
        .def("with_returns", &PerformanceMetricsCalculator::withReturns,
             py::return_value_policy::reference_internal)
        .def("with_risk_free_rate", &PerformanceMetricsCalculator::withRiskFreeRate,
             py::return_value_policy::reference_internal)
        .def("with_period", &PerformanceMetricsCalculator::withPeriod,
             py::return_value_policy::reference_internal)
        .def("with_target_return", &PerformanceMetricsCalculator::withTargetReturn,
             py::return_value_policy::reference_internal)
        .def("calculate", [](PerformanceMetricsCalculator const& calc) {
            auto result = calc.calculate();
            if (!result) {
                throw std::runtime_error("Performance metrics calculation failed");
            }
            return *result;
        })
        .def_static("from_equity_curve",
            [](std::vector<double> const& equity, double rfr) {
                auto result = PerformanceMetricsCalculator::fromEquityCurve(equity, rfr);
                if (!result) {
                    throw std::runtime_error("Equity curve analysis failed");
                }
                return *result;
            },
            py::arg("equity"), py::arg("rfr") = 0.0);

    // ========================================================================
    // Correlation Analyzer
    // ========================================================================

    py::class_<CorrelationMatrix>(m, "CorrelationMatrix")
        .def_readonly("symbols", &CorrelationMatrix::symbols)
        .def_readonly("matrix", &CorrelationMatrix::matrix)
        .def_readonly("dimension", &CorrelationMatrix::dimension)
        .def("get_correlation", &CorrelationMatrix::getCorrelation)
        .def("is_valid", &CorrelationMatrix::isValid);

    py::class_<DiversificationMetrics>(m, "DiversificationMetrics")
        .def_readonly("avg_correlation", &DiversificationMetrics::avg_correlation)
        .def_readonly("max_correlation", &DiversificationMetrics::max_correlation)
        .def_readonly("min_correlation", &DiversificationMetrics::min_correlation)
        .def_readonly("diversification_ratio", &DiversificationMetrics::diversification_ratio)
        .def_readonly("concentration_index", &DiversificationMetrics::concentration_index)
        .def_readonly("highly_correlated_pairs", &DiversificationMetrics::highly_correlated_pairs)
        .def("is_diversified", &DiversificationMetrics::isDiversified)
        .def("get_rating", &DiversificationMetrics::getRating);

    py::class_<CorrelationAnalyzer, std::shared_ptr<CorrelationAnalyzer>>(m, "CorrelationAnalyzer")
        .def(py::init<>())
        .def_static("create", []() {
            return std::make_shared<CorrelationAnalyzer>();
        })
        .def("add_series", &CorrelationAnalyzer::addSeries,
             py::return_value_policy::reference_internal)
        .def("clear_series", &CorrelationAnalyzer::clearSeries,
             py::return_value_policy::reference_internal)
        .def("compute_correlation_matrix", [](CorrelationAnalyzer const& analyzer) {
            auto result = analyzer.computeCorrelationMatrix();
            if (!result) {
                throw std::runtime_error("Correlation matrix computation failed");
            }
            return *result;
        })
        .def("analyze_diversification", [](CorrelationAnalyzer const& analyzer,
                                          std::vector<double> const& weights) {
            auto result = analyzer.analyzeDiversification(weights);
            if (!result) {
                throw std::runtime_error("Diversification analysis failed");
            }
            return *result;
        })
        .def("find_highly_correlated_pairs", &CorrelationAnalyzer::findHighlyCorrelatedPairs,
             py::arg("threshold") = 0.7)
        .def("compute_rolling_correlation", [](CorrelationAnalyzer const& analyzer,
                                              std::string const& sym1,
                                              std::string const& sym2,
                                              size_t window_size) {
            auto result = analyzer.computeRollingCorrelation(sym1, sym2, window_size);
            if (!result) {
                throw std::runtime_error("Rolling correlation computation failed");
            }
            return *result;
        });
}
