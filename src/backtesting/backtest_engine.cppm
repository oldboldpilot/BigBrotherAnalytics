/**
 * BigBrotherAnalytics - Backtest Engine Module (C++23)
 *
 * Historical strategy validation with comprehensive metrics.
 * Consolidates: backtest_engine.hpp and backtest_engine_impl.cpp
 *
 * Following C++ Core Guidelines:
 * - R.1: RAII
 * - C.21: Rule of Five
 * - Trailing return syntax
 * - Fluent API
 */

// Global module fragment
module;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric> // Added for std::accumulate
#include <string>
#include <vector>

// Module declaration
export module bigbrother.backtest_engine;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
// import bigbrother.backtest;  // Removed - this module is bigbrother.backtest_engine
import bigbrother.strategy;
import bigbrother.risk_management;

export namespace bigbrother::backtest {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using bigbrother::strategy::IStrategy;       // Explicit
using bigbrother::strategy::StrategyContext; // Explicit
using bigbrother::strategy::TradingSignal;   // Explicit to avoid ambiguity
using namespace bigbrother::risk;

// BacktestConfig (moved from removed bigbrother.backtest)
struct BacktestConfig {
    Timestamp start_date{0};
    Timestamp end_date{0};
    double initial_capital{30'000.0};
    double commission_per_trade{0.65};
    double slippage_bps{2.0};
    bool allow_short_selling{false};
};

// ============================================================================
// Backtest Results
// ============================================================================

struct BacktestResults {
    double initial_capital{30'000.0};
    double final_capital{0.0};
    double total_return{0.0};
    double annualized_return{0.0};
    double sharpe_ratio{0.0};
    double sortino_ratio{0.0};
    double max_drawdown{0.0};
    double win_rate{0.0};
    double profit_factor{0.0};
    int total_trades{0};
    int winning_trades{0};
    int losing_trades{0};
    double avg_win{0.0};
    double avg_loss{0.0};
    double largest_win{0.0};
    double largest_loss{0.0};
    std::vector<double> daily_returns;
    std::vector<double> equity_curve;

    [[nodiscard]] auto calculateSharpeRatio(double risk_free_rate = 0.041) const noexcept
        -> double {
        if (daily_returns.empty())
            return 0.0;

        auto const mean_return = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) /
                                 static_cast<double>(daily_returns.size());

        auto const variance = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0,
                                              [mean = mean_return](double acc, double ret) {
                                                  return acc + (ret - mean) * (ret - mean);
                                              }) /
                              static_cast<double>(daily_returns.size());

        auto const std_dev = std::sqrt(variance);

        if (std_dev == 0.0)
            return 0.0;

        return (mean_return - risk_free_rate / 252.0) / std_dev * std::sqrt(252.0);
    }

    [[nodiscard]] auto calculateMaxDrawdown() const noexcept -> double {
        if (equity_curve.empty())
            return 0.0;

        double max_dd = 0.0;
        double peak = equity_curve[0];

        for (auto const equity : equity_curve) {
            if (equity > peak) {
                peak = equity;
            }

            auto const drawdown = (peak - equity) / peak;
            if (drawdown > max_dd) {
                max_dd = drawdown;
            }
        }

        return max_dd;
    }
};

// ============================================================================
// Backtest Engine (Fluent API)
// ============================================================================

class BacktestEngine {
  public:
    explicit BacktestEngine(BacktestConfig config = {})
        : config_{config}, capital_{config.initial_capital} {}

    // C.21: Rule of Five
    BacktestEngine(BacktestEngine const&) = delete;
    auto operator=(BacktestEngine const&) -> BacktestEngine& = delete;
    BacktestEngine(BacktestEngine&&) noexcept = default;
    auto operator=(BacktestEngine&&) noexcept -> BacktestEngine& = default;
    ~BacktestEngine() = default;

    // Fluent API
    [[nodiscard]] auto withStrategy(std::shared_ptr<IStrategy> strategy) -> BacktestEngine& {
        strategy_ = std::move(strategy);
        return *this;
    }

    [[nodiscard]] auto withConfig(BacktestConfig config) -> BacktestEngine& {
        config_ = config;
        capital_ = config.initial_capital;
        return *this;
    }

    [[nodiscard]] auto withDataSource(std::string path) -> BacktestEngine& {
        data_source_ = std::move(path);
        return *this;
    }

    [[nodiscard]] auto run() -> Result<BacktestResults> {
        if (!strategy_) {
            return makeError<BacktestResults>(ErrorCode::InvalidParameter, "No strategy specified");
        }

        Logger::getInstance().info("Starting backtest: {} to {}", config_.start_date,
                                   config_.end_date);

        BacktestResults results;
        results.initial_capital = config_.initial_capital;
        results.equity_curve.push_back(capital_);

        // Simulate trading days
        auto const days = (config_.end_date - config_.start_date) / (24 * 3600 * 1000000);

        for (int day = 0; day < days; ++day) {
            // Generate signals
            StrategyContext context;
            auto signals_result = strategy_->generateSignals(context);

            if (!signals_result.empty()) {
                for (auto const& signal : signals_result) {
                    processTrade(signal, results);
                }
            }

            // Update equity curve
            results.equity_curve.push_back(capital_);

            // Calculate daily return
            if (results.equity_curve.size() > 1) {
                auto const prev = results.equity_curve[results.equity_curve.size() - 2];
                auto const curr = results.equity_curve.back();
                results.daily_returns.push_back((curr - prev) / prev);
            }
        }

        // Calculate final metrics
        results.final_capital = capital_;
        results.total_return = (capital_ - config_.initial_capital) / config_.initial_capital;
        results.annualized_return = results.total_return * (365.0 / days);
        results.sharpe_ratio = results.calculateSharpeRatio();
        results.max_drawdown = results.calculateMaxDrawdown();
        results.win_rate = results.total_trades > 0 ? static_cast<double>(results.winning_trades) /
                                                          static_cast<double>(results.total_trades)
                                                    : 0.0;
        results.profit_factor =
            results.avg_loss != 0.0 ? std::abs(results.avg_win / results.avg_loss) : 0.0;

        Logger::getInstance().info("Backtest complete: Final capital = ${:.2f}", capital_);

        return results;
    }

  private:
    auto processTrade(TradingSignal const& signal, BacktestResults& results) -> void {
        // Simplified trade processing
        auto const position_size = capital_ * 0.02; // 2% per trade
        auto const commission = config_.commission_per_trade;

        // Simulate random outcome for demo
        auto const random_return = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.10;
        auto const pnl = position_size * random_return - commission;

        capital_ += pnl;
        results.total_trades++;

        if (pnl > 0) {
            results.winning_trades++;
            results.avg_win += pnl;
            if (pnl > results.largest_win) {
                results.largest_win = pnl;
            }
        } else {
            results.losing_trades++;
            results.avg_loss += pnl;
            if (pnl < results.largest_loss) {
                results.largest_loss = pnl;
            }
        }
    }

    BacktestConfig config_;
    double capital_;
    std::shared_ptr<IStrategy> strategy_;
    std::string data_source_;
};

// ============================================================================
// Backtest Runner (Fluent Builder)
// ============================================================================

class BacktestRunner {
  public:
    [[nodiscard]] static auto create() -> BacktestRunner { return BacktestRunner{}; }

    [[nodiscard]] auto withInitialCapital(double capital) -> BacktestRunner& {
        config_.initial_capital = capital;
        return *this;
    }

    [[nodiscard]] auto withDateRange(Timestamp start, Timestamp end) -> BacktestRunner& {
        config_.start_date = start;
        config_.end_date = end;
        return *this;
    }

    [[nodiscard]] auto withCommission(double commission) -> BacktestRunner& {
        config_.commission_per_trade = commission;
        return *this;
    }

    [[nodiscard]] auto withStrategy(std::shared_ptr<IStrategy> strategy) -> BacktestRunner& {
        strategy_ = std::move(strategy);
        return *this;
    }

    [[nodiscard]] auto execute() -> Result<BacktestResults> {
        BacktestEngine engine{config_};

        if (strategy_) {
            engine.withStrategy(strategy_);
        }

        return engine.run();
    }

  private:
    BacktestRunner() = default;

    BacktestConfig config_;
    std::shared_ptr<IStrategy> strategy_;
};

} // namespace bigbrother::backtest
