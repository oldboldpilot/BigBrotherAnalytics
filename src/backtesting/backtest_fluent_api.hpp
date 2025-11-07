#pragma once

#include "backtest_engine.hpp"
#include <optional>

namespace bigbrother::backtest {

/**
 * Fluent API for Backtesting
 *
 * Provides intuitive, chainable interface for running backtests.
 *
 * Example Usage:
 *
 *   // Run simple backtest
 *   auto results = BacktestRunner()
 *       .from("2020-01-01")
 *       .to("2024-01-01")
 *       .withCapital(30000.0)
 *       .addStrategy<DeltaNeutralStraddleStrategy>()
 *       .addStrategy<VolatilityArbitrageStrategy>()
 *       .loadData("data/historical/")
 *       .run();
 *
 *   if (results->passesThresholds()) {
 *       std::cout << "Strategy is profitable!" << std::endl;
 *   }
 *
 *   // Advanced backtest with custom parameters
 *   auto advanced = BacktestRunner()
 *       .from("2020-01-01")
 *       .to("2024-01-01")
 *       .withCapital(30000.0)
 *       .commission(0.50)
 *       .slippage(5.0)  // 5 bps
 *       .allowShortSelling()
 *       .reinvestProfits()
 *       .loadData("data/")
 *       .run();
 *
 *   // Walk-forward optimization
 *   auto walkforward = BacktestRunner()
 *       .from("2020-01-01")
 *       .to("2024-01-01")
 *       .withCapital(30000.0)
 *       .walkForward(90, 30)  // 90-day train, 30-day test
 *       .run();
 */

class BacktestRunner {
public:
    BacktestRunner()
        : config_{
            .start_date = 0,
            .end_date = 0,
            .initial_capital = 30'000.0,
            .commission_per_trade = 0.0,  // Schwab: $0 commissions
            .slippage_bps = 2.0,           // 2 basis points
            .allow_short_selling = false,
            .reinvest_profits = true,
            .data_frequency = "1day"
        } {}

    // Date range
    [[nodiscard]] auto from(std::string const& date) -> BacktestRunner& {
        // Parse date string YYYY-MM-DD
        // Convert to timestamp
        // TODO: Implement date parsing
        return *this;
    }

    [[nodiscard]] auto to(std::string const& date) -> BacktestRunner& {
        // Parse date string
        // TODO: Implement date parsing
        return *this;
    }

    [[nodiscard]] auto fromTimestamp(Timestamp ts) noexcept -> BacktestRunner& {
        config_.start_date = ts;
        return *this;
    }

    [[nodiscard]] auto toTimestamp(Timestamp ts) noexcept -> BacktestRunner& {
        config_.end_date = ts;
        return *this;
    }

    // Capital and costs
    [[nodiscard]] auto withCapital(double capital) noexcept -> BacktestRunner& {
        config_.initial_capital = capital;
        return *this;
    }

    [[nodiscard]] auto commission(double amount) noexcept -> BacktestRunner& {
        config_.commission_per_trade = amount;
        return *this;
    }

    [[nodiscard]] auto slippage(double bps) noexcept -> BacktestRunner& {
        config_.slippage_bps = bps;
        return *this;
    }

    // Trading rules
    [[nodiscard]] auto allowShortSelling(bool allow = true) noexcept -> BacktestRunner& {
        config_.allow_short_selling = allow;
        return *this;
    }

    [[nodiscard]] auto reinvestProfits(bool reinvest = true) noexcept -> BacktestRunner& {
        config_.reinvest_profits = reinvest;
        return *this;
    }

    // Data frequency
    [[nodiscard]] auto daily() noexcept -> BacktestRunner& {
        config_.data_frequency = "1day";
        return *this;
    }

    [[nodiscard]] auto intraday(int minutes) noexcept -> BacktestRunner& {
        config_.data_frequency = std::to_string(minutes) + "min";
        return *this;
    }

    // Strategy registration
    template<typename StrategyType, typename... Args>
    [[nodiscard]] auto addStrategy(Args&&... args) -> BacktestRunner& {
        if (!engine_) {
            engine_ = std::make_unique<BacktestEngine>(config_);
        }

        auto strategy = std::make_unique<StrategyType>(std::forward<Args>(args)...);
        engine_->addStrategy(std::move(strategy));

        return *this;
    }

    // Data loading
    [[nodiscard]] auto loadData(std::string data_path) -> BacktestRunner& {
        data_path_ = std::move(data_path);
        return *this;
    }

    [[nodiscard]] auto forSymbols(std::vector<std::string> symbols) -> BacktestRunner& {
        symbols_ = std::move(symbols);
        return *this;
    }

    // Walk-forward optimization
    [[nodiscard]] auto walkForward(int train_days, int test_days) -> BacktestRunner& {
        walk_forward_train_days_ = train_days;
        walk_forward_test_days_ = test_days;
        use_walk_forward_ = true;
        return *this;
    }

    // Terminal operation: run backtest
    [[nodiscard]] auto run() -> Result<BacktestMetrics> {
        if (!engine_) {
            engine_ = std::make_unique<BacktestEngine>(config_);
        }

        // Load data
        if (!data_path_.empty() && !symbols_.empty()) {
            auto load_result = engine_->loadHistoricalData(symbols_, data_path_);

            if (!load_result) {
                return std::unexpected(load_result.error());
            }
        }

        // Run backtest
        return engine_->run();
    }

    // Get detailed results
    [[nodiscard]] auto getTrades() const -> std::vector<BacktestTrade> const& {
        if (engine_) {
            return engine_->getTrades();
        }
        static std::vector<BacktestTrade> empty;
        return empty;
    }

    [[nodiscard]] auto getEquityCurve() const
        -> std::vector<std::pair<Timestamp, double>> const& {
        if (engine_) {
            return engine_->getEquityCurve();
        }
        static std::vector<std::pair<Timestamp, double>> empty;
        return empty;
    }

    // Export results
    [[nodiscard]] auto exportTrades(std::string const& filename) const -> Result<void> {
        if (!engine_) {
            return makeError<void>(ErrorCode::InvalidParameter, "No backtest run");
        }
        return engine_->exportTrades(filename);
    }

    [[nodiscard]] auto exportMetrics(std::string const& filename) const -> Result<void> {
        if (!engine_) {
            return makeError<void>(ErrorCode::InvalidParameter, "No backtest run");
        }
        return engine_->exportMetrics(filename);
    }

private:
    BacktestConfig config_;
    std::unique_ptr<BacktestEngine> engine_;

    std::string data_path_;
    std::vector<std::string> symbols_;

    bool use_walk_forward_{false};
    int walk_forward_train_days_{90};
    int walk_forward_test_days_{30};
};

/**
 * Convenience function for quick backtests
 */
[[nodiscard]] inline auto quickBacktest(
    std::string const& start_date,
    std::string const& end_date,
    std::vector<std::string> const& symbols
) -> Result<BacktestMetrics> {

    return BacktestRunner()
        .from(start_date)
        .to(end_date)
        .withCapital(30'000.0)
        .forSymbols(symbols)
        .daily()
        .run();
}

} // namespace bigbrother::backtest
