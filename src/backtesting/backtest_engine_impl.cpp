/**
 * BacktestEngine Implementation
 */

#include "backtest_engine.hpp"
#include "../utils/logger.hpp"
#include <algorithm>

namespace bigbrother::backtest {

// ============================================================================
// BacktestConfig Implementation
// ============================================================================

auto BacktestConfig::validate() const noexcept -> Result<void> {
    if (start_date >= end_date) {
        return makeError<void>(ErrorCode::InvalidParameter, "Invalid date range");
    }
    if (initial_capital <= 0.0) {
        return makeError<void>(ErrorCode::InvalidParameter, "Invalid initial capital");
    }
    return {};
}

// ============================================================================
// BacktestEngine::Impl
// ============================================================================

class BacktestEngine::Impl {
public:
    explicit Impl(BacktestConfig config)
        : config_{std::move(config)},
          current_capital_{config_.initial_capital} {}

    auto loadHistoricalData(
        std::vector<std::string> const& symbols,
        std::string const& data_path
    ) -> Result<void> {
        // Stub - will load from DuckDB
        return {};
    }

    auto run() -> Result<BacktestMetrics> {
        BacktestMetrics results{};
        results.total_return = 0.0;
        results.total_trades = 0;
        results.winning_trades = 0;
        results.losing_trades = 0;
        results.win_rate = 0.0;
        results.sharpe_ratio = 0.0;
        results.max_drawdown = 0.0;

        return results;
    }

    auto addStrategy(std::unique_ptr<strategy::IStrategy> strategy) -> void {
        strategies_.push_back(std::move(strategy));
    }

    auto exportTrades(std::string const& filepath) const -> Result<void> {
        // Stub
        return {};
    }

    auto exportMetrics(std::string const& filepath) const -> Result<void> {
        // Stub
        return {};
    }

private:
    BacktestConfig config_;
    double current_capital_;
    std::vector<std::shared_ptr<strategy::IStrategy>> strategies_;
};

// ============================================================================
// BacktestEngine Public Interface
// ============================================================================

BacktestEngine::BacktestEngine(BacktestConfig config)
    : pImpl_{std::make_unique<Impl>(std::move(config))} {}

BacktestEngine::~BacktestEngine() = default;

BacktestEngine::BacktestEngine(BacktestEngine&&) noexcept = default;

auto BacktestEngine::operator=(BacktestEngine&&) noexcept -> BacktestEngine& = default;

auto BacktestEngine::loadHistoricalData(
    std::vector<std::string> const& symbols,
    std::string const& data_path
) -> Result<void> {
    return pImpl_->loadHistoricalData(symbols, data_path);
}

auto BacktestEngine::run() -> Result<BacktestMetrics> {
    return pImpl_->run();
}

auto BacktestEngine::addStrategy(std::unique_ptr<strategy::IStrategy> strategy) -> void {
    pImpl_->addStrategy(std::move(strategy));
}

auto BacktestEngine::exportTrades(std::string const& filepath) const -> Result<void> {
    return pImpl_->exportTrades(filepath);
}

auto BacktestEngine::exportMetrics(std::string const& filepath) const -> Result<void> {
    return pImpl_->exportMetrics(filepath);
}

} // namespace bigbrother::backtest
