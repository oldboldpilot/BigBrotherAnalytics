/**
 * BigBrotherAnalytics - Backtesting Module (C++23)
 *
 * Historical validation of trading strategies with comprehensive metrics.
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - Fluent API for backtest configuration
 * - Modern error handling with std::expected
 * - Performance-optimized simulation
 * - Comprehensive metrics calculation
 */

// Global module fragment
module;

#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <string>
#include <algorithm>

// Module declaration
export module bigbrother.backtest;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.strategy;
import bigbrother.risk;

export namespace bigbrother::backtest {

using namespace bigbrother::types;

// ============================================================================
// Backtest Configuration
// ============================================================================

/**
 * Backtest Configuration
 * C.1: Struct for passive data
 */
struct BacktestConfig {
    Timestamp start_date{0};
    Timestamp end_date{0};
    double initial_capital{30'000.0};
    double commission_per_trade{0.65};
    double slippage_bps{2.0};
    bool allow_short_selling{false};
    bool reinvest_profits{true};
    std::string data_frequency{"1day"};

    [[nodiscard]] auto validate() const noexcept -> Result<void>;
};

/**
 * Backtest Metrics
 */
struct BacktestMetrics {
    // Returns
    double total_return{0.0};
    double total_return_percent{0.0};
    double annualized_return{0.0};
    double cagr{0.0};

    // Risk-adjusted
    double sharpe_ratio{0.0};
    double sortino_ratio{0.0};
    double max_drawdown{0.0};
    double max_drawdown_percent{0.0};

    // Trade statistics
    int64_t total_trades{0};
    int64_t winning_trades{0};
    int64_t losing_trades{0};
    double win_rate{0.0};

    // P&L
    double total_gross_pnl{0.0};
    double total_net_pnl{0.0};
    double avg_win{0.0};
    double avg_loss{0.0};
    double profit_factor{0.0};
    double expectancy{0.0};

    [[nodiscard]] auto passesThresholds() const noexcept -> bool {
        return total_return > 0.0 &&
               win_rate >= 0.60 &&
               sharpe_ratio >= 2.0 &&
               max_drawdown_percent <= 0.15;
    }
};

// ============================================================================
// Backtest Engine
// ============================================================================

/**
 * Backtest Engine
 *
 * Main backtesting engine with pImpl pattern.
 */
class BacktestEngine {
public:
    explicit BacktestEngine(BacktestConfig config);
    ~BacktestEngine();

    BacktestEngine(BacktestEngine const&) = delete;
    auto operator=(BacktestEngine const&) = delete;
    BacktestEngine(BacktestEngine&&) noexcept;
    auto operator=(BacktestEngine&&) noexcept -> BacktestEngine&;

    [[nodiscard]] auto loadHistoricalData(
        std::vector<std::string> const& symbols,
        std::string const& data_path
    ) -> Result<void>;

    auto addStrategy(std::unique_ptr<strategy::IStrategy> strategy) -> void;

    [[nodiscard]] auto run() -> Result<BacktestMetrics>;

    [[nodiscard]] auto exportTrades(std::string const& filepath) const -> Result<void>;

    [[nodiscard]] auto exportMetrics(std::string const& filepath) const -> Result<void>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

// ============================================================================
// Fluent API for Backtesting
// ============================================================================

/**
 * Backtest Runner - Fluent API
 *
 * Example Usage:
 *
 *   auto metrics = BacktestRunner()
 *       .from("2020-01-01")
 *       .to("2024-01-01")
 *       .withCapital(30000.0)
 *       .forSymbols({"SPY", "QQQ"})
 *       .addStrategy<DeltaNeutralStraddleStrategy>()
 *       .addStrategy<IronCondorStrategy>()
 *       .run();
 *
 *   if (metrics.passesThresholds()) {
 *       std::println("Strategy is profitable!");
 *   }
 */
class BacktestRunner {
public:
    BacktestRunner();

    [[nodiscard]] auto from(std::string const& date) -> BacktestRunner&;
    [[nodiscard]] auto to(std::string const& date) -> BacktestRunner&;
    [[nodiscard]] auto withCapital(double capital) -> BacktestRunner&;
    [[nodiscard]] auto commission(double comm) -> BacktestRunner&;
    [[nodiscard]] auto slippage(double slip) -> BacktestRunner&;
    [[nodiscard]] auto forSymbols(std::vector<std::string> symbols) -> BacktestRunner&;
    [[nodiscard]] auto loadData(std::string const& path) -> BacktestRunner&;

    template<typename StrategyT, typename... Args>
    [[nodiscard]] auto addStrategy(Args&&... args) -> BacktestRunner& {
        if (!engine_) {
            engine_ = std::make_unique<BacktestEngine>(config_);
        }
        engine_->addStrategy(
            std::make_unique<StrategyT>(std::forward<Args>(args)...)
        );
        return *this;
    }

    [[nodiscard]] auto run() -> BacktestMetrics;
    [[nodiscard]] auto exportTrades(std::string const& path) -> BacktestRunner&;
    [[nodiscard]] auto exportMetrics(std::string const& path) -> BacktestRunner&;

private:
    BacktestConfig config_;
    std::vector<std::string> symbols_;
    std::string data_path_;
    std::unique_ptr<BacktestEngine> engine_;
};

} // export namespace bigbrother::backtest
