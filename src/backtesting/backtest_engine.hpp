#pragma once

#include "../utils/types.hpp"
#include "../correlation_engine/options_pricing.hpp"
#include "../trading_decision/strategy.hpp"
#include "../risk_management/risk_manager.hpp"

#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace bigbrother::backtest {

using namespace types;

/**
 * Backtest Configuration
 */
struct BacktestConfig {
    Timestamp start_date;
    Timestamp end_date;
    double initial_capital;
    double commission_per_trade;
    double slippage_bps;            // Slippage in basis points
    bool allow_short_selling;
    bool reinvest_profits;
    std::string data_frequency;     // "1min", "5min", "1day"

    [[nodiscard]] auto validate() const noexcept -> Result<void>;
};

/**
 * Simulated Fill
 *
 * Represents how an order would have been filled historically
 */
struct SimulatedFill {
    std::string order_id;
    std::string symbol;
    Quantity quantity;
    Price fill_price;
    double commission;
    double slippage;
    Timestamp fill_time;

    [[nodiscard]] auto totalCost() const noexcept -> double {
        return fill_price * static_cast<double>(std::abs(quantity)) +
               commission + slippage;
    }
};

/**
 * Backtest Trade
 *
 * Complete record of a simulated trade
 */
struct BacktestTrade {
    std::string trade_id;
    std::string symbol;
    std::string strategy;

    // Entry
    Timestamp entry_time;
    Price entry_price;
    Quantity quantity;
    double entry_cost;

    // Exit
    Timestamp exit_time;
    Price exit_price;
    double exit_proceeds;

    // P&L
    double gross_pnl;
    double net_pnl;           // After commissions and slippage
    double return_percent;

    // Hold time
    Duration hold_time_us;

    // Rationale
    std::string entry_rationale;
    std::string exit_rationale;

    [[nodiscard]] auto holdTimeDays() const noexcept -> double {
        return static_cast<double>(hold_time_us) / (1'000'000.0 * 86400.0);
    }

    [[nodiscard]] auto isWinner() const noexcept -> bool {
        return net_pnl > 0.0;
    }
};

/**
 * Backtest Performance Metrics (WITH TAX CALCULATIONS)
 */
struct BacktestMetrics {
    // Pre-tax returns
    double total_return;
    double total_return_percent;
    double annualized_return;
    double cagr;

    // TAX CALCULATIONS (CRITICAL FOR REAL PROFITABILITY)
    double total_tax_owed{0.0};              // Total taxes on gains
    double effective_tax_rate{0.0};          // Actual tax rate paid
    double after_tax_return{0.0};            // Return after taxes
    double after_tax_return_percent{0.0};    // % return after taxes
    double after_tax_sharpe_ratio{0.0};      // Sharpe using after-tax returns
    double tax_efficiency{0.0};              // Net/Gross (higher = better)
    int wash_sales_disallowed{0};            // Wash sale rule violations
    double wash_sale_loss_disallowed{0.0};   // Losses disallowed by wash sales

    // Risk-adjusted returns (pre-tax)
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;

    // Drawdown
    double max_drawdown;
    double max_drawdown_percent;
    Duration max_drawdown_duration_us;

    // Trade statistics
    int64_t total_trades;
    int64_t winning_trades;
    int64_t losing_trades;
    double win_rate;

    // P&L statistics
    double total_gross_pnl;
    double total_net_pnl;
    double total_commissions;
    double total_slippage;
    double avg_win;
    double avg_loss;
    double largest_win;
    double largest_loss;

    // Risk metrics
    double profit_factor;
    double expectancy;
    double kelly_criterion;
    double var_95;

    // Time metrics
    Duration total_time_in_market_us;
    double avg_hold_time_days;
    double max_hold_time_days;

    // Strategy breakdown
    std::map<std::string, double> pnl_by_strategy;
    std::map<std::string, double> win_rate_by_strategy;

    /**
     * Check if passes success criteria (AFTER TAX)
     */
    [[nodiscard]] auto passesThresholds() const noexcept -> bool {
        // Per PRD success criteria - must be AFTER TAX
        return after_tax_return > 0.0 &&      // Profitable after tax
               win_rate >= 0.60 &&             // 60% win rate
               after_tax_sharpe_ratio >= 2.0 && // Sharpe > 2.0 (after tax!)
               max_drawdown_percent <= 0.15;   // Max DD < 15%
    }

    /**
     * Check if truly profitable (after taxes)
     */
    [[nodiscard]] auto isProfitableAfterTax() const noexcept -> bool {
        return after_tax_return > 0.0;
    }

    [[nodiscard]] auto meetsTargets() const noexcept -> bool {
        // Target: $150/day profit on $30k account
        double const target_daily_return = 150.0 / 30'000.0;  // 0.5% per day

        return annualized_return / 252.0 >= target_daily_return &&
               passesThresholds();
    }
};

/**
 * Order Simulator
 *
 * Simulates how orders would have been filled using historical data
 */
class OrderSimulator {
public:
    /**
     * Simulate market order fill
     *
     * Fills at next bar's open price + slippage
     */
    [[nodiscard]] static auto simulateMarketOrder(
        Order const& order,
        Bar const& next_bar,
        double commission_per_trade,
        double slippage_bps
    ) noexcept -> SimulatedFill;

    /**
     * Simulate limit order fill
     *
     * Fills only if limit price is touched
     */
    [[nodiscard]] static auto simulateLimitOrder(
        Order const& order,
        std::vector<Bar> const& bars,
        double commission_per_trade,
        double slippage_bps
    ) noexcept -> std::optional<SimulatedFill>;

    /**
     * Simulate stop order fill
     */
    [[nodiscard]] static auto simulateStopOrder(
        Order const& order,
        std::vector<Bar> const& bars,
        double commission_per_trade,
        double slippage_bps
    ) noexcept -> std::optional<SimulatedFill>;

private:
    [[nodiscard]] static auto calculateSlippage(
        Price price,
        Quantity quantity,
        double slippage_bps
    ) noexcept -> double;
};

/**
 * Backtest Engine
 *
 * Runs historical simulations to validate trading strategies
 */
class BacktestEngine {
public:
    explicit BacktestEngine(BacktestConfig config);
    ~BacktestEngine();

    // Delete copy, allow move
    BacktestEngine(BacktestEngine const&) = delete;
    auto operator=(BacktestEngine const&) = delete;
    BacktestEngine(BacktestEngine&&) noexcept;
    auto operator=(BacktestEngine&&) noexcept -> BacktestEngine&;

    /**
     * Load historical data
     */
    [[nodiscard]] auto loadHistoricalData(
        std::vector<std::string> const& symbols,
        std::string const& data_path
    ) -> Result<void>;

    /**
     * Add strategy to backtest
     */
    auto addStrategy(std::unique_ptr<strategy::IStrategy> strategy) -> void;

    /**
     * Run backtest
     *
     * @return Backtest metrics
     */
    [[nodiscard]] auto run() -> Result<BacktestMetrics>;

    /**
     * Get all trades
     */
    [[nodiscard]] auto getTrades() const noexcept -> std::vector<BacktestTrade> const&;

    /**
     * Get equity curve (account value over time)
     */
    [[nodiscard]] auto getEquityCurve() const noexcept
        -> std::vector<std::pair<Timestamp, double>> const&;

    /**
     * Export results to CSV
     */
    [[nodiscard]] auto exportTrades(std::string const& filename) const
        -> Result<void>;

    [[nodiscard]] auto exportMetrics(std::string const& filename) const
        -> Result<void>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Fluent API for Backtest Execution
 *
 * Simplified interface for running backtests.
 *
 * Example:
 *   BacktestRunner runner;
 *   runner.from("2020-01-01")
 *         .to("2024-01-01")
 *         .withCapital(30000.0)
 *         .forSymbols({"SPY", "QQQ"})
 *         .addStrategy<DeltaNeutralStraddleStrategy>()
 *         .run();
 */
class BacktestRunner {
public:
    BacktestRunner() : config_{} {
        config_.initial_capital = 30000.0;
        config_.commission_per_trade = 0.65;
        config_.slippage_bps = 2.0;
        config_.allow_short_selling = false;
        config_.reinvest_profits = true;
    }

    [[nodiscard]] auto from(std::string const& date) -> BacktestRunner& {
        // Parse date and set start_date
        return *this;
    }

    [[nodiscard]] auto to(std::string const& date) -> BacktestRunner& {
        // Parse date and set end_date
        return *this;
    }

    [[nodiscard]] auto withCapital(double capital) -> BacktestRunner& {
        config_.initial_capital = capital;
        return *this;
    }

    [[nodiscard]] auto commission(double comm) -> BacktestRunner& {
        config_.commission_per_trade = comm;
        return *this;
    }

    [[nodiscard]] auto slippage(double slip) -> BacktestRunner& {
        config_.slippage_bps = slip;
        return *this;
    }

    [[nodiscard]] auto forSymbols(std::vector<std::string> symbols) -> BacktestRunner& {
        symbols_ = std::move(symbols);
        return *this;
    }

    [[nodiscard]] auto loadData(std::string const& path) -> BacktestRunner& {
        data_path_ = path;
        return *this;
    }

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

    [[nodiscard]] auto run() -> BacktestMetrics {
        if (!engine_) {
            engine_ = std::make_unique<BacktestEngine>(config_);
        }

        // Load data
        if (!data_path_.empty()) {
            engine_->loadHistoricalData(symbols_, data_path_);
        }

        // Run backtest
        auto result = engine_->run();
        if (result) {
            return *result;
        }

        return BacktestMetrics{};  // Empty metrics on error
    }

    [[nodiscard]] auto exportTrades(std::string const& path) -> BacktestRunner& {
        if (engine_) {
            engine_->exportTrades(path);
        }
        return *this;
    }

    [[nodiscard]] auto exportMetrics(std::string const& path) -> BacktestRunner& {
        if (engine_) {
            engine_->exportMetrics(path);
        }
        return *this;
    }

private:
    BacktestConfig config_;
    std::vector<std::string> symbols_;
    std::string data_path_;
    std::unique_ptr<BacktestEngine> engine_;
};

} // namespace bigbrother::backtest
