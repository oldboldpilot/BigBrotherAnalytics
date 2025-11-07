#pragma once

#include "strategy.hpp"
#include <memory>
#include <vector>
#include <mutex>

namespace bigbrother::strategy {

/**
 * Strategy Manager
 *
 * Orchestrates multiple trading strategies.
 * Aggregates signals, prioritizes by expected value, and manages execution.
 *
 * Features:
 * - Run multiple strategies simultaneously
 * - Aggregate and deduplicate signals
 * - Prioritize by expected value
 * - Risk-adjusted position sizing
 * - Conflict resolution (multiple strategies want same symbol)
 * - Performance tracking per strategy
 * - Dynamic strategy enable/disable
 */
class StrategyManager {
public:
    StrategyManager() = default;
    ~StrategyManager() = default;

    // Delete copy, allow move
    StrategyManager(StrategyManager const&) = delete;
    auto operator=(StrategyManager const&) = delete;
    StrategyManager(StrategyManager&&) noexcept;
    auto operator=(StrategyManager&&) noexcept -> StrategyManager&;

    /**
     * Register a strategy
     */
    auto addStrategy(std::unique_ptr<IStrategy> strategy) -> void;

    /**
     * Remove a strategy
     */
    auto removeStrategy(std::string const& strategy_name) -> void;

    /**
     * Generate signals from all active strategies
     *
     * @param context Market context
     * @return Aggregated and prioritized signals
     */
    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal>;

    /**
     * Get all registered strategies
     */
    [[nodiscard]] auto getStrategies() const
        -> std::vector<IStrategy const*>;

    /**
     * Enable/disable strategy
     */
    auto setStrategyActive(std::string const& name, bool active) -> void;

    /**
     * Get strategy performance
     */
    struct StrategyPerformance {
        std::string name;
        int signals_generated;
        int trades_executed;
        double total_pnl;
        double win_rate;
        double sharpe_ratio;
        bool active;
    };

    [[nodiscard]] auto getPerformance(std::string const& strategy_name) const
        -> std::optional<StrategyPerformance>;

    /**
     * Get all strategy performance
     */
    [[nodiscard]] auto getAllPerformance() const
        -> std::vector<StrategyPerformance>;

private:
    /**
     * Deduplicate signals (same symbol from multiple strategies)
     */
    [[nodiscard]] auto deduplicateSignals(std::vector<TradingSignal> signals) const
        -> std::vector<TradingSignal>;

    /**
     * Resolve conflicts when multiple strategies want same symbol
     */
    [[nodiscard]] auto resolveConflicts(std::vector<TradingSignal> signals) const
        -> std::vector<TradingSignal>;

    std::vector<std::unique_ptr<IStrategy>> strategies_;
    mutable std::mutex mutex_;

    // Performance tracking
    std::map<std::string, StrategyPerformance> performance_;
};

/**
 * Fluent API for Strategy Execution
 *
 * Example Usage:
 *
 *   // Create strategy manager
 *   StrategyManager manager;
 *
 *   // Add strategies
 *   StrategyExecutor(manager)
 *       .addStraddle()
 *       .addStrangle()
 *       .addVolatilityArb()
 *       .configure();
 *
 *   // Generate signals
 *   auto signals = StrategyExecutor(manager)
 *       .withContext(context)
 *       .minConfidence(0.70)
 *       .minExpectedReturn(100.0)
 *       .maxSignals(5)
 *       .generateSignals();
 *
 *   // Execute top signals
 *   StrategyExecutor(manager)
 *       .withRiskManager(risk_mgr)
 *       .withSchwabClient(schwab)
 *       .signals(signals)
 *       .validateWithMonteCarlo()
 *       .execute();
 */
class StrategyExecutor {
public:
    explicit StrategyExecutor(StrategyManager& manager)
        : manager_{manager} {}

    // Strategy registration
    [[nodiscard]] auto addStraddle(
        DeltaNeutralStraddleStrategy::Parameters params = {}
    ) -> StrategyExecutor& {
        auto strategy = std::make_unique<DeltaNeutralStraddleStrategy>(params);
        manager_.addStrategy(std::move(strategy));
        return *this;
    }

    [[nodiscard]] auto addStrangle(
        DeltaNeutralStrangleStrategy::Parameters params = {}
    ) -> StrategyExecutor& {
        auto strategy = std::make_unique<DeltaNeutralStrangleStrategy>(params);
        manager_.addStrategy(std::move(strategy));
        return *this;
    }

    [[nodiscard]] auto addVolatilityArb(
        VolatilityArbitrageStrategy::Parameters params = {}
    ) -> StrategyExecutor& {
        auto strategy = std::make_unique<VolatilityArbitrageStrategy>(params);
        manager_.addStrategy(std::move(strategy));
        return *this;
    }

    [[nodiscard]] auto addMeanReversion(
        MeanReversionStrategy::Parameters params = {}
    ) -> StrategyExecutor& {
        auto strategy = std::make_unique<MeanReversionStrategy>(params);
        manager_.addStrategy(std::move(strategy));
        return *this;
    }

    // Configuration
    [[nodiscard]] auto configure() -> StrategyExecutor& {
        // Configuration complete
        return *this;
    }

    // Signal generation
    [[nodiscard]] auto withContext(StrategyContext context) -> StrategyExecutor& {
        context_ = std::move(context);
        return *this;
    }

    [[nodiscard]] auto minConfidence(double threshold) -> StrategyExecutor& {
        min_confidence_ = threshold;
        return *this;
    }

    [[nodiscard]] auto minExpectedReturn(double threshold) -> StrategyExecutor& {
        min_expected_return_ = threshold;
        return *this;
    }

    [[nodiscard]] auto maxSignals(int max) -> StrategyExecutor& {
        max_signals_ = max;
        return *this;
    }

    // Terminal: generate signals
    [[nodiscard]] auto generateSignals() -> std::vector<TradingSignal> {
        if (!context_) {
            LOG_ERROR("Strategy context not provided");
            return {};
        }

        auto signals = manager_.generateSignals(*context_);

        // Filter by criteria
        std::vector<TradingSignal> filtered;

        for (auto& signal : signals) {
            if (signal.confidence >= min_confidence_.value_or(0.0) &&
                signal.expected_return >= min_expected_return_.value_or(0.0)) {
                filtered.push_back(std::move(signal));
            }
        }

        // Limit to max signals
        if (max_signals_ && filtered.size() > static_cast<size_t>(*max_signals_)) {
            filtered.resize(*max_signals_);
        }

        return filtered;
    }

    // Execution
    [[nodiscard]] auto withRiskManager(risk::RiskManager& risk_mgr) -> StrategyExecutor& {
        risk_manager_ = &risk_mgr;
        return *this;
    }

    [[nodiscard]] auto withSchwabClient(schwab::SchwabClient& client) -> StrategyExecutor& {
        schwab_client_ = &client;
        return *this;
    }

    [[nodiscard]] auto signals(std::vector<TradingSignal> sigs) -> StrategyExecutor& {
        signals_ = std::move(sigs);
        return *this;
    }

    [[nodiscard]] auto validateWithMonteCarlo(bool validate = true) -> StrategyExecutor& {
        validate_monte_carlo_ = validate;
        return *this;
    }

    // Terminal: execute signals
    [[nodiscard]] auto execute() -> Result<std::vector<std::string>>;

private:
    StrategyManager& manager_;
    std::optional<StrategyContext> context_;
    std::optional<double> min_confidence_;
    std::optional<double> min_expected_return_;
    std::optional<int> max_signals_;
    std::vector<TradingSignal> signals_;

    risk::RiskManager* risk_manager_{nullptr};
    schwab::SchwabClient* schwab_client_{nullptr};
    bool validate_monte_carlo_{true};
};

/**
 * Convenience functions
 */

// Create default strategy manager with all strategies
[[nodiscard]] inline auto createDefaultStrategyManager()
    -> std::unique_ptr<StrategyManager> {

    auto manager = std::make_unique<StrategyManager>();

    StrategyExecutor(*manager)
        .addStraddle()
        .addStrangle()
        .addVolatilityArb()
        .addMeanReversion()
        .configure();

    return manager;
}

} // namespace bigbrother::strategy
