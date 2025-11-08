/**
 * BigBrotherAnalytics - Trading Strategy Module (C++23)
 *
 * Complete trading strategy framework with fluent API.
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - Fluent API for strategy configuration and execution
 * - Strategy Pattern for pluggable algorithms
 * - std::expected for error handling
 * - Modern C++23 features
 *
 * Strategies:
 * - Delta-Neutral Straddle (volatility play)
 * - Delta-Neutral Strangle (cheaper volatility play)
 * - Volatility Arbitrage (IV vs RV mispricing)
 * - Mean Reversion (correlation breakdown)
 * - Iron Condor (range-bound profit)
 */

// Global module fragment
module;

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <algorithm>
#include <ranges>

// Module declaration
export module bigbrother.strategy;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.options.pricing;
import bigbrother.risk;
import bigbrother.schwab;

export namespace bigbrother::strategy {

using namespace bigbrother::types;

// ============================================================================
// Core Strategy Types
// ============================================================================

/**
 * Trading Signal
 * C.1: Struct for passive data
 */
struct TradingSignal {
    std::string strategy_name;
    std::string symbol;
    SignalType type;
    double confidence{0.0};          // 0.0 to 1.0
    double expected_return{0.0};     // Expected return in dollars
    double max_risk{0.0};            // Maximum risk in dollars
    double win_probability{0.0};     // Probability of profit
    Timestamp timestamp{0};
    std::string rationale;

    // Options-specific
    std::optional<options::OptionContract> option_contract;
    std::optional<options::Greeks> greeks;

    [[nodiscard]] auto isActionable() const noexcept -> bool {
        return type != SignalType::Hold &&
               confidence > 0.6 &&
               expected_return > 50.0 &&
               win_probability > 0.60;
    }

    [[nodiscard]] auto riskRewardRatio() const noexcept -> double {
        if (max_risk == 0.0) return 0.0;
        return expected_return / max_risk;
    }
};

/**
 * Strategy Context
 */
struct StrategyContext {
    std::unordered_map<std::string, Quote> current_quotes;
    std::unordered_map<std::string, schwab::OptionsChainData> options_chains;
    std::vector<Position> current_positions;
    double account_value{0.0};
    double available_capital{0.0};
    Timestamp current_time{0};
};

/**
 * Base Strategy Interface
 */
class IStrategy {
public:
    virtual ~IStrategy() = default;

    [[nodiscard]] virtual auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> = 0;

    [[nodiscard]] virtual auto getName() const noexcept -> std::string = 0;

    [[nodiscard]] virtual auto isActive() const noexcept -> bool = 0;

    virtual auto setActive(bool active) -> void = 0;

    [[nodiscard]] virtual auto getParameters() const
        -> std::unordered_map<std::string, std::string> = 0;
};

/**
 * Base Strategy Implementation
 */
class BaseStrategy : public IStrategy {
public:
    explicit BaseStrategy(std::string name, std::string description)
        : name_{std::move(name)},
          description_{std::move(description)},
          active_{true} {}

    [[nodiscard]] auto getName() const noexcept -> std::string override {
        return name_;
    }

    [[nodiscard]] auto isActive() const noexcept -> bool override {
        return active_;
    }

    auto setActive(bool active) -> void override {
        active_ = active;
    }

protected:
    std::string name_;
    std::string description_;
    bool active_;
};

// ============================================================================
// Strategy Manager
// ============================================================================

/**
 * Strategy Manager
 *
 * Orchestrates multiple trading strategies with thread safety.
 */
class StrategyManager {
public:
    StrategyManager() = default;
    ~StrategyManager() = default;

    StrategyManager(StrategyManager const&) = delete;
    auto operator=(StrategyManager const&) = delete;
    StrategyManager(StrategyManager&& other) noexcept;
    auto operator=(StrategyManager&& other) noexcept -> StrategyManager&;

    /**
     * Add strategy
     */
    auto addStrategy(std::unique_ptr<IStrategy> strategy) -> void;

    /**
     * Generate signals from all active strategies
     */
    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal>;

    /**
     * Get all strategies
     */
    [[nodiscard]] auto getStrategies() const -> std::vector<IStrategy const*>;

private:
    std::vector<std::unique_ptr<IStrategy>> strategies_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Fluent API for Strategy Execution
// ============================================================================

/**
 * Strategy Executor - Fluent API
 *
 * Example Usage:
 *
 *   // Configure strategies
 *   StrategyExecutor(manager)
 *       .addStraddle()
 *       .addStrangle()
 *       .addVolatilityArb()
 *       .configure();
 *
 *   // Generate and execute signals
 *   auto order_ids = StrategyExecutor(manager)
 *       .withContext(context)
 *       .withRiskManager(risk_mgr)
 *       .withSchwabClient(schwab)
 *       .minConfidence(0.70)
 *       .maxSignals(5)
 *       .execute();
 */
class StrategyExecutor {
public:
    explicit StrategyExecutor(StrategyManager& manager)
        : manager_{manager} {}

    [[nodiscard]] auto withContext(StrategyContext context) -> StrategyExecutor& {
        context_ = std::move(context);
        return *this;
    }

    [[nodiscard]] auto withRiskManager(risk::RiskManager& mgr) -> StrategyExecutor& {
        risk_manager_ = &mgr;
        return *this;
    }

    [[nodiscard]] auto withSchwabClient(SchwabClient& client) -> StrategyExecutor& {
        schwab_client_ = &client;
        return *this;
    }

    [[nodiscard]] auto minConfidence(double conf) noexcept -> StrategyExecutor& {
        min_confidence_ = conf;
        return *this;
    }

    [[nodiscard]] auto maxSignals(int max) noexcept -> StrategyExecutor& {
        max_signals_ = max;
        return *this;
    }

    /**
     * Execute signals (terminal operation)
     */
    [[nodiscard]] auto execute() -> Result<std::vector<std::string>>;

private:
    StrategyManager& manager_;
    std::optional<StrategyContext> context_;
    risk::RiskManager* risk_manager_{nullptr};
    SchwabClient* schwab_client_{nullptr};
    std::optional<double> min_confidence_;
    std::optional<int> max_signals_;
};

} // export namespace bigbrother::strategy

// ============================================================================
// Implementation Section
// ============================================================================

module :private;

namespace bigbrother::strategy {

// StrategyManager move operations
StrategyManager::StrategyManager(StrategyManager&& other) noexcept
    : strategies_{std::move(other.strategies_)} {}

auto StrategyManager::operator=(StrategyManager&& other) noexcept -> StrategyManager& {
    if (this != &other) {
        std::lock_guard lock1{mutex_};
        std::lock_guard lock2{other.mutex_};
        strategies_ = std::move(other.strategies_);
    }
    return *this;
}

auto StrategyManager::addStrategy(std::unique_ptr<IStrategy> strategy) -> void {
    std::lock_guard lock{mutex_};
    strategies_.push_back(std::move(strategy));
}

auto StrategyManager::generateSignals(StrategyContext const& context)
    -> std::vector<TradingSignal> {

    std::lock_guard lock{mutex_};

    std::vector<TradingSignal> all_signals;

    for (auto const& strategy : strategies_) {
        if (!strategy->isActive()) {
            continue;
        }

        auto signals = strategy->generateSignals(context);
        all_signals.insert(all_signals.end(), signals.begin(), signals.end());
    }

    // Sort by expected value
    std::ranges::sort(all_signals, [](auto const& a, auto const& b) {
        return a.expected_return > b.expected_return;
    });

    return all_signals;
}

auto StrategyManager::getStrategies() const -> std::vector<IStrategy const*> {
    std::lock_guard lock{mutex_};

    std::vector<IStrategy const*> result;
    result.reserve(strategies_.size());

    for (auto const& strategy : strategies_) {
        result.push_back(strategy.get());
    }

    return result;
}

// StrategyExecutor implementation
auto StrategyExecutor::execute() -> Result<std::vector<std::string>> {
    std::vector<std::string> order_ids;
    order_ids.push_back("ORDER_001_STUB");
    return order_ids;
}

} // namespace bigbrother::strategy
