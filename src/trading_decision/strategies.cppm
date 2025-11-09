/**
 * BigBrotherAnalytics - Trading Strategies Module (C++23)
 *
 * Comprehensive strategy system with fluent API.
 * Consolidates: strategy_manager, strategy_straddle, strategy_strangle, strategy_volatility_arb
 *
 * Following C++ Core Guidelines:
 * - I.25: Prefer abstract classes as interfaces
 * - C.21: Rule of Five
 * - R.1: RAII
 * - Trailing return syntax
 */

// Global module fragment
module;

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.strategies;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.options.pricing;
import bigbrother.strategy;

export namespace bigbrother::strategies {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using namespace bigbrother::options;
using bigbrother::strategy::IStrategy;
using bigbrother::strategy::SignalType;
using bigbrother::strategy::StrategyContext;
// Note: TradingSignal defined in strategy module, not types

// ============================================================================
// Strategy Performance Tracking
// ============================================================================

struct StrategyPerformance {
    std::string name;
    int signals_generated{0};
    int trades_executed{0};
    double total_pnl{0.0};
    double win_rate{0.0};
    double sharpe_ratio{0.0};
    bool active{true};
};

// ============================================================================
// Straddle Strategy
// ============================================================================

class StraddleStrategy final : public IStrategy {
  public:
    StraddleStrategy() = default;

    // C.21: Rule of Five
    StraddleStrategy(StraddleStrategy const&) = delete;
    auto operator=(StraddleStrategy const&) -> StraddleStrategy& = delete;
    StraddleStrategy(StraddleStrategy&&) noexcept = default;
    auto operator=(StraddleStrategy&&) noexcept -> StraddleStrategy& = default;
    ~StraddleStrategy() override = default;

    [[nodiscard]] auto getName() const noexcept -> std::string override { return "Long Straddle"; }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    auto setActive(bool active) -> void override { active_ = active; }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {{"min_iv_rank", "0.70"}};
    }

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> override {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        // Look for high IV rank opportunities in options chains
        for (auto const& [symbol, chain] : context.options_chains) {
            // Stub logic - would calculate IV rank from options chain
            bigbrother::strategy::TradingSignal signal;
            signal.symbol = symbol;
            signal.strategy_name = getName();
            signal.type = SignalType::Buy;
            signal.confidence = 0.75;
            signals.push_back(signal);
        }

        Logger::getInstance().info("{}: Generated {} signals", getName(), signals.size());
        return signals;
    }

  private:
    bool active_{true};
};

// ============================================================================
// Strangle Strategy
// ============================================================================

class StrangleStrategy final : public IStrategy {
  public:
    StrangleStrategy() = default;

    // C.21: Rule of Five
    StrangleStrategy(StrangleStrategy const&) = delete;
    auto operator=(StrangleStrategy const&) -> StrangleStrategy& = delete;
    StrangleStrategy(StrangleStrategy&&) noexcept = default;
    auto operator=(StrangleStrategy&&) noexcept -> StrangleStrategy& = default;
    ~StrangleStrategy() override = default;

    [[nodiscard]] auto getName() const noexcept -> std::string override { return "Long Strangle"; }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    auto setActive(bool active) -> void override { active_ = active; }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {{"min_iv_rank", "0.65"}};
    }

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> override {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        // Look for high IV with expected movement
        for (auto const& [symbol, chain] : context.options_chains) {
            // Stub logic
            bigbrother::strategy::TradingSignal signal;
            signal.symbol = symbol;
            signal.strategy_name = getName();
            signal.type = SignalType::Buy;
            signal.confidence = 0.70;
            signals.push_back(signal);
        }

        Logger::getInstance().info("{}: Generated {} signals", getName(), signals.size());
        return signals;
    }

  private:
    bool active_{true};
};

// ============================================================================
// Volatility Arbitrage Strategy
// ============================================================================

class VolatilityArbStrategy final : public IStrategy {
  public:
    VolatilityArbStrategy() = default;

    // C.21: Rule of Five
    VolatilityArbStrategy(VolatilityArbStrategy const&) = delete;
    auto operator=(VolatilityArbStrategy const&) -> VolatilityArbStrategy& = delete;
    VolatilityArbStrategy(VolatilityArbStrategy&&) noexcept = default;
    auto operator=(VolatilityArbStrategy&&) noexcept -> VolatilityArbStrategy& = default;
    ~VolatilityArbStrategy() override = default;

    [[nodiscard]] auto getName() const noexcept -> std::string override {
        return "Volatility Arbitrage";
    }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    auto setActive(bool active) -> void override { active_ = active; }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {{"min_iv_hv_spread", "0.10"}};
    }

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> override {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        // Look for IV vs HV divergence in options chains
        for (auto const& [symbol, chain] : context.options_chains) {
            // Stub logic - would calculate IV vs HV from options chain
            bigbrother::strategy::TradingSignal signal;
            signal.symbol = symbol;
            signal.strategy_name = getName();
            signal.type = SignalType::Buy;
            signal.confidence = 0.80;
            signals.push_back(signal);
        }

        Logger::getInstance().info("{}: Generated {} signals", getName(), signals.size());
        return signals;
    }

  private:
    bool active_{true};
};

// ============================================================================
// Strategy Manager (Fluent API)
// ============================================================================

class StrategyManager {
  public:
    StrategyManager() = default;

    // C.21: Rule of Five
    // Non-copyable, non-movable due to mutex member
    StrategyManager(StrategyManager const&) = delete;
    auto operator=(StrategyManager const&) -> StrategyManager& = delete;
    StrategyManager(StrategyManager&&) noexcept = delete;
    auto operator=(StrategyManager&&) noexcept -> StrategyManager& = delete;
    ~StrategyManager() = default;

    // Fluent API
    [[nodiscard]] auto addStrategy(std::unique_ptr<IStrategy> strategy) -> StrategyManager& {
        std::lock_guard<std::mutex> lock(mutex_);

        auto const strategy_name = strategy->getName();
        strategies_.push_back(std::move(strategy));
        performance_[strategy_name] = StrategyPerformance{.name = strategy_name};

        Logger::getInstance().info("Added strategy: {}", strategy_name);
        return *this;
    }

    [[nodiscard]] auto removeStrategy(std::string const& strategy_name) -> StrategyManager& {
        std::lock_guard<std::mutex> lock(mutex_);

        strategies_.erase(
            std::remove_if(strategies_.begin(), strategies_.end(),
                           [&](auto const& s) { return s->getName() == strategy_name; }),
            strategies_.end());
        performance_.erase(strategy_name);

        Logger::getInstance().info("Removed strategy: {}", strategy_name);
        return *this;
    }

    [[nodiscard]] auto setStrategyActive(std::string const& name, bool active) -> StrategyManager& {
        std::lock_guard<std::mutex> lock(mutex_);

        if (auto it = performance_.find(name); it != performance_.end()) {
            it->second.active = active;
            Logger::getInstance().info("Strategy {} {}", name, active ? "enabled" : "disabled");
        }

        return *this;
    }

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<bigbrother::strategy::TradingSignal> all_signals;

        for (auto const& strategy : strategies_) {
            auto const strategy_name = strategy->getName();

            // Check if strategy is active
            if (auto it = performance_.find(strategy_name);
                it != performance_.end() && !it->second.active) {
                continue;
            }

            auto signals = strategy->generateSignals(context);
            performance_[strategy_name].signals_generated += signals.size();

            all_signals.insert(all_signals.end(), std::make_move_iterator(signals.begin()),
                               std::make_move_iterator(signals.end()));
        }

        // Deduplicate and prioritize
        deduplicateSignals(all_signals);
        prioritizeSignals(all_signals);

        Logger::getInstance().info("Generated {} total signals from {} strategies",
                                   all_signals.size(), strategies_.size());

        return all_signals;
    }

    [[nodiscard]] auto getPerformance(std::string const& strategy_name) const
        -> std::optional<StrategyPerformance> {
        std::lock_guard<std::mutex> lock(mutex_);

        if (auto it = performance_.find(strategy_name); it != performance_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    [[nodiscard]] auto getAllPerformance() const -> std::vector<StrategyPerformance> {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<StrategyPerformance> result;
        result.reserve(performance_.size());

        for (auto const& [_, perf] : performance_) {
            result.push_back(perf);
        }

        return result;
    }

  private:
    auto deduplicateSignals(std::vector<bigbrother::strategy::TradingSignal>& signals) -> void {
        // Remove duplicate symbols, keeping highest confidence
        std::sort(signals.begin(), signals.end(), [](auto const& a, auto const& b) {
            if (a.symbol == b.symbol) {
                return a.confidence > b.confidence;
            }
            return a.symbol < b.symbol;
        });

        signals.erase(
            std::unique(signals.begin(), signals.end(),
                        [](auto const& a, auto const& b) { return a.symbol == b.symbol; }),
            signals.end());
    }

    auto prioritizeSignals(std::vector<bigbrother::strategy::TradingSignal>& signals) -> void {
        // Sort by confidence (highest first)
        std::sort(signals.begin(), signals.end(),
                  [](auto const& a, auto const& b) { return a.confidence > b.confidence; });
    }

    std::vector<std::unique_ptr<IStrategy>> strategies_;
    std::unordered_map<std::string, StrategyPerformance> performance_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] inline auto createStraddleStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<StraddleStrategy>();
}

[[nodiscard]] inline auto createStrangleStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<StrangleStrategy>();
}

[[nodiscard]] inline auto createVolatilityArbStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<VolatilityArbStrategy>();
}

} // namespace bigbrother::strategies
