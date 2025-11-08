/**
 * Strategy Manager Implementation
 * C++23 module implementation unit
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-08
 */

// Global module fragment
module;

#include <algorithm>

// Module implementation unit declaration
module bigbrother.strategies;

import bigbrother.utils.logger;

namespace bigbrother::strategy {

// ============================================================================
// StrategyManager Implementation
// ============================================================================

// Cannot default move ops because of mutex member
StrategyManager::StrategyManager(StrategyManager&& other) noexcept
    : strategies_{std::move(other.strategies_)},
      performance_{std::move(other.performance_)} {}

auto StrategyManager::operator=(StrategyManager&& other) noexcept -> StrategyManager& {
    if (this != &other) {
        std::lock_guard lock1{mutex_};
        std::lock_guard lock2{other.mutex_};
        strategies_ = std::move(other.strategies_);
        performance_ = std::move(other.performance_);
    }
    return *this;
}

auto StrategyManager::addStrategy(std::unique_ptr<IStrategy> strategy) -> void {
    std::lock_guard lock{mutex_};

    auto const name = strategy->getName();
    strategies_.push_back(std::move(strategy));

    // Initialize performance tracking
    performance_[name] = StrategyPerformance{
        .name = name,
        .signals_generated = 0,
        .trades_executed = 0,
        .total_pnl = 0.0,
        .win_rate = 0.0,
        .sharpe_ratio = 0.0,
        .active = true
    };
}

auto StrategyManager::removeStrategy(std::string const& strategy_name) -> void {
    std::lock_guard lock{mutex_};

    strategies_.erase(
        std::remove_if(strategies_.begin(), strategies_.end(),
            [&](auto const& s) { return s->getName() == strategy_name; }),
        strategies_.end()
    );

    performance_.erase(strategy_name);
}

auto StrategyManager::generateSignals(StrategyContext const& context)
    -> std::vector<TradingSignal> {

    std::lock_guard lock{mutex_};

    std::vector<TradingSignal> all_signals;

    // Generate signals from each active strategy
    for (auto const& strategy : strategies_) {
        if (!strategy->isActive()) {
            continue;
        }

        auto signals = strategy->generateSignals(context);

        // Track signal generation
        if (auto it = performance_.find(strategy->getName()); it != performance_.end()) {
            it->second.signals_generated += static_cast<int>(signals.size());
        }

        // Add strategy name to signals
        for (auto& signal : signals) {
            if (signal.strategy_name.empty()) {
                signal.strategy_name = strategy->getName();
            }
        }

        all_signals.insert(all_signals.end(), signals.begin(), signals.end());
    }

    // Deduplicate and resolve conflicts
    all_signals = deduplicateSignals(std::move(all_signals));
    all_signals = resolveConflicts(std::move(all_signals));

    // Sort by expected value (highest first)
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

auto StrategyManager::setStrategyActive(std::string const& name, bool active) -> void {
    std::lock_guard lock{mutex_};

    for (auto& strategy : strategies_) {
        if (strategy->getName() == name) {
            strategy->setActive(active);
            if (auto it = performance_.find(name); it != performance_.end()) {
                it->second.active = active;
            }
            break;
        }
    }
}

auto StrategyManager::getPerformance(std::string const& strategy_name) const
    -> std::optional<StrategyPerformance> {
    std::lock_guard lock{mutex_};

    if (auto it = performance_.find(strategy_name); it != performance_.end()) {
        return it->second;
    }

    return std::nullopt;
}

auto StrategyManager::getAllPerformance() const -> std::vector<StrategyPerformance> {
    std::lock_guard lock{mutex_};

    std::vector<StrategyPerformance> result;
    result.reserve(performance_.size());

    for (auto const& [name, perf] : performance_) {
        result.push_back(perf);
    }

    return result;
}

auto StrategyManager::deduplicateSignals(std::vector<TradingSignal> signals) const
    -> std::vector<TradingSignal> {

    // Keep highest confidence signal for each symbol
    std::unordered_map<std::string, TradingSignal> best_signals;

    for (auto& signal : signals) {
        auto const& symbol = signal.symbol;

        if (auto it = best_signals.find(symbol); it == best_signals.end()) {
            best_signals[symbol] = std::move(signal);
        } else if (signal.confidence > it->second.confidence) {
            it->second = std::move(signal);
        }
    }

    std::vector<TradingSignal> result;
    result.reserve(best_signals.size());

    for (auto& [symbol, signal] : best_signals) {
        result.push_back(std::move(signal));
    }

    return result;
}

auto StrategyManager::resolveConflicts(std::vector<TradingSignal> signals) const
    -> std::vector<TradingSignal> {
    // For now, just return as-is (already deduplicated)
    // Future: Add sophisticated conflict resolution
    return signals;
}

// ============================================================================
// StrategyExecutor Implementation
// ============================================================================

auto StrategyExecutor::execute() -> Result<std::vector<std::string>> {
    // Stub implementation
    std::vector<std::string> order_ids;
    order_ids.push_back("ORDER_001_STUB");
    return order_ids;
}

} // namespace bigbrother::strategy
