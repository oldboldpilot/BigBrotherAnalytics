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
// Sector Rotation Strategy
// ============================================================================

class SectorRotationStrategy final : public IStrategy {
  public:
    struct SectorScore {
        std::string sector_name;
        std::string etf_ticker;
        double employment_score{0.0};
        double composite_score{0.0};
        bool is_improving{false};
        bool is_declining{false};
    };

    SectorRotationStrategy() = default;

    // C.21: Rule of Five
    SectorRotationStrategy(SectorRotationStrategy const&) = delete;
    auto operator=(SectorRotationStrategy const&) -> SectorRotationStrategy& = delete;
    SectorRotationStrategy(SectorRotationStrategy&&) noexcept = default;
    auto operator=(SectorRotationStrategy&&) noexcept -> SectorRotationStrategy& = default;
    ~SectorRotationStrategy() override = default;

    [[nodiscard]] auto getName() const noexcept -> std::string override {
        return "Sector Rotation";
    }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    auto setActive(bool active) -> void override { active_ = active; }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {{"min_employment_score", "0.60"}, {"rotation_threshold", "0.70"}};
    }

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> override {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        // Define 11 GICS sectors with ETF mappings
        std::vector<SectorScore> sectors = {
            {"Energy", "XLE", 0.0, 0.0, false, false},
            {"Materials", "XLB", 0.0, 0.0, false, false},
            {"Industrials", "XLI", 0.0, 0.0, false, false},
            {"Consumer Discretionary", "XLY", 0.0, 0.0, false, false},
            {"Consumer Staples", "XLP", 0.0, 0.0, false, false},
            {"Health Care", "XLV", 0.0, 0.0, false, false},
            {"Financials", "XLF", 0.0, 0.0, false, false},
            {"Information Technology", "XLK", 0.0, 0.0, false, false},
            {"Communication Services", "XLC", 0.0, 0.0, false, false},
            {"Utilities", "XLU", 0.0, 0.0, false, false},
            {"Real Estate", "XLRE", 0.0, 0.0, false, false},
        };

        // Score each sector based on employment data
        scoreSectors(sectors);

        // Generate Buy signals for improving sectors
        for (auto const& sector : sectors) {
            if (sector.is_improving && sector.composite_score > 0.70) {
                bigbrother::strategy::TradingSignal signal;
                signal.symbol = sector.etf_ticker;
                signal.strategy_name = getName();
                signal.type = SignalType::Buy;
                signal.confidence = sector.composite_score;
                signal.expected_return = sector.composite_score * 100.0;
                signal.max_risk = 50.0;
                signal.win_probability = sector.composite_score;
                signal.timestamp = context.current_time;
                signal.rationale = "Sector rotation: " + sector.sector_name +
                                   " showing strong employment growth (score: " +
                                   std::to_string(sector.employment_score) + ")";
                signals.push_back(signal);
            }
        }

        // Generate Sell signals for declining sectors
        for (auto const& sector : sectors) {
            if (sector.is_declining && sector.composite_score < -0.70) {
                bigbrother::strategy::TradingSignal signal;
                signal.symbol = sector.etf_ticker;
                signal.strategy_name = getName();
                signal.type = SignalType::Sell;
                signal.confidence = std::abs(sector.composite_score);
                signal.expected_return = std::abs(sector.composite_score) * 80.0;
                signal.max_risk = 60.0;
                signal.win_probability = std::abs(sector.composite_score);
                signal.timestamp = context.current_time;
                signal.rationale = "Sector rotation: " + sector.sector_name +
                                   " showing employment weakness (score: " +
                                   std::to_string(sector.employment_score) + ")";
                signals.push_back(signal);
            }
        }

        Logger::getInstance().info("{}: Generated {} signals from sector analysis", getName(),
                                   signals.size());
        return signals;
    }

  private:
    bool active_{true};

    auto scoreSectors(std::vector<SectorScore>& sectors) -> void {
        // Scoring logic based on employment trends
        // This is a simplified scoring model - in production would query DuckDB
        // for actual employment data from sector_employment table

        for (auto& sector : sectors) {
            // Simulate employment score calculation
            // In production: Query BLS employment data, calculate trends, momentum
            // For now: Generate realistic stub scores

            // Cyclical sectors (Energy, Materials, Industrials, Financials)
            if (sector.sector_name == "Energy") {
                sector.employment_score = 0.45; // Moderate
                sector.composite_score = 0.50;
                sector.is_improving = false;
                sector.is_declining = false;
            } else if (sector.sector_name == "Materials") {
                sector.employment_score = 0.55;
                sector.composite_score = 0.60;
                sector.is_improving = false;
                sector.is_declining = false;
            } else if (sector.sector_name == "Industrials") {
                sector.employment_score = 0.75;
                sector.composite_score = 0.78;
                sector.is_improving = true;
                sector.is_declining = false;
            } else if (sector.sector_name == "Financials") {
                sector.employment_score = 0.65;
                sector.composite_score = 0.68;
                sector.is_improving = false;
                sector.is_declining = false;
            }
            // Technology and Communication (high growth)
            else if (sector.sector_name == "Information Technology") {
                sector.employment_score = 0.80;
                sector.composite_score = 0.85;
                sector.is_improving = true;
                sector.is_declining = false;
            } else if (sector.sector_name == "Communication Services") {
                sector.employment_score = 0.40;
                sector.composite_score = 0.35;
                sector.is_improving = false;
                sector.is_declining = false;
            }
            // Healthcare (stable growth)
            else if (sector.sector_name == "Health Care") {
                sector.employment_score = 0.82;
                sector.composite_score = 0.88;
                sector.is_improving = true;
                sector.is_declining = false;
            }
            // Consumer sectors
            else if (sector.sector_name == "Consumer Discretionary") {
                sector.employment_score = 0.35;
                sector.composite_score = -0.75;
                sector.is_improving = false;
                sector.is_declining = true;
            } else if (sector.sector_name == "Consumer Staples") {
                sector.employment_score = 0.50;
                sector.composite_score = 0.55;
                sector.is_improving = false;
                sector.is_declining = false;
            }
            // Defensive sectors
            else if (sector.sector_name == "Utilities") {
                sector.employment_score = 0.48;
                sector.composite_score = 0.52;
                sector.is_improving = false;
                sector.is_declining = false;
            } else if (sector.sector_name == "Real Estate") {
                sector.employment_score = 0.30;
                sector.composite_score = -0.78;
                sector.is_improving = false;
                sector.is_declining = true;
            }

            // TODO: Replace with actual database query
            // Query: SELECT employment_count, unemployment_rate, job_openings
            //        FROM sector_employment WHERE sector_id = ?
            //        ORDER BY report_date DESC LIMIT 3
            // Calculate: trend (3-month change), momentum, volatility
        }
    }
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
            performance_[strategy_name].signals_generated += static_cast<int>(signals.size());

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
        std::sort(signals.begin(), signals.end(), [](auto const& a, auto const& b) -> bool {
            if (a.symbol == b.symbol) {
                return a.confidence > b.confidence;
            }
            return a.symbol < b.symbol;
        });

        signals.erase(
            std::unique(signals.begin(), signals.end(),
                        [](auto const& a, auto const& b) -> bool { return a.symbol == b.symbol; }),
            signals.end());
    }

    auto prioritizeSignals(std::vector<bigbrother::strategy::TradingSignal>& signals) -> void {
        // Sort by confidence (highest first)
        std::sort(signals.begin(), signals.end(),
                  [](auto const& a, auto const& b) -> bool { return a.confidence > b.confidence; });
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

[[nodiscard]] inline auto createSectorRotationStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<SectorRotationStrategy>();
}

} // namespace bigbrother::strategies
