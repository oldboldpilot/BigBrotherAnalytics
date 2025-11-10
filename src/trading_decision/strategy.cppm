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

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.strategy;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.options.pricing;
import bigbrother.risk_management;
import bigbrother.schwab_api; // Changed from bigbrother.schwab
import bigbrother.employment.signals;

export namespace bigbrother::strategy {

using namespace bigbrother::types;
using namespace bigbrother::options;
using namespace bigbrother::schwab;
using namespace bigbrother::employment;

// ============================================================================
// Forward Declarations
// ============================================================================

struct StrategyContext;
class ContextBuilder;
class SignalBuilder;
class PerformanceQueryBuilder;
class ReportBuilder;
class StrategyManager;

// ============================================================================
// Core Strategy Types
// ============================================================================

/**
 * Signal Type Enum
 */
enum class SignalType { Buy, Sell, Hold, ClosePosition };

/**
 * Performance Metrics
 *
 * Aggregated performance statistics for a strategy.
 */
struct PerformanceMetrics {
    std::string strategy_name;
    int signals_generated{0};
    int trades_executed{0};
    int winning_trades{0};
    int losing_trades{0};
    double total_pnl{0.0};
    double win_rate{0.0}; // 0.0 to 1.0
    double sharpe_ratio{0.0};
    double max_drawdown{0.0};
    double profit_factor{0.0};
    Timestamp period_start{0};
    Timestamp period_end{0};
};

/**
 * Trading Signal
 * C.1: Struct for passive data
 * C.21: Explicitly defaulted special members for proper move semantics
 */
struct TradingSignal {
    std::string strategy_name;
    std::string symbol;
    SignalType type;
    double confidence{0.0};      // 0.0 to 1.0
    double expected_return{0.0}; // Expected return in dollars
    double max_risk{0.0};        // Maximum risk in dollars
    double win_probability{0.0}; // Probability of profit
    Timestamp timestamp{0};
    std::string rationale;

    // Options-specific
    std::optional<OptionContract> option_contract;
    std::optional<Greeks> greeks;

    // C.21: Rule of Five - explicitly defaulted for move semantics
    TradingSignal() = default;
    TradingSignal(TradingSignal const&) = default;
    auto operator=(TradingSignal const&) -> TradingSignal& = default;
    TradingSignal(TradingSignal&&) noexcept = default;
    auto operator=(TradingSignal&&) noexcept -> TradingSignal& = default;
    ~TradingSignal() = default;

    [[nodiscard]] auto isActionable() const noexcept -> bool {
        return type != SignalType::Hold && confidence > 0.6 && expected_return > 50.0 &&
               win_probability > 0.60;
    }

    [[nodiscard]] auto riskRewardRatio() const noexcept -> double {
        if (max_risk == 0.0)
            return 0.0;
        return expected_return / max_risk;
    }
};

/**
 * Strategy Context
 *
 * Contains all market data and signals needed for strategy decision-making.
 * Provides employment signals alongside price data for fundamental analysis.
 *
 * Employment Signals Usage:
 * - employment_signals: Individual sector employment signals with strength indicators
 * - rotation_signals: Sector rotation recommendations (Overweight/Underweight)
 * - jobless_claims_alert: Optional recession warning from jobless claims spike
 *
 * Example:
 *   // Check employment signals for a specific sector
 *   for (auto const& signal : context.employment_signals) {
 *       if (signal.sector_name == "Information Technology" && signal.isActionable()) {
 *           // Use signal.signal_strength (-1.0 to +1.0) in strategy logic
 *       }
 *   }
 *
 *   // Use rotation signals for sector ETF trading
 *   for (auto const& rotation : context.rotation_signals) {
 *       if (rotation.isStrongSignal() &&
 *           rotation.action == SectorRotationSignal::Action::Overweight) {
 *           // Generate buy signal for rotation.sector_etf
 *       }
 *   }
 *
 *   // Fluent API example
 *   auto context = StrategyContext::builder()
 *       .withAccountValue(100000.0)
 *       .withAvailableCapital(20000.0)
 *       .withQuotes(quotes)
 *       .withEmploymentSignals(signals)
 *       .build();
 */
struct StrategyContext {
    // Price Data & Positions
    std::unordered_map<std::string, Quote> current_quotes;
    std::unordered_map<std::string, OptionsChainData> options_chains;
    std::vector<AccountPosition> current_positions;
    double account_value{0.0};
    double available_capital{0.0};
    Timestamp current_time{0};

    // Employment Signals (from BLS data)
    // Individual sector employment signals with detailed metrics
    std::vector<EmploymentSignal> employment_signals;

    // Sector rotation recommendations based on employment trends
    std::vector<SectorRotationSignal> rotation_signals;

    // Optional recession warning from jobless claims spike
    std::optional<EmploymentSignal> jobless_claims_alert;

    /**
     * Create fluent builder for constructing context
     */
    [[nodiscard]] static auto builder() -> ContextBuilder;

    /**
     * Get employment signals for a specific sector
     *
     * @param sector_name Sector name (e.g., "Information Technology")
     * @return Vector of employment signals for that sector
     */
    [[nodiscard]] auto getEmploymentSignalsForSector(std::string const& sector_name) const
        -> std::vector<EmploymentSignal> {
        std::vector<EmploymentSignal> results;
        for (auto const& signal : employment_signals) {
            if (signal.sector_name == sector_name) {
                results.push_back(signal);
            }
        }
        return results;
    }

    /**
     * Get rotation signal for a specific sector
     *
     * @param sector_name Sector name (e.g., "Financials")
     * @return Optional rotation signal if available
     */
    [[nodiscard]] auto getRotationSignalForSector(std::string const& sector_name) const
        -> std::optional<SectorRotationSignal> {
        for (auto const& rotation : rotation_signals) {
            if (rotation.sector_name == sector_name) {
                return rotation;
            }
        }
        return std::nullopt;
    }

    /**
     * Check if there's a recession warning active
     *
     * @return True if jobless claims spike detected
     */
    [[nodiscard]] auto hasRecessionWarning() const noexcept -> bool {
        return jobless_claims_alert.has_value();
    }

    /**
     * Get aggregate employment health score
     *
     * Calculates overall employment trend across all sectors.
     * Positive = improving employment, Negative = deteriorating
     *
     * @return Score from -1.0 (very negative) to +1.0 (very positive)
     */
    [[nodiscard]] auto getAggregateEmploymentScore() const noexcept -> double {
        if (employment_signals.empty()) {
            return 0.0;
        }

        double total = 0.0;
        for (auto const& signal : employment_signals) {
            total += signal.signal_strength;
        }

        return total / static_cast<double>(employment_signals.size());
    }

    /**
     * Get strongest employment signals (actionable only)
     *
     * @param limit Maximum number of signals to return
     * @return Vector of top N actionable employment signals
     */
    [[nodiscard]] auto getStrongestEmploymentSignals(int limit = 5) const
        -> std::vector<EmploymentSignal> {
        std::vector<EmploymentSignal> actionable;

        // Filter actionable signals
        for (auto const& signal : employment_signals) {
            if (signal.isActionable()) {
                actionable.push_back(signal);
            }
        }

        // Sort by absolute signal strength (strongest first)
        std::ranges::sort(actionable, [](auto const& a, auto const& b) -> bool {
            return std::abs(a.signal_strength) > std::abs(b.signal_strength);
        });

        // Return top N
        if (actionable.size() > static_cast<size_t>(limit)) {
            actionable.resize(limit);
        }

        return actionable;
    }
};

/**
 * Strategy Context Builder
 *
 * Fluent API for building strategy contexts with all required data.
 * Provides type-safe configuration and validation.
 *
 * Example Usage:
 *   auto context = StrategyContext::builder()
 *       .withAccountValue(50000.0)
 *       .withAvailableCapital(10000.0)
 *       .withCurrentTime(getTimestamp())
 *       .withQuotes(quotes_map)
 *       .withOptions(options_chains_map)
 *       .withEmploymentSignals(emp_signals)
 *       .build();
 */
class ContextBuilder {
  public:
    [[nodiscard]] auto withAccountValue(double value) noexcept -> ContextBuilder& {
        context_.account_value = value;
        return *this;
    }

    [[nodiscard]] auto withAvailableCapital(double capital) noexcept -> ContextBuilder& {
        context_.available_capital = capital;
        return *this;
    }

    [[nodiscard]] auto withCurrentTime(Timestamp time) noexcept -> ContextBuilder& {
        context_.current_time = time;
        return *this;
    }

    [[nodiscard]] auto withQuotes(std::unordered_map<std::string, Quote> quotes)
        -> ContextBuilder& {
        context_.current_quotes = std::move(quotes);
        return *this;
    }

    [[nodiscard]] auto withOptions(std::unordered_map<std::string, OptionsChainData> chains)
        -> ContextBuilder& {
        context_.options_chains = std::move(chains);
        return *this;
    }

    [[nodiscard]] auto withPositions(std::vector<AccountPosition> positions) -> ContextBuilder& {
        context_.current_positions = std::move(positions);
        return *this;
    }

    [[nodiscard]] auto withEmploymentSignals(std::vector<EmploymentSignal> signals)
        -> ContextBuilder& {
        context_.employment_signals = std::move(signals);
        return *this;
    }

    [[nodiscard]] auto withRotationSignals(std::vector<SectorRotationSignal> signals)
        -> ContextBuilder& {
        context_.rotation_signals = std::move(signals);
        return *this;
    }

    [[nodiscard]] auto withJoblessClaims(std::optional<EmploymentSignal> alert) -> ContextBuilder& {
        context_.jobless_claims_alert = alert;
        return *this;
    }

    [[nodiscard]] auto addQuote(std::string symbol, Quote quote) -> ContextBuilder& {
        context_.current_quotes[std::move(symbol)] = std::move(quote);
        return *this;
    }

    [[nodiscard]] auto addPosition(AccountPosition position) -> ContextBuilder& {
        context_.current_positions.push_back(std::move(position));
        return *this;
    }

    [[nodiscard]] auto addEmploymentSignal(EmploymentSignal signal) -> ContextBuilder& {
        context_.employment_signals.push_back(std::move(signal));
        return *this;
    }

    [[nodiscard]] auto build() -> StrategyContext { return std::move(context_); }

  private:
    StrategyContext context_;
};

// Implementation of StrategyContext::builder() after ContextBuilder is defined
inline auto StrategyContext::builder() -> ContextBuilder {
    return ContextBuilder{};
}

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
    BaseStrategy(std::string name, std::string description)
        : name_{std::move(name)}, description_{std::move(description)}, active_{true} {}

    [[nodiscard]] auto getName() const noexcept -> std::string override { return name_; }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    auto setActive(bool active) -> void override { active_ = active; }

  protected:
    std::string name_;
    std::string description_;
    bool active_;
};

// ============================================================================
// Signal Builder - Fluent API for signal generation
// ============================================================================

/**
 * SignalBuilder - Fluent API for generating and filtering signals
 *
 * Example Usage:
 *   auto signals = mgr.signalBuilder()
 *       .forContext(context)
 *       .fromStrategies({"SectorRotation", "Straddle"})
 *       .withMinConfidence(0.70)
 *       .limitTo(5)
 *       .generate();
 */
class SignalBuilder {
  public:
    explicit SignalBuilder(StrategyManager& manager) : manager_{manager} {}

    [[nodiscard]] auto forContext(StrategyContext const& ctx) -> SignalBuilder& {
        context_ = &ctx;
        return *this;
    }

    [[nodiscard]] auto fromStrategies(std::vector<std::string> names) -> SignalBuilder& {
        strategy_filter_ = std::move(names);
        return *this;
    }

    [[nodiscard]] auto withMinConfidence(double conf) noexcept -> SignalBuilder& {
        min_confidence_ = conf;
        return *this;
    }

    [[nodiscard]] auto withMinRiskRewardRatio(double ratio) noexcept -> SignalBuilder& {
        min_risk_reward_ = ratio;
        return *this;
    }

    [[nodiscard]] auto limitTo(int count) noexcept -> SignalBuilder& {
        max_signals_ = count;
        return *this;
    }

    [[nodiscard]] auto onlyActionable(bool actionable = true) noexcept -> SignalBuilder& {
        only_actionable_ = actionable;
        return *this;
    }

    /**
     * Terminal operation - generates signals
     */
    [[nodiscard]] auto generate() -> std::vector<TradingSignal>;

  private:
    StrategyManager& manager_;
    StrategyContext const* context_{nullptr};
    std::optional<std::vector<std::string>> strategy_filter_;
    std::optional<double> min_confidence_;
    std::optional<double> min_risk_reward_;
    std::optional<int> max_signals_;
    bool only_actionable_{false};
};

// ============================================================================
// Performance Query Builder
// ============================================================================

/**
 * PerformanceQueryBuilder - Fluent API for performance analysis
 *
 * Example Usage:
 *   auto perf = mgr.performanceBuilder()
 *       .forStrategy("SectorRotation")
 *       .inPeriod(start_date, end_date)
 *       .calculate();
 */
class PerformanceQueryBuilder {
  public:
    explicit PerformanceQueryBuilder(StrategyManager const& manager) : manager_{manager} {}

    [[nodiscard]] auto forStrategy(std::string const& name) -> PerformanceQueryBuilder& {
        strategy_name_ = name;
        return *this;
    }

    [[nodiscard]] auto inPeriod(Timestamp start, Timestamp end) noexcept
        -> PerformanceQueryBuilder& {
        start_time_ = start;
        end_time_ = end;
        return *this;
    }

    [[nodiscard]] auto minTradeCount(int count) noexcept -> PerformanceQueryBuilder& {
        min_trades_ = count;
        return *this;
    }

    /**
     * Terminal operation - calculates performance
     */
    [[nodiscard]] auto calculate() const -> std::optional<PerformanceMetrics>;

  private:
    StrategyManager const& manager_;
    std::optional<std::string> strategy_name_;
    std::optional<Timestamp> start_time_;
    std::optional<Timestamp> end_time_;
    std::optional<int> min_trades_;
};

// ============================================================================
// Report Builder
// ============================================================================

/**
 * ReportBuilder - Fluent API for strategy reports
 *
 * Example Usage:
 *   auto report = mgr.reportBuilder()
 *       .allStrategies()
 *       .withMetrics({"sharpe", "win_rate", "max_drawdown"})
 *       .sortBy("sharpe_ratio")
 *       .generate();
 */
class ReportBuilder {
  public:
    explicit ReportBuilder(StrategyManager const& manager) : manager_{manager} {}

    [[nodiscard]] auto allStrategies() -> ReportBuilder& {
        include_all_ = true;
        return *this;
    }

    [[nodiscard]] auto forStrategy(std::string const& name) -> ReportBuilder& {
        strategy_filter_.push_back(name);
        return *this;
    }

    [[nodiscard]] auto withMetrics(std::vector<std::string> metrics) -> ReportBuilder& {
        metrics_ = std::move(metrics);
        return *this;
    }

    [[nodiscard]] auto sortBy(std::string const& field) -> ReportBuilder& {
        sort_field_ = field;
        return *this;
    }

    [[nodiscard]] auto descending(bool desc = true) noexcept -> ReportBuilder& {
        descending_ = desc;
        return *this;
    }

    /**
     * Terminal operation - generates report
     */
    [[nodiscard]] auto generate() const -> std::string;

  private:
    StrategyManager const& manager_;
    bool include_all_{false};
    std::vector<std::string> strategy_filter_;
    std::vector<std::string> metrics_;
    std::optional<std::string> sort_field_;
    bool descending_{false};
};

// ============================================================================
// Strategy Manager
// ============================================================================

/**
 * Strategy Manager - Fluent API for strategy orchestration
 *
 * Orchestrates multiple trading strategies with thread safety.
 * Provides fluent builders for signal generation, performance analysis, and reporting.
 *
 * Example Usage:
 *   StrategyManager mgr;
 *   mgr.addStrategy(std::make_unique<SectorRotationStrategy>())
 *      .addStrategy(std::make_unique<StraddleStrategy>())
 *      .addStrategy(std::make_unique<VolatilityArbStrategy>());
 *
 *   auto signals = mgr.signalBuilder()
 *       .forContext(context)
 *       .withMinConfidence(0.70)
 *       .limitTo(10)
 *       .generate();
 *
 *   auto report = mgr.reportBuilder()
 *       .allStrategies()
 *       .withMetrics({"sharpe", "win_rate"})
 *       .generate();
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
     * Add strategy - fluent API
     */
    [[nodiscard]] auto addStrategy(std::unique_ptr<IStrategy> strategy) -> StrategyManager& {
        strategies_.push_back(std::move(strategy));
        return *this;
    }

    /**
     * Remove strategy - fluent API
     */
    [[nodiscard]] auto removeStrategy(std::string const& name) -> StrategyManager& {
        std::lock_guard lock{mutex_};
        strategies_.erase(std::remove_if(strategies_.begin(), strategies_.end(),
                                         [&name](auto const& s) { return s->getName() == name; }),
                          strategies_.end());
        return *this;
    }

    /**
     * Set strategy active/inactive - fluent API
     */
    [[nodiscard]] auto setStrategyActive(std::string const& name, bool active) -> StrategyManager& {
        std::lock_guard lock{mutex_};
        for (auto const& strategy : strategies_) {
            if (strategy->getName() == name) {
                strategy->setActive(active);
                break;
            }
        }
        return *this;
    }

    /**
     * Enable all strategies - fluent API
     */
    [[nodiscard]] auto enableAll() -> StrategyManager& {
        std::lock_guard lock{mutex_};
        for (auto const& strategy : strategies_) {
            strategy->setActive(true);
        }
        return *this;
    }

    /**
     * Disable all strategies - fluent API
     */
    [[nodiscard]] auto disableAll() -> StrategyManager& {
        std::lock_guard lock{mutex_};
        for (auto const& strategy : strategies_) {
            strategy->setActive(false);
        }
        return *this;
    }

    /**
     * Generate signals from all active strategies
     */
    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal>;

    /**
     * Get signal builder for fluent signal generation
     */
    [[nodiscard]] auto signalBuilder() -> SignalBuilder { return SignalBuilder{*this}; }

    /**
     * Get performance query builder
     */
    [[nodiscard]] auto performanceBuilder() const -> PerformanceQueryBuilder {
        return PerformanceQueryBuilder{*this};
    }

    /**
     * Get report builder
     */
    [[nodiscard]] auto reportBuilder() const -> ReportBuilder { return ReportBuilder{*this}; }

    /**
     * Get all strategies
     */
    [[nodiscard]] auto getStrategies() const -> std::vector<IStrategy const*>;

    /**
     * Get strategy count
     */
    [[nodiscard]] auto getStrategyCount() const noexcept -> size_t {
        std::lock_guard lock{mutex_};
        return strategies_.size();
    }

    /**
     * Get strategy by name
     */
    [[nodiscard]] auto getStrategy(std::string const& name) const -> IStrategy const*;

  private:
    friend class SignalBuilder;
    friend class PerformanceQueryBuilder;
    friend class ReportBuilder;

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
    StrategyExecutor(StrategyManager& manager) : manager_{manager} {}

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

} // namespace bigbrother::strategy

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
    std::ranges::sort(all_signals, [](auto const& a, auto const& b) -> bool {
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

auto StrategyManager::getStrategy(std::string const& name) const -> IStrategy const* {
    std::lock_guard lock{mutex_};

    for (auto const& strategy : strategies_) {
        if (strategy->getName() == name) {
            return strategy.get();
        }
    }

    return nullptr;
}

// SignalBuilder implementation
auto SignalBuilder::generate() -> std::vector<TradingSignal> {
    if (!context_) {
        return {};
    }

    // Get all signals from manager
    auto all_signals = manager_.generateSignals(*context_);

    // Filter by strategy names if specified
    if (strategy_filter_) {
        std::vector<TradingSignal> filtered;
        for (auto const& signal : all_signals) {
            for (auto const& strategy_name : *strategy_filter_) {
                if (signal.strategy_name == strategy_name) {
                    filtered.push_back(signal);
                    break;
                }
            }
        }
        all_signals = std::move(filtered);
    }

    // Filter by minimum confidence
    if (min_confidence_) {
        auto it = std::remove_if(
            all_signals.begin(), all_signals.end(),
            [conf = *min_confidence_](auto const& s) -> bool { return s.confidence < conf; });
        all_signals.erase(it, all_signals.end());
    }

    // Filter by minimum risk/reward ratio
    if (min_risk_reward_) {
        auto it = std::remove_if(all_signals.begin(), all_signals.end(),
                                 [ratio = *min_risk_reward_](auto const& s) -> bool {
                                     return s.riskRewardRatio() < ratio;
                                 });
        all_signals.erase(it, all_signals.end());
    }

    // Filter by actionable signals
    if (only_actionable_) {
        auto it = std::remove_if(all_signals.begin(), all_signals.end(),
                                 [](auto const& s) -> bool { return !s.isActionable(); });
        all_signals.erase(it, all_signals.end());
    }

    // Sort by confidence (highest first)
    std::ranges::sort(all_signals, [](auto const& a, auto const& b) -> bool {
        return a.confidence > b.confidence;
    });

    // Limit to max signals if specified
    if (max_signals_ && all_signals.size() > static_cast<size_t>(*max_signals_)) {
        all_signals.resize(*max_signals_);
    }

    return all_signals;
}

// PerformanceQueryBuilder implementation
auto PerformanceQueryBuilder::calculate() const -> std::optional<PerformanceMetrics> {
    // Stub implementation - would integrate with actual trade history
    if (!strategy_name_) {
        return std::nullopt;
    }

    PerformanceMetrics metrics;
    metrics.strategy_name = *strategy_name_;
    metrics.period_start = start_time_.value_or(0);
    metrics.period_end = end_time_.value_or(0);
    metrics.signals_generated = 0;
    metrics.win_rate = 0.0;
    metrics.sharpe_ratio = 0.0;

    return metrics;
}

// ReportBuilder implementation
auto ReportBuilder::generate() const -> std::string {
    std::ostringstream oss;

    oss << "=== Strategy Performance Report ===\n\n";

    // Stub implementation - would gather actual performance data
    oss << "Strategies: ";
    if (include_all_) {
        oss << "All\n";
    } else {
        for (auto const& name : strategy_filter_) {
            oss << name << " ";
        }
        oss << "\n";
    }

    if (!metrics_.empty()) {
        oss << "Metrics: ";
        for (auto const& metric : metrics_) {
            oss << metric << " ";
        }
        oss << "\n";
    }

    if (sort_field_) {
        oss << "Sort by: " << *sort_field_;
        if (descending_) {
            oss << " (DESC)";
        }
        oss << "\n";
    }

    oss << "\n[Report data would be populated from strategy performance history]\n";

    return oss.str();
}

// StrategyExecutor implementation
auto StrategyExecutor::execute() -> Result<std::vector<std::string>> {
    // Validate prerequisites
    if (!context_) {
        return makeError<std::vector<std::string>>(ErrorCode::InvalidParameter,
                                                   "StrategyExecutor: Context not set");
    }
    if (!schwab_client_) {
        return makeError<std::vector<std::string>>(ErrorCode::InvalidParameter,
                                                   "StrategyExecutor: SchwabClient not set");
    }
    if (!risk_manager_) {
        return makeError<std::vector<std::string>>(ErrorCode::InvalidParameter,
                                                   "StrategyExecutor: RiskManager not set");
    }

    // 1. Generate signals from all strategies
    auto signals = manager_.generateSignals(*context_);

    utils::Logger::getInstance().info("Generated {} signals from strategies", signals.size());

    // 2. Filter signals by confidence threshold
    if (min_confidence_) {
        auto it = std::remove_if(signals.begin(), signals.end(),
                                 [min_conf = *min_confidence_](auto const& s) -> bool {
                                     return s.confidence < min_conf;
                                 });
        signals.erase(it, signals.end());
        utils::Logger::getInstance().info("After confidence filter: {} signals", signals.size());
    }

    // 3. Filter to only actionable signals
    auto it = std::remove_if(signals.begin(), signals.end(),
                             [](auto const& s) -> bool { return !s.isActionable(); });
    signals.erase(it, signals.end());

    utils::Logger::getInstance().info("Actionable signals: {}", signals.size());

    // 4. Limit number of signals if specified
    if (max_signals_ && signals.size() > static_cast<size_t>(*max_signals_)) {
        signals.resize(*max_signals_);
        utils::Logger::getInstance().info("Limited to {} signals", signals.size());
    }

    // 5. Execute each signal (with risk checks)
    std::vector<std::string> order_ids;
    order_ids.reserve(signals.size());

    for (auto const& signal : signals) {
        // Pre-trade risk check using RiskManager::assessTrade
        double entry_price = context_->current_quotes.contains(signal.symbol)
                                 ? context_->current_quotes[signal.symbol].last
                                 : 100.0;

        double stop_price = entry_price * 0.90; // 10% stop loss
        double target_price = entry_price * (1.0 + signal.expected_return / entry_price);
        double position_size = signal.max_risk / (entry_price * 0.10); // Size for 10% risk

        auto risk_assessment =
            risk_manager_->assessTrade(signal.symbol, position_size, entry_price, stop_price,
                                       target_price, signal.win_probability);

        if (!risk_assessment || !risk_assessment->approved) {
            std::string reason =
                risk_assessment ? risk_assessment->rejection_reason : "Assessment failed";
            utils::Logger::getInstance().warn("Signal rejected by risk manager: {} - {}",
                                              signal.symbol, reason);
            continue;
        }

        // Convert signal to order
        Order order;
        order.symbol = signal.symbol;
        order.type = OrderType::Limit;
        order.duration = OrderDuration::Day;

        // Determine quantity and price from signal
        if (signal.option_contract) {
            // Options order
            order.quantity =
                static_cast<Quantity>(signal.max_risk / 100.0); // 1 contract = $100 risk approx
            if (signal.option_contract->type == OptionType::Call ||
                signal.option_contract->type == OptionType::Put) {
                // Get current quote for option
                auto quote_result = schwab_client_->marketData().getQuote(signal.symbol);
                if (quote_result) {
                    order.limit_price = quote_result->ask; // Buy at ask
                } else {
                    utils::Logger::getInstance().error("Failed to get quote for {}: {}",
                                                       signal.symbol, quote_result.error().message);
                    continue;
                }
            }
        } else {
            // Stock order
            order.quantity = static_cast<Quantity>(signal.max_risk /
                                                   context_->current_quotes[signal.symbol].last);
            order.limit_price = context_->current_quotes[signal.symbol].ask;
        }

        // Validate order quantity
        if (order.quantity <= quantity_epsilon) {
            utils::Logger::getInstance().warn("Invalid order quantity for {}: {}", signal.symbol,
                                              order.quantity);
            continue;
        }

        // Place order via Schwab API
        utils::Logger::getInstance().info(
            "Placing order: {} {} @ ${:.2f} (Strategy: {}, Confidence: {:.1f}%)", order.quantity,
            order.symbol, order.limit_price, signal.strategy_name, signal.confidence * 100.0);

        auto order_result = schwab_client_->orders().placeOrder(order);

        if (order_result) {
            std::string order_id = *order_result;
            order_ids.push_back(order_id);

            utils::Logger::getInstance().info("✓ Order placed successfully: {} (ID: {})",
                                              signal.symbol, order_id);

            // Log trade decision for explainability
            utils::Logger::getInstance().info("  Rationale: {}", signal.rationale);
            utils::Logger::getInstance().info(
                "  Expected Return: ${:.2f}, Max Risk: ${:.2f}, Win Prob: {:.1f}%",
                signal.expected_return, signal.max_risk, signal.win_probability * 100.0);

        } else {
            utils::Logger::getInstance().error("✗ Order failed for {}: {}", signal.symbol,
                                               order_result.error().message);
        }
    }

    utils::Logger::getInstance().info("Execution complete: {}/{} orders placed successfully",
                                      order_ids.size(), signals.size());

    return order_ids;
}

} // namespace bigbrother::strategy
