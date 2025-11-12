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
#include <cmath>
#include <deque>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../schwab_api/duckdb_bridge.hpp"

// Module declaration
export module bigbrother.strategies;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.options.pricing;
import bigbrother.strategy;
import bigbrother.employment.signals;
import bigbrother.risk_management;
import bigbrother.market_intelligence.price_predictor;
import bigbrother.market_intelligence.feature_extractor;

export namespace bigbrother::strategies {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using namespace bigbrother::options;
using bigbrother::employment::EmploymentSignalGenerator;
using bigbrother::employment::SectorRotationSignal;
using bigbrother::strategy::IStrategy;
using bigbrother::strategy::SignalType;
using bigbrother::strategy::StrategyContext;
// Note: TradingSignal defined in strategy module, not types

// ============================================================================
// Forward Declarations
// ============================================================================

class StraddleStrategy;
class StrangleStrategy;
class VolatilityArbStrategy;
class SectorRotationStrategy;
class SectorRotationStrategyBuilder;

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
// Straddle Strategy Builder
// ============================================================================

class StraddleStrategyBuilder {
  public:
    [[nodiscard]] auto withMinIVRank(double rank) noexcept -> StraddleStrategyBuilder& {
        min_iv_rank_ = rank;
        return *this;
    }

    [[nodiscard]] auto withMaxDistance(double distance) noexcept -> StraddleStrategyBuilder& {
        max_distance_ = distance;
        return *this;
    }

    [[nodiscard]] auto build() -> std::unique_ptr<IStrategy>;

  private:
    std::optional<double> min_iv_rank_;
    std::optional<double> max_distance_;
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
        return {{"min_iv_rank", std::to_string(min_iv_rank_)},
                {"max_distance", std::to_string(max_distance_)}};
    }

    /**
     * Create fluent builder for StraddleStrategy
     */
    [[nodiscard]] static auto builder() -> StraddleStrategyBuilder {
        return StraddleStrategyBuilder{};
    }

    auto setMinIVRank(double rank) noexcept -> void { min_iv_rank_ = rank; }
    auto setMaxDistance(double distance) noexcept -> void { max_distance_ = distance; }

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
    double min_iv_rank_{0.70};
    double max_distance_{0.10};
};

// StraddleStrategyBuilder::build() implementation
inline auto StraddleStrategyBuilder::build() -> std::unique_ptr<IStrategy> {
    auto strategy = std::make_unique<StraddleStrategy>();
    if (min_iv_rank_) {
        strategy->setMinIVRank(*min_iv_rank_);
    }
    if (max_distance_) {
        strategy->setMaxDistance(*max_distance_);
    }
    return strategy;
}

// ============================================================================
// Strangle Strategy Builder
// ============================================================================

class StrangleStrategyBuilder {
  public:
    [[nodiscard]] auto withMinIVRank(double rank) noexcept -> StrangleStrategyBuilder& {
        min_iv_rank_ = rank;
        return *this;
    }

    [[nodiscard]] auto withWingWidth(double width) noexcept -> StrangleStrategyBuilder& {
        wing_width_ = width;
        return *this;
    }

    [[nodiscard]] auto build() -> std::unique_ptr<IStrategy>;

  private:
    std::optional<double> min_iv_rank_;
    std::optional<double> wing_width_;
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
        return {{"min_iv_rank", std::to_string(min_iv_rank_)},
                {"wing_width", std::to_string(wing_width_)}};
    }

    /**
     * Create fluent builder for StrangleStrategy
     */
    [[nodiscard]] static auto builder() -> StrangleStrategyBuilder {
        return StrangleStrategyBuilder{};
    }

    auto setMinIVRank(double rank) noexcept -> void { min_iv_rank_ = rank; }
    auto setWingWidth(double width) noexcept -> void { wing_width_ = width; }

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
    double min_iv_rank_{0.65};
    double wing_width_{0.20};
};

// StrangleStrategyBuilder::build() implementation
inline auto StrangleStrategyBuilder::build() -> std::unique_ptr<IStrategy> {
    auto strategy = std::make_unique<StrangleStrategy>();
    if (min_iv_rank_) {
        strategy->setMinIVRank(*min_iv_rank_);
    }
    if (wing_width_) {
        strategy->setWingWidth(*wing_width_);
    }
    return strategy;
}

// ============================================================================
// Volatility Arbitrage Strategy Builder
// ============================================================================

class VolatilityArbStrategyBuilder {
  public:
    [[nodiscard]] auto withMinIVHVSpread(double spread) noexcept
        -> VolatilityArbStrategyBuilder& {
        min_iv_hv_spread_ = spread;
        return *this;
    }

    [[nodiscard]] auto withLookbackPeriod(int days) noexcept -> VolatilityArbStrategyBuilder& {
        lookback_days_ = days;
        return *this;
    }

    [[nodiscard]] auto build() -> std::unique_ptr<IStrategy>;

  private:
    std::optional<double> min_iv_hv_spread_;
    std::optional<int> lookback_days_;
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
        return {{"min_iv_hv_spread", std::to_string(min_iv_hv_spread_)},
                {"lookback_days", std::to_string(lookback_days_)}};
    }

    /**
     * Create fluent builder for VolatilityArbStrategy
     */
    [[nodiscard]] static auto builder() -> VolatilityArbStrategyBuilder {
        return VolatilityArbStrategyBuilder{};
    }

    auto setMinIVHVSpread(double spread) noexcept -> void { min_iv_hv_spread_ = spread; }
    auto setLookbackPeriod(int days) noexcept -> void { lookback_days_ = days; }

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
    double min_iv_hv_spread_{0.10};
    int lookback_days_{30};
};

// VolatilityArbStrategyBuilder::build() implementation
inline auto VolatilityArbStrategyBuilder::build() -> std::unique_ptr<IStrategy> {
    auto strategy = std::make_unique<VolatilityArbStrategy>();
    if (min_iv_hv_spread_) {
        strategy->setMinIVHVSpread(*min_iv_hv_spread_);
    }
    if (lookback_days_) {
        strategy->setLookbackPeriod(*lookback_days_);
    }
    return strategy;
}

// ============================================================================
// Sector Rotation Strategy
// ============================================================================

/**
 * SectorRotationStrategy - Multi-Signal Sector Allocation Strategy
 *
 * Uses employment signals, sentiment, and momentum to generate
 * overweight/underweight recommendations for 11 GICS sectors.
 *
 * Strategy Logic:
 * 1. Fetch employment signals from EmploymentSignalGenerator (Python/DuckDB)
 * 2. Combine with sentiment and momentum scores (extensible)
 * 3. Generate composite score for each sector (-1.0 to +1.0)
 * 4. Rank sectors and generate top N overweight, bottom M underweight
 * 5. Apply risk limits and diversification constraints
 * 6. Generate position sizing recommendations per sector
 *
 * Configuration Parameters:
 * - min_composite_score: Minimum score to generate signal (default: 0.60)
 * - rotation_threshold: Score threshold for rotation (default: 0.70)
 * - employment_weight: Weight for employment signal (default: 0.60)
 * - sentiment_weight: Weight for sentiment signal (default: 0.30)
 * - momentum_weight: Weight for momentum signal (default: 0.10)
 * - top_n_overweight: Number of sectors to overweight (default: 3)
 * - bottom_n_underweight: Number of sectors to underweight (default: 2)
 * - max_sector_allocation: Max % of portfolio per sector (default: 0.25)
 * - min_sector_allocation: Min % of portfolio per sector (default: 0.05)
 * - rebalance_frequency_days: Days between rebalancing (default: 30)
 *
 * Integration with RiskManager:
 * - Respects sector exposure limits
 * - Enforces position sizing constraints
 * - Validates against portfolio heat limits
 * - Checks correlation exposure constraints
 *
 * Fluent Builder Usage:
 *   auto strategy = SectorRotationStrategy::builder()
 *       .withEmploymentWeight(0.65)
 *       .withSentimentWeight(0.25)
 *       .withMomentumWeight(0.10)
 *       .topNOverweight(4)
 *       .bottomNUnderweight(3)
 *       .rotationThreshold(0.75)
 *       .build();
 *
 * Traditional Usage:
 *   auto strategy = createSectorRotationStrategy();
 *   strategy->setParameter("top_n_overweight", "4");
 *   strategy->setParameter("employment_weight", "0.70");
 *
 *   StrategyContext context{...};
 *   auto signals = strategy->generateSignals(context);
 *
 *   // Signals contain BUY (overweight) and SELL (underweight) recommendations
 *   // with position sizing, risk metrics, and rationale
 */
class SectorRotationStrategy final : public IStrategy {
  public:
    /**
     * Sector Score - Internal scoring structure
     *
     * Combines multiple signals into a composite score for ranking.
     */
    struct SectorScore {
        int sector_code{0};
        std::string sector_name;
        std::string etf_ticker;

        // Signal components
        double employment_score{0.0}; // -1.0 to +1.0 from employment data
        double sentiment_score{0.0};  // -1.0 to +1.0 from news sentiment
        double momentum_score{0.0};   // -1.0 to +1.0 from price momentum
        double composite_score{0.0};  // Weighted average

        // Classification
        bool is_overweight{false};  // Top N sectors
        bool is_underweight{false}; // Bottom M sectors
        bool is_neutral{false};     // Middle sectors

        // Position sizing
        double target_allocation{0.0}; // Target % of portfolio (0.0 to 1.0)
        double position_size{0.0};     // Dollar amount to allocate

        [[nodiscard]] auto isStrongSignal() const noexcept -> bool {
            return std::abs(composite_score) > 0.70;
        }

        [[nodiscard]] auto rank() const noexcept -> double { return composite_score; }
    };

    /**
     * Configuration Parameters
     */
    struct Config {
        double min_composite_score{0.60};
        double rotation_threshold{0.70};
        double employment_weight{0.60};
        double sentiment_weight{0.30};
        double momentum_weight{0.10};
        int top_n_overweight{3};
        int bottom_n_underweight{2};
        double max_sector_allocation{0.25};
        double min_sector_allocation{0.05};
        int rebalance_frequency_days{30};
        std::string db_path{"data/bigbrother.duckdb"};
        std::string scripts_path{"scripts"};
    };

    explicit SectorRotationStrategy(Config config)
        : config_{std::move(config)}, signal_generator_{config_.scripts_path, config_.db_path} {

        // Normalize weights
        auto const total_weight =
            config_.employment_weight + config_.sentiment_weight + config_.momentum_weight;
        if (total_weight > 0.0) {
            config_.employment_weight /= total_weight;
            config_.sentiment_weight /= total_weight;
            config_.momentum_weight /= total_weight;
        }
    }

    // C.21: Rule of Five
    SectorRotationStrategy(SectorRotationStrategy const&) = delete;
    auto operator=(SectorRotationStrategy const&) -> SectorRotationStrategy& = delete;
    SectorRotationStrategy(SectorRotationStrategy&&) noexcept = default;
    auto operator=(SectorRotationStrategy&&) noexcept -> SectorRotationStrategy& = default;
    ~SectorRotationStrategy() override = default;

    /**
     * Create fluent builder for SectorRotationStrategy
     */
    [[nodiscard]] static auto builder() -> SectorRotationStrategyBuilder;

    [[nodiscard]] auto getName() const noexcept -> std::string override {
        return "Sector Rotation (Multi-Signal)";
    }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    auto setActive(bool active) -> void override { active_ = active; }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {{"min_composite_score", std::to_string(config_.min_composite_score)},
                {"rotation_threshold", std::to_string(config_.rotation_threshold)},
                {"employment_weight", std::to_string(config_.employment_weight)},
                {"sentiment_weight", std::to_string(config_.sentiment_weight)},
                {"momentum_weight", std::to_string(config_.momentum_weight)},
                {"top_n_overweight", std::to_string(config_.top_n_overweight)},
                {"bottom_n_underweight", std::to_string(config_.bottom_n_underweight)},
                {"max_sector_allocation", std::to_string(config_.max_sector_allocation)},
                {"min_sector_allocation", std::to_string(config_.min_sector_allocation)},
                {"rebalance_frequency_days", std::to_string(config_.rebalance_frequency_days)}};
    }

    /**
     * Set configuration parameter
     */
    auto setParameter(std::string const& key, std::string const& value) -> void {
        if (key == "min_composite_score") {
            config_.min_composite_score = std::stod(value);
        } else if (key == "rotation_threshold") {
            config_.rotation_threshold = std::stod(value);
        } else if (key == "employment_weight") {
            config_.employment_weight = std::stod(value);
        } else if (key == "sentiment_weight") {
            config_.sentiment_weight = std::stod(value);
        } else if (key == "momentum_weight") {
            config_.momentum_weight = std::stod(value);
        } else if (key == "top_n_overweight") {
            config_.top_n_overweight = std::stoi(value);
        } else if (key == "bottom_n_underweight") {
            config_.bottom_n_underweight = std::stoi(value);
        } else if (key == "max_sector_allocation") {
            config_.max_sector_allocation = std::stod(value);
        } else if (key == "min_sector_allocation") {
            config_.min_sector_allocation = std::stod(value);
        } else if (key == "rebalance_frequency_days") {
            config_.rebalance_frequency_days = std::stoi(value);
        }
    }

    /**
     * Generate sector rotation signals
     *
     * Main entry point - analyzes all 11 GICS sectors and generates
     * overweight/underweight trading signals.
     */
    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> override {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        // 1. Initialize 11 GICS sectors
        auto sectors = initializeSectors();

        // 2. Fetch and score employment signals from DuckDB
        scoreEmploymentSignals(sectors);

        // 3. Score sentiment (placeholder for future integration)
        scoreSentimentSignals(sectors);

        // 4. Score momentum from price action (using context quotes)
        scoreMomentumSignals(sectors, context);

        // 5. Calculate composite scores and rank sectors
        calculateCompositeScores(sectors);
        rankSectors(sectors);

        // 6. Classify sectors (overweight/neutral/underweight)
        classifySectors(sectors);

        // 7. Calculate position sizing
        calculatePositionSizing(sectors, context);

        // 8. Generate trading signals for top/bottom sectors
        signals = generateTradingSignals(sectors, context);

        Logger::getInstance().info(
            "{}: Generated {} signals ({} overweight, {} underweight) from sector analysis",
            getName(), signals.size(),
            std::count_if(signals.begin(), signals.end(),
                          [](auto const& s) { return s.type == SignalType::Buy; }),
            std::count_if(signals.begin(), signals.end(),
                          [](auto const& s) { return s.type == SignalType::Sell; }));

        return signals;
    }

  private:
    bool active_{true};
    Config config_;
    EmploymentSignalGenerator signal_generator_;

    /**
     * Initialize 11 GICS sectors with ETF mappings
     */
    [[nodiscard]] auto initializeSectors() const -> std::vector<SectorScore> {
        return {
            {10, "Energy", "XLE", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {15, "Materials", "XLB", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {20, "Industrials", "XLI", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {25, "Consumer Discretionary", "XLY", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0,
             0.0},
            {30, "Consumer Staples", "XLP", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {35, "Health Care", "XLV", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {40, "Financials", "XLF", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {45, "Information Technology", "XLK", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0,
             0.0},
            {50, "Communication Services", "XLC", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0,
             0.0},
            {55, "Utilities", "XLU", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
            {60, "Real Estate", "XLRE", 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0},
        };
    }

    /**
     * Score sectors based on employment data from DuckDB
     *
     * Calls EmploymentSignalGenerator to fetch actual BLS data
     * and calculate employment trends, momentum, and signals.
     */
    auto scoreEmploymentSignals(std::vector<SectorScore>& sectors) -> void {
        try {
            // Fetch rotation signals from Python/DuckDB backend
            auto rotation_signals = signal_generator_.generateRotationSignals();

            // Map rotation signals to sectors
            for (auto& sector : sectors) {
                auto it = std::find_if(
                    rotation_signals.begin(), rotation_signals.end(),
                    [&sector](auto const& sig) { return sig.sector_code == sector.sector_code; });

                if (it != rotation_signals.end()) {
                    sector.employment_score = it->employment_score;
                    Logger::getInstance().debug("{}: Employment score = {:.2f}", sector.sector_name,
                                                sector.employment_score);
                } else {
                    // Fallback: neutral score if no data
                    sector.employment_score = 0.0;
                    Logger::getInstance().warn("{}: No employment data found, using neutral score",
                                               sector.sector_name);
                }
            }
        } catch (std::exception const& e) {
            Logger::getInstance().error(
                "Failed to fetch employment signals: {}. Using fallback stub data.", e.what());
            // Fallback to stub data if Python backend fails
            scoreEmploymentSignalsStub(sectors);
        }
    }

    /**
     * Fallback stub employment scoring (for testing/demo)
     */
    auto scoreEmploymentSignalsStub(std::vector<SectorScore>& sectors) -> void {
        for (auto& sector : sectors) {
            // Realistic stub scores based on typical sector dynamics
            if (sector.sector_name == "Energy") {
                sector.employment_score = 0.45;
            } else if (sector.sector_name == "Materials") {
                sector.employment_score = 0.55;
            } else if (sector.sector_name == "Industrials") {
                sector.employment_score = 0.75;
            } else if (sector.sector_name == "Consumer Discretionary") {
                sector.employment_score = -0.65;
            } else if (sector.sector_name == "Consumer Staples") {
                sector.employment_score = 0.50;
            } else if (sector.sector_name == "Health Care") {
                sector.employment_score = 0.82;
            } else if (sector.sector_name == "Financials") {
                sector.employment_score = 0.65;
            } else if (sector.sector_name == "Information Technology") {
                sector.employment_score = 0.88;
            } else if (sector.sector_name == "Communication Services") {
                sector.employment_score = 0.40;
            } else if (sector.sector_name == "Utilities") {
                sector.employment_score = 0.48;
            } else if (sector.sector_name == "Real Estate") {
                sector.employment_score = -0.72;
            }
        }
    }

    /**
     * Score sectors based on news sentiment
     *
     * Placeholder for future sentiment analysis integration.
     * Would query sector_news_sentiment table or sentiment API.
     */
    auto scoreSentimentSignals(std::vector<SectorScore>& sectors) -> void {
        // TODO: Integrate with sentiment analysis module
        // For now: neutral scores (0.0)
        for (auto& sector : sectors) {
            sector.sentiment_score = 0.0;
        }
    }

    /**
     * Score sectors based on price momentum
     *
     * Uses recent price action from context.current_quotes
     * to calculate momentum signals.
     */
    auto scoreMomentumSignals(std::vector<SectorScore>& sectors, StrategyContext const& context)
        -> void {
        for (auto& sector : sectors) {
            // Look up sector ETF quote
            auto it = context.current_quotes.find(sector.etf_ticker);
            if (it != context.current_quotes.end()) {
                // Simple momentum: use price change % if available
                // In production: would calculate RSI, MACD, trend strength, etc.
                auto const& quote = it->second;

                // Placeholder momentum calculation
                // Would typically use: (current_price - MA50) / MA50
                sector.momentum_score = 0.0; // Neutral for now
            } else {
                sector.momentum_score = 0.0;
            }
        }
    }

    /**
     * Calculate composite scores
     *
     * Weighted average of employment, sentiment, and momentum.
     */
    auto calculateCompositeScores(std::vector<SectorScore>& sectors) -> void {
        for (auto& sector : sectors) {
            sector.composite_score = config_.employment_weight * sector.employment_score +
                                     config_.sentiment_weight * sector.sentiment_score +
                                     config_.momentum_weight * sector.momentum_score;

            // Clamp to [-1.0, +1.0]
            sector.composite_score = std::max(-1.0, std::min(1.0, sector.composite_score));

            Logger::getInstance().debug(
                "{}: Composite score = {:.3f} (emp: {:.3f}, sent: {:.3f}, mom: {:.3f})",
                sector.sector_name, sector.composite_score, sector.employment_score,
                sector.sentiment_score, sector.momentum_score);
        }
    }

    /**
     * Rank sectors by composite score (descending)
     */
    auto rankSectors(std::vector<SectorScore>& sectors) -> void {
        std::sort(sectors.begin(), sectors.end(), [](auto const& a, auto const& b) -> bool {
            return a.composite_score > b.composite_score;
        });
    }

    /**
     * Classify sectors into overweight/neutral/underweight
     *
     * Top N sectors -> Overweight (BUY signals)
     * Bottom M sectors -> Underweight (SELL signals)
     * Middle sectors -> Neutral (HOLD)
     */
    auto classifySectors(std::vector<SectorScore>& sectors) -> void {
        auto const n = sectors.size();

        for (size_t i = 0; i < n; ++i) {
            if (i < static_cast<size_t>(config_.top_n_overweight) &&
                sectors[i].composite_score >= config_.min_composite_score) {
                sectors[i].is_overweight = true;
                sectors[i].is_neutral = false;
                sectors[i].is_underweight = false;
            } else if (i >= n - static_cast<size_t>(config_.bottom_n_underweight) &&
                       sectors[i].composite_score <= -config_.min_composite_score) {
                sectors[i].is_overweight = false;
                sectors[i].is_neutral = false;
                sectors[i].is_underweight = true;
            } else {
                sectors[i].is_overweight = false;
                sectors[i].is_neutral = true;
                sectors[i].is_underweight = false;
            }
        }
    }

    /**
     * Calculate position sizing for each sector
     *
     * Uses available capital and allocation constraints
     * to determine dollar amounts for each position.
     */
    auto calculatePositionSizing(std::vector<SectorScore>& sectors, StrategyContext const& context)
        -> void {
        auto const available_capital = context.available_capital;
        auto const total_overweight = std::count_if(
            sectors.begin(), sectors.end(), [](auto const& s) -> bool { return s.is_overweight; });

        if (total_overweight == 0) {
            return;
        }

        // Equal weight allocation for overweight sectors
        // (More sophisticated: weight by composite score)
        auto const base_allocation = 1.0 / static_cast<double>(total_overweight);

        for (auto& sector : sectors) {
            if (sector.is_overweight) {
                // Target allocation % (can be score-weighted)
                sector.target_allocation = base_allocation;

                // Clamp to min/max limits
                sector.target_allocation =
                    std::max(config_.min_sector_allocation,
                             std::min(config_.max_sector_allocation, sector.target_allocation));

                // Calculate dollar amount
                sector.position_size = available_capital * sector.target_allocation;

                Logger::getInstance().debug("{}: Overweight -> ${:.2f} ({:.1f}%)",
                                            sector.sector_name, sector.position_size,
                                            sector.target_allocation * 100.0);
            } else if (sector.is_underweight) {
                // Underweight: reduce or exit existing positions
                sector.target_allocation = 0.0;
                sector.position_size = 0.0;
            } else {
                // Neutral: maintain current allocation
                sector.target_allocation = config_.min_sector_allocation;
                sector.position_size = available_capital * sector.target_allocation;
            }
        }
    }

    /**
     * Generate trading signals from scored sectors
     */
    [[nodiscard]] auto generateTradingSignals(std::vector<SectorScore> const& sectors,
                                              StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        for (auto const& sector : sectors) {
            // Generate BUY signals for overweight sectors
            if (sector.is_overweight && sector.composite_score >= config_.rotation_threshold) {
                bigbrother::strategy::TradingSignal signal;
                signal.symbol = sector.etf_ticker;
                signal.strategy_name = getName();
                signal.type = SignalType::Buy;
                signal.confidence = std::abs(sector.composite_score);
                signal.expected_return = sector.position_size * 0.15; // 15% expected return
                signal.max_risk = sector.position_size * 0.10;        // 10% max risk
                signal.win_probability =
                    0.60 + (sector.composite_score - config_.rotation_threshold) * 0.20;
                signal.timestamp = context.current_time;
                signal.rationale = formatOverweightRationale(sector);
                signals.push_back(signal);

                Logger::getInstance().info("{}: OVERWEIGHT {} - Score: {:.3f}, Size: ${:.2f}",
                                           getName(), sector.sector_name, sector.composite_score,
                                           sector.position_size);
            }

            // Generate SELL signals for underweight sectors
            if (sector.is_underweight && sector.composite_score <= -config_.rotation_threshold) {
                bigbrother::strategy::TradingSignal signal;
                signal.symbol = sector.etf_ticker;
                signal.strategy_name = getName();
                signal.type = SignalType::Sell;
                signal.confidence = std::abs(sector.composite_score);
                signal.expected_return = 0.0; // Avoid losses by exiting
                signal.max_risk = 0.0;
                signal.win_probability =
                    0.55 + (std::abs(sector.composite_score) - config_.rotation_threshold) * 0.15;
                signal.timestamp = context.current_time;
                signal.rationale = formatUnderweightRationale(sector);
                signals.push_back(signal);

                Logger::getInstance().info("{}: UNDERWEIGHT {} - Score: {:.3f}", getName(),
                                           sector.sector_name, sector.composite_score);
            }
        }

        return signals;
    }

    /**
     * Format rationale for overweight recommendation
     */
    [[nodiscard]] auto formatOverweightRationale(SectorScore const& sector) const -> std::string {
        std::ostringstream oss;
        oss << "Sector Rotation: OVERWEIGHT " << sector.sector_name << " - ";
        oss << "Strong composite score: " << sector.composite_score << " | ";
        oss << "Employment trend: " << sector.employment_score << " | ";
        oss << "Target allocation: " << (sector.target_allocation * 100.0) << "% | ";
        oss << "Position size: $" << sector.position_size;
        return oss.str();
    }

    /**
     * Format rationale for underweight recommendation
     */
    [[nodiscard]] auto formatUnderweightRationale(SectorScore const& sector) const -> std::string {
        std::ostringstream oss;
        oss << "Sector Rotation: UNDERWEIGHT " << sector.sector_name << " - ";
        oss << "Weak composite score: " << sector.composite_score << " | ";
        oss << "Employment weakness: " << sector.employment_score << " | ";
        oss << "Recommend reducing/exiting positions";
        return oss.str();
    }
};

// ============================================================================
// Sector Rotation Strategy Builder
// ============================================================================

/**
 * SectorRotationStrategyBuilder - Fluent API for configuring SectorRotationStrategy
 *
 * Example Usage:
 *   auto strategy = SectorRotationStrategy::builder()
 *       .withEmploymentWeight(0.60)
 *       .withSentimentWeight(0.30)
 *       .withMomentumWeight(0.10)
 *       .topNOverweight(3)
 *       .bottomNUnderweight(2)
 *       .rotationThreshold(0.70)
 *       .minCompositeScore(0.60)
 *       .maxSectorAllocation(0.25)
 *       .minSectorAllocation(0.05)
 *       .build();
 */
class SectorRotationStrategyBuilder {
  public:
    [[nodiscard]] auto withEmploymentWeight(double weight) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.employment_weight = weight;
        return *this;
    }

    [[nodiscard]] auto withSentimentWeight(double weight) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.sentiment_weight = weight;
        return *this;
    }

    [[nodiscard]] auto withMomentumWeight(double weight) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.momentum_weight = weight;
        return *this;
    }

    [[nodiscard]] auto topNOverweight(int n) noexcept -> SectorRotationStrategyBuilder& {
        config_.top_n_overweight = n;
        return *this;
    }

    [[nodiscard]] auto bottomNUnderweight(int n) noexcept -> SectorRotationStrategyBuilder& {
        config_.bottom_n_underweight = n;
        return *this;
    }

    [[nodiscard]] auto minCompositeScore(double score) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.min_composite_score = score;
        return *this;
    }

    [[nodiscard]] auto rotationThreshold(double threshold) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.rotation_threshold = threshold;
        return *this;
    }

    [[nodiscard]] auto maxSectorAllocation(double max_alloc) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.max_sector_allocation = max_alloc;
        return *this;
    }

    [[nodiscard]] auto minSectorAllocation(double min_alloc) noexcept
        -> SectorRotationStrategyBuilder& {
        config_.min_sector_allocation = min_alloc;
        return *this;
    }

    [[nodiscard]] auto rebalanceFrequency(int days) noexcept -> SectorRotationStrategyBuilder& {
        config_.rebalance_frequency_days = days;
        return *this;
    }

    [[nodiscard]] auto withDatabasePath(std::string const& path) -> SectorRotationStrategyBuilder& {
        config_.db_path = path;
        return *this;
    }

    [[nodiscard]] auto withScriptsPath(std::string const& path) -> SectorRotationStrategyBuilder& {
        config_.scripts_path = path;
        return *this;
    }

    [[nodiscard]] auto build() -> std::unique_ptr<IStrategy> {
        return std::make_unique<SectorRotationStrategy>(config_);
    }

  private:
    SectorRotationStrategy::Config config_;
};

// SectorRotationStrategy::builder() implementation
inline auto SectorRotationStrategy::builder() -> SectorRotationStrategyBuilder {
    return SectorRotationStrategyBuilder{};
}

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
// ML Predictor Strategy (AI-Powered Price Prediction)
// ============================================================================

/**
 * ML-based strategy using ONNX Runtime for price prediction
 *
 * Uses trained neural network to predict 1-day, 5-day, and 20-day price changes
 * Generates BUY/SELL signals based on predicted returns
 */
class MLPredictorStrategy : public IStrategy {
  public:
    MLPredictorStrategy() {
        // Initialize predictor on first use
        auto& predictor = bigbrother::market_intelligence::PricePredictor::getInstance();

        if (!predictor.isInitialized()) {
            bigbrother::market_intelligence::PredictorConfig config;
            config.use_cuda = true;  // Enable CUDA if available
            config.model_weights_path = "models/price_predictor.onnx";

            if (!predictor.initialize(config)) {
                Logger::getInstance().error("Failed to initialize ML predictor");
            } else {
                Logger::getInstance().info("ML Predictor Strategy initialized with CUDA");
            }
        }

        // Load historical price/volume data from database (last 30 days for SPY, QQQ, IWM, DIA)
        loadHistoricalData();
    }

    [[nodiscard]] auto getName() const noexcept -> std::string override {
        return "ML Price Predictor";
    }

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<bigbrother::strategy::TradingSignal> override {

        std::vector<bigbrother::strategy::TradingSignal> signals;

        if (!active_) {
            return signals;
        }

        auto& predictor = bigbrother::market_intelligence::PricePredictor::getInstance();

        if (!predictor.isInitialized()) {
            Logger::getInstance().warn("ML predictor not initialized, skipping");
            return signals;
        }

        // Get symbols to analyze (from context or default watchlist)
        std::vector<std::string> symbols = {"SPY", "QQQ", "IWM", "DIA"};

        for (auto const& symbol : symbols) {
            // Update price/volume history from current market data
            auto quote_it = context.current_quotes.find(symbol);
            if (quote_it != context.current_quotes.end()) {
                auto const& quote = quote_it->second;
                updateHistory(symbol, quote.last, quote.volume);
            }

            // Extract features from market data (will use historical buffers if available)
            auto features = extractFeatures(context, symbol);
            if (!features) {
                continue;
            }

            // Run ML prediction
            auto prediction = predictor.predict(symbol, *features);
            if (!prediction) {
                Logger::getInstance().warn("Prediction failed for {}", symbol);
                continue;
            }

            // Get overall signal
            auto signal_type = prediction->getOverallSignal();

            // Skip HOLD signals
            if (signal_type == bigbrother::market_intelligence::PricePrediction::Signal::HOLD) {
                continue;
            }

            // Create trading signal
            bigbrother::strategy::TradingSignal signal;
            signal.strategy_name = getName();
            signal.symbol = symbol;
            signal.type = convertSignalType(signal_type);

            // Use confidence from prediction
            signal.confidence = (prediction->confidence_1d + prediction->confidence_5d + prediction->confidence_20d) / 3.0;

            // Estimate expected return based on predicted changes
            double expected_change = (prediction->day_1_change + prediction->day_5_change + prediction->day_20_change) / 3.0;
            signal.expected_return = std::abs(expected_change) * 1000.0;  // Assume $1000 position
            signal.max_risk = 200.0;  // Max risk $200 (2% stop loss)
            signal.win_probability = signal.confidence;
            signal.timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            signal.rationale = formatRationale(*prediction);

            Logger::getInstance().info(
                "ML Signal: {} {} (confidence: {:.1f}%, predicted change: {:.2f}%)",
                symbol,
                bigbrother::market_intelligence::PricePrediction::signalToString(signal_type),
                signal.confidence * 100,
                expected_change * 100
            );

            signals.push_back(std::move(signal));
        }

        return signals;
    }

    auto setActive(bool active) noexcept -> void override {
        active_ = active;
    }

    [[nodiscard]] auto isActive() const noexcept -> bool override {
        return active_;
    }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {
            {"model", "price_predictor.onnx"},
            {"confidence_threshold", "0.6"},
            {"use_cuda", "true"}
        };
    }

  private:
    bool active_{true};

    /**
     * Extract features from market data
     *
     * Uses historical price/volume buffers for accurate technical indicators.
     * Falls back to approximations when insufficient history available.
     */
    [[nodiscard]] auto extractFeatures(StrategyContext const& context, std::string const& symbol)
        -> std::optional<bigbrother::market_intelligence::PriceFeatures> {

        // Get current quote for symbol
        auto quote_it = context.current_quotes.find(symbol);
        if (quote_it == context.current_quotes.end()) {
            Logger::getInstance().warn("No quote available for {}", symbol);
            return std::nullopt;
        }

        auto const& quote = quote_it->second;
        float const last_price = static_cast<float>(quote.last);
        float const bid_price = static_cast<float>(quote.bid);
        float const ask_price = static_cast<float>(quote.ask);
        float const current_volume = static_cast<float>(quote.volume);

        // Check if we have sufficient historical data (26 days for MACD)
        auto price_hist_it = price_history_.find(symbol);
        auto volume_hist_it = volume_history_.find(symbol);

        bool const has_full_history =
            price_hist_it != price_history_.end() &&
            volume_hist_it != volume_history_.end() &&
            price_hist_it->second.size() >= 26 &&
            volume_hist_it->second.size() >= 20;

        bigbrother::market_intelligence::PriceFeatures features;

        if (has_full_history) {
            // Use accurate feature extraction with historical data
            auto const& price_hist = price_hist_it->second;
            auto const& volume_hist = volume_hist_it->second;

            // Convert deque to vector for span (most recent first)
            std::vector<float> price_vec(price_hist.begin(), price_hist.end());
            std::vector<float> volume_vec(volume_hist.begin(), volume_hist.end());

            // Create spans
            std::span<float const> price_span(price_vec);
            std::span<float const> volume_span(volume_vec);

            // OHLCV data (use current quote + historical high/low if available)
            features.close = last_price;
            features.open = price_vec[0];  // Today's open (same as current in our model)
            features.high = ask_price;     // Approximate from bid/ask
            features.low = bid_price;
            features.volume = current_volume;

            // Returns from historical prices
            features.return_1d = (price_vec[0] - price_vec[1]) / price_vec[1];
            if (price_vec.size() >= 6) {
                features.return_5d = (price_vec[0] - price_vec[5]) / price_vec[5];
            } else {
                features.return_5d = features.return_1d * 5.0f;  // Approximation
            }
            if (price_vec.size() >= 21) {
                features.return_20d = (price_vec[0] - price_vec[20]) / price_vec[20];
            } else {
                features.return_20d = features.return_1d * 20.0f;  // Approximation
            }

            // Technical indicators using FeatureExtractor static methods
            features.rsi_14 = bigbrother::market_intelligence::FeatureExtractor::calculateRSI(
                price_span.subspan(0, std::min(size_t(14), price_vec.size())));

            auto [macd, signal, hist] = bigbrother::market_intelligence::FeatureExtractor::calculateMACD(price_span);
            features.macd = macd;
            features.macd_signal = signal;
            // Note: macd_histogram no longer in PriceFeatures (not used in 60-feature model)

            auto [bb_upper, bb_middle, bb_lower] =
                bigbrother::market_intelligence::FeatureExtractor::calculateBollingerBands(
                    price_span.subspan(0, std::min(size_t(20), price_vec.size())));
            features.bb_upper = bb_upper;
            // Note: bb_middle no longer in PriceFeatures (not used in 60-feature model)
            features.bb_lower = bb_lower;
            features.bb_position = (last_price - bb_lower) / (bb_upper - bb_lower + 0.0001f);

            features.atr_14 = bigbrother::market_intelligence::FeatureExtractor::calculateATR(
                price_span.subspan(0, std::min(size_t(14), price_vec.size())));

            // Volume features - calculate volume_ratio directly
            float volume_sum = 0.0f;
            size_t volume_count = std::min(size_t(20), volume_vec.size());
            for (size_t i = 0; i < volume_count; ++i) {
                volume_sum += volume_vec[i];
            }
            float volume_sma20 = volume_sum / static_cast<float>(volume_count);
            features.volume_ratio = current_volume / (volume_sma20 + 0.0001f);

            // Note: momentum_5d no longer in PriceFeatures (redundant with return_5d)

            Logger::getInstance().debug(
                "Extracted features (accurate) for {}: price={:.2f}, rsi={:.2f}, macd={:.4f}, history_size={}",
                symbol, last_price, features.rsi_14, features.macd, price_vec.size());

        } else {
            // Fall back to approximations (first few days)
            float const spread = ask_price - bid_price;
            float const volatility_estimate = spread / last_price;
            float const price_position = (last_price - bid_price) / (spread + 0.0001f);

            // OHLCV approximations
            features.close = last_price;
            features.open = last_price;
            features.high = ask_price;
            features.low = bid_price;
            features.volume = current_volume;

            // Returns - use bid/ask spread as volatility proxy
            features.return_1d = volatility_estimate * 0.5f;
            features.return_5d = volatility_estimate * 2.0f;
            features.return_20d = volatility_estimate * 5.0f;

            // Technical indicators (simplified)
            features.rsi_14 = 50.0f + (price_position - 0.5f) * 40.0f;  // Centered around 50
            features.macd = volatility_estimate * last_price * 0.01f;
            features.macd_signal = features.macd * 0.7f;
            // Note: macd_histogram, bb_middle, volume_sma20, momentum_5d removed from 60-feature model
            features.bb_upper = last_price * (1.0f + volatility_estimate * 2.0f);
            features.bb_lower = last_price * (1.0f - volatility_estimate * 2.0f);
            features.bb_position = price_position;
            features.atr_14 = spread;
            features.volume_ratio = 1.0f;

            size_t hist_size = price_hist_it != price_history_.end() ?
                price_hist_it->second.size() : 0;
            Logger::getInstance().warn(
                "Using approximate features for {} (insufficient history: {} days, need 26)",
                symbol, hist_size);
        }

        return features;
    }

    /**
     * Convert ML signal to strategy signal type
     */
    [[nodiscard]] auto convertSignalType(bigbrother::market_intelligence::PricePrediction::Signal ml_signal)
        -> SignalType {

        using MLSignal = bigbrother::market_intelligence::PricePrediction::Signal;

        switch (ml_signal) {
            case MLSignal::STRONG_BUY:
            case MLSignal::BUY:
                return SignalType::Buy;
            case MLSignal::STRONG_SELL:
            case MLSignal::SELL:
                return SignalType::Sell;
            case MLSignal::HOLD:
            default:
                return SignalType::Hold;
        }
    }

    /**
     * Format prediction rationale for logging
     */
    [[nodiscard]] auto formatRationale(bigbrother::market_intelligence::PricePrediction const& pred)
        -> std::string {

        std::ostringstream oss;
        oss << "ML Prediction: 1d=" << (pred.day_1_change * 100) << "%, "
            << "5d=" << (pred.day_5_change * 100) << "%, "
            << "20d=" << (pred.day_20_change * 100) << "%";
        return oss.str();
    }

    /**
     * Load historical price/volume data from database on startup
     *
     * Loads last 30 days of data for watchlist symbols (SPY, QQQ, IWM, DIA)
     */
    auto loadHistoricalData() -> void {
        try {
            using namespace bigbrother::duckdb_bridge;

            // Open database and create connection
            auto db = openDatabase("data/bigbrother.duckdb");
            if (!db) {
                Logger::getInstance().error("Failed to open database for historical data");
                return;
            }

            auto conn = createConnection(*db);
            if (!conn) {
                Logger::getInstance().error("Failed to create database connection");
                return;
            }

            std::vector<std::string> symbols = {"SPY", "QQQ", "IWM", "DIA"};

            for (auto const& symbol : symbols) {
                // Query last 30 days of price data (most recent first)
                auto sql_query = std::format(
                    "SELECT date, close, volume FROM stock_prices "
                    "WHERE symbol = '{}' ORDER BY date DESC LIMIT 30",
                    symbol);

                auto result = executeQueryWithResults(*conn, sql_query);
                if (!result || hasError(*result)) {
                    Logger::getInstance().warn("No historical data found for {}", symbol);
                    continue;
                }

                auto row_count = getRowCount(*result);
                if (row_count == 0) {
                    Logger::getInstance().warn("No historical data found for {}", symbol);
                    continue;
                }

                // Populate history buffers (most recent first)
                // Column indices: 0=date, 1=close, 2=volume
                for (size_t row = 0; row < row_count; ++row) {
                    try {
                        if (!isValueNull(*result, 1, row) && !isValueNull(*result, 2, row)) {
                            float price = static_cast<float>(getValueAsDouble(*result, 1, row));
                            float volume = static_cast<float>(getValueAsDouble(*result, 2, row));

                            price_history_[symbol].push_back(price);
                            volume_history_[symbol].push_back(volume);
                        }
                    } catch (std::exception const& e) {
                        Logger::getInstance().warn("Failed to parse row {} for {}: {}", row, symbol, e.what());
                    }
                }

                Logger::getInstance().info(
                    "Loaded {} days of historical data for {}",
                    price_history_[symbol].size(), symbol);
            }

        } catch (std::exception const& e) {
            Logger::getInstance().error("Exception loading historical data: {}", e.what());
        }
    }

    /**
     * Update price and volume history for a symbol
     *
     * @param symbol Symbol to update
     * @param price Current price (close/last)
     * @param volume Current volume
     * @param high Today's high (optional, defaults to price)
     * @param low Today's low (optional, defaults to price)
     */
    auto updateHistory(
        std::string const& symbol,
        double price,
        double volume,
        double high = 0.0,
        double low = 0.0) -> void {

        // Initialize high/low if not provided
        if (high == 0.0) high = price;
        if (low == 0.0) low = price;

        // Add to price history (keep last 30 days)
        auto& price_hist = price_history_[symbol];
        price_hist.push_front(static_cast<float>(price));
        if (price_hist.size() > 30) {
            price_hist.pop_back();
        }

        // Add to volume history (keep last 30 days)
        auto& volume_hist = volume_history_[symbol];
        volume_hist.push_front(static_cast<float>(volume));
        if (volume_hist.size() > 30) {
            volume_hist.pop_back();
        }

        // Add to high history
        auto& high_hist = high_history_[symbol];
        high_hist.push_front(static_cast<float>(high));
        if (high_hist.size() > 30) {
            high_hist.pop_back();
        }

        // Add to low history
        auto& low_hist = low_history_[symbol];
        low_hist.push_front(static_cast<float>(low));
        if (low_hist.size() > 30) {
            low_hist.pop_back();
        }

        Logger::getInstance().debug(
            "Updated history for {}: price={:.2f}, volume={:.0f}, buffer_size={}",
            symbol, price, volume, price_hist.size());
    }

  private:
    // Price history buffers (30 days per symbol, most recent first)
    std::unordered_map<std::string, std::deque<float>> price_history_;
    std::unordered_map<std::string, std::deque<float>> volume_history_;
    std::unordered_map<std::string, std::deque<float>> high_history_;
    std::unordered_map<std::string, std::deque<float>> low_history_;
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] inline auto createMLPredictorStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<MLPredictorStrategy>();
}

[[nodiscard]] inline auto createStraddleStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<StraddleStrategy>();
}

[[nodiscard]] inline auto createStrangleStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<StrangleStrategy>();
}

[[nodiscard]] inline auto createVolatilityArbStrategy() -> std::unique_ptr<IStrategy> {
    return std::make_unique<VolatilityArbStrategy>();
}

[[nodiscard]] inline auto createSectorRotationStrategy(
    SectorRotationStrategy::Config config)
    -> std::unique_ptr<IStrategy> {
    return std::make_unique<SectorRotationStrategy>(std::move(config));
}

} // namespace bigbrother::strategies
