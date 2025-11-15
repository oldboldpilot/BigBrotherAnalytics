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
#include <chrono>
#include <cmath>
#include <ctime>
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
import bigbrother.utils.validation;
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
            signal.max_risk = 1000.0; // $1000 max risk per trade
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
            signal.max_risk = 1000.0; // $1000 max risk per trade
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
    [[nodiscard]] auto withMinIVHVSpread(double spread) noexcept -> VolatilityArbStrategyBuilder& {
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
            signal.max_risk = 1000.0; // $1000 max risk per trade
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

    [[nodiscard]] auto minCompositeScore(double score) noexcept -> SectorRotationStrategyBuilder& {
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
        // Initialize v4.0 predictor (85-feature INT32 SIMD model)
        auto& predictor = bigbrother::market_intelligence::PricePredictor::getInstance();

        if (!predictor.isInitialized()) {
            bigbrother::market_intelligence::PredictorConfigV4 config;
            config.model_weights_path = "models/weights";

            if (!predictor.initialize(config)) {
                Logger::getInstance().error("Failed to initialize ML predictor v4.0");
            } else {
                Logger::getInstance().info("ML Predictor v4.0 Strategy initialized (INT32 SIMD, 85 features)");
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
            Logger::getInstance().warn("ML strategy not active");
            return signals;
        }

        auto& predictor = bigbrother::market_intelligence::PricePredictor::getInstance();

        if (!predictor.isInitialized()) {
            Logger::getInstance().warn("ML predictor v4.0 not initialized, skipping");
            return signals;
        }

        Logger::getInstance().info("ML Price Predictor: Analyzing {} symbols", 4);

        // Get symbols to analyze (from context or default watchlist)
        std::vector<std::string> symbols = {"SPY", "QQQ", "IWM", "DIA"};

        for (auto const& symbol : symbols) {
            Logger::getInstance().info("ML: Processing symbol {}", symbol);

            // DON'T update history here - causes duplicates when multiple instances run!
            // Historical data is already loaded from database at startup (loadHistoricalData)
            // Only update history once per new bar, not on every prediction cycle

            // Extract features from market data (will use historical buffers if available)
            auto features = extractFeatures(context, symbol);
            if (!features) {
                Logger::getInstance().warn("ML: Feature extraction failed for {}", symbol);
                continue;
            }

            Logger::getInstance().info("ML: Features extracted for {}", symbol);

            // Convert deque history to vectors for 85-feature extraction
            auto price_hist_it = price_history_.find(symbol);
            auto volume_hist_it = volume_history_.find(symbol);

            if (price_hist_it == price_history_.end() || volume_hist_it == volume_history_.end()) {
                Logger::getInstance().warn("No price/volume history for {}, skipping", symbol);
                continue;
            }

            // Check if we have enough history (need 21 days minimum for 20-day lags)
            if (price_hist_it->second.size() < 21 || volume_hist_it->second.size() < 21) {
                Logger::getInstance().warn("Insufficient history for {} ({} days, need 21), skipping",
                                           symbol, price_hist_it->second.size());
                continue;
            }

            std::vector<float> price_vec(price_hist_it->second.begin(), price_hist_it->second.end());
            std::vector<float> volume_vec(volume_hist_it->second.begin(), volume_hist_it->second.end());

            Logger::getInstance().info("ML: Converting {} days of history to 85 features for {} - "
                                       "most recent prices: [{:.2f}, {:.2f}, {:.2f}]",
                                       price_vec.size(), symbol,
                                       price_vec.size() > 0 ? price_vec[0] : 0.0f,
                                       price_vec.size() > 1 ? price_vec[1] : 0.0f,
                                       price_vec.size() > 2 ? price_vec[2] : 0.0f);

            // Extract 85 features for v4.0 model
            auto features_85 = features->toArray85(price_vec, volume_vec, std::chrono::system_clock::now());

            // Debug: Log ALL 85 features to diagnose why predictions are identical
            std::string all_features = "[";
            for (size_t i = 0; i < 85; ++i) {
                if (i > 0) all_features += ", ";
                all_features += std::to_string(features_85[i]);
            }
            all_features += "]";
            Logger::getInstance().info("ML: ALL 85 features for {}: {}", symbol, all_features);

            Logger::getInstance().info("ML: Running prediction for {}", symbol);

            // Run ML prediction with 85-feature array
            auto prediction = predictor.predict(symbol, features_85);

            Logger::getInstance().info("ML: Prediction complete for {} (success: {})",
                                       symbol, prediction.has_value());
            if (!prediction) {
                Logger::getInstance().warn("Prediction failed for {}", symbol);
                continue;
            }

            Logger::getInstance().info("ML: Prediction for {} - 1d:{:.2f}%, 5d:{:.2f}%, 20d:{:.2f}%",
                                       symbol,
                                       prediction->day_1_change * 100,
                                       prediction->day_5_change * 100,
                                       prediction->day_20_change * 100);

            // CRITICAL: Sanity check predictions to catch model errors
            // Reject predictions outside reasonable range (-50% to +50%)
            constexpr double MAX_REASONABLE_CHANGE = 0.50; // 50%
            bool prediction_invalid = std::abs(prediction->day_1_change) > MAX_REASONABLE_CHANGE ||
                                      std::abs(prediction->day_5_change) > MAX_REASONABLE_CHANGE ||
                                      std::abs(prediction->day_20_change) > MAX_REASONABLE_CHANGE;

            if (prediction_invalid) {
                Logger::getInstance().error("REJECTED: Nonsensical prediction for {} (1d={:.2f}%, "
                                            "5d={:.2f}%, 20d={:.2f}%) - exceeds +/-50% threshold",
                                            symbol, prediction->day_1_change * 100,
                                            prediction->day_5_change * 100,
                                            prediction->day_20_change * 100);
                continue; // Skip this nonsensical prediction
            }

            // Get overall signal
            auto signal_type = prediction->getOverallSignal();

            Logger::getInstance().info("ML: Signal for {} is {}",
                                       symbol,
                                       bigbrother::market_intelligence::PricePrediction::signalToString(signal_type));

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
            signal.confidence = (prediction->confidence_1d + prediction->confidence_5d +
                                 prediction->confidence_20d) /
                                3.0;

            // Estimate expected return based on predicted changes
            double expected_change =
                (prediction->day_1_change + prediction->day_5_change + prediction->day_20_change) /
                3.0;
            signal.expected_return = std::abs(expected_change) * 1000.0; // Assume $1000 position
            signal.max_risk = 200.0; // Max risk $200 (2% stop loss)
            signal.win_probability = signal.confidence;
            signal.timestamp =
                std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            signal.rationale = formatRationale(*prediction);

            Logger::getInstance().info(
                "ML Signal: {} {} (confidence: {:.1f}%, predicted change: {:.2f}%)", symbol,
                bigbrother::market_intelligence::PricePrediction::signalToString(signal_type),
                signal.confidence * 100, expected_change * 100);

            signals.push_back(std::move(signal));
        }

        return signals;
    }

    auto setActive(bool active) noexcept -> void override { active_ = active; }

    [[nodiscard]] auto isActive() const noexcept -> bool override { return active_; }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {{"model", "price_predictor.onnx"},
                {"confidence_threshold", "0.6"},
                {"use_cuda", "true"}};
    }

  private:
    bool active_{true};

    /**
     * Symbol encoding map for identification features (top 20 symbols)
     * Maps symbol to integer ID (0-19)
     */
    static auto getSymbolId(std::string const& symbol) -> float {
        static std::unordered_map<std::string, float> const symbol_map = {
            {"SPY", 0.0f},  {"QQQ", 1.0f},  {"IWM", 2.0f},  {"DIA", 3.0f},
            {"AAPL", 4.0f}, {"MSFT", 5.0f}, {"AMZN", 6.0f}, {"GOOGL", 7.0f},
            {"META", 8.0f}, {"TSLA", 9.0f}, {"NVDA", 10.0f}, {"JPM", 11.0f},
            {"BAC", 12.0f}, {"WMT", 13.0f}, {"V", 14.0f},   {"JNJ", 15.0f},
            {"PG", 16.0f},  {"XOM", 17.0f}, {"CVX", 18.0f}, {"UNH", 19.0f}};

        auto it = symbol_map.find(symbol);
        return it != symbol_map.end() ? it->second : -1.0f;
    }

    /**
     * Sector encoding based on symbol prefix/heuristics
     * Returns sector ID or -1 for unknown
     */
    static auto getSectorId(std::string const& symbol) -> float {
        // ETFs and indices
        if (symbol == "SPY" || symbol == "DIA" || symbol == "IWM") return 0.0f; // Broad Market
        if (symbol == "QQQ") return 1.0f; // Technology

        // Tech sector
        if (symbol == "AAPL" || symbol == "MSFT" || symbol == "GOOGL" ||
            symbol == "META" || symbol == "NVDA") return 1.0f;

        // Consumer Discretionary
        if (symbol == "AMZN" || symbol == "TSLA" || symbol == "WMT") return 2.0f;

        // Financials
        if (symbol == "JPM" || symbol == "BAC" || symbol == "V") return 3.0f;

        // Healthcare
        if (symbol == "JNJ" || symbol == "UNH") return 4.0f;

        // Consumer Staples
        if (symbol == "PG") return 5.0f;

        // Energy
        if (symbol == "XOM" || symbol == "CVX") return 6.0f;

        return -1.0f; // Unknown
    }

    /**
     * Check if symbol is an option (contains digits in specific pattern)
     */
    static auto isOption(std::string const& symbol) -> float {
        // Options typically have format: AAPL230120C00150000
        // Check for digits indicating strike/expiry
        size_t digit_count = 0;
        for (char c : symbol) {
            if (std::isdigit(c)) digit_count++;
        }
        // If more than 4 digits, likely an option contract
        return digit_count > 4 ? 1.0f : 0.0f;
    }

    /**
     * Extract time features from current timestamp
     * Returns struct with hour, minute, day_of_week, etc.
     */
    struct TimeFeatures {
        float hour_of_day{0.0f};
        float minute_of_hour{0.0f};
        float day_of_week{0.0f};
        float day_of_month{0.0f};
        float month_of_year{0.0f};
        float quarter{0.0f};
        float day_of_year{0.0f};
        float is_market_open{0.0f};
    };

    static auto extractTimeFeatures() -> TimeFeatures {
        TimeFeatures tf;

        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_local;

        // Use localtime_r for thread safety on Linux
        #if defined(__linux__) || defined(__APPLE__)
        localtime_r(&time_t_now, &tm_local);
        #else
        localtime_s(&tm_local, &time_t_now);
        #endif

        tf.hour_of_day = static_cast<float>(tm_local.tm_hour);
        tf.minute_of_hour = static_cast<float>(tm_local.tm_min);
        tf.day_of_week = static_cast<float>(tm_local.tm_wday == 0 ? 6 : tm_local.tm_wday - 1); // 0=Mon, 6=Sun
        tf.day_of_month = static_cast<float>(tm_local.tm_mday);
        tf.month_of_year = static_cast<float>(tm_local.tm_mon + 1); // 1-12
        tf.quarter = static_cast<float>((tm_local.tm_mon / 3) + 1); // 1-4
        tf.day_of_year = static_cast<float>(tm_local.tm_yday + 1); // 1-365

        // Market open: 9:30-16:00 ET weekdays
        // Simplified: weekday (Mon-Fri) and hour 9-15 (roughly 9:30-16:00)
        bool is_weekday = tm_local.tm_wday >= 1 && tm_local.tm_wday <= 5;
        bool is_market_hours = tm_local.tm_hour >= 9 && tm_local.tm_hour < 16;
        tf.is_market_open = (is_weekday && is_market_hours) ? 1.0f : 0.0f;

        return tf;
    }

    /**
     * Treasury rates structure (realistic defaults as of 2024)
     * IMPORTANT: Rates stored as DECIMALS (0.0525 = 5.25%)
     * TODO: Fetch from FRED API or cached data
     */
    struct TreasuryRates {
        float fed_funds_rate{0.0525f};      // 5.25% as decimal
        float treasury_3mo{0.0530f};        // 5.30% as decimal
        float treasury_2yr{0.0450f};        // 4.50% as decimal
        float treasury_5yr{0.0420f};        // 4.20% as decimal
        float treasury_10yr{0.0430f};       // 4.30% as decimal
        float yield_curve_slope{-0.0020f};  // 10yr - 2yr (decimal)
        float yield_curve_inversion{1.0f};  // 1.0 if inverted
    };

    static auto getTreasuryRates() -> TreasuryRates {
        TreasuryRates rates;
        // Calculate derived features
        rates.yield_curve_slope = rates.treasury_10yr - rates.treasury_2yr;
        rates.yield_curve_inversion = rates.yield_curve_slope < 0.0f ? 1.0f : 0.0f;
        return rates;
    }

    /**
     * Options Greeks structure (simplified approximation for stocks)
     * Greeks are normalized/scaled appropriately for ML model
     * TODO: Implement proper Black-Scholes when option data available
     */
    struct Greeks {
        float delta{0.5f};              // ATM delta for stocks (dimensionless, -1 to 1)
        float gamma{0.01f};             // Small gamma (dimensionless, ~0.01 for ATM)
        float theta{-0.05f};            // Time decay per day (normalized)
        float vega{0.20f};              // Volatility sensitivity (normalized)
        float rho{0.01f};               // Rate sensitivity (normalized)
        float implied_volatility{0.25f}; // Default 25% IV (as decimal)
    };

    static auto estimateGreeks(float price, float atr) -> Greeks {
        Greeks greeks;
        // Estimate IV from ATR (volatility proxy)
        // IV should be decimal (0.25 = 25% volatility)
        greeks.implied_volatility = (atr / price) * 15.87f; // Annualized (sqrt(252))

        // Normalize theta, vega, rho to reasonable ranges for ML
        // These are scaled estimates, not exact dollar Greeks
        greeks.theta = -0.05f;  // Time decay (negative for long positions)
        greeks.vega = 0.20f;    // Volatility sensitivity
        greeks.rho = 0.01f;     // Rate sensitivity

        return greeks;
    }

    /**
     * Calculate moving average from price history
     */
    static auto calculateMA(std::span<float const> prices, size_t period) -> float {
        if (prices.empty() || period == 0) return 0.0f;
        size_t count = std::min(period, prices.size());
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum += prices[i];
        }
        return sum / static_cast<float>(count);
    }

    /**
     * Calculate win rate (proportion of positive returns) over period
     */
    static auto calculateWinRate(std::span<float const> prices, size_t period) -> float {
        if (prices.size() < period + 1) return 0.0f;
        size_t wins = 0;
        for (size_t i = 0; i < period && i + 1 < prices.size(); ++i) {
            if (prices[i] > prices[i + 1]) wins++;
        }
        return static_cast<float>(wins) / static_cast<float>(period);
    }

    /**
     * Extract features from market data
     *
     * Uses historical price/volume buffers for accurate technical indicators.
     * Falls back to approximations when insufficient history available.
     */
    [[nodiscard]] auto extractFeatures(StrategyContext const& context, std::string const& symbol)
        -> std::optional<bigbrother::market_intelligence::PriceFeatures> {

        // DEFENSIVE CHECK: Validate symbol is not empty and looks like a stock ticker
        if (symbol.empty() || symbol.length() > 10) {
            Logger::getInstance().error(
                "MLPredictor: Invalid symbol for feature extraction: '{}' (empty or too long)",
                symbol);
            return std::nullopt;
        }

        // DEFENSIVE CHECK: Reject JSON field names that might have leaked into current_quotes
        if (bigbrother::utils::looksLikeJsonField(symbol)) {
            Logger::getInstance().error(
                "MLPredictor: Symbol '{}' looks like a JSON field name, not a stock ticker",
                symbol);
            return std::nullopt;
        }

        // DEFENSIVE CHECK: Validate symbol looks like a valid stock ticker
        if (!bigbrother::utils::isValidStockSymbol(symbol)) {
            Logger::getInstance().error(
                "MLPredictor: Symbol '{}' does not look like a valid ticker", symbol);
            return std::nullopt;
        }

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
            price_hist_it != price_history_.end() && volume_hist_it != volume_history_.end() &&
            price_hist_it->second.size() >= 26 && volume_hist_it->second.size() >= 20;

        bigbrother::market_intelligence::PriceFeatures features;

        // ========================================================================
        // PART 1: IDENTIFICATION FEATURES (3 features)
        // ========================================================================
        features.symbol_encoded = getSymbolId(symbol);
        features.sector_encoded = getSectorId(symbol);
        features.is_option = isOption(symbol);

        // ========================================================================
        // PART 2: TIME FEATURES (8 features)
        // ========================================================================
        auto time_features = extractTimeFeatures();
        features.hour_of_day = time_features.hour_of_day;
        features.minute_of_hour = time_features.minute_of_hour;
        features.day_of_week = time_features.day_of_week;
        features.day_of_month = time_features.day_of_month;
        features.month_of_year = time_features.month_of_year;
        features.quarter = time_features.quarter;
        features.day_of_year = time_features.day_of_year;
        features.is_market_open = time_features.is_market_open;

        // ========================================================================
        // PART 3: TREASURY RATES (7 features)
        // ========================================================================
        auto treasury_rates = getTreasuryRates();
        features.fed_funds_rate = treasury_rates.fed_funds_rate;
        features.treasury_3mo = treasury_rates.treasury_3mo;
        features.treasury_2yr = treasury_rates.treasury_2yr;
        features.treasury_5yr = treasury_rates.treasury_5yr;
        features.treasury_10yr = treasury_rates.treasury_10yr;
        features.yield_curve_slope = treasury_rates.yield_curve_slope;
        features.yield_curve_inversion = treasury_rates.yield_curve_inversion;

        // ========================================================================
        // PART 4: SENTIMENT FEATURES (2 features)
        // ========================================================================
        // TODO: Integrate with news sentiment data from database
        features.avg_sentiment = 0.0f;  // Neutral sentiment (no news data yet)
        features.news_count = 0.0f;     // No news articles

        // ========================================================================
        // PART 5: PRICE, MOMENTUM, VOLATILITY FEATURES (16 features)
        // ========================================================================
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

            // [26-30] OHLCV data (use historical data for realistic values)
            features.close = last_price;

            // Open: Use yesterday's close as proxy for today's open
            features.open = price_vec.size() >= 2 ? price_vec[1] : last_price;

            // High/Low: Use recent price range (last 5 bars)
            size_t lookback = std::min(size_t(5), price_vec.size());
            features.high = *std::max_element(price_vec.begin(), price_vec.begin() + lookback);
            features.low = *std::min_element(price_vec.begin(), price_vec.begin() + lookback);

            // Volume: Use most recent historical volume (volume_vec[0] is today)
            features.volume = volume_vec.size() > 0 ? volume_vec[0] : 0.0f;

            // [31-37] Returns from historical prices
            features.return_1d = (price_vec[0] - price_vec[1]) / price_vec[1];
            if (price_vec.size() >= 6) {
                features.return_5d = (price_vec[0] - price_vec[5]) / price_vec[5];
            } else {
                features.return_5d = features.return_1d * 5.0f; // Approximation
            }
            if (price_vec.size() >= 21) {
                features.return_20d = (price_vec[0] - price_vec[20]) / price_vec[20];
            } else {
                features.return_20d = features.return_1d * 20.0f; // Approximation
            }

            // Technical indicators using FeatureExtractor static methods
            features.rsi_14 = bigbrother::market_intelligence::FeatureExtractor::calculateRSI(
                price_span.subspan(0, std::min(size_t(14), price_vec.size())));

            auto [macd, signal, hist] =
                bigbrother::market_intelligence::FeatureExtractor::calculateMACD(price_span);
            features.macd = macd;
            features.macd_signal = signal;

            auto [bb_upper, bb_middle, bb_lower] =
                bigbrother::market_intelligence::FeatureExtractor::calculateBollingerBands(
                    price_span.subspan(0, std::min(size_t(20), price_vec.size())));
            features.bb_upper = bb_upper;
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

            // ====================================================================
            // PART 6: OPTIONS GREEKS (6 features) - estimated from ATR
            // ====================================================================
            auto greeks = estimateGreeks(last_price, features.atr_14);
            features.delta = greeks.delta;
            features.gamma = greeks.gamma;
            features.theta = greeks.theta;
            features.vega = greeks.vega;
            features.rho = greeks.rho;
            features.implied_volatility = greeks.implied_volatility;

            // ====================================================================
            // PART 7: INTERACTION FEATURES (10 features)
            // ====================================================================
            features.sentiment_momentum = features.avg_sentiment * features.return_5d;
            features.volume_rsi_signal = features.volume_ratio * (features.rsi_14 - 50.0f) / 50.0f;
            features.yield_volatility = features.yield_curve_slope * features.atr_14;
            features.delta_iv = features.delta * features.implied_volatility;
            features.macd_volume = features.macd * features.volume_ratio;
            features.bb_momentum = features.bb_position * features.return_1d;
            features.sentiment_strength = features.avg_sentiment * std::log(features.news_count + 1.0f);
            features.rate_return = features.fed_funds_rate * features.return_20d;
            features.gamma_volatility = features.gamma * features.atr_14;
            features.rsi_bb_signal = (features.rsi_14 / 100.0f) * features.bb_position;

            // ====================================================================
            // PART 8: DIRECTIONALITY FEATURES (8 features)
            // ====================================================================
            features.price_direction = features.return_1d > 0.0f ? 1.0f : 0.0f;

            // Calculate MA5 and MA20 for comparison
            float ma5 = calculateMA(price_span, 5);
            float ma20 = calculateMA(price_span, 20);
            features.price_above_ma5 = last_price > ma5 ? 1.0f : 0.0f;
            features.price_above_ma20 = last_price > ma20 ? 1.0f : 0.0f;

            // Calculate 3-day return if history allows
            if (price_vec.size() >= 4) {
                features.momentum_3d = (price_vec[0] - price_vec[3]) / price_vec[3];
            } else {
                features.momentum_3d = features.return_1d * 3.0f;
            }

            // Calculate trend strength (5-day win rate - 0.5)
            features.trend_strength = calculateWinRate(price_span, 5) - 0.5f;

            // Calculate recent win rate (10-day)
            features.recent_win_rate = calculateWinRate(price_span, 10);

            // Signal directions
            features.macd_signal_direction = features.macd > features.macd_signal ? 1.0f : 0.0f;
            features.volume_trend = features.volume_ratio > 1.0f ? 1.0f : 0.0f;

            Logger::getInstance().debug("Extracted ALL 60 features (accurate) for {}: price={:.2f}, "
                                        "rsi={:.2f}, macd={:.4f}, history_size={}",
                                        symbol, last_price, features.rsi_14, features.macd,
                                        price_vec.size());

        } else {
            // Fall back to approximations (first few days)
            float const spread = ask_price - bid_price;
            float const volatility_estimate = spread / last_price;
            float const price_position = (last_price - bid_price) / (spread + 0.0001f);

            // [26-30] OHLCV approximations
            features.close = last_price;
            features.open = last_price;
            features.high = ask_price;
            features.low = bid_price;
            features.volume = current_volume;

            // [31-37] Returns - use bid/ask spread as volatility proxy
            features.return_1d = volatility_estimate * 0.5f;
            features.return_5d = volatility_estimate * 2.0f;
            features.return_20d = volatility_estimate * 5.0f;

            // Technical indicators (simplified)
            features.rsi_14 = 50.0f + (price_position - 0.5f) * 40.0f; // Centered around 50
            features.macd = volatility_estimate * last_price * 0.01f;
            features.macd_signal = features.macd * 0.7f;
            features.bb_upper = last_price * (1.0f + volatility_estimate * 2.0f);
            features.bb_lower = last_price * (1.0f - volatility_estimate * 2.0f);
            features.bb_position = price_position;
            features.atr_14 = spread;
            features.volume_ratio = 1.0f;

            // [18-23] OPTIONS GREEKS - simplified estimates
            auto greeks = estimateGreeks(last_price, features.atr_14);
            features.delta = greeks.delta;
            features.gamma = greeks.gamma;
            features.theta = greeks.theta;
            features.vega = greeks.vega;
            features.rho = greeks.rho;
            features.implied_volatility = greeks.implied_volatility;

            // [42-51] INTERACTION FEATURES (all zeros due to no sentiment/limited data)
            features.sentiment_momentum = 0.0f;
            features.volume_rsi_signal = features.volume_ratio * (features.rsi_14 - 50.0f) / 50.0f;
            features.yield_volatility = features.yield_curve_slope * features.atr_14;
            features.delta_iv = features.delta * features.implied_volatility;
            features.macd_volume = features.macd * features.volume_ratio;
            features.bb_momentum = features.bb_position * features.return_1d;
            features.sentiment_strength = 0.0f;
            features.rate_return = features.fed_funds_rate * features.return_20d;
            features.gamma_volatility = features.gamma * features.atr_14;
            features.rsi_bb_signal = (features.rsi_14 / 100.0f) * features.bb_position;

            // [52-59] DIRECTIONALITY FEATURES (simplified)
            features.price_direction = features.return_1d > 0.0f ? 1.0f : 0.0f;
            features.trend_strength = 0.0f; // Unknown without history
            features.price_above_ma5 = 0.5f; // Neutral assumption
            features.price_above_ma20 = 0.5f; // Neutral assumption
            features.momentum_3d = features.return_1d * 3.0f;
            features.macd_signal_direction = features.macd > features.macd_signal ? 1.0f : 0.0f;
            features.volume_trend = features.volume_ratio > 1.0f ? 1.0f : 0.0f;
            features.recent_win_rate = 0.5f; // Neutral assumption

            size_t hist_size =
                price_hist_it != price_history_.end() ? price_hist_it->second.size() : 0;
            Logger::getInstance().warn(
                "Using approximate features for {} (insufficient history: {} days, need 26)",
                symbol, hist_size);
        }

        return features;
    }

    /**
     * Convert ML signal to strategy signal type
     */
    [[nodiscard]] auto
    convertSignalType(bigbrother::market_intelligence::PricePrediction::Signal ml_signal)
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
        Logger::getInstance().info("ML: Starting loadHistoricalData()...");

        try {
            using namespace bigbrother::duckdb_bridge;

            Logger::getInstance().info("ML: Opening database data/bigbrother.duckdb in READ-ONLY mode...");

            // Open database in READ-ONLY mode (allows concurrent access with dashboard)
            auto db = openDatabase("data/bigbrother.duckdb", true);  // read_only = true
            if (!db) {
                Logger::getInstance().error("Failed to open database for historical data");
                return;
            }

            Logger::getInstance().info("ML: Database opened successfully, creating connection...");

            auto conn = createConnection(*db);
            if (!conn) {
                Logger::getInstance().error("Failed to create database connection");
                return;
            }

            Logger::getInstance().info("ML: Connection created successfully");

            std::vector<std::string> symbols = {"SPY", "QQQ", "IWM", "DIA"};

            for (auto const& symbol : symbols) {
                Logger::getInstance().info("ML: Querying historical data for {}...", symbol);

                // Query last 100 days of price data (most recent first)
                // Need sufficient history for 85-feature model (requires 21+ days)
                auto sql_query = std::format("SELECT date, close, volume FROM stock_prices "
                                             "WHERE symbol = '{}' ORDER BY date DESC LIMIT 100",
                                             symbol);

                Logger::getInstance().info("ML: Executing query: {}", sql_query);

                auto result = executeQueryWithResults(*conn, sql_query);
                if (!result) {
                    Logger::getInstance().warn("Query returned null result for {}", symbol);
                    continue;
                }

                if (hasError(*result)) {
                    Logger::getInstance().warn("Query has error for {}", symbol);
                    continue;
                }

                auto row_count = getRowCount(*result);
                Logger::getInstance().info("ML: Query returned {} rows for {}", row_count, symbol);

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
                        Logger::getInstance().warn("Failed to parse row {} for {}: {}", row, symbol,
                                                   e.what());
                    }
                }

                Logger::getInstance().info("Loaded {} days of historical data for {} - first 3 prices: [{:.2f}, {:.2f}, {:.2f}]",
                                           price_history_[symbol].size(), symbol,
                                           price_history_[symbol].size() > 0 ? price_history_[symbol][0] : 0.0f,
                                           price_history_[symbol].size() > 1 ? price_history_[symbol][1] : 0.0f,
                                           price_history_[symbol].size() > 2 ? price_history_[symbol][2] : 0.0f);
            }

            Logger::getInstance().info("ML: loadHistoricalData() completed successfully");

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
    auto updateHistory(std::string const& symbol, double price, double volume, double high = 0.0,
                       double low = 0.0) -> void {

        // Initialize high/low if not provided
        if (high == 0.0)
            high = price;
        if (low == 0.0)
            low = price;

        // Add to price history (keep last 30 days)
        auto& price_hist = price_history_[symbol];

        Logger::getInstance().info("ML: updateHistory() for {} - adding price={:.2f}, buffer_size_before={}, "
                                   "first_3_before=[{:.2f}, {:.2f}, {:.2f}]",
                                   symbol, price, price_hist.size(),
                                   price_hist.size() > 0 ? price_hist[0] : 0.0f,
                                   price_hist.size() > 1 ? price_hist[1] : 0.0f,
                                   price_hist.size() > 2 ? price_hist[2] : 0.0f);

        price_hist.push_front(static_cast<float>(price));
        if (price_hist.size() > 30) {
            price_hist.pop_back();
        }

        Logger::getInstance().info("ML: updateHistory() for {} - buffer_size_after={}, "
                                   "first_3_after=[{:.2f}, {:.2f}, {:.2f}]",
                                   symbol, price_hist.size(),
                                   price_hist.size() > 0 ? price_hist[0] : 0.0f,
                                   price_hist.size() > 1 ? price_hist[1] : 0.0f,
                                   price_hist.size() > 2 ? price_hist[2] : 0.0f);

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
            "Updated history for {}: price={:.2f}, volume={:.0f}, buffer_size={}", symbol, price,
            volume, price_hist.size());
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

[[nodiscard]] inline auto createSectorRotationStrategy(SectorRotationStrategy::Config config)
    -> std::unique_ptr<IStrategy> {
    return std::make_unique<SectorRotationStrategy>(std::move(config));
}

} // namespace bigbrother::strategies
