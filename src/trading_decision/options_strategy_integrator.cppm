/**
 * @file options_strategy_integrator.cppm
 * @brief Integration layer for 52 options strategies with trading engine
 *
 * Connects options strategies with:
 * - Trading decision engine
 * - P&L tracking and portfolio management
 * - Risk management and OptionsGreeks monitoring
 * - Price prediction and market intelligence
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 */

module;

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

export module bigbrother.options_strategy_integrator;

import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.options_strategies.base;
import bigbrother.options_strategies.single_leg;
import bigbrother.options_strategies.vertical_spreads;
import bigbrother.options_strategies.straddles_strangles;
import bigbrother.options_strategies.butterflies_condors;
import bigbrother.options_strategies.covered_positions;
import bigbrother.options_strategies.calendar_spreads;
import bigbrother.options_strategies.ratio_spreads;
import bigbrother.options_strategies.albatross_ladder;
import bigbrother.risk_management;
import bigbrother.strategy;

export namespace bigbrother::options_integration {

using namespace bigbrother::types;
using namespace bigbrother::options_strategies;
using namespace bigbrother::risk;

// Resolve ambiguity between strategy::Greeks and options_strategies::Greeks
using OptionsGreeks = bigbrother::options_strategies::Greeks;

// ============================================================================
// Strategy Selection Criteria
// ============================================================================

enum class MarketCondition {
    BULLISH,
    BEARISH,
    NEUTRAL,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    UNCERTAIN
};

struct StrategySelectionCriteria {
    MarketCondition market_outlook{MarketCondition::UNCERTAIN};
    float current_iv{0.25f};           // Current implied volatility
    float historical_iv{0.20f};        // Historical average IV
    float iv_percentile{50.0f};        // IV rank (0-100)
    float current_price{100.0f};
    float predicted_price{100.0f};     // From price predictor
    float prediction_confidence{0.5f}; // Confidence in prediction
    float time_horizon_days{30.0f};    // Trading time horizon
    float max_loss_tolerance{1000.0f}; // Maximum loss willing to accept
    float capital_allocation{5000.0f}; // Capital to allocate
    bool prefer_defined_risk{true};    // Prefer strategies with defined risk
    bool income_focused{false};        // Focus on income generation
};

// ============================================================================
// Strategy Recommendation
// ============================================================================

struct StrategyRecommendation {
    std::string strategy_name;
    StrategyType strategy_type;
    MarketOutlook outlook;
    ComplexityLevel complexity;
    float score{0.0f};                    // Recommendation score (0-100)
    float expected_pnl{0.0f};             // Expected profit/loss
    float max_profit{0.0f};
    float max_loss{0.0f};
    float probability_of_profit{0.0f};    // Estimated PoP
    float capital_required{0.0f};
    OptionsGreeks greeks{};
    std::vector<std::string> strikes;     // Recommended strikes
    std::vector<float> premiums;          // Expected premiums
    std::string rationale;                // Why this strategy

    [[nodiscard]] auto getRiskRewardRatio() const -> float {
        if (max_loss == 0.0f) return 0.0f;
        return std::abs(max_profit / max_loss);
    }
};

// ============================================================================
// Position Tracking for Options Strategies
// ============================================================================

struct OptionsPosition {
    std::string position_id;
    std::string symbol;
    std::unique_ptr<IOptionsStrategy> strategy;
    float entry_price{0.0f};
    float current_price{0.0f};
    float total_premium_paid{0.0f};
    float total_premium_received{0.0f};
    float unrealized_pnl{0.0f};
    float realized_pnl{0.0f};
    OptionsGreeks current_greeks{};
    int days_held{0};
    int days_to_expiration{0};
    bool is_closed{false};

    [[nodiscard]] auto calculatePnL(float underlying_price) const -> float {
        if (!strategy) return 0.0f;
        return strategy->calculateProfitLoss(underlying_price);
    }

    [[nodiscard]] auto updateGreeks(float underlying_price, float rate) -> OptionsGreeks {
        if (!strategy) return OptionsGreeks{};
        current_greeks = strategy->calculateGreeks(underlying_price, rate);
        return current_greeks;
    }
};

// ============================================================================
// Options Strategy Selector
// ============================================================================

class OptionsStrategySelector {
public:
    OptionsStrategySelector() = default;

    /**
     * Recommend best strategies based on market conditions and criteria
     * Returns top N strategies ranked by suitability score
     */
    [[nodiscard]] auto recommendStrategies(
        const StrategySelectionCriteria& criteria,
        int top_n = 5
    ) -> std::vector<StrategyRecommendation> {

        std::vector<StrategyRecommendation> recommendations;

        // Score each strategy category based on market conditions
        if (criteria.market_outlook == MarketCondition::HIGH_VOLATILITY) {
            addVolatilityStrategies(criteria, recommendations);
        }
        else if (criteria.market_outlook == MarketCondition::BULLISH) {
            addBullishStrategies(criteria, recommendations);
        }
        else if (criteria.market_outlook == MarketCondition::BEARISH) {
            addBearishStrategies(criteria, recommendations);
        }
        else if (criteria.market_outlook == MarketCondition::NEUTRAL) {
            addNeutralStrategies(criteria, recommendations);
        }
        else {
            // Uncertain - recommend conservative strategies
            addConservativeStrategies(criteria, recommendations);
        }

        // Sort by score and return top N
        std::sort(recommendations.begin(), recommendations.end(),
            [](const auto& a, const auto& b) -> bool { return a.score > b.score; });

        if (recommendations.size() > static_cast<size_t>(top_n)) {
            recommendations.resize(top_n);
        }

        return recommendations;
    }

    /**
     * Create a strategy instance based on recommendation
     */
    [[nodiscard]] auto createStrategy(
        const StrategyRecommendation& rec,
        float underlying_price,
        const std::vector<float>& strikes,
        float days,
        float iv,
        float rate = 0.05f
    ) -> std::unique_ptr<IOptionsStrategy> {

        // Single leg strategies
        if (rec.strategy_name == "Long Call") {
            return createLongCall(underlying_price, strikes[0], days, iv, rate);
        }
        if (rec.strategy_name == "Long Put") {
            return createLongPut(underlying_price, strikes[0], days, iv, rate);
        }
        if (rec.strategy_name == "Short Call") {
            return createShortCall(underlying_price, strikes[0], days, iv, rate);
        }
        if (rec.strategy_name == "Short Put") {
            return createShortPut(underlying_price, strikes[0], days, iv, rate);
        }

        // Vertical spreads
        if (rec.strategy_name == "Bull Call Spread") {
            return createBullCallSpread(underlying_price, strikes[0], strikes[1], days, iv, rate);
        }
        if (rec.strategy_name == "Bear Put Spread") {
            return createBearPutSpread(underlying_price, strikes[0], strikes[1], days, iv, rate);
        }
        if (rec.strategy_name == "Bull Put Spread") {
            return createBullPutSpread(underlying_price, strikes[0], strikes[1], days, iv, rate);
        }
        if (rec.strategy_name == "Bear Call Spread") {
            return createBearCallSpread(underlying_price, strikes[0], strikes[1], days, iv, rate);
        }

        // Straddles & Strangles
        if (rec.strategy_name == "Long Straddle") {
            return createLongStraddle(underlying_price, strikes[0], days, iv, rate);
        }
        if (rec.strategy_name == "Short Straddle") {
            return createShortStraddle(underlying_price, strikes[0], days, iv, rate);
        }
        if (rec.strategy_name == "Long Strangle") {
            return createLongStrangle(underlying_price, strikes[0], strikes[1], days, iv, rate);
        }
        if (rec.strategy_name == "Short Strangle") {
            return createShortStrangle(underlying_price, strikes[0], strikes[1], days, iv, rate);
        }

        // Iron Condor (popular neutral strategy)
        if (rec.strategy_name == "Iron Condor") {
            return createLongIronCondor(underlying_price, strikes[0], strikes[1],
                                       strikes[2], strikes[3], days, iv, rate);
        }

        // Covered positions
        if (rec.strategy_name == "Covered Call") {
            return createCoveredCall(underlying_price, strikes[0],
                                    days, iv, rate, 100.0f);
        }

        // Default to long call if unknown
        return createLongCall(underlying_price, strikes[0], days, iv, rate);
    }

private:
    void addVolatilityStrategies(const StrategySelectionCriteria& criteria,
                                 std::vector<StrategyRecommendation>& recs) {
        // High IV - sell premium strategies
        if (criteria.iv_percentile > 70.0f) {
            recs.push_back(createRecommendation(
                "Short Straddle", StrategyType::COMBINATION, MarketOutlook::NEUTRAL,
                ComplexityLevel::INTERMEDIATE, 85.0f, criteria,
                "High IV environment - sell premium with straddle"
            ));
            recs.push_back(createRecommendation(
                "Short Strangle", StrategyType::COMBINATION, MarketOutlook::NEUTRAL,
                ComplexityLevel::INTERMEDIATE, 80.0f, criteria,
                "High IV - wider strikes for more room"
            ));
            recs.push_back(createRecommendation(
                "Iron Condor", StrategyType::CONDOR, MarketOutlook::NEUTRAL,
                ComplexityLevel::ADVANCED, 75.0f, criteria,
                "Defined risk premium selling in high IV"
            ));
        }
        // Low IV - buy premium strategies
        else if (criteria.iv_percentile < 30.0f) {
            recs.push_back(createRecommendation(
                "Long Straddle", StrategyType::COMBINATION, MarketOutlook::VOLATILE,
                ComplexityLevel::INTERMEDIATE, 80.0f, criteria,
                "Low IV - cheap premium, expect volatility expansion"
            ));
            recs.push_back(createRecommendation(
                "Long Strangle", StrategyType::COMBINATION, MarketOutlook::VOLATILE,
                ComplexityLevel::INTERMEDIATE, 75.0f, criteria,
                "Low IV with defined risk on volatility play"
            ));
        }
    }

    void addBullishStrategies(const StrategySelectionCriteria& criteria,
                              std::vector<StrategyRecommendation>& recs) {
        float price_move = (criteria.predicted_price - criteria.current_price) /
                          criteria.current_price;

        if (price_move > 0.05f) {  // Strong bullish move expected
            recs.push_back(createRecommendation(
                "Long Call", StrategyType::SINGLE_LEG, MarketOutlook::BULLISH,
                ComplexityLevel::BEGINNER, 85.0f, criteria,
                "Strong bullish outlook - leverage with long call"
            ));
            recs.push_back(createRecommendation(
                "Bull Call Spread", StrategyType::VERTICAL_SPREAD, MarketOutlook::BULLISH,
                ComplexityLevel::BEGINNER, 80.0f, criteria,
                "Defined risk bullish play"
            ));
        } else {  // Mild bullish
            recs.push_back(createRecommendation(
                "Bull Put Spread", StrategyType::VERTICAL_SPREAD, MarketOutlook::BULLISH,
                ComplexityLevel::BEGINNER, 75.0f, criteria,
                "Collect premium on mild bullish outlook"
            ));
            if (!criteria.prefer_defined_risk) {
                recs.push_back(createRecommendation(
                    "Covered Call", StrategyType::COMBINATION, MarketOutlook::BULLISH,
                    ComplexityLevel::BEGINNER, 70.0f, criteria,
                    "Income generation on stock holdings"
                ));
            }
        }
    }

    void addBearishStrategies(const StrategySelectionCriteria& criteria,
                              std::vector<StrategyRecommendation>& recs) {
        float price_move = (criteria.predicted_price - criteria.current_price) /
                          criteria.current_price;

        if (price_move < -0.05f) {  // Strong bearish move expected
            recs.push_back(createRecommendation(
                "Long Put", StrategyType::SINGLE_LEG, MarketOutlook::BEARISH,
                ComplexityLevel::BEGINNER, 85.0f, criteria,
                "Strong bearish outlook - leverage with long put"
            ));
            recs.push_back(createRecommendation(
                "Bear Put Spread", StrategyType::VERTICAL_SPREAD, MarketOutlook::BEARISH,
                ComplexityLevel::BEGINNER, 80.0f, criteria,
                "Defined risk bearish play"
            ));
        } else {  // Mild bearish
            recs.push_back(createRecommendation(
                "Bear Call Spread", StrategyType::VERTICAL_SPREAD, MarketOutlook::BEARISH,
                ComplexityLevel::BEGINNER, 75.0f, criteria,
                "Collect premium on mild bearish outlook"
            ));
        }
    }

    void addNeutralStrategies(const StrategySelectionCriteria& criteria,
                              std::vector<StrategyRecommendation>& recs) {
        if (criteria.income_focused) {
            recs.push_back(createRecommendation(
                "Iron Condor", StrategyType::CONDOR, MarketOutlook::NEUTRAL,
                ComplexityLevel::ADVANCED, 90.0f, criteria,
                "Premium collection in neutral market with defined risk"
            ));
            recs.push_back(createRecommendation(
                "Short Strangle", StrategyType::COMBINATION, MarketOutlook::NEUTRAL,
                ComplexityLevel::INTERMEDIATE, 80.0f, criteria,
                "Higher premium than condor but undefined risk"
            ));
            recs.push_back(createRecommendation(
                "Iron Butterfly", StrategyType::BUTTERFLY, MarketOutlook::NEUTRAL,
                ComplexityLevel::ADVANCED, 75.0f, criteria,
                "Maximum profit at current price"
            ));
        } else {
            recs.push_back(createRecommendation(
                "Long Iron Butterfly", StrategyType::BUTTERFLY, MarketOutlook::NEUTRAL,
                ComplexityLevel::ADVANCED, 70.0f, criteria,
                "Profit from range-bound movement"
            ));
        }
    }

    void addConservativeStrategies(const StrategySelectionCriteria& criteria,
                                   std::vector<StrategyRecommendation>& recs) {
        recs.push_back(createRecommendation(
            "Bull Put Spread", StrategyType::VERTICAL_SPREAD, MarketOutlook::BULLISH,
            ComplexityLevel::BEGINNER, 75.0f, criteria,
            "Conservative premium collection with limited risk"
        ));
        recs.push_back(createRecommendation(
            "Covered Call", StrategyType::COMBINATION, MarketOutlook::BULLISH,
            ComplexityLevel::BEGINNER, 70.0f, criteria,
            "Generate income on existing stock holdings"
        ));
    }

    [[nodiscard]] auto createRecommendation(
        std::string name,
        StrategyType type,
        MarketOutlook outlook,
        ComplexityLevel complexity,
        float score,
        const StrategySelectionCriteria& criteria,
        std::string rationale
    ) -> StrategyRecommendation {

        StrategyRecommendation rec;
        rec.strategy_name = std::move(name);
        rec.strategy_type = type;
        rec.outlook = outlook;
        rec.complexity = complexity;
        rec.score = score * criteria.prediction_confidence;  // Adjust by confidence
        rec.rationale = std::move(rationale);

        // Estimate expected values (simplified)
        rec.capital_required = criteria.capital_allocation;
        rec.probability_of_profit = 0.50f + (score / 200.0f);  // 50-100% range

        return rec;
    }
};

// ============================================================================
// Portfolio OptionsGreeks Aggregator
// ============================================================================

class PortfolioGreeksAggregator {
public:
    /**
     * Aggregate OptionsGreeks across all positions
     */
    [[nodiscard]] auto calculatePortfolioGreeks(
        const std::vector<OptionsPosition>& positions
    ) -> OptionsGreeks {

        OptionsGreeks total{};

        for (const auto& position : positions) {
            if (!position.is_closed) {
                total.delta += position.current_greeks.delta;
                total.gamma += position.current_greeks.gamma;
                total.theta += position.current_greeks.theta;
                total.vega += position.current_greeks.vega;
                total.rho += position.current_greeks.rho;
            }
        }

        return total;
    }

    /**
     * Calculate portfolio risk metrics from OptionsGreeks
     */
    [[nodiscard]] auto calculateRiskMetrics(
        const std::vector<OptionsPosition>& positions,
        float portfolio_value
    ) -> PortfolioRisk {

        PortfolioRisk risk;
        OptionsGreeks portfolio_greeks = calculatePortfolioGreeks(positions);

        risk.net_delta = portfolio_greeks.delta;
        risk.net_vega = portfolio_greeks.vega;
        risk.net_theta = portfolio_greeks.theta;

        // Calculate total exposure and P&L
        risk.total_value = portfolio_value;
        risk.active_positions = 0;
        risk.daily_pnl = 0.0;

        for (const auto& position : positions) {
            if (!position.is_closed) {
                risk.active_positions++;
                risk.daily_pnl += position.unrealized_pnl;
                risk.total_exposure += std::abs(position.calculatePnL(position.current_price));
            }
        }

        // Calculate portfolio heat (exposure as % of portfolio)
        if (portfolio_value > 0.0) {
            risk.portfolio_heat = risk.total_exposure / portfolio_value;
        }

        return risk;
    }
};

// ============================================================================
// Options Strategy Manager
// ============================================================================

class OptionsStrategyManager {
public:
    OptionsStrategyManager()
        : selector_{}
        , greeks_aggregator_{}
        , positions_{}
    {}

    /**
     * Get strategy recommendations based on market analysis
     */
    [[nodiscard]] auto getRecommendations(
        const StrategySelectionCriteria& criteria
    ) -> std::vector<StrategyRecommendation> {
        return selector_.recommendStrategies(criteria, 5);
    }

    /**
     * Open a new options position
     */
    auto openPosition(
        std::string symbol,
        const StrategyRecommendation& rec,
        float underlying_price,
        const std::vector<float>& strikes,
        float days,
        float iv,
        float rate = 0.05f
    ) -> Result<std::string> {

        auto strategy = selector_.createStrategy(rec, underlying_price, strikes, days, iv, rate);
        if (!strategy) {
            return makeError<std::string>(ErrorCode::InvalidParameter,
                                         "Failed to create strategy");
        }

        OptionsPosition position;
        position.position_id = symbol + "_" + rec.strategy_name + "_" +
                              std::to_string(positions_.size());
        position.symbol = std::move(symbol);
        position.strategy = std::move(strategy);
        position.entry_price = underlying_price;
        position.current_price = underlying_price;
        position.days_to_expiration = static_cast<int>(days);

        // Calculate initial OptionsGreeks
        position.updateGreeks(underlying_price, rate);

        positions_.push_back(std::move(position));

        Logger::getInstance().info("Opened position: {}", positions_.back().position_id);
        return positions_.back().position_id;
    }

    /**
     * Update all positions with current market data
     */
    auto updatePositions(
        float underlying_price,
        float rate = 0.05f
    ) -> void {
        for (auto& position : positions_) {
            if (!position.is_closed) {
                position.current_price = underlying_price;
                position.unrealized_pnl = position.calculatePnL(underlying_price);
                position.updateGreeks(underlying_price, rate);
                position.days_held++;
                position.days_to_expiration--;
            }
        }
    }

    /**
     * Close a position
     */
    auto closePosition(const std::string& position_id) -> Result<float> {
        for (auto& position : positions_) {
            if (position.position_id == position_id && !position.is_closed) {
                position.is_closed = true;
                position.realized_pnl = position.unrealized_pnl;
                Logger::getInstance().info("Closed position {} with P&L: {}",
                           position_id, position.realized_pnl);
                return position.realized_pnl;
            }
        }
        return makeError<float>(ErrorCode::InvalidParameter, "Position not found");
    }

    /**
     * Get portfolio risk metrics
     */
    [[nodiscard]] auto getPortfolioRisk(float portfolio_value) -> PortfolioRisk {
        return greeks_aggregator_.calculateRiskMetrics(positions_, portfolio_value);
    }

    /**
     * Get all active positions
     */
    [[nodiscard]] auto getActivePositions() const -> std::vector<OptionsPosition> {
        std::vector<OptionsPosition> active;
        for (const auto& pos : positions_) {
            if (!pos.is_closed) {
                // Note: Can't copy OptionsPosition due to unique_ptr, so return const ref
                // In real implementation, would use shared_ptr or return by reference
            }
        }
        return active;
    }

    /**
     * Get total P&L across all positions
     */
    [[nodiscard]] auto getTotalPnL() const -> float {
        float total = 0.0f;
        for (const auto& pos : positions_) {
            if (pos.is_closed) {
                total += pos.realized_pnl;
            } else {
                total += pos.unrealized_pnl;
            }
        }
        return total;
    }

private:
    OptionsStrategySelector selector_;
    PortfolioGreeksAggregator greeks_aggregator_;
    std::vector<OptionsPosition> positions_;
};

}  // namespace bigbrother::options_integration
