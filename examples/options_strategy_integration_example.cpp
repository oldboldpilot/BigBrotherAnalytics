/**
 * @file options_strategy_integration_example.cpp
 * @brief Example of integrating 52 options strategies with trading engine
 *
 * Demonstrates:
 * - Strategy selection based on market conditions
 * - Risk management with Greeks monitoring
 * - P&L tracking across positions
 * - Integration with price prediction
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 */

#include <iostream>
#include <vector>
#include <iomanip>

import bigbrother.options_strategy_integrator;
import bigbrother.options_strategies.base;
import bigbrother.utils.logger;
import bigbrother.risk_management;

using namespace bigbrother::options_integration;
using namespace bigbrother::risk;
using namespace bigbrother::utils;

// ============================================================================
// Example 1: Strategy Selection Based on Market Outlook
// ============================================================================

auto demonstrateStrategySelection() -> void {
    std::cout << "\n=== Example 1: Strategy Selection ===\n\n";

    OptionsStrategyManager manager;

    // Scenario: High volatility environment, neutral outlook
    StrategySelectionCriteria criteria;
    criteria.market_outlook = MarketCondition::HIGH_VOLATILITY;
    criteria.current_iv = 0.45f;           // 45% IV (high)
    criteria.historical_iv = 0.25f;        // Historical average 25%
    criteria.iv_percentile = 85.0f;        // 85th percentile (very high)
    criteria.current_price = 450.0f;       // SPY at $450
    criteria.predicted_price = 452.0f;     // Mild bullish prediction
    criteria.prediction_confidence = 0.65f; // 65% confidence
    criteria.time_horizon_days = 30.0f;    // 30 days to expiration
    criteria.max_loss_tolerance = 2000.0f; // Max $2000 loss
    criteria.capital_allocation = 5000.0f; // $5000 to allocate
    criteria.prefer_defined_risk = true;
    criteria.income_focused = true;

    // Get recommendations
    auto recommendations = manager.getRecommendations(criteria);

    std::cout << "Market Conditions:\n";
    std::cout << "  Current Price: $" << criteria.current_price << "\n";
    std::cout << "  Current IV: " << (criteria.current_iv * 100) << "%\n";
    std::cout << "  IV Percentile: " << criteria.iv_percentile << "%\n";
    std::cout << "  Prediction: $" << criteria.predicted_price
              << " (confidence: " << (criteria.prediction_confidence * 100) << "%)\n\n";

    std::cout << "Top Strategy Recommendations:\n";
    for (size_t i = 0; i < recommendations.size(); ++i) {
        const auto& rec = recommendations[i];
        std::cout << (i + 1) << ". " << rec.strategy_name
                  << " (Score: " << std::fixed << std::setprecision(1) << rec.score << "/100)\n";
        std::cout << "   Rationale: " << rec.rationale << "\n";
        std::cout << "   Risk/Reward: " << rec.getRiskRewardRatio() << ":1\n";
        std::cout << "   Probability of Profit: "
                  << (rec.probability_of_profit * 100) << "%\n\n";
    }
}

// ============================================================================
// Example 2: Opening and Managing Positions
// ============================================================================

auto demonstratePositionManagement() -> void {
    std::cout << "\n=== Example 2: Position Management ===\n\n";

    OptionsStrategyManager manager;

    // Open an Iron Condor position (popular neutral strategy)
    StrategySelectionCriteria criteria;
    criteria.market_outlook = MarketCondition::NEUTRAL;
    criteria.current_price = 450.0f;
    criteria.current_iv = 0.30f;
    criteria.income_focused = true;

    auto recommendations = manager.getRecommendations(criteria);
    if (recommendations.empty()) {
        std::cout << "No recommendations available\n";
        return;
    }

    // Take the top recommendation (Iron Condor)
    const auto& top_rec = recommendations[0];
    std::cout << "Opening position: " << top_rec.strategy_name << "\n\n";

    // Define strikes for Iron Condor: 440/445/455/460
    std::vector<float> strikes = {440.0f, 445.0f, 455.0f, 460.0f};
    float underlying_price = 450.0f;
    float days = 30.0f;
    float iv = 0.30f;
    float rate = 0.05f;

    auto result = manager.openPosition(
        "SPY",
        top_rec,
        underlying_price,
        strikes,
        days,
        iv,
        rate
    );

    if (result.has_value()) {
        std::cout << "Position opened successfully: " << result.value() << "\n\n";

        // Simulate market moves and position updates
        std::cout << "Day-by-day P&L and Greeks:\n";
        std::cout << std::setw(5) << "Day"
                  << std::setw(10) << "Price"
                  << std::setw(12) << "P&L"
                  << std::setw(10) << "Delta"
                  << std::setw(10) << "Theta"
                  << std::setw(10) << "Vega" << "\n";
        std::cout << std::string(65, '-') << "\n";

        // Simulate 10 days of price movement
        for (int day = 0; day < 10; ++day) {
            // Price oscillates around 450
            float price = 450.0f + 3.0f * std::sin(day * 0.5f);

            manager.updatePositions(price, rate);
            auto portfolio_risk = manager.getPortfolioRisk(50000.0f);  // $50k portfolio

            std::cout << std::setw(5) << day
                      << std::setw(10) << std::fixed << std::setprecision(2) << price
                      << std::setw(12) << std::fixed << std::setprecision(2) << portfolio_risk.daily_pnl
                      << std::setw(10) << std::fixed << std::setprecision(4) << portfolio_risk.net_delta
                      << std::setw(10) << std::fixed << std::setprecision(2) << portfolio_risk.net_theta
                      << std::setw(10) << std::fixed << std::setprecision(2) << portfolio_risk.net_vega
                      << "\n";
        }

        // Close position
        std::cout << "\nClosing position...\n";
        auto close_result = manager.closePosition(result.value());
        if (close_result.has_value()) {
            std::cout << "Position closed. Realized P&L: $"
                      << std::fixed << std::setprecision(2) << close_result.value() << "\n";
        }
    } else {
        std::cout << "Failed to open position: " << result.error().message << "\n";
    }
}

// ============================================================================
// Example 3: Portfolio Risk Management
// ============================================================================

auto demonstrateRiskManagement() -> void {
    std::cout << "\n=== Example 3: Portfolio Risk Management ===\n\n";

    OptionsStrategyManager manager;

    // Open multiple positions with different strategies
    float spy_price = 450.0f;
    float qqq_price = 380.0f;
    float iwm_price = 190.0f;

    std::cout << "Opening diverse portfolio of options positions:\n\n";

    // Position 1: SPY Iron Condor (neutral, income)
    {
        StrategySelectionCriteria criteria;
        criteria.market_outlook = MarketCondition::NEUTRAL;
        criteria.current_price = spy_price;
        criteria.income_focused = true;

        auto recs = manager.getRecommendations(criteria);
        if (!recs.empty()) {
            std::vector<float> strikes = {440.0f, 445.0f, 455.0f, 460.0f};
            auto result = manager.openPosition("SPY", recs[0], spy_price, strikes,
                                              30.0f, 0.30f);
            if (result.has_value()) {
                std::cout << "✓ Opened SPY Iron Condor\n";
            }
        }
    }

    // Position 2: QQQ Bull Call Spread (bullish)
    {
        StrategySelectionCriteria criteria;
        criteria.market_outlook = MarketCondition::BULLISH;
        criteria.current_price = qqq_price;
        criteria.predicted_price = 390.0f;
        criteria.prefer_defined_risk = true;

        auto recs = manager.getRecommendations(criteria);
        if (!recs.empty()) {
            std::vector<float> strikes = {375.0f, 385.0f};
            auto result = manager.openPosition("QQQ", recs[0], qqq_price, strikes,
                                              30.0f, 0.35f);
            if (result.has_value()) {
                std::cout << "✓ Opened QQQ Bull Call Spread\n";
            }
        }
    }

    // Position 3: IWM Bear Put Spread (bearish)
    {
        StrategySelectionCriteria criteria;
        criteria.market_outlook = MarketCondition::BEARISH;
        criteria.current_price = iwm_price;
        criteria.predicted_price = 185.0f;
        criteria.prefer_defined_risk = true;

        auto recs = manager.getRecommendations(criteria);
        if (!recs.empty()) {
            std::vector<float> strikes = {185.0f, 195.0f};
            auto result = manager.openPosition("IWM", recs[0], iwm_price, strikes,
                                              30.0f, 0.40f);
            if (result.has_value()) {
                std::cout << "✓ Opened IWM Bear Put Spread\n";
            }
        }
    }

    std::cout << "\nPortfolio Risk Metrics:\n";
    std::cout << std::string(60, '-') << "\n";

    float portfolio_value = 50000.0f;  // $50k portfolio
    auto risk = manager.getPortfolioRisk(portfolio_value);

    std::cout << "Total Portfolio Value: $" << std::fixed << std::setprecision(2)
              << risk.total_value << "\n";
    std::cout << "Active Positions: " << risk.active_positions << "\n";
    std::cout << "Total Exposure: $" << risk.total_exposure << "\n";
    std::cout << "Portfolio Heat: " << (risk.portfolio_heat * 100) << "%\n";
    std::cout << "Risk Level: " << risk.getRiskLevel() << "\n\n";

    std::cout << "Portfolio Greeks:\n";
    std::cout << "  Net Delta: " << std::fixed << std::setprecision(4) << risk.net_delta << "\n";
    std::cout << "  Net Theta: $" << std::fixed << std::setprecision(2) << risk.net_theta << "/day\n";
    std::cout << "  Net Vega: $" << std::fixed << std::setprecision(2) << risk.net_vega << " per 1% IV\n\n";

    std::cout << "Daily P&L: $" << std::fixed << std::setprecision(2) << risk.daily_pnl << "\n";

    // Check risk limits
    RiskLimits limits = RiskLimits::forThirtyKAccount();
    if (risk.portfolio_heat > limits.max_portfolio_heat) {
        std::cout << "\n⚠️ WARNING: Portfolio heat exceeds limit!\n";
        std::cout << "   Current: " << (risk.portfolio_heat * 100) << "%\n";
        std::cout << "   Limit: " << (limits.max_portfolio_heat * 100) << "%\n";
    } else {
        std::cout << "\n✓ Portfolio within risk limits\n";
    }
}

// ============================================================================
// Example 4: Strategy Selection with Price Prediction
// ============================================================================

auto demonstratePricePredictionIntegration() -> void {
    std::cout << "\n=== Example 4: Price Prediction Integration ===\n\n";

    OptionsStrategyManager manager;

    // Simulate price prediction from ML model
    float current_price = 450.0f;
    float predicted_price_5d = 455.0f;   // +5 days: +1.1%
    float predicted_price_30d = 462.0f;  // +30 days: +2.7%
    float confidence = 0.70f;            // 70% confidence

    std::cout << "Price Prediction:\n";
    std::cout << "  Current: $" << current_price << "\n";
    std::cout << "  5-day prediction: $" << predicted_price_5d
              << " (+" << ((predicted_price_5d / current_price - 1.0f) * 100) << "%)\n";
    std::cout << "  30-day prediction: $" << predicted_price_30d
              << " (+" << ((predicted_price_30d / current_price - 1.0f) * 100) << "%)\n";
    std::cout << "  Confidence: " << (confidence * 100) << "%\n\n";

    // Use prediction to select strategy
    StrategySelectionCriteria criteria;
    criteria.market_outlook = MarketCondition::BULLISH;  // Based on prediction
    criteria.current_price = current_price;
    criteria.predicted_price = predicted_price_30d;
    criteria.prediction_confidence = confidence;
    criteria.time_horizon_days = 30.0f;

    auto recommendations = manager.getRecommendations(criteria);

    std::cout << "Recommended Strategies (adjusted by prediction confidence):\n";
    for (size_t i = 0; i < std::min(recommendations.size(), size_t(3)); ++i) {
        const auto& rec = recommendations[i];
        std::cout << (i + 1) << ". " << rec.strategy_name
                  << " (Score: " << rec.score << "/100)\n";
        std::cout << "   Expected move aligns with "
                  << ((rec.outlook == MarketOutlook::BULLISH) ? "bullish" : "neutral")
                  << " outlook\n";
    }
}

// ============================================================================
// Main Example Runner
// ============================================================================

auto main() -> int {
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Options Strategy Integration Examples                   ║\n";
    std::cout << "║   52 Strategies × Trading Engine × Risk Management        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    try {
        // Run examples
        demonstrateStrategySelection();
        demonstratePositionManagement();
        demonstrateRiskManagement();
        demonstratePricePredictionIntegration();

        std::cout << "\n✓ All examples completed successfully!\n\n";
        std::cout << "Key Takeaways:\n";
        std::cout << "• Strategy selection adapts to market conditions\n";
        std::cout << "• Greeks monitoring provides real-time risk metrics\n";
        std::cout << "• P&L tracking across multiple positions\n";
        std::cout << "• Integration with price prediction for informed decisions\n";
        std::cout << "• Portfolio heat and risk limits enforced automatically\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
