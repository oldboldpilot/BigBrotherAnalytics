/**
 * Examples: Fluent Risk Manager API
 *
 * Demonstrates practical usage of the new fluent API pattern
 * for risk management following Schwab design philosophy.
 *
 * Compile with:
 * g++ -std=c++23 -fmodules-ts fluent_risk_examples.cpp -o fluent_examples
 */

#include <iostream>
#include <vector>
#include <iomanip>

import bigbrother.risk_management;

using namespace bigbrother::risk;
using namespace bigbrother::types;

// ============================================================================
// Example 1: Basic Trade Assessment
// ============================================================================

void example_basic_trade_assessment() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 1: Basic Trade Assessment\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;

    // Assess a simple stock trade
    auto risk = risk_mgr.assessTrade()
        .forSymbol("SPY")
        .withQuantity(10)
        .atPrice(450.00)
        .withStop(440.00)
        .withTarget(465.00)
        .withProbability(0.65)
        .assess();

    if (risk) {
        std::cout << "Trade: SPY\n";
        std::cout << "  Position Size: $" << std::fixed << std::setprecision(2)
                  << risk->position_size << "\n";
        std::cout << "  Max Loss: $" << risk->max_loss << "\n";
        std::cout << "  Expected Return: $" << risk->expected_return << "\n";
        std::cout << "  Expected Value: $" << risk->expected_value << "\n";
        std::cout << "  Risk/Reward Ratio: " << risk->risk_reward_ratio << "\n";
        std::cout << "  Kelly Fraction: " << risk->kelly_fraction << "\n";
        std::cout << "  Status: " << (risk->approved ? "APPROVED" : "REJECTED") << "\n";
        if (!risk->approved) {
            std::cout << "  Reason: " << risk->rejection_reason << "\n";
        }
    } else {
        std::cerr << "Error: " << risk.error().message << "\n";
    }
}

// ============================================================================
// Example 2: Configuration and Fluent Setup
// ============================================================================

void example_configuration() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 2: Fluent Configuration\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;

    // Configure with fluent chaining
    risk_mgr.setMaxDailyLoss(1000.0)
        .setMaxPositionSize(2000.0)
        .setMaxPortfolioHeat(0.20)
        .setMaxConcurrentPositions(15)
        .setAccountValue(50000.0)
        .requireStopLoss(true);

    std::cout << "Risk Manager configured:\n";
    std::cout << "  Max Daily Loss: $1000.00\n";
    std::cout << "  Max Position Size: $2000.00\n";
    std::cout << "  Max Portfolio Heat: 20%\n";
    std::cout << "  Max Concurrent Positions: 15\n";
    std::cout << "  Account Value: $50,000.00\n";
    std::cout << "  Stop Loss Required: Yes\n";
}

// ============================================================================
// Example 3: Portfolio Analysis
// ============================================================================

void example_portfolio_analysis() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 3: Portfolio Analysis\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;

    // Build portfolio with multiple positions
    auto portfolio = risk_mgr.portfolio()
        .addPosition("SPY", 10, 450.00, 0.05)   // Equities: S&P 500
        .addPosition("QQQ", 15, 350.00, 0.08)   // Equities: Nasdaq
        .addPosition("XLE", 20, 80.00, 0.10)    // Energy
        .addPosition("TLT", 50, 100.00, 0.03)   // Bonds
        .addPosition("GLD", 30, 200.00, 0.04)   // Gold
        .calculateHeat()
        .calculateVaR(0.95)
        .analyze();

    if (portfolio) {
        std::cout << "Portfolio Summary:\n";
        std::cout << "  Active Positions: " << portfolio->active_positions << "\n";
        std::cout << "  Total Value: $" << std::fixed << std::setprecision(2)
                  << portfolio->total_value << "\n";
        std::cout << "  Total Exposure: $" << portfolio->total_exposure << "\n";
        std::cout << "  Portfolio Heat: " << std::setprecision(1) << std::fixed
                  << (portfolio->portfolio_heat * 100) << "%\n";
        std::cout << "  Risk Level: " << portfolio->getRiskLevel() << "\n";
        std::cout << "  Daily P&L: $" << std::setprecision(2) << portfolio->daily_pnl << "\n";
        std::cout << "  Loss Remaining: $" << portfolio->daily_loss_remaining << "\n";
    }
}

// ============================================================================
// Example 4: Kelly Criterion Calculator
// ============================================================================

void example_kelly_criterion() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 4: Kelly Criterion Calculator\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;

    // Test different scenarios
    struct Scenario {
        const char* name;
        double win_rate;
        double win_loss_ratio;
    };

    std::vector<Scenario> scenarios = {
        {"Conservative (50% WR, 1:1)", 0.50, 1.0},
        {"Balanced (55% WR, 1.5:1)", 0.55, 1.5},
        {"Aggressive (60% WR, 2:1)", 0.60, 2.0},
        {"Very Aggressive (70% WR, 3:1)", 0.70, 3.0},
    };

    for (auto const& scenario : scenarios) {
        auto kelly = risk_mgr.kelly()
            .withWinRate(scenario.win_rate)
            .withWinLossRatio(scenario.win_loss_ratio)
            .calculate();

        if (kelly) {
            std::cout << scenario.name << "\n";
            std::cout << "  Kelly Fraction: " << std::setprecision(3) << std::fixed
                      << *kelly << " (" << (*kelly * 100) << "%)\n";
            std::cout << "  Half Kelly: " << (*kelly * 0.5) << "\n\n";
        }
    }
}

// ============================================================================
// Example 5: Position Sizing Methods
// ============================================================================

void example_position_sizing() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 5: Position Sizing Methods\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;

    struct SizingScenario {
        const char* name;
        SizingMethod method;
    };

    std::vector<SizingScenario> methods = {
        {"Fixed Dollar", SizingMethod::FixedDollar},
        {"Fixed Percent", SizingMethod::FixedPercent},
        {"Kelly Criterion", SizingMethod::KellyCriterion},
        {"Kelly Half", SizingMethod::KellyHalf},
        {"Volatility Adjusted", SizingMethod::VolatilityAdjusted},
        {"Risk Parity", SizingMethod::RiskParity},
    };

    for (auto const& scenario : methods) {
        auto size = risk_mgr.positionSizer()
            .withMethod(scenario.method)
            .withAccountValue(30000.0)
            .withWinProbability(0.58)
            .withWinAmount(150.0)
            .withLossAmount(100.0)
            .withVolatility(0.25)
            .calculate();

        if (size) {
            std::cout << scenario.name << ": $"
                      << std::setprecision(2) << std::fixed << *size << "\n";
        }
    }
}

// ============================================================================
// Example 6: Daily P&L Tracking
// ============================================================================

void example_daily_pnl_tracking() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 6: Daily P&L Tracking\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;
    risk_mgr.setMaxDailyLoss(900.0);

    // Simulate a trading day
    std::vector<double> trades = {500.0, -200.0, 300.0, -150.0, 600.0};

    std::cout << "Trading Day Simulation:\n";
    std::cout << "  Max Daily Loss Allowed: $900.00\n\n";

    for (size_t i = 0; i < trades.size(); ++i) {
        risk_mgr.updateDailyPnL(trades[i]);

        std::cout << "Trade " << (i + 1) << ": " << (trades[i] > 0 ? "+" : "")
                  << std::setprecision(2) << std::fixed << trades[i] << "\n";
        std::cout << "  Cumulative P&L: $" << risk_mgr.getDailyPnL() << "\n";
        std::cout << "  Loss Remaining: $" << risk_mgr.getDailyLossRemaining() << "\n";
        std::cout << "  Limit Reached: " << (risk_mgr.isDailyLossLimitReached() ? "YES" : "NO")
                  << "\n\n";
    }

    // Reset for tomorrow
    std::cout << "Day Summary: $" << std::setprecision(2) << std::fixed
              << risk_mgr.getDailyPnL() << "\n";
    std::cout << "Resetting for tomorrow...\n";

    risk_mgr.resetDaily();
    std::cout << "New Daily P&L: $" << risk_mgr.getDailyPnL() << "\n";
}

// ============================================================================
// Example 7: Multiple Trade Workflow
// ============================================================================

void example_multiple_trades() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 7: Multiple Trade Workflow\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;
    risk_mgr.setMaxDailyLoss(1000.0).setMaxPositionSize(2000.0);

    struct TradeSetup {
        const char* symbol;
        int quantity;
        double price;
        double stop;
        double target;
        double win_prob;
    };

    std::vector<TradeSetup> trades = {
        {"SPY", 10, 450.0, 440.0, 465.0, 0.65},
        {"QQQ", 15, 350.0, 340.0, 370.0, 0.60},
        {"XLE", 20, 80.0, 75.0, 90.0, 0.55},
    };

    int approved = 0, rejected = 0;

    std::cout << "Evaluating " << trades.size() << " trades:\n\n";

    for (auto const& trade : trades) {
        auto result = risk_mgr.assessTrade()
            .forSymbol(trade.symbol)
            .withQuantity(trade.quantity)
            .atPrice(trade.price)
            .withStop(trade.stop)
            .withTarget(trade.target)
            .withProbability(trade.win_prob)
            .assess();

        if (result) {
            std::cout << trade.symbol << ": "
                      << (result->approved ? "APPROVED" : "REJECTED") << "\n";
            if (result->approved) {
                approved++;
                std::cout << "  Position: $" << std::setprecision(2) << std::fixed
                          << result->position_size << "\n";
                std::cout << "  Max Loss: $" << result->max_loss << "\n";
                std::cout << "  Expected Value: $" << result->expected_value << "\n";
            } else {
                rejected++;
                std::cout << "  Reason: " << result->rejection_reason << "\n";
            }
            std::cout << "\n";
        }
    }

    std::cout << "Summary: " << approved << " approved, " << rejected << " rejected\n";

    // Analyze combined portfolio
    auto portfolio = risk_mgr.portfolio()
        .addPosition("SPY", 10, 450.0)
        .addPosition("QQQ", 15, 350.0)
        .addPosition("XLE", 20, 80.0)
        .calculateHeat()
        .analyze();

    if (portfolio) {
        std::cout << "\nPortfolio Heat: " << std::setprecision(1) << std::fixed
                  << (portfolio->portfolio_heat * 100) << "%\n";
        std::cout << "Total Value: $" << std::setprecision(2) << portfolio->total_value << "\n";
    }
}

// ============================================================================
// Example 8: Risk Management Strategy
// ============================================================================

void example_risk_strategy() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 8: Integrated Risk Strategy\n";
    std::cout << std::string(70, '=') << "\n";

    RiskManager risk_mgr;

    // Step 1: Configure for the day
    std::cout << "Step 1: Configure Risk Parameters\n";
    risk_mgr.setMaxDailyLoss(900.0)
        .setMaxPositionSize(1500.0)
        .setMaxPortfolioHeat(0.15)
        .setMaxConcurrentPositions(10)
        .requireStopLoss(true);
    std::cout << "  Risk Manager configured\n\n";

    // Step 2: Evaluate trade opportunity
    std::cout << "Step 2: Evaluate Trade\n";
    auto risk = risk_mgr.assessTrade()
        .forSymbol("AAPL")
        .withQuantity(20)
        .atPrice(150.00)
        .withStop(145.00)
        .withTarget(165.00)
        .withProbability(0.62)
        .assess();

    std::cout << "  Trade assessed\n";
    if (risk && risk->approved) {
        std::cout << "  Status: APPROVED\n\n";

        // Step 3: Calculate position sizing
        std::cout << "Step 3: Calculate Position Size\n";
        auto kelly = risk_mgr.kelly()
            .withWinRate(0.62)
            .withWinLossRatio(2.0)
            .calculate();

        std::cout << "  Kelly fraction: " << std::setprecision(3) << *kelly << "\n";

        auto size = risk_mgr.positionSizer()
            .withMethod(SizingMethod::KellyHalf)
            .withAccountValue(30000.0)
            .withWinProbability(0.62)
            .withWinAmount(200.0)
            .withLossAmount(100.0)
            .calculate();

        if (size) {
            std::cout << "  Position size: $" << std::setprecision(2) << std::fixed << *size << "\n\n";
        }

        // Step 4: Execute and track
        std::cout << "Step 4: Simulate Trade Execution\n";
        std::cout << "  Trade executed at $150.00\n";
        std::cout << "  Stop loss set at $145.00\n";
        std::cout << "  Target set at $165.00\n\n";

        // Step 5: Analyze portfolio
        std::cout << "Step 5: Portfolio Analysis\n";
        auto portfolio = risk_mgr.portfolio()
            .addPosition("AAPL", 20, 150.00, 0.20)
            .calculateHeat()
            .analyze();

        if (portfolio) {
            std::cout << "  Portfolio value: $" << std::setprecision(2)
                      << portfolio->total_value << "\n";
            std::cout << "  Portfolio heat: " << std::setprecision(1) << std::fixed
                      << (portfolio->portfolio_heat * 100) << "%\n";
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main() {
    std::cout << "BigBrotherAnalytics - Fluent Risk Manager API Examples\n";
    std::cout << "========================================================\n";

    try {
        example_basic_trade_assessment();
        example_configuration();
        example_portfolio_analysis();
        example_kelly_criterion();
        example_position_sizing();
        example_daily_pnl_tracking();
        example_multiple_trades();
        example_risk_strategy();

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << std::string(70, '=') << "\n";
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
