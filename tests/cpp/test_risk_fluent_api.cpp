/**
 * Unit Tests for RiskManager Fluent API
 *
 * Tests fluent method chaining patterns following Schwab API design.
 * Validates:
 * - Fluent configuration methods
 * - Builder patterns (TradeRiskBuilder, PortfolioRiskBuilder)
 * - Specialized calculators (Kelly, Monte Carlo, Position Sizer)
 * - Method chaining consistency
 * - Terminal operations
 */

#include <gtest/gtest.h>
#include <cmath>

// Note: When compiled as module, include the compiled .cppm file
// For now, we'll test against the public API definitions
import bigbrother.risk_management;

using namespace bigbrother::risk;
using namespace bigbrother::types;

// Test fixtures
class RiskManagerFluentTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize with standard 30k account limits
        risk_mgr = std::make_unique<RiskManager>(RiskLimits::forThirtyKAccount());
    }

    std::unique_ptr<RiskManager> risk_mgr;
};

// ============================================================================
// Fluent Configuration Tests
// ============================================================================

TEST_F(RiskManagerFluentTest, FluentConfigurationChaining) {
    // Test method chaining for configuration
    auto& result = risk_mgr->setMaxDailyLoss(1000.0)
                       .setMaxPositionSize(2000.0)
                       .setMaxPortfolioHeat(0.20)
                       .setMaxConcurrentPositions(15);

    EXPECT_EQ(&result, risk_mgr.get()) << "Fluent methods should return reference to self";
}

TEST_F(RiskManagerFluentTest, SetMaxDailyLoss) {
    risk_mgr->setMaxDailyLoss(1200.0);

    // Verify the limit was set correctly
    // (Would need getter methods or internal inspection)
    auto portfolio = risk_mgr->getPortfolioRisk();
    ASSERT_TRUE(portfolio.has_value());
}

TEST_F(RiskManagerFluentTest, SetMaxPositionSize) {
    risk_mgr->setMaxPositionSize(2500.0);
    // Configuration should succeed without throwing
}

TEST_F(RiskManagerFluentTest, SetMaxPortfolioHeat) {
    risk_mgr->setMaxPortfolioHeat(0.25);
    // Configuration should succeed
}

TEST_F(RiskManagerFluentTest, SetAccountValue) {
    risk_mgr->setAccountValue(50000.0);
    // Configuration should succeed
}

TEST_F(RiskManagerFluentTest, RequireStopLoss) {
    risk_mgr->requireStopLoss(true);
    risk_mgr->requireStopLoss(false);
    // Configuration should succeed
}

TEST_F(RiskManagerFluentTest, ComplexFluentConfiguration) {
    // Chain multiple configuration methods
    auto& config = risk_mgr->setAccountValue(50000.0)
                       .setMaxDailyLoss(1500.0)
                       .setMaxPositionSize(2500.0)
                       .setMaxPortfolioHeat(0.20)
                       .setMaxConcurrentPositions(20)
                       .requireStopLoss(true);

    EXPECT_EQ(&config, risk_mgr.get());
}

// ============================================================================
// Trade Risk Builder Tests (Fluent API)
// ============================================================================

TEST_F(RiskManagerFluentTest, TradeRiskBuilderChaining) {
    // Test that assessTrade() returns builder and methods chain
    auto builder = risk_mgr->assessTrade();
    auto result = builder.forSymbol("SPY")
                      .withQuantity(10)
                      .atPrice(450.00)
                      .withStop(440.00)
                      .withTarget(465.00)
                      .withProbability(0.65)
                      .assess();

    ASSERT_TRUE(result.has_value()) << "Trade assessment should succeed";
    EXPECT_TRUE(result->approved) << "Trade should be approved";
    EXPECT_GT(result->position_size, 0.0) << "Position size should be positive";
}

TEST_F(RiskManagerFluentTest, TradeRiskBuilderSymbolOnly) {
    // Test with minimal configuration
    auto result = risk_mgr->assessTrade()
                      .forSymbol("AAPL")
                      .atPrice(150.00)
                      .withStop(145.00)
                      .withTarget(160.00)
                      .assess();

    ASSERT_TRUE(result.has_value());
}

TEST_F(RiskManagerFluentTest, TradeRiskBuilderMultipleSymbols) {
    // Test different symbols
    auto spy_result = risk_mgr->assessTrade()
                          .forSymbol("SPY")
                          .withQuantity(10)
                          .atPrice(450.00)
                          .withStop(440.00)
                          .withTarget(465.00)
                          .withProbability(0.60)
                          .assess();

    ASSERT_TRUE(spy_result.has_value());

    auto qqq_result = risk_mgr->assessTrade()
                          .forSymbol("QQQ")
                          .withQuantity(20)
                          .atPrice(350.00)
                          .withStop(340.00)
                          .withTarget(370.00)
                          .withProbability(0.55)
                          .assess();

    ASSERT_TRUE(qqq_result.has_value());
}

TEST_F(RiskManagerFluentTest, TradeRiskBuilderHighProbability) {
    auto result = risk_mgr->assessTrade()
                      .forSymbol("XLE")
                      .withQuantity(50)
                      .atPrice(80.00)
                      .withStop(75.00)
                      .withTarget(90.00)
                      .withProbability(0.75)
                      .assess();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->win_probability, 0.75);
    EXPECT_GT(result->expected_value, 0.0) << "Expected value should be positive with 75% probability";
}

TEST_F(RiskManagerFluentTest, TradeRiskBuilderLowProbability) {
    auto result = risk_mgr->assessTrade()
                      .forSymbol("GLD")
                      .withQuantity(30)
                      .atPrice(200.00)
                      .withStop(190.00)
                      .withTarget(215.00)
                      .withProbability(0.45)
                      .assess();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->win_probability, 0.45);
}

// ============================================================================
// Portfolio Risk Builder Tests
// ============================================================================

TEST_F(RiskManagerFluentTest, PortfolioRiskBuilderChaining) {
    auto result = risk_mgr->portfolio()
                      .addPosition("SPY", 10, 450.00, 0.05)
                      .addPosition("XLE", 20, 80.00, 0.08)
                      .calculateHeat()
                      .calculateVaR(0.95)
                      .analyze();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->total_value, 0.0) << "Portfolio value should be calculated";
    EXPECT_GT(result->active_positions, 0) << "Should have active positions";
}

TEST_F(RiskManagerFluentTest, PortfolioRiskBuilderMultiplePositions) {
    auto result = risk_mgr->portfolio()
                      .addPosition("SPY", 10, 450.00, 0.05)
                      .addPosition("QQQ", 15, 350.00, 0.08)
                      .addPosition("XLE", 20, 80.00, 0.10)
                      .addPosition("GLD", 30, 200.00, 0.04)
                      .calculateHeat()
                      .analyze();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->active_positions, 4) << "Should have 4 positions";
    // SPY: 10*450 + QQQ: 15*350 + XLE: 20*80 + GLD: 30*200 = 4500+5250+1600+6000 = 17350
    EXPECT_NEAR(result->total_value, 17350.0, 0.01);
}

TEST_F(RiskManagerFluentTest, PortfolioRiskBuilderCalculateHeat) {
    auto result = risk_mgr->portfolio()
                      .addPosition("SPY", 10, 450.00)
                      .addPosition("XLE", 20, 80.00)
                      .calculateHeat()
                      .analyze();

    ASSERT_TRUE(result.has_value());
    // Heat = total_value / 30000
    // total_value = 10*450 + 20*80 = 4500 + 1600 = 6100
    // heat = 6100 / 30000 = 0.203...
    EXPECT_GT(result->portfolio_heat, 0.0);
}

TEST_F(RiskManagerFluentTest, PortfolioRiskBuilderCalculateVaR) {
    auto result = risk_mgr->portfolio()
                      .addPosition("SPY", 10, 450.00, 0.05)
                      .calculateVaR(0.95)
                      .analyze();

    ASSERT_TRUE(result.has_value());
}

TEST_F(RiskManagerFluentTest, PortfolioRiskBuilderHighVolatilityPositions) {
    auto result = risk_mgr->portfolio()
                      .addPosition("TSLA", 5, 250.00, 0.40)  // High volatility
                      .addPosition("AMZN", 10, 170.00, 0.25) // Medium volatility
                      .calculateHeat()
                      .analyze();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->active_positions, 2);
}

// ============================================================================
// Kelly Calculator Tests
// ============================================================================

TEST_F(RiskManagerFluentTest, KellyCalculatorChaining) {
    auto result = risk_mgr->kelly()
                      .withWinRate(0.55)
                      .withWinLossRatio(1.8)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0) << "Kelly fraction should be positive";
    EXPECT_LT(*result, 0.25) << "Kelly fraction should be bounded";
}

TEST_F(RiskManagerFluentTest, KellyCalculatorBreakeven) {
    // With 50% win rate and 1:1 win/loss ratio, Kelly should be 0
    auto result = risk_mgr->kelly()
                      .withWinRate(0.50)
                      .withWinLossRatio(1.0)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, 0.001) << "Kelly at breakeven should be ~0";
}

TEST_F(RiskManagerFluentTest, KellyCalculatorHighWinRate) {
    // 70% win rate with 2:1 ratio
    auto result = risk_mgr->kelly()
                      .withWinRate(0.70)
                      .withWinLossRatio(2.0)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.1) << "Kelly should be significant with 70% win rate";
}

TEST_F(RiskManagerFluentTest, KellyCalculatorWithDrawdownLimit) {
    auto result = risk_mgr->kelly()
                      .withWinRate(0.60)
                      .withWinLossRatio(2.0)
                      .withDrawdownLimit(0.5)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

TEST_F(RiskManagerFluentTest, KellyCalculatorInvalidWinRate) {
    // Win rate > 1.0 should error
    auto result = risk_mgr->kelly()
                      .withWinRate(1.5)
                      .withWinLossRatio(2.0)
                      .calculate();

    EXPECT_FALSE(result.has_value()) << "Invalid win rate should fail";
}

TEST_F(RiskManagerFluentTest, KellyCalculatorInvalidRatio) {
    // Negative ratio should error
    auto result = risk_mgr->kelly()
                      .withWinRate(0.55)
                      .withWinLossRatio(-1.0)
                      .calculate();

    EXPECT_FALSE(result.has_value()) << "Invalid ratio should fail";
}

// ============================================================================
// Position Sizer Builder Tests
// ============================================================================

TEST_F(RiskManagerFluentTest, PositionSizerBuilderChaining) {
    auto result = risk_mgr->positionSizer()
                      .withMethod(SizingMethod::KellyCriterion)
                      .withAccountValue(30000.0)
                      .withWinProbability(0.55)
                      .withWinAmount(100.0)
                      .withLossAmount(100.0)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0) << "Position size should be positive";
}

TEST_F(RiskManagerFluentTest, PositionSizerFixedDollar) {
    auto result = risk_mgr->positionSizer()
                      .withMethod(SizingMethod::FixedDollar)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 1000.0, 0.01);
}

TEST_F(RiskManagerFluentTest, PositionSizerFixedPercent) {
    auto result = risk_mgr->positionSizer()
                      .withMethod(SizingMethod::FixedPercent)
                      .withAccountValue(30000.0)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    // 5% of 30k = 1500
    EXPECT_NEAR(*result, 1500.0, 0.01);
}

TEST_F(RiskManagerFluentTest, PositionSizerKellyCriterion) {
    auto result = risk_mgr->positionSizer()
                      .withMethod(SizingMethod::KellyCriterion)
                      .withAccountValue(30000.0)
                      .withWinProbability(0.60)
                      .withWinAmount(100.0)
                      .withLossAmount(100.0)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

TEST_F(RiskManagerFluentTest, PositionSizerKellyHalf) {
    auto result = risk_mgr->positionSizer()
                      .withMethod(SizingMethod::KellyHalf)
                      .withAccountValue(30000.0)
                      .withWinProbability(0.60)
                      .withWinAmount(100.0)
                      .withLossAmount(100.0)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

TEST_F(RiskManagerFluentTest, PositionSizerVolatilityAdjusted) {
    auto result = risk_mgr->positionSizer()
                      .withMethod(SizingMethod::VolatilityAdjusted)
                      .withAccountValue(30000.0)
                      .withVolatility(0.25)
                      .calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

// ============================================================================
// Monte Carlo Simulator Builder Tests
// ============================================================================

TEST_F(RiskManagerFluentTest, MonteCarloBuilderChaining) {
    // Create basic pricing params for testing
    PricingParams params{
        .spot_price = 450.0,
        .strike_price = 450.0,
        .time_to_expiration = 0.25,
        .risk_free_rate = 0.05,
        .volatility = 0.20,
        .option_type = OptionType::Call};

    auto result = risk_mgr->monteCarlo()
                      .forOption(params)
                      .withSimulations(1000)
                      .withSteps(50)
                      .withPositionSize(100.0)
                      .run();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->expected_value, -1000.0);
    EXPECT_LT(result->max_loss, 0.0);
}

TEST_F(RiskManagerFluentTest, MonteCarloWithDifferentSimulations) {
    PricingParams params{
        .spot_price = 100.0,
        .strike_price = 100.0,
        .time_to_expiration = 0.5,
        .risk_free_rate = 0.05,
        .volatility = 0.30,
        .option_type = OptionType::Call};

    // Test with various simulation counts
    auto result_100 = risk_mgr->monteCarlo()
                          .forOption(params)
                          .withSimulations(100)
                          .run();

    auto result_1000 = risk_mgr->monteCarlo()
                           .forOption(params)
                           .withSimulations(1000)
                           .run();

    ASSERT_TRUE(result_100.has_value());
    ASSERT_TRUE(result_1000.has_value());
}

// ============================================================================
// Daily P&L Management Tests
// ============================================================================

TEST_F(RiskManagerFluentTest, UpdateDailyPnL) {
    risk_mgr->updateDailyPnL(500.0);

    auto daily_pnl = risk_mgr->getDailyPnL();
    EXPECT_NEAR(daily_pnl, 500.0, 0.01);
}

TEST_F(RiskManagerFluentTest, UpdateDailyPnLChaining) {
    auto& result = risk_mgr->updateDailyPnL(500.0)
                       .updateDailyPnL(300.0)
                       .updateDailyPnL(-100.0);

    EXPECT_EQ(&result, risk_mgr.get()) << "updateDailyPnL should return reference";
    EXPECT_NEAR(risk_mgr->getDailyPnL(), 700.0, 0.01);
}

TEST_F(RiskManagerFluentTest, ResetDaily) {
    risk_mgr->updateDailyPnL(500.0);
    auto& result = risk_mgr->resetDaily();

    EXPECT_EQ(&result, risk_mgr.get());
    EXPECT_NEAR(risk_mgr->getDailyPnL(), 0.0, 0.01);
}

TEST_F(RiskManagerFluentTest, DailyLossRemainingTracking) {
    risk_mgr->setMaxDailyLoss(900.0);

    auto remaining_initial = risk_mgr->getDailyLossRemaining();
    EXPECT_NEAR(remaining_initial, 900.0, 0.01);
}

TEST_F(RiskManagerFluentTest, IsDailyLossLimitReached) {
    risk_mgr->setMaxDailyLoss(100.0);

    bool reached = risk_mgr->isDailyLossLimitReached();
    EXPECT_FALSE(reached) << "Should not reach limit initially";
}

// ============================================================================
// Integration Tests (Complex Scenarios)
// ============================================================================

TEST_F(RiskManagerFluentTest, CompleteTradeWorkflow) {
    // Setup risk manager with specific limits
    risk_mgr->setMaxDailyLoss(1000.0)
        .setMaxPositionSize(2000.0)
        .setMaxPortfolioHeat(0.15);

    // Assess trade using fluent API
    auto trade_risk = risk_mgr->assessTrade()
                          .forSymbol("SPY")
                          .withQuantity(10)
                          .atPrice(450.00)
                          .withStop(440.00)
                          .withTarget(465.00)
                          .withProbability(0.65)
                          .assess();

    ASSERT_TRUE(trade_risk.has_value());
    EXPECT_TRUE(trade_risk->approved);

    // Calculate Kelly fraction for position sizing
    auto kelly = risk_mgr->kelly()
                     .withWinRate(0.65)
                     .withWinLossRatio(2.0)
                     .calculate();

    ASSERT_TRUE(kelly.has_value());
    EXPECT_GT(*kelly, 0.0);
}

TEST_F(RiskManagerFluentTest, CompletePortfolioAnalysis) {
    // Build portfolio with multiple positions
    auto portfolio = risk_mgr->portfolio()
                         .addPosition("SPY", 10, 450.00, 0.05)
                         .addPosition("XLE", 20, 80.00, 0.08)
                         .addPosition("TLT", 50, 100.00, 0.03)
                         .calculateHeat()
                         .calculateVaR(0.95)
                         .analyze();

    ASSERT_TRUE(portfolio.has_value());
    EXPECT_EQ(portfolio->active_positions, 3);
    EXPECT_GT(portfolio->total_value, 0.0);
    EXPECT_GT(portfolio->portfolio_heat, 0.0);
}

TEST_F(RiskManagerFluentTest, KellyPositionSizingIntegration) {
    // Use Kelly calculator to inform position sizer
    auto kelly = risk_mgr->kelly()
                     .withWinRate(0.58)
                     .withWinLossRatio(1.5)
                     .calculate();

    ASSERT_TRUE(kelly.has_value());

    // Now use that information in position sizer
    auto size = risk_mgr->positionSizer()
                    .withMethod(SizingMethod::KellyCriterion)
                    .withWinProbability(0.58)
                    .withWinAmount(150.0)
                    .withLossAmount(100.0)
                    .calculate();

    ASSERT_TRUE(size.has_value());
    EXPECT_GT(*size, 0.0);
}

TEST_F(RiskManagerFluentTest, SequentialTrades) {
    // Execute multiple trade assessments in sequence
    std::vector<std::string> symbols = {"SPY", "QQQ", "XLE"};
    std::vector<double> prices = {450.0, 350.0, 80.0};
    std::vector<double> stops = {440.0, 340.0, 75.0};
    std::vector<double> targets = {465.0, 370.0, 90.0};

    for (size_t i = 0; i < symbols.size(); ++i) {
        auto result = risk_mgr->assessTrade()
                          .forSymbol(symbols[i])
                          .withQuantity(10)
                          .atPrice(prices[i])
                          .withStop(stops[i])
                          .withTarget(targets[i])
                          .withProbability(0.60)
                          .assess();

        ASSERT_TRUE(result.has_value()) << "Trade " << i << " should be assessed";
    }
}

} // namespace
