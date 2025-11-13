/**
 * BigBrotherAnalytics - Risk Management Modules Test
 *
 * Regression tests for newly converted C++23 risk management modules.
 * Tests fluent API, calculations, and integration.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 */

#include <gtest/gtest.h>
#include <cmath>

// Import the new risk management modules
import bigbrother.risk.position_sizer;
import bigbrother.risk.stop_loss;
import bigbrother.risk.monte_carlo;
import bigbrother.risk.manager;
import bigbrother.utils.types;
import bigbrother.options.pricing;

using namespace bigbrother::risk;
using namespace bigbrother::types;
using namespace bigbrother::options;

// ============================================================================
// Position Sizer Tests
// ============================================================================

TEST(PositionSizerTest, FluentAPIBasic) {
    auto sizer = PositionSizer::create()
        .withMethod(SizingMethod::KellyHalf)
        .withAccountValue(30000.0)
        .withWinProbability(0.60)
        .withExpectedGain(500.0)
        .withExpectedLoss(300.0)
        .withMaxPosition(2000.0);

    auto result = sizer.calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->dollar_amount, 0.0);
    EXPECT_LE(result->dollar_amount, 2000.0);
    EXPECT_GT(result->kelly_fraction, 0.0);
    EXPECT_EQ(result->method_used, SizingMethod::KellyHalf);
}

TEST(PositionSizerTest, KellyFraction) {
    // Test Kelly Criterion calculation
    // Win prob 60%, win $2 for every $1 risked
    double kelly = PositionSizer::kellyFraction(0.60, 2.0, 1.0);

    // Kelly formula: (p*b - q) / b = (0.6*2 - 0.4) / 2 = 0.4
    EXPECT_NEAR(kelly, 0.40, 0.01);
}

TEST(PositionSizerTest, PositionLimits) {
    auto sizer = PositionSizer::create()
        .withMethod(SizingMethod::FixedDollar)
        .withMaxPosition(500.0);

    auto result = sizer.calculate();

    ASSERT_TRUE(result.has_value());
    EXPECT_LE(result->dollar_amount, 500.0);
}

TEST(PositionSizerTest, InvalidInputs) {
    auto sizer = PositionSizer::create()
        .withAccountValue(-1000.0);  // Invalid

    auto result = sizer.calculate();

    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Stop Loss Manager Tests
// ============================================================================

TEST(StopLossTest, AddHardStop) {
    auto manager = StopLossManager::create();
    auto& ref = manager.addHardStop("AAPL", 145.0, 150.0);

    EXPECT_EQ(ref.getStopCount(), 1);
    EXPECT_TRUE(ref.hasStop("AAPL"));
}

TEST(StopLossTest, TrailingStopUpdate) {
    auto manager = StopLossManager::create();
    manager.addTrailingStop("MSFT", 295.0, 300.0, 5.0);

    // Price moves up to $310
    std::unordered_map<std::string, Price> prices = {{"MSFT", 310.0}};
    auto triggered = manager.update(prices);

    // Should not trigger, trailing stop should move up
    EXPECT_TRUE(triggered.empty());

    // Price drops to $300
    prices["MSFT"] = 300.0;
    triggered = manager.update(prices);

    // Should still not trigger (stop at ~305)
    EXPECT_TRUE(triggered.empty());

    // Price drops to $299
    prices["MSFT"] = 299.0;
    triggered = manager.update(prices);

    // Should trigger now
    EXPECT_EQ(triggered.size(), 1);
}

TEST(StopLossTest, ClearOperations) {
    auto manager = StopLossManager::create();
    auto& ref = manager.addHardStop("AAPL", 145.0, 150.0)
                       .addHardStop("MSFT", 295.0, 300.0);

    EXPECT_EQ(ref.getStopCount(), 2);

    auto& cleared = ref.clearAll();
    EXPECT_EQ(cleared.getStopCount(), 0);
}

// ============================================================================
// Monte Carlo Simulator Tests
// ============================================================================

TEST(MonteCarloTest, StockSimulationBasic) {
    auto simulator = MonteCarloSimulator::create()
        .withSimulations(1000)
        .withParallel(false)  // Sequential for deterministic testing
        .withSeed(42);

    auto result = simulator.simulateStock(
        100.0,  // entry
        110.0,  // target
        95.0,   // stop
        0.20    // volatility
    );

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->num_simulations, 1000);
    EXPECT_GT(result->win_probability, 0.0);
    EXPECT_LT(result->win_probability, 1.0);
    EXPECT_NE(result->var_95, 0.0);
}

TEST(MonteCarloTest, SimulationStatistics) {
    auto simulator = MonteCarloSimulator::create()
        .withSimulations(10000)
        .withParallel(true);

    auto result = simulator.simulateStock(100.0, 110.0, 95.0, 0.15);

    ASSERT_TRUE(result.has_value());

    // Mean should be roughly between max loss and max gain
    EXPECT_GT(result->mean_pnl, -5.0);
    EXPECT_LT(result->mean_pnl, 10.0);

    // VaR should be positive
    EXPECT_GT(result->var_95, 0.0);

    // CVaR should be >= VaR
    EXPECT_GE(result->cvar_95, result->var_95);
}

TEST(MonteCarloTest, ParallelVsSequential) {
    auto sim_parallel = MonteCarloSimulator::create()
        .withSimulations(5000)
        .withParallel(true)
        .withSeed(42);

    auto sim_sequential = MonteCarloSimulator::create()
        .withSimulations(5000)
        .withParallel(false)
        .withSeed(42);

    auto result_parallel = sim_parallel.simulateStock(100.0, 110.0, 95.0, 0.20);
    auto result_sequential = sim_sequential.simulateStock(100.0, 110.0, 95.0, 0.20);

    ASSERT_TRUE(result_parallel.has_value());
    ASSERT_TRUE(result_sequential.has_value());

    // Results should be statistically similar (within 10%)
    EXPECT_NEAR(result_parallel->mean_pnl, result_sequential->mean_pnl,
                std::abs(result_sequential->mean_pnl) * 0.1);
}

// ============================================================================
// Risk Manager Tests
// ============================================================================

TEST(RiskManagerTest, CreateWithValidLimits) {
    RiskLimits limits{
        .account_value = 50000.0,
        .max_daily_loss = 1000.0,
        .max_position_size = 3000.0,
        .max_concurrent_positions = 10
    };

    auto manager = RiskManager::create(limits);

    ASSERT_TRUE(manager.has_value());
    EXPECT_EQ(manager->getRiskLimits().account_value, 50000.0);
}

TEST(RiskManagerTest, FluentConfiguration) {
    RiskLimits limits;
    auto manager_result = RiskManager::create(limits);
    ASSERT_TRUE(manager_result.has_value());

    auto& manager = *manager_result;
    auto& updated = manager.withAccountValue(60000.0)
                           .withDailyLossLimit(1500.0)
                           .withPositionSizeLimit(4000.0);

    EXPECT_EQ(updated.getRiskLimits().account_value, 60000.0);
    EXPECT_EQ(updated.getRiskLimits().max_daily_loss, 1500.0);
}

TEST(RiskManagerTest, TradeAssessment) {
    RiskLimits limits{
        .account_value = 30000.0,
        .max_daily_loss = 900.0,
        .max_position_size = 2000.0
    };

    auto manager_result = RiskManager::create(limits);
    ASSERT_TRUE(manager_result.has_value());
    auto& manager = *manager_result;

    auto risk = manager.assessTrade(
        "AAPL",     // symbol
        1500.0,     // position size
        150.0,      // entry
        145.0,      // stop
        160.0,      // target
        0.65        // win probability
    );

    ASSERT_TRUE(risk.has_value());
    EXPECT_LE(risk->position_size, 2000.0);
    EXPECT_GT(risk->risk_reward_ratio, 0.0);
}

TEST(RiskManagerTest, TradeRejection) {
    RiskLimits limits{
        .account_value = 30000.0,
        .max_daily_loss = 900.0,
        .max_position_size = 2000.0
    };

    auto manager_result = RiskManager::create(limits);
    ASSERT_TRUE(manager_result.has_value());
    auto& manager = *manager_result;

    // Try to place a trade that would exceed position limit
    auto risk = manager.assessTrade(
        "AAPL", 5000.0, 150.0, 145.0, 160.0, 0.65
    );

    ASSERT_TRUE(risk.has_value());
    EXPECT_FALSE(risk->approved);
    EXPECT_FALSE(risk->rejection_reason.empty());
}

TEST(RiskManagerTest, PortfolioRiskQuery) {
    RiskLimits limits;
    auto manager_result = RiskManager::create(limits);
    ASSERT_TRUE(manager_result.has_value());
    auto& manager = *manager_result;

    auto portfolio_risk = manager.getPortfolioRisk();

    EXPECT_EQ(portfolio_risk.active_positions, 0);
    EXPECT_TRUE(portfolio_risk.canOpenNewPosition());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(IntegrationTest, PositionSizingToRiskManager) {
    // Size a position
    auto sizer = PositionSizer::create()
        .withMethod(SizingMethod::KellyHalf)
        .withAccountValue(30000.0)
        .withWinProbability(0.60)
        .withExpectedGain(500.0)
        .withExpectedLoss(300.0);

    auto size_result = sizer.calculate();
    ASSERT_TRUE(size_result.has_value());

    // Assess with risk manager
    RiskLimits limits;
    auto manager_result = RiskManager::create(limits);
    ASSERT_TRUE(manager_result.has_value());
    auto& manager = *manager_result;

    auto risk = manager.assessTrade(
        "AAPL",
        size_result->dollar_amount,
        150.0, 145.0, 160.0, 0.60
    );

    ASSERT_TRUE(risk.has_value());
}

TEST(IntegrationTest, MonteCarloToStopLoss) {
    // Run Monte Carlo to find optimal stop
    auto simulator = MonteCarloSimulator::create()
        .withSimulations(1000);

    auto mc_result = simulator.simulateStock(100.0, 110.0, 95.0, 0.20);
    ASSERT_TRUE(mc_result.has_value());

    // Set stop loss based on VaR
    auto manager = StopLossManager::create();
    auto& ref = manager.addHardStop("STOCK", 100.0 - mc_result->var_95, 100.0);

    EXPECT_TRUE(ref.hasStop("STOCK"));
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST(PerformanceTest, MonteCarloThroughput) {
    auto start = std::chrono::high_resolution_clock::now();

    auto simulator = MonteCarloSimulator::create()
        .withSimulations(100000)
        .withParallel(true);

    auto result = simulator.simulateStock(100.0, 110.0, 95.0, 0.20);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ASSERT_TRUE(result.has_value());

    // Should complete 100K simulations in under 1 second with OpenMP
    EXPECT_LT(duration.count(), 1000);

    std::cout << "Monte Carlo: 100K simulations in "
              << duration.count() << "ms" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
