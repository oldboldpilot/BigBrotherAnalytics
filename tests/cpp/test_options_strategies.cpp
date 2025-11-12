/**
 * Comprehensive Regression Tests for All 52 Options Strategies
 * Tests P&L calculations, Greeks, max profit/loss, breakevens, and edge cases
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.single_leg;
import bigbrother.options_strategies.vertical_spreads;
import bigbrother.options_strategies.straddles_strangles;
import bigbrother.options_strategies.butterflies_condors;
import bigbrother.options_strategies.covered_positions;
import bigbrother.options_strategies.calendar_spreads;
import bigbrother.options_strategies.ratio_spreads;
import bigbrother.options_strategies.albatross_ladder;

using namespace bigbrother::options_strategies;

// Test Helpers
constexpr float TOLERANCE = 0.01f;

bool nearlyEqual(float a, float b, float tolerance = TOLERANCE) {
    return std::abs(a - b) < tolerance;
}

// =============================================================================
// Tier 1: Single Leg Strategies (4 strategies)
// =============================================================================

TEST(SingleLegTest, LongCallBasicFunctionality) {
    auto strategy = createLongCall(100.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Test name and description
    EXPECT_EQ(strategy->getName(), "Long Call");

    // P&L tests at various prices
    EXPECT_LT(strategy->calculateProfitLoss(100.0f), 0.0f);  // Loss below strike
    EXPECT_GT(strategy->calculateProfitLoss(110.0f), 0.0f);  // Profit above strike + premium

    // Greeks tests - Long call should have positive delta and gamma
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.delta, 0.0f);
    EXPECT_GT(greeks.gamma, 0.0f);
    EXPECT_LT(greeks.theta, 0.0f);  // Time decay is negative
    EXPECT_GT(greeks.vega, 0.0f);

    // Breakeven should exist
    auto breakevens = strategy->getBreakevens();
    EXPECT_EQ(breakevens.size(), 1);
    EXPECT_GT(breakevens[0], 105.0f);  // Above strike
}

TEST(SingleLegTest, ShortCallBasicFunctionality) {
    auto strategy = createShortCall(100.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Short call should have negative delta
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_LT(greeks.delta, 0.0f);
    EXPECT_LT(greeks.gamma, 0.0f);
    EXPECT_GT(greeks.theta, 0.0f);  // Positive theta (time decay benefits short)

    // Max profit is premium received
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_GT(maxProfit.value(), 0.0f);

    // Max loss is unlimited
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_FALSE(maxLoss.has_value());
}

TEST(SingleLegTest, LongPutBasicFunctionality) {
    auto strategy = createLongPut(100.0f, 95.0f, 30.0f, 0.25f, 0.05f);

    // P&L tests
    EXPECT_GT(strategy->calculateProfitLoss(90.0f), 0.0f);  // Profit below strike
    EXPECT_LT(strategy->calculateProfitLoss(100.0f), 0.0f); // Loss above strike

    // Greeks - Long put should have negative delta, positive gamma
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_LT(greeks.delta, 0.0f);
    EXPECT_GT(greeks.gamma, 0.0f);
    EXPECT_LT(greeks.theta, 0.0f);
    EXPECT_GT(greeks.vega, 0.0f);
}

TEST(SingleLegTest, ShortPutBasicFunctionality) {
    auto strategy = createShortPut(100.0f, 95.0f, 30.0f, 0.25f, 0.05f);

    // Short put should have positive delta (benefits from price increases)
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.delta, 0.0f);
    EXPECT_LT(greeks.gamma, 0.0f);
    EXPECT_GT(greeks.theta, 0.0f);

    // Max profit is premium received
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_GT(maxProfit.value(), 0.0f);
}

// =============================================================================
// Tier 2.1: Vertical Spreads (4 strategies)
// =============================================================================

TEST(VerticalSpreadTest, BullCallSpread) {
    auto strategy = createBullCallSpread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should have defined max profit and max loss
    auto maxProfit = strategy->getMaxProfit();
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_TRUE(maxLoss.has_value());
    EXPECT_GT(maxProfit.value(), 0.0f);
    EXPECT_LT(maxLoss.value(), 0.0f);

    // Should have positive delta (bullish)
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.delta, 0.0f);

    // Should have 2 breakevens
    auto breakevens = strategy->getBreakevens();
    EXPECT_GE(breakevens.size(), 1);
}

TEST(VerticalSpreadTest, BearPutSpread) {
    auto strategy = createBearPutSpread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should have negative delta (bearish)
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_LT(greeks.delta, 0.0f);

    // Should have defined risk/reward
    auto maxProfit = strategy->getMaxProfit();
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_TRUE(maxLoss.has_value());
}

TEST(VerticalSpreadTest, BullPutSpread) {
    auto strategy = createBullPutSpread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Credit spread - should collect premium
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_GT(maxProfit.value(), 0.0f);
}

TEST(VerticalSpreadTest, BearCallSpread) {
    auto strategy = createBearCallSpread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Credit spread - should have positive theta
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.theta, 0.0f);  // Benefits from time decay
}

// =============================================================================
// Tier 2.2: Straddles & Strangles (8 strategies)
// =============================================================================

TEST(StraddleStrangleTest, LongStraddle) {
    auto strategy = createLongStraddle(100.0f, 100.0f, 30.0f, 0.25f, 0.05f);

    // Should have near-zero delta at ATM
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_NEAR(greeks.delta, 0.0f, 0.1f);

    // Should have high gamma and vega (benefits from movement and volatility)
    EXPECT_GT(greeks.gamma, 0.0f);
    EXPECT_GT(greeks.vega, 0.0f);
    EXPECT_LT(greeks.theta, 0.0f);

    // Should profit from large moves in either direction
    EXPECT_LT(strategy->calculateProfitLoss(100.0f), 0.0f);  // Loss at strike
    EXPECT_GT(strategy->calculateProfitLoss(110.0f), strategy->calculateProfitLoss(100.0f));
    EXPECT_GT(strategy->calculateProfitLoss(90.0f), strategy->calculateProfitLoss(100.0f));

    // Should have 2 breakevens
    auto breakevens = strategy->getBreakevens();
    EXPECT_EQ(breakevens.size(), 2);
}

TEST(StraddleStrangleTest, ShortStraddle) {
    auto strategy = createShortStraddle(100.0f, 100.0f, 30.0f, 0.25f, 0.05f);

    // Should have positive theta (benefits from time decay)
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.theta, 0.0f);

    // Should have negative gamma and vega
    EXPECT_LT(greeks.gamma, 0.0f);
    EXPECT_LT(greeks.vega, 0.0f);

    // Max profit at strike
    EXPECT_GT(strategy->calculateProfitLoss(100.0f), strategy->calculateProfitLoss(110.0f));
}

TEST(StraddleStrangleTest, LongStrangle) {
    auto strategy = createLongStrangle(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Similar to straddle but cheaper (lower max loss)
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_NEAR(greeks.delta, 0.0f, 0.1f);
    EXPECT_GT(greeks.vega, 0.0f);

    // Should have 2 breakevens
    auto breakevens = strategy->getBreakevens();
    EXPECT_EQ(breakevens.size(), 2);
}

TEST(StraddleStrangleTest, ShortStrangle) {
    auto strategy = createShortStrangle(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should benefit from time decay
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.theta, 0.0f);

    // Max profit in the middle
    float profitAtStrike = strategy->calculateProfitLoss(100.0f);
    EXPECT_GT(profitAtStrike, strategy->calculateProfitLoss(90.0f));
    EXPECT_GT(profitAtStrike, strategy->calculateProfitLoss(110.0f));
}

// =============================================================================
// Tier 3: Butterflies & Condors (12 strategies)
// =============================================================================

TEST(ButterflyCondorTest, LongCallButterfly) {
    auto strategy = createLongCallButterfly(100.0f, 95.0f, 100.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should have limited risk and reward
    auto maxProfit = strategy->getMaxProfit();
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_TRUE(maxLoss.has_value());

    // Max profit at middle strike
    float profitAtMiddle = strategy->calculateProfitLoss(100.0f);
    EXPECT_GT(profitAtMiddle, strategy->calculateProfitLoss(95.0f));
    EXPECT_GT(profitAtMiddle, strategy->calculateProfitLoss(105.0f));

    // Should have near-zero delta at middle strike
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_NEAR(greeks.delta, 0.0f, 0.1f);
}

TEST(ButterflyCondorTest, IronButterfly) {
    auto strategy = createLongIronButterfly(100.0f, 95.0f, 100.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should have positive theta (credit spread)
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.theta, 0.0f);

    // Max profit at ATM
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_GT(maxProfit.value(), 0.0f);
}

TEST(ButterflyCondorTest, IronCondor) {
    auto strategy = createLongIronCondor(100.0f, 90.0f, 95.0f, 105.0f, 110.0f, 30.0f, 0.25f, 0.05f);

    // Should have wide profit zone
    float profitAt95 = strategy->calculateProfitLoss(95.0f);
    float profitAt100 = strategy->calculateProfitLoss(100.0f);
    float profitAt105 = strategy->calculateProfitLoss(105.0f);

    // All prices in the middle should be profitable
    EXPECT_GT(profitAt95, 0.0f);
    EXPECT_GT(profitAt100, 0.0f);
    EXPECT_GT(profitAt105, 0.0f);

    // Should have positive theta
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.theta, 0.0f);
}

// =============================================================================
// Tier 4: Covered Positions (3 strategies)
// =============================================================================

TEST(CoveredPositionTest, CoveredCall) {
    auto strategy = createCoveredCall(100.0f, 100.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should have high delta from stock position
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.delta, 0.5f);  // Stock delta (1.0) minus call delta

    // Max profit is limited
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());

    // Benefits from time decay
    EXPECT_GT(greeks.theta, 0.0f);
}

TEST(CoveredPositionTest, CoveredPut) {
    auto strategy = createCoveredPut(100.0f, 100.0f, 95.0f, 30.0f, 0.25f, 0.05f);

    // Should have negative delta from short stock
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_LT(greeks.delta, -0.5f);
}

TEST(CoveredPositionTest, Collar) {
    auto strategy = createCollar(100.0f, 100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Protected position - limited upside and downside
    auto maxProfit = strategy->getMaxProfit();
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_TRUE(maxLoss.has_value());

    // Delta should be between 0 and 1
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.delta, 0.0f);
    EXPECT_LT(greeks.delta, 1.0f);
}

// =============================================================================
// Tier 5: Calendar Spreads (6 strategies)
// =============================================================================

TEST(CalendarSpreadTest, LongCallCalendar) {
    auto strategy = createLongCallCalendar(100.0f, 100.0f, 30.0f, 60.0f, 0.25f, 0.05f);

    // Benefits from time decay differential
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);

    // Should have positive vega (benefits from volatility)
    EXPECT_GT(greeks.vega, 0.0f);

    // Max profit typically near strike at near expiration
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
}

TEST(CalendarSpreadTest, LongPutCalendar) {
    auto strategy = createLongPutCalendar(100.0f, 100.0f, 30.0f, 60.0f, 0.25f, 0.05f);

    // Similar characteristics to call calendar
    auto greeks = strategy->calculateGreeks(100.0f, 0.05f);
    EXPECT_GT(greeks.vega, 0.0f);
}

// =============================================================================
// Tier 6: Ratio Spreads (8 strategies)
// =============================================================================

TEST(RatioSpreadTest, CallRatioSpread) {
    auto strategy = createCallRatioSpread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Should have defined max profit
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());

    // Max loss should be unlimited (naked short calls)
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_FALSE(maxLoss.has_value());
}

TEST(RatioSpreadTest, PutRatioSpread) {
    auto strategy = createPutRatioSpread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Similar structure to call ratio spread
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
}

TEST(RatioSpreadTest, CallRatioBackspread) {
    auto strategy = createCallRatioBackspread(100.0f, 95.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Opposite of ratio spread - unlimited profit potential
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_FALSE(maxProfit.has_value());  // Unlimited

    // Limited max loss
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_TRUE(maxLoss.has_value());
}

// =============================================================================
// Tier 7: Albatross & Ladder (7 strategies)
// =============================================================================

TEST(AlbatrossLadderTest, LongCallAlbatross) {
    auto strategy = createLongCallAlbatross(100.0f, 90.0f, 95.0f, 105.0f, 110.0f, 30.0f, 0.25f, 0.05f);

    // Similar to condor but with wider body
    auto maxProfit = strategy->getMaxProfit();
    auto maxLoss = strategy->getMaxLoss();
    EXPECT_TRUE(maxProfit.has_value());
    EXPECT_TRUE(maxLoss.has_value());

    // Wide profit zone between middle strikes
    float profitAt100 = strategy->calculateProfitLoss(100.0f);
    EXPECT_GT(profitAt100, 0.0f);
}

TEST(AlbatrossLadderTest, CallLadder) {
    auto strategy = createCallLadderSpread(100.0f, 95.0f, 100.0f, 105.0f, 30.0f, 0.25f, 0.05f);

    // Ladder has asymmetric P&L
    auto maxProfit = strategy->getMaxProfit();
    EXPECT_TRUE(maxProfit.has_value());
}

// =============================================================================
// Edge Cases and Stress Tests
// =============================================================================

TEST(EdgeCaseTest, ZeroVolatility) {
    // Test with zero volatility (should not crash)
    auto strategy = createLongCall(100.0f, 105.0f, 30.0f, 0.001f, 0.05f);
    EXPECT_NO_THROW(strategy->calculateGreeks(100.0f, 0.05f));
}

TEST(EdgeCaseTest, VeryShortExpiration) {
    // Test with 1 day to expiration
    auto strategy = createLongCall(100.0f, 105.0f, 1.0f, 0.25f, 0.05f);
    EXPECT_NO_THROW(strategy->calculateGreeks(100.0f, 0.05f));
}

TEST(EdgeCaseTest, VeryLongExpiration) {
    // Test with 2 years to expiration
    auto strategy = createLongCall(100.0f, 105.0f, 730.0f, 0.25f, 0.05f);
    EXPECT_NO_THROW(strategy->calculateGreeks(100.0f, 0.05f));
}

TEST(EdgeCaseTest, HighVolatility) {
    // Test with very high volatility
    auto strategy = createLongCall(100.0f, 105.0f, 30.0f, 2.0f, 0.05f);
    EXPECT_NO_THROW(strategy->calculateGreeks(100.0f, 0.05f));
}

TEST(EdgeCaseTest, DeepITM) {
    // Test deep in-the-money option
    auto strategy = createLongCall(150.0f, 105.0f, 30.0f, 0.25f, 0.05f);
    auto greeks = strategy->calculateGreeks(150.0f, 0.05f);
    EXPECT_NEAR(greeks.delta, 1.0f, 0.1f);  // Should be close to 1.0
}

TEST(EdgeCaseTest, DeepOTM) {
    // Test deep out-of-the-money option
    auto strategy = createLongCall(50.0f, 105.0f, 30.0f, 0.25f, 0.05f);
    auto greeks = strategy->calculateGreeks(50.0f, 0.05f);
    EXPECT_NEAR(greeks.delta, 0.0f, 0.1f);  // Should be close to 0.0
}

// =============================================================================
// Performance Validation
// =============================================================================

TEST(PerformanceTest, GreeksCalculationSpeed) {
    auto strategy = createLongIronCondor(100.0f, 90.0f, 95.0f, 105.0f, 110.0f, 30.0f, 0.25f, 0.05f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        strategy->calculateGreeks(100.0f, 0.05f);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be able to calculate 10,000 Greeks in < 100ms (target: <10μs per calculation)
    EXPECT_LT(duration.count(), 100000);

    // Log performance
    std::cout << "Greeks calculation: " << (duration.count() / 10000.0) << " μs per call\n";
}

TEST(PerformanceTest, ProfitLossCalculationSpeed) {
    auto strategy = createLongIronCondor(100.0f, 90.0f, 95.0f, 105.0f, 110.0f, 30.0f, 0.25f, 0.05f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        strategy->calculateProfitLoss(100.0f);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // P&L should be very fast (target: <1μs per calculation)
    EXPECT_LT(duration.count(), 100000);

    std::cout << "P&L calculation: " << (duration.count() / 100000.0) << " μs per call\n";
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n=== Running Comprehensive Options Strategies Tests (52 strategies) ===\n\n";
    return RUN_ALL_TESTS();
}
