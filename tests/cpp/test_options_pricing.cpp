/**
 * Unit Tests for Options Pricing Engine
 *
 * Tests all pricing models and validates against known values.
 */

#include "../../src/correlation_engine/options_pricing.hpp"
#include "../../src/correlation_engine/options_fluent_api.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace bigbrother::options;
using namespace bigbrother::types;

// Test tolerance
constexpr double PRICE_TOLERANCE = 0.01;   // $0.01
constexpr double GREEK_TOLERANCE = 0.001;  // 0.001
constexpr double IV_TOLERANCE = 0.0001;    // 0.01%

/**
 * Test Black-Scholes pricing against known values
 */
TEST(BlackScholesTest, CallOptionPricing) {
    // Known test case: European call
    // S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0
    // Expected price ~= $10.45

    auto result = OptionBuilder()
        .call()
        .european()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .dividendYield(0.0)
        .useBlackScholes()
        .price();

    ASSERT_TRUE(result.has_value()) << "Pricing failed: " << result.error().message;

    double const price = *result;
    EXPECT_NEAR(price, 10.45, PRICE_TOLERANCE) << "Call price incorrect";
    EXPECT_GT(price, 0.0) << "Price should be positive";
}

TEST(BlackScholesTest, PutOptionPricing) {
    // Known test case: European put
    // S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0
    // Expected price ~= $5.57

    auto result = OptionBuilder()
        .put()
        .european()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .dividendYield(0.0)
        .useBlackScholes()
        .price();

    ASSERT_TRUE(result.has_value()) << "Pricing failed: " << result.error().message;

    double const price = *result;
    EXPECT_NEAR(price, 5.57, PRICE_TOLERANCE) << "Put price incorrect";
    EXPECT_GT(price, 0.0) << "Price should be positive";
}

TEST(BlackScholesTest, PutCallParity) {
    // Test put-call parity for European options:
    // C - P = S*e^(-qT) - K*e^(-rT)

    double const S = 105.0;
    double const K = 100.0;
    double const T = 0.5;
    double const r = 0.05;
    double const sigma = 0.25;
    double const q = 0.02;

    auto call_result = OptionBuilder()
        .call()
        .european()
        .spot(S)
        .strike(K)
        .yearsToExpiration(T)
        .volatility(sigma)
        .riskFreeRate(r)
        .dividendYield(q)
        .useBlackScholes()
        .price();

    auto put_result = OptionBuilder()
        .put()
        .european()
        .spot(S)
        .strike(K)
        .yearsToExpiration(T)
        .volatility(sigma)
        .riskFreeRate(r)
        .dividendYield(q)
        .useBlackScholes()
        .price();

    ASSERT_TRUE(call_result.has_value());
    ASSERT_TRUE(put_result.has_value());

    double const call_price = *call_result;
    double const put_price = *put_result;

    double const parity_left = call_price - put_price;
    double const parity_right = S * std::exp(-q * T) - K * std::exp(-r * T);

    EXPECT_NEAR(parity_left, parity_right, PRICE_TOLERANCE)
        << "Put-call parity violated";
}

/**
 * Test Greeks calculation
 */
TEST(BlackScholesTest, GreeksCalculation) {
    // ATM call option
    auto greeks_result = OptionBuilder()
        .call()
        .european()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .useBlackScholes()
        .greeks();

    ASSERT_TRUE(greeks_result.has_value());

    auto const& greeks = *greeks_result;

    // Delta for ATM call should be around 0.5-0.6
    EXPECT_GT(greeks.delta, 0.4);
    EXPECT_LT(greeks.delta, 0.7);

    // Gamma should be positive
    EXPECT_GT(greeks.gamma, 0.0);

    // Theta should be negative (time decay)
    EXPECT_LT(greeks.theta, 0.0);

    // Vega should be positive
    EXPECT_GT(greeks.vega, 0.0);

    // Rho for call should be positive
    EXPECT_GT(greeks.rho, 0.0);
}

TEST(BlackScholesTest, DeltaMonotonicity) {
    // Delta should increase as spot increases (for calls)
    std::vector<double> spots = {90.0, 95.0, 100.0, 105.0, 110.0};
    std::vector<double> deltas;

    for (auto spot : spots) {
        auto greeks_result = OptionBuilder()
            .call()
            .spot(spot)
            .strike(100.0)
            .yearsToExpiration(0.25)
            .volatility(0.25)
            .riskFreeRate(0.05)
            .useBlackScholes()
            .greeks();

        ASSERT_TRUE(greeks_result.has_value());
        deltas.push_back(greeks_result->delta);
    }

    // Delta should be monotonically increasing
    for (size_t i = 1; i < deltas.size(); ++i) {
        EXPECT_GT(deltas[i], deltas[i-1])
            << "Delta should increase with spot price";
    }
}

/**
 * Test Trinomial Tree pricing
 */
TEST(TrinomialTreeTest, EuropeanCallPricing) {
    // Should match Black-Scholes for European options
    auto bs_result = OptionBuilder()
        .call()
        .european()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .useBlackScholes()
        .price();

    auto trinomial_result = OptionBuilder()
        .call()
        .european()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .useTrinomial(200)  // More steps for accuracy
        .price();

    ASSERT_TRUE(bs_result.has_value());
    ASSERT_TRUE(trinomial_result.has_value());

    // Should converge to Black-Scholes price
    EXPECT_NEAR(*trinomial_result, *bs_result, 0.10)
        << "Trinomial should converge to Black-Scholes";
}

TEST(TrinomialTreeTest, AmericanCallNoDividend) {
    // American call with no dividends should equal European call
    auto european_result = OptionBuilder()
        .call()
        .european()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .dividendYield(0.0)
        .useTrinomial(100)
        .price();

    auto american_result = OptionBuilder()
        .call()
        .american()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .dividendYield(0.0)
        .useTrinomial(100)
        .price();

    ASSERT_TRUE(european_result.has_value());
    ASSERT_TRUE(american_result.has_value());

    // Should be very close (early exercise not optimal without dividends)
    EXPECT_NEAR(*american_result, *european_result, PRICE_TOLERANCE);
}

TEST(TrinomialTreeTest, AmericanPutEarlyExercise) {
    // Deep ITM American put should be worth more than European
    // (early exercise is valuable)

    auto european_result = OptionBuilder()
        .put()
        .european()
        .spot(80.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .useTrinomial(100)
        .price();

    auto american_result = OptionBuilder()
        .put()
        .american()
        .spot(80.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .useTrinomial(100)
        .price();

    ASSERT_TRUE(european_result.has_value());
    ASSERT_TRUE(american_result.has_value());

    // American put should be worth more
    EXPECT_GT(*american_result, *european_result)
        << "American put should be worth more due to early exercise";
}

/**
 * Test Implied Volatility calculation
 */
TEST(ImpliedVolatilityTest, RecoverVolatility) {
    // Price an option, then recover the volatility
    double const original_vol = 0.25;

    auto price_result = OptionBuilder()
        .call()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(0.5)
        .volatility(original_vol)
        .riskFreeRate(0.05)
        .useBlackScholes()
        .price();

    ASSERT_TRUE(price_result.has_value());

    // Now recover the volatility
    auto iv_result = OptionBuilder()
        .call()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(0.5)
        .riskFreeRate(0.05)
        .marketPrice(*price_result)
        .impliedVolatility();

    ASSERT_TRUE(iv_result.has_value()) << "IV calculation failed: "
                                        << iv_result.error().message;

    // Should recover original volatility
    EXPECT_NEAR(*iv_result, original_vol, IV_TOLERANCE)
        << "Failed to recover original volatility";
}

TEST(ImpliedVolatilityTest, MultipleSt rikes) {
    // Test IV calculation across different strikes
    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};

    for (auto strike : strikes) {
        auto price_result = OptionBuilder()
            .call()
            .spot(100.0)
            .strike(strike)
            .yearsToExpiration(0.25)
            .volatility(0.30)
            .riskFreeRate(0.05)
            .price();

        ASSERT_TRUE(price_result.has_value());

        auto iv_result = OptionBuilder()
            .call()
            .spot(100.0)
            .strike(strike)
            .yearsToExpiration(0.25)
            .riskFreeRate(0.05)
            .marketPrice(*price_result)
            .impliedVolatility();

        ASSERT_TRUE(iv_result.has_value())
            << "IV calculation failed for strike " << strike;

        EXPECT_NEAR(*iv_result, 0.30, IV_TOLERANCE)
            << "IV incorrect for strike " << strike;
    }
}

/**
 * Test edge cases
 */
TEST(EdgeCasesTest, ExpiredOption) {
    // Expired call option
    auto result = OptionBuilder()
        .call()
        .spot(105.0)
        .strike(100.0)
        .yearsToExpiration(0.0)  // Expired
        .volatility(0.25)
        .riskFreeRate(0.05)
        .price();

    ASSERT_TRUE(result.has_value());

    // Should equal intrinsic value
    EXPECT_NEAR(*result, 5.0, PRICE_TOLERANCE);
}

TEST(EdgeCasesTest, DeepITMCall) {
    // Deep in-the-money call
    auto result = OptionBuilder()
        .call()
        .spot(150.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .price();

    ASSERT_TRUE(result.has_value());

    // Should be close to intrinsic value + time value
    EXPECT_GT(*result, 50.0);  // At least intrinsic value
    EXPECT_LT(*result, 52.0);  // Not much time value at this level
}

TEST(EdgeCasesTest, DeepOTMCall) {
    // Deep out-of-the-money call
    auto result = OptionBuilder()
        .call()
        .spot(50.0)
        .strike(100.0)
        .yearsToExpiration(0.5)
        .volatility(0.20)
        .riskFreeRate(0.05)
        .price();

    ASSERT_TRUE(result.has_value());

    // Should be near zero
    EXPECT_LT(*result, 0.50);
    EXPECT_GT(*result, 0.0);
}

TEST(EdgeCasesTest, InvalidParameters) {
    // Negative spot price
    auto result1 = OptionBuilder()
        .call()
        .spot(-100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(0.25)
        .riskFreeRate(0.05)
        .price();

    EXPECT_FALSE(result1.has_value()) << "Should reject negative spot";

    // Negative volatility
    auto result2 = OptionBuilder()
        .call()
        .spot(100.0)
        .strike(100.0)
        .yearsToExpiration(1.0)
        .volatility(-0.25)
        .riskFreeRate(0.05)
        .price();

    EXPECT_FALSE(result2.has_value()) << "Should reject negative volatility";
}

/**
 * Test convenience functions
 */
TEST(ConvenienceFunctionsTest, PriceCall) {
    auto result = priceCall(100.0, 100.0, 1.0, 0.25, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

TEST(ConvenienceFunctionsTest, PricePut) {
    auto result = pricePut(100.0, 100.0, 1.0, 0.25, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

TEST(ConvenienceFunctionsTest, CallGreeks) {
    auto result = callGreeks(100.0, 100.0, 1.0, 0.25, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->isValid());
}

/**
 * Performance benchmarks
 */
TEST(PerformanceTest, BlackScholesSpeed) {
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERATIONS = 10000;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto result = OptionBuilder()
            .call()
            .spot(100.0)
            .strike(100.0 + i * 0.01)  // Vary strike slightly
            .yearsToExpiration(1.0)
            .volatility(0.25)
            .riskFreeRate(0.05)
            .useBlackScholes()
            .price();

        ASSERT_TRUE(result.has_value());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = static_cast<double>(duration.count()) / ITERATIONS;

    std::cout << "Black-Scholes average time: " << avg_time_us << " μs" << std::endl;

    // Should be < 1 microsecond per pricing
    EXPECT_LT(avg_time_us, 1.0) << "Black-Scholes too slow";
}

TEST(PerformanceTest, TrinomialTreeSpeed) {
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERATIONS = 1000;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto result = OptionBuilder()
            .call()
            .american()
            .spot(100.0)
            .strike(100.0 + i * 0.01)
            .yearsToExpiration(1.0)
            .volatility(0.25)
            .riskFreeRate(0.05)
            .useTrinomial(100)
            .price();

        ASSERT_TRUE(result.has_value());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = static_cast<double>(duration.count()) / ITERATIONS;

    std::cout << "Trinomial tree (100 steps) average time: " << avg_time_us << " μs" << std::endl;

    // Should be < 100 microseconds per pricing
    EXPECT_LT(avg_time_us, 100.0) << "Trinomial tree too slow";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
