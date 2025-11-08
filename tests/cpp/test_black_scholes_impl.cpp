/**
 * Black-Scholes Model Unit Tests
 *
 * Comprehensive test suite for Black-Scholes pricing and Greeks.
 * Following C++23 best practices with trailing return types.
 */

#include <gtest/gtest.h>
#include "../../src/correlation_engine/options_pricing.hpp"
#include <cmath>
#include <chrono>

using namespace bigbrother::options;
using namespace bigbrother::types;

/**
 * Test fixture for Black-Scholes tests
 */
class BlackScholesTest : public ::testing::Test {
protected:
    BlackScholesModel bs_model;

    // Helper: Check if two doubles are approximately equal
    [[nodiscard]] auto isClose(double a, double b, double tolerance = 0.01) const -> bool {
        return std::abs(a - b) < tolerance;
    }
};

/**
 * Test ATM Call Option Pricing
 */
TEST_F(BlackScholesTest, ATMCallPrice) {
    PricingParams params{
        .spot_price = 100.0,
        .strike_price = 100.0,
        .risk_free_rate = 0.05,
        .time_to_expiration = 1.0,
        .volatility = 0.20,
        .dividend_yield = 0.0,
        .option_type = OptionType::Call
    };

    auto result = bs_model.price(params);

    ASSERT_TRUE(result.has_value()) << "ATM call pricing should succeed";

    double call_price = *result;

    // Expected ~$10.45 (Hull's textbook value)
    EXPECT_TRUE(isClose(call_price, 10.45, 0.50))
        << "Call price: " << call_price << ", expected ~10.45";

    EXPECT_GT(call_price, 0.0);
    EXPECT_LT(call_price, params.spot_price);
}

/**
 * Test Put-Call Parity
 */
TEST_F(BlackScholesTest, PutCallParity) {
    PricingParams call_params{
        .spot_price = 100.0,
        .strike_price = 100.0,
        .risk_free_rate = 0.05,
        .time_to_expiration = 1.0,
        .volatility = 0.20,
        .dividend_yield = 0.0,
        .option_type = OptionType::Call
    };

    PricingParams put_params = call_params;
    put_params.option_type = OptionType::Put;

    auto call_result = bs_model.price(call_params);
    auto put_result = bs_model.price(put_params);

    ASSERT_TRUE(call_result.has_value());
    ASSERT_TRUE(put_result.has_value());

    // P = C - S + K*e^(-rT)
    double expected_put = *call_result - 100.0 + 100.0 * std::exp(-0.05);

    EXPECT_TRUE(isClose(*put_result, expected_put, 0.01))
        << "Put-call parity violated";
}

/**
 * Test Greeks - Delta
 */
TEST_F(BlackScholesTest, CallDelta) {
    PricingParams params{
        .spot_price = 100.0,
        .strike_price = 100.0,
        .risk_free_rate = 0.05,
        .time_to_expiration = 1.0,
        .volatility = 0.20,
        .dividend_yield = 0.0,
        .option_type = OptionType::Call
    };

    auto result = bs_model.greeks(params);
    ASSERT_TRUE(result.has_value());

    auto greeks = *result;

    // ATM call delta ~0.5-0.6
    EXPECT_GT(greeks.delta, 0.45);
    EXPECT_LT(greeks.delta, 0.65);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\nRunning Black-Scholes Tests with C++23...\n\n";
    return RUN_ALL_TESTS();
}
