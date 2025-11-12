/**
 * BigBrotherAnalytics - Straddle & Strangle Options Strategies (C++23)
 *
 * Tier 2: Volatility Strategies (8 strategies)
 * - Long Straddle (profit from volatility)
 * - Short Straddle (profit from low volatility)
 * - Long Strangle (cheaper volatility play)
 * - Short Strangle (premium collection)
 * - Long Gut (expensive volatility play)
 * - Short Gut (high premium collection)
 * - Strap (bullish volatility bias - 2 calls, 1 put)
 * - Strip (bearish volatility bias - 1 call, 2 puts)
 *
 * These strategies profit from volatility (or lack thereof) rather than
 * directional moves. Same expiration, potentially different strikes.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 */

module;

#include <algorithm>
#include <cmath>
#include <immintrin.h>  // AVX2 intrinsics
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

export module bigbrother.options_strategies.straddles_strangles;

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// ============================================================================
// Long Straddle Strategy
// ============================================================================

/**
 * Long Straddle - Volatility strategy
 *
 * Description: Buy ATM call + ATM put (same strike)
 * Complexity: Intermediate
 * Market Outlook: Volatile (big move expected)
 *
 * Max Profit: Unlimited (large price move either direction)
 * Max Loss: Total premium paid
 * Breakevens: Strike ± Total Premium
 *
 * Legs:
 * - Buy 1 Call at strike K
 * - Buy 1 Put at strike K
 *
 * Use Case: Expecting big move but unsure of direction (earnings, FDA approval, etc.)
 */
class LongStraddleStrategy final : public BaseOptionsStrategy<LongStraddleStrategy> {
public:
    LongStraddleStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Straddle",
            "Buy ATM call + put - profit from big moves in either direction",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Long call at strike
        float call_premium = simd::blackScholesCall(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Long put at same strike
        float put_premium = simd::blackScholesPut(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(long_call));
        addLeg(std::move(long_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_intrinsic - call_leg.premium) * call_leg.quantity;

        // Long put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_intrinsic - put_leg.premium) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            // Gamma and Vega are same for calls and puts
            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        // Total premium paid
        return legs_[0].premium + legs_[1].premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        float strike = legs_[0].strike;
        return {strike - total_premium, strike + total_premium};
    }
};

// ============================================================================
// Short Straddle Strategy
// ============================================================================

/**
 * Short Straddle - Low volatility strategy
 *
 * Description: Sell ATM call + ATM put (same strike)
 * Complexity: Advanced (unlimited risk both directions)
 * Market Outlook: Neutral (low volatility expected)
 *
 * Max Profit: Total premium received
 * Max Loss: Unlimited
 * Breakevens: Strike ± Total Premium
 *
 * Use Case: Expecting low volatility, range-bound trading
 */
class ShortStraddleStrategy final : public BaseOptionsStrategy<ShortStraddleStrategy> {
public:
    ShortStraddleStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Straddle",
            "Sell ATM call + put - collect premium, profit from low volatility",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Short call
        float call_premium = simd::blackScholesCall(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_call{
            .is_call = true,
            .is_long = false,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Short put
        float put_premium = simd::blackScholesPut(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_put{
            .is_call = false,
            .is_long = false,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(short_call));
        addLeg(std::move(short_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Short call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_leg.premium - call_intrinsic) * call_leg.quantity;

        // Short put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_leg.premium - put_intrinsic) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[0].premium + legs_[1].premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited both directions
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        float strike = legs_[0].strike;
        return {strike - total_premium, strike + total_premium};
    }
};

// ============================================================================
// Long Strangle Strategy
// ============================================================================

/**
 * Long Strangle - Volatility strategy (cheaper than straddle)
 *
 * Description: Buy OTM call + OTM put (different strikes)
 * Complexity: Intermediate
 * Market Outlook: Volatile (big move expected)
 *
 * Max Profit: Unlimited
 * Max Loss: Total premium paid
 * Breakevens: Lower Strike - Premium, Upper Strike + Premium
 *
 * Use Case: Expecting big move, want lower cost than straddle
 */
class LongStrangleStrategy final : public BaseOptionsStrategy<LongStrangleStrategy> {
public:
    LongStrangleStrategy(
        float underlying_price,
        float lower_strike,   // OTM put strike
        float upper_strike,   // OTM call strike
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Strangle",
            "Buy OTM call + OTM put - cheaper volatility play",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Long call at upper strike (OTM)
        float call_premium = simd::blackScholesCall(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Long put at lower strike (OTM)
        float put_premium = simd::blackScholesPut(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(long_call));
        addLeg(std::move(long_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_intrinsic - call_leg.premium) * call_leg.quantity;

        // Long put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_intrinsic - put_leg.premium) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return legs_[0].premium + legs_[1].premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        return {legs_[1].strike - total_premium, legs_[0].strike + total_premium};
    }
};

// ============================================================================
// Short Strangle Strategy
// ============================================================================

/**
 * Short Strangle - Premium collection strategy
 *
 * Description: Sell OTM call + OTM put
 * Complexity: Advanced
 * Market Outlook: Neutral (range-bound expected)
 *
 * Max Profit: Total premium received
 * Max Loss: Unlimited
 * Breakevens: Lower Strike - Premium, Upper Strike + Premium
 *
 * Use Case: Collect premium, expect price to stay in range
 */
class ShortStrangleStrategy final : public BaseOptionsStrategy<ShortStrangleStrategy> {
public:
    ShortStrangleStrategy(
        float underlying_price,
        float lower_strike,
        float upper_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Strangle",
            "Sell OTM call + OTM put - collect premium from range-bound trading",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Short call at upper strike
        float call_premium = simd::blackScholesCall(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_call{
            .is_call = true,
            .is_long = false,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Short put at lower strike
        float put_premium = simd::blackScholesPut(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_put{
            .is_call = false,
            .is_long = false,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(short_call));
        addLeg(std::move(short_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Short call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_leg.premium - call_intrinsic) * call_leg.quantity;

        // Short put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_leg.premium - put_intrinsic) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[0].premium + legs_[1].premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        return {legs_[1].strike - total_premium, legs_[0].strike + total_premium};
    }
};

// ============================================================================
// Long Gut Strategy
// ============================================================================

/**
 * Long Gut - Expensive volatility strategy
 *
 * Description: Buy ITM call + ITM put
 * Complexity: Advanced
 * Market Outlook: Volatile
 *
 * Max Profit: Unlimited
 * Max Loss: Total premium - (Upper Strike - Lower Strike)
 * Breakevens: Complex calculation
 *
 * Use Case: Rare - expecting huge move, ITM options have higher delta
 */
class LongGutStrategy final : public BaseOptionsStrategy<LongGutStrategy> {
public:
    LongGutStrategy(
        float underlying_price,
        float lower_strike,   // ITM put strike (below current price)
        float upper_strike,   // ITM call strike (above current price)
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Gut",
            "Buy ITM call + ITM put - expensive volatility play",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Long ITM call
        float call_premium = simd::blackScholesCall(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Long ITM put
        float put_premium = simd::blackScholesPut(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(long_call));
        addLeg(std::move(long_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_intrinsic - call_leg.premium) * call_leg.quantity;

        // Long put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_intrinsic - put_leg.premium) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        float intrinsic_value = legs_[1].strike - legs_[0].strike;
        return total_premium - intrinsic_value;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        // Complex - approximate
        float total_premium = legs_[0].premium + legs_[1].premium;
        float net_cost = total_premium - (legs_[1].strike - legs_[0].strike);
        return {legs_[0].strike - net_cost, legs_[1].strike + net_cost};
    }
};

// ============================================================================
// Short Gut Strategy
// ============================================================================

/**
 * Short Gut - High premium collection, high risk
 *
 * Description: Sell ITM call + ITM put
 * Complexity: Complex
 * Market Outlook: Neutral (very low volatility expected)
 *
 * Max Profit: Total premium - (Upper Strike - Lower Strike)
 * Max Loss: Unlimited
 *
 * Use Case: Very rare - extreme premium collection
 */
class ShortGutStrategy final : public BaseOptionsStrategy<ShortGutStrategy> {
public:
    ShortGutStrategy(
        float underlying_price,
        float lower_strike,
        float upper_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Gut",
            "Sell ITM call + ITM put - high premium, very high risk",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::COMPLEX)
    {
        float T = days_to_expiration / 365.0f;

        // Short ITM call
        float call_premium = simd::blackScholesCall(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_call{
            .is_call = true,
            .is_long = false,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Short ITM put
        float put_premium = simd::blackScholesPut(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_put{
            .is_call = false,
            .is_long = false,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(short_call));
        addLeg(std::move(short_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Short call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_leg.premium - call_intrinsic) * call_leg.quantity;

        // Short put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_leg.premium - put_intrinsic) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        float intrinsic_value = legs_[1].strike - legs_[0].strike;
        return total_premium - intrinsic_value;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = legs_[0].premium + legs_[1].premium;
        float net_credit = total_premium - (legs_[1].strike - legs_[0].strike);
        return {legs_[0].strike - net_credit, legs_[1].strike + net_credit};
    }
};

// ============================================================================
// Strap Strategy (Bullish Volatility Bias)
// ============================================================================

/**
 * Strap - Bullish volatility strategy
 *
 * Description: Buy 2 ATM calls + 1 ATM put (same strike)
 * Complexity: Intermediate
 * Market Outlook: Bullish Volatile
 *
 * Max Profit: Unlimited (favors upside)
 * Max Loss: Total premium paid
 * Breakevens: 2 points (asymmetric)
 *
 * Use Case: Expect volatility with bullish bias
 */
class StrapStrategy final : public BaseOptionsStrategy<StrapStrategy> {
public:
    StrapStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Strap",
            "Buy 2 ATM calls + 1 ATM put - bullish volatility bias",
            StrategyType::COMBINATION,
            MarketOutlook::BULLISH_VOLATILE,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Long 2 calls
        float call_premium = simd::blackScholesCall(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = strike,
            .quantity = 2.0f,  // 2 calls
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Long 1 put
        float put_premium = simd::blackScholesPut(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(long_call));
        addLeg(std::move(long_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long calls (2x)
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_intrinsic - call_leg.premium) * call_leg.quantity;

        // Long put
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_intrinsic - put_leg.premium) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited upside
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return (legs_[0].premium * 2.0f) + legs_[1].premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = (legs_[0].premium * 2.0f) + legs_[1].premium;
        float strike = legs_[0].strike;
        // Downside: strike - total_premium
        // Upside: strike + (total_premium / 2) because 2 calls
        return {strike - total_premium, strike + (total_premium / 2.0f)};
    }
};

// ============================================================================
// Strip Strategy (Bearish Volatility Bias)
// ============================================================================

/**
 * Strip - Bearish volatility strategy
 *
 * Description: Buy 1 ATM call + 2 ATM puts (same strike)
 * Complexity: Intermediate
 * Market Outlook: Bearish Volatile
 *
 * Max Profit: Limited downside, unlimited upside
 * Max Loss: Total premium paid
 * Breakevens: 2 points (asymmetric, favors downside)
 *
 * Use Case: Expect volatility with bearish bias
 */
class StripStrategy final : public BaseOptionsStrategy<StripStrategy> {
public:
    StripStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Strip",
            "Buy 1 ATM call + 2 ATM puts - bearish volatility bias",
            StrategyType::COMBINATION,
            MarketOutlook::BEARISH_VOLATILE,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Long 1 call
        float call_premium = simd::blackScholesCall(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        };

        // Long 2 puts
        float put_premium = simd::blackScholesPut(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = strike,
            .quantity = 2.0f,  // 2 puts
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        };

        addLeg(std::move(long_call));
        addLeg(std::move(long_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long call
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        pnl += (call_intrinsic - call_leg.premium) * call_leg.quantity;

        // Long puts (2x)
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        pnl += (put_intrinsic - put_leg.premium) * put_leg.quantity;

        return pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * leg.quantity;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * leg.quantity;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * leg.quantity;
            }

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        // Max profit on downside: strike * 2 - total premium
        float total_premium = legs_[0].premium + (legs_[1].premium * 2.0f);
        return (legs_[0].strike * 2.0f) - total_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return legs_[0].premium + (legs_[1].premium * 2.0f);
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float total_premium = legs_[0].premium + (legs_[1].premium * 2.0f);
        float strike = legs_[0].strike;
        // Downside: strike - (total_premium / 2) because 2 puts
        // Upside: strike + total_premium
        return {strike - (total_premium / 2.0f), strike + total_premium};
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] inline auto createLongStraddle(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongStraddleStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortStraddle(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortStraddleStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createLongStrangle(
    float underlying_price,
    float lower_strike,
    float upper_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongStrangleStrategy>(
        underlying_price, lower_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortStrangle(
    float underlying_price,
    float lower_strike,
    float upper_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortStrangleStrategy>(
        underlying_price, lower_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createLongGut(
    float underlying_price,
    float lower_strike,
    float upper_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongGutStrategy>(
        underlying_price, lower_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortGut(
    float underlying_price,
    float lower_strike,
    float upper_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortGutStrategy>(
        underlying_price, lower_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createStrap(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<StrapStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createStrip(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<StripStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

} // namespace bigbrother::options_strategies
