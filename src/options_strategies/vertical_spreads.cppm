/**
 * BigBrotherAnalytics - Vertical Spread Options Strategies (C++23)
 *
 * Tier 2: Vertical Spreads (4 strategies)
 * - Bull Call Spread (Bullish, limited risk/reward)
 * - Bull Put Spread (Bullish, credit spread)
 * - Bear Call Spread (Bearish, credit spread)
 * - Bear Put Spread (Bearish, limited risk/reward)
 *
 * Vertical spreads combine two options of the same type (both calls or both puts)
 * with the same expiration but different strike prices. They offer defined
 * risk/reward profiles ideal for directional trading.
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

export module bigbrother.options_strategies.vertical_spreads;

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// ============================================================================
// Bull Call Spread Strategy
// ============================================================================

/**
 * Bull Call Spread - Bullish vertical spread
 *
 * Description: Buy lower strike call, sell higher strike call
 * Complexity: Beginner/Intermediate
 * Market Outlook: Moderately Bullish
 *
 * Max Profit: (Higher Strike - Lower Strike) - Net Debit
 * Max Loss: Net Debit (premium paid)
 * Breakeven: Lower Strike + Net Debit
 *
 * Legs:
 * - Buy 1 Call at lower strike (K1)
 * - Sell 1 Call at higher strike (K2, K2 > K1)
 *
 * Use Case: Expect moderate upside, want to reduce cost vs long call
 */
class BullCallSpreadStrategy final : public BaseOptionsStrategy<BullCallSpreadStrategy> {
public:
    /**
     * Constructor
     *
     * @param underlying_price Current price of underlying
     * @param lower_strike Lower strike price (BUY)
     * @param higher_strike Higher strike price (SELL)
     * @param days_to_expiration Days until expiration
     * @param implied_volatility Implied volatility (e.g., 0.25 = 25%)
     * @param risk_free_rate Risk-free rate (e.g., 0.05 = 5%)
     */
    BullCallSpreadStrategy(
        float underlying_price,
        float lower_strike,
        float higher_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Bull Call Spread",
            "Buy lower strike call, sell higher strike call - limited risk, limited reward",
            StrategyType::VERTICAL_SPREAD,
            MarketOutlook::BULLISH,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Long call at lower strike
        float premium_long = simd::blackScholesCall(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_long
        };

        // Short call at higher strike
        float premium_short = simd::blackScholesCall(
            underlying_price, higher_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_call{
            .is_call = true,
            .is_long = false,
            .strike = higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_short
        };

        addLeg(std::move(long_call));
        addLeg(std::move(short_call));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long call (lower strike)
        auto const& long_leg = legs_[0];
        float intrinsic_long = std::max(0.0f, underlying_price - long_leg.strike);
        pnl += (intrinsic_long - long_leg.premium) * long_leg.quantity;

        // Short call (higher strike)
        auto const& short_leg = legs_[1];
        float intrinsic_short = std::max(0.0f, underlying_price - short_leg.strike);
        pnl += (short_leg.premium - intrinsic_short) * short_leg.quantity;

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

            // Delta
            float delta = simd::deltaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            // Gamma
            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            // Theta
            float theta = simd::thetaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            // Vega
            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

            // Rho
            float rho = simd::rhoCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.rho += rho * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float spread_width = legs_[1].strike - legs_[0].strike;
        float net_debit = legs_[0].premium - legs_[1].premium;
        return spread_width - net_debit;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        // Net debit paid
        return legs_[0].premium - legs_[1].premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net_debit = legs_[0].premium - legs_[1].premium;
        return {legs_[0].strike + net_debit};
    }
};

// ============================================================================
// Bull Put Spread Strategy
// ============================================================================

/**
 * Bull Put Spread - Bullish vertical spread (credit spread)
 *
 * Description: Sell higher strike put, buy lower strike put
 * Complexity: Intermediate
 * Market Outlook: Moderately Bullish
 *
 * Max Profit: Net Credit (premium received)
 * Max Loss: (Higher Strike - Lower Strike) - Net Credit
 * Breakeven: Higher Strike - Net Credit
 *
 * Legs:
 * - Sell 1 Put at higher strike (K2)
 * - Buy 1 Put at lower strike (K1, K1 < K2)
 *
 * Use Case: Collect premium with defined risk, expect upside or neutral
 */
class BullPutSpreadStrategy final : public BaseOptionsStrategy<BullPutSpreadStrategy> {
public:
    BullPutSpreadStrategy(
        float underlying_price,
        float lower_strike,
        float higher_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Bull Put Spread",
            "Sell higher strike put, buy lower strike put - credit spread",
            StrategyType::VERTICAL_SPREAD,
            MarketOutlook::BULLISH,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Short put at higher strike
        float premium_short = simd::blackScholesPut(
            underlying_price, higher_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_put{
            .is_call = false,
            .is_long = false,
            .strike = higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_short
        };

        // Long put at lower strike
        float premium_long = simd::blackScholesPut(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_long
        };

        addLeg(std::move(short_put));
        addLeg(std::move(long_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Short put (higher strike)
        auto const& short_leg = legs_[0];
        float intrinsic_short = std::max(0.0f, short_leg.strike - underlying_price);
        pnl += (short_leg.premium - intrinsic_short) * short_leg.quantity;

        // Long put (lower strike)
        auto const& long_leg = legs_[1];
        float intrinsic_long = std::max(0.0f, long_leg.strike - underlying_price);
        pnl += (intrinsic_long - long_leg.premium) * long_leg.quantity;

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

            // Delta
            float delta = simd::deltaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            // Gamma
            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            // Theta
            float theta = simd::thetaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            // Vega
            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

            // Rho
            float rho = simd::rhoPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.rho += rho * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        // Net credit received
        return legs_[0].premium - legs_[1].premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float spread_width = legs_[0].strike - legs_[1].strike;
        float net_credit = legs_[0].premium - legs_[1].premium;
        return spread_width - net_credit;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net_credit = legs_[0].premium - legs_[1].premium;
        return {legs_[0].strike - net_credit};
    }
};

// ============================================================================
// Bear Call Spread Strategy
// ============================================================================

/**
 * Bear Call Spread - Bearish vertical spread (credit spread)
 *
 * Description: Sell lower strike call, buy higher strike call
 * Complexity: Intermediate
 * Market Outlook: Moderately Bearish
 *
 * Max Profit: Net Credit (premium received)
 * Max Loss: (Higher Strike - Lower Strike) - Net Credit
 * Breakeven: Lower Strike + Net Credit
 *
 * Legs:
 * - Sell 1 Call at lower strike (K1)
 * - Buy 1 Call at higher strike (K2, K2 > K1)
 *
 * Use Case: Collect premium with defined risk, expect downside or neutral
 */
class BearCallSpreadStrategy final : public BaseOptionsStrategy<BearCallSpreadStrategy> {
public:
    BearCallSpreadStrategy(
        float underlying_price,
        float lower_strike,
        float higher_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Bear Call Spread",
            "Sell lower strike call, buy higher strike call - credit spread",
            StrategyType::VERTICAL_SPREAD,
            MarketOutlook::BEARISH,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Short call at lower strike
        float premium_short = simd::blackScholesCall(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_call{
            .is_call = true,
            .is_long = false,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_short
        };

        // Long call at higher strike
        float premium_long = simd::blackScholesCall(
            underlying_price, higher_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_call{
            .is_call = true,
            .is_long = true,
            .strike = higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_long
        };

        addLeg(std::move(short_call));
        addLeg(std::move(long_call));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Short call (lower strike)
        auto const& short_leg = legs_[0];
        float intrinsic_short = std::max(0.0f, underlying_price - short_leg.strike);
        pnl += (short_leg.premium - intrinsic_short) * short_leg.quantity;

        // Long call (higher strike)
        auto const& long_leg = legs_[1];
        float intrinsic_long = std::max(0.0f, underlying_price - long_leg.strike);
        pnl += (intrinsic_long - long_leg.premium) * long_leg.quantity;

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

            // Delta
            float delta = simd::deltaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            // Gamma
            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            // Theta
            float theta = simd::thetaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            // Vega
            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

            // Rho
            float rho = simd::rhoCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.rho += rho * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        // Net credit received
        return legs_[0].premium - legs_[1].premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float spread_width = legs_[1].strike - legs_[0].strike;
        float net_credit = legs_[0].premium - legs_[1].premium;
        return spread_width - net_credit;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net_credit = legs_[0].premium - legs_[1].premium;
        return {legs_[0].strike + net_credit};
    }
};

// ============================================================================
// Bear Put Spread Strategy
// ============================================================================

/**
 * Bear Put Spread - Bearish vertical spread
 *
 * Description: Buy higher strike put, sell lower strike put
 * Complexity: Intermediate
 * Market Outlook: Moderately Bearish
 *
 * Max Profit: (Higher Strike - Lower Strike) - Net Debit
 * Max Loss: Net Debit (premium paid)
 * Breakeven: Higher Strike - Net Debit
 *
 * Legs:
 * - Buy 1 Put at higher strike (K2)
 * - Sell 1 Put at lower strike (K1, K1 < K2)
 *
 * Use Case: Expect moderate downside, want to reduce cost vs long put
 */
class BearPutSpreadStrategy final : public BaseOptionsStrategy<BearPutSpreadStrategy> {
public:
    BearPutSpreadStrategy(
        float underlying_price,
        float lower_strike,
        float higher_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Bear Put Spread",
            "Buy higher strike put, sell lower strike put - limited risk, limited reward",
            StrategyType::VERTICAL_SPREAD,
            MarketOutlook::BEARISH,
            ComplexityLevel::INTERMEDIATE)
    {
        float T = days_to_expiration / 365.0f;

        // Long put at higher strike
        float premium_long = simd::blackScholesPut(
            underlying_price, higher_strike, T, risk_free_rate, implied_volatility);

        OptionLeg long_put{
            .is_call = false,
            .is_long = true,
            .strike = higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_long
        };

        // Short put at lower strike
        float premium_short = simd::blackScholesPut(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);

        OptionLeg short_put{
            .is_call = false,
            .is_long = false,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium_short
        };

        addLeg(std::move(long_put));
        addLeg(std::move(short_put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        float pnl = 0.0f;

        // Long put (higher strike)
        auto const& long_leg = legs_[0];
        float intrinsic_long = std::max(0.0f, long_leg.strike - underlying_price);
        pnl += (intrinsic_long - long_leg.premium) * long_leg.quantity;

        // Short put (lower strike)
        auto const& short_leg = legs_[1];
        float intrinsic_short = std::max(0.0f, short_leg.strike - underlying_price);
        pnl += (short_leg.premium - intrinsic_short) * short_leg.quantity;

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

            // Delta
            float delta = simd::deltaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            // Gamma
            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            // Theta
            float theta = simd::thetaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            // Vega
            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

            // Rho
            float rho = simd::rhoPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.rho += rho * sign * leg.quantity;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float spread_width = legs_[0].strike - legs_[1].strike;
        float net_debit = legs_[0].premium - legs_[1].premium;
        return spread_width - net_debit;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        // Net debit paid
        return legs_[0].premium - legs_[1].premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net_debit = legs_[0].premium - legs_[1].premium;
        return {legs_[0].strike - net_debit};
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] inline auto createBullCallSpread(
    float underlying_price,
    float lower_strike,
    float higher_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<BullCallSpreadStrategy>(
        underlying_price, lower_strike, higher_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createBullPutSpread(
    float underlying_price,
    float lower_strike,
    float higher_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<BullPutSpreadStrategy>(
        underlying_price, lower_strike, higher_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createBearCallSpread(
    float underlying_price,
    float lower_strike,
    float higher_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<BearCallSpreadStrategy>(
        underlying_price, lower_strike, higher_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createBearPutSpread(
    float underlying_price,
    float lower_strike,
    float higher_strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<BearPutSpreadStrategy>(
        underlying_price, lower_strike, higher_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

} // namespace bigbrother::options_strategies
