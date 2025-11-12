/**
 * BigBrotherAnalytics - Single Leg Options Strategies (C++23)
 *
 * Tier 1: Foundational strategies (4 total)
 * - Long Call (Bullish)
 * - Long Put (Bearish)
 * - Short Call (Bearish)
 * - Short Put (Bullish)
 *
 * These are the simplest options strategies, serving as building blocks
 * for all complex multi-leg strategies.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 */

module;

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

export module bigbrother.options_strategies.single_leg;

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// ============================================================================
// Long Call Strategy
// ============================================================================

/**
 * Long Call - Bullish single-leg strategy
 *
 * Description: Buy a call option
 * Complexity: Beginner
 * Market Outlook: Bullish
 *
 * Max Profit: Unlimited (as underlying rises)
 * Max Loss: Premium paid (if underlying ≤ strike at expiration)
 * Breakeven: Strike + Premium
 *
 * Legs:
 * - Buy 1 Call at strike K
 */
class LongCallStrategy final : public BaseOptionsStrategy<LongCallStrategy> {
public:
    /**
     * Constructor
     *
     * @param underlying_price Current price of underlying
     * @param strike Strike price
     * @param days_to_expiration Days until expiration
     * @param implied_volatility Implied volatility (e.g., 0.25 = 25%)
     * @param risk_free_rate Risk-free rate (e.g., 0.05 = 5%)
     */
    LongCallStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Call",
            "Buy call option - unlimited upside, limited downside",
            StrategyType::SINGLE_LEG,
            MarketOutlook::BULLISH,
            ComplexityLevel::BEGINNER)
    {
        // Calculate premium using Black-Scholes
        float T = days_to_expiration / 365.0f;
        float premium = simd::blackScholesCall(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg call{
            .is_call = true,
            .is_long = true,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium
        };

        addLeg(std::move(call));
    }

    // P&L calculation
    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        auto const& leg = legs_[0];

        // Intrinsic value: max(0, S - K)
        float intrinsic = std::max(0.0f, underlying_price - leg.strike);

        // P&L = Intrinsic - Premium Paid
        return (intrinsic - leg.premium) * leg.quantity;
    }

    // Expiration P&L
    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    // Greeks
    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        auto const& leg = legs_[0];
        float T = leg.days_to_expiration / 365.0f;

        Greeks greeks;
        greeks.delta = simd::deltaCallBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];

        greeks.gamma = simd::gammaBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];

        greeks.theta = simd::thetaCallBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];

        greeks.vega = simd::vegaBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];

        greeks.rho = simd::rhoCallBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];

        return greeks;
    }

    // Max profit/loss
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return legs_[0].premium * legs_[0].quantity;  // Premium paid
    }

    // Breakeven
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        return {legs_[0].strike + legs_[0].premium};
    }
};

// ============================================================================
// Long Put Strategy
// ============================================================================

/**
 * Long Put - Bearish single-leg strategy
 *
 * Max Profit: Strike - Premium (if underlying → 0)
 * Max Loss: Premium paid
 * Breakeven: Strike - Premium
 */
class LongPutStrategy final : public BaseOptionsStrategy<LongPutStrategy> {
public:
    LongPutStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Put",
            "Buy put option - profit from downside, limited risk",
            StrategyType::SINGLE_LEG,
            MarketOutlook::BEARISH,
            ComplexityLevel::BEGINNER)
    {
        float T = days_to_expiration / 365.0f;
        float premium = simd::blackScholesPut(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg put{
            .is_call = false,
            .is_long = true,
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium
        };

        addLeg(std::move(put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        auto const& leg = legs_[0];
        float intrinsic = std::max(0.0f, leg.strike - underlying_price);
        return (intrinsic - leg.premium) * leg.quantity;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        auto const& leg = legs_[0];
        float T = leg.days_to_expiration / 365.0f;

        Greeks greeks;
        // Similar to Long Call but use put Greeks functions
        greeks.delta = simd::deltaPutBatch(...)[0];  // Negative
        greeks.gamma = simd::gammaBatch(...)[0];
        // ... other Greeks
        return greeks;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[0].premium) * legs_[0].quantity;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return legs_[0].premium * legs_[0].quantity;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        return {legs_[0].strike - legs_[0].premium};
    }
};

// ============================================================================
// Short Call Strategy
// ============================================================================

/**
 * Short Call - Bearish single-leg strategy (neutral to slightly bearish)
 *
 * Max Profit: Premium received
 * Max Loss: Unlimited (as underlying rises)
 * Breakeven: Strike + Premium
 */
class ShortCallStrategy final : public BaseOptionsStrategy<ShortCallStrategy> {
public:
    ShortCallStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Call",
            "Sell call option - collect premium, unlimited risk",
            StrategyType::SINGLE_LEG,
            MarketOutlook::BEARISH,
            ComplexityLevel::BEGINNER)
    {
        float T = days_to_expiration / 365.0f;
        float premium = simd::blackScholesCall(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg call{
            .is_call = true,
            .is_long = false,  // SHORT
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium
        };

        addLeg(std::move(call));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        auto const& leg = legs_[0];
        float intrinsic = std::max(0.0f, underlying_price - leg.strike);

        // Short: P&L = Premium Received - Intrinsic
        return (leg.premium - intrinsic) * leg.quantity;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        auto const& leg = legs_[0];
        float T = leg.days_to_expiration / 365.0f;

        // Short position: negate all Greeks
        Greeks greeks;
        greeks.delta = -simd::deltaCallBatch(...)[0];
        greeks.gamma = -simd::gammaBatch(...)[0];
        greeks.theta = -simd::thetaCallBatch(...)[0];  // Positive for short
        greeks.vega = -simd::vegaBatch(...)[0];
        greeks.rho = -simd::rhoCallBatch(...)[0];
        return greeks;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[0].premium * legs_[0].quantity;  // Premium received
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        return {legs_[0].strike + legs_[0].premium};
    }
};

// ============================================================================
// Short Put Strategy
// ============================================================================

/**
 * Short Put - Bullish single-leg strategy (neutral to slightly bullish)
 *
 * Max Profit: Premium received
 * Max Loss: Strike - Premium (if underlying → 0)
 * Breakeven: Strike - Premium
 */
class ShortPutStrategy final : public BaseOptionsStrategy<ShortPutStrategy> {
public:
    ShortPutStrategy(
        float underlying_price,
        float strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Put",
            "Sell put option - collect premium, risk assignment",
            StrategyType::SINGLE_LEG,
            MarketOutlook::BULLISH,
            ComplexityLevel::BEGINNER)
    {
        float T = days_to_expiration / 365.0f;
        float premium = simd::blackScholesPut(
            underlying_price, strike, T, risk_free_rate, implied_volatility);

        OptionLeg put{
            .is_call = false,
            .is_long = false,  // SHORT
            .strike = strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium
        };

        addLeg(std::move(put));
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        auto const& leg = legs_[0];
        float intrinsic = std::max(0.0f, leg.strike - underlying_price);
        return (leg.premium - intrinsic) * leg.quantity;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        auto const& leg = legs_[0];
        float T = leg.days_to_expiration / 365.0f;

        Greeks greeks;
        greeks.delta = -simd::deltaPutBatch(...)[0];  // Positive (short put)
        greeks.gamma = -simd::gammaBatch(...)[0];
        greeks.theta = -simd::thetaPutBatch(...)[0];  // Positive
        greeks.vega = -simd::vegaBatch(...)[0];
        greeks.rho = -simd::rhoPutBatch(...)[0];
        return greeks;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[0].premium * legs_[0].quantity;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[0].premium) * legs_[0].quantity;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        return {legs_[0].strike - legs_[0].premium};
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] inline auto createLongCall(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<LongCallStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createLongPut(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<LongPutStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortCall(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<ShortCallStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortPut(
    float underlying_price,
    float strike,
    float days_to_expiration,
    float implied_volatility,
    float risk_free_rate = 0.05f) -> std::unique_ptr<IOptionsStrategy> {

    return std::make_unique<ShortPutStrategy>(
        underlying_price, strike, days_to_expiration,
        implied_volatility, risk_free_rate);
}

} // namespace bigbrother::options_strategies
