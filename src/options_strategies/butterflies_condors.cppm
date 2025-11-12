/**
 * @file butterflies_condors.cppm
 * @brief Butterfly and Condor options strategies (Tier 3)
 *
 * Implements 12 advanced 4-leg strategies with symmetric risk profiles:
 * - 4 Butterfly spreads (call/put, long/short)
 * - 4 Condor spreads (call/put, long/short)
 * - 2 Iron Butterfly spreads (long/short)
 * - 2 Iron Condor spreads (long/short)
 *
 * All strategies use AVX2 SIMD intrinsics for high-performance Greeks calculations.
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

export module bigbrother.options_strategies.butterflies_condors;

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// ============================================================================
// BUTTERFLY SPREADS (4 strategies)
// ============================================================================

/**
 * Long Call Butterfly
 *
 * Construction:
 * - Buy 1 call at lower strike (K1)
 * - Sell 2 calls at middle strike (K2)
 * - Buy 1 call at higher strike (K3)
 *
 * Market Outlook: Neutral (profit from minimal price movement)
 * Max Profit: (K2 - K1) - net premium (when S = K2 at expiration)
 * Max Loss: Net premium paid (when S <= K1 or S >= K3)
 * Breakevens: K1 + net premium, K3 - net premium
 *
 * Characteristics:
 * - Limited risk, limited profit
 * - Low cost entry
 * - Delta neutral near K2
 * - Positive theta (time decay benefits short strikes)
 */
class LongCallButterflyStrategy final : public BaseOptionsStrategy<LongCallButterflyStrategy> {
public:
    LongCallButterflyStrategy(
        float underlying_price,
        float lower_strike,     // K1
        float middle_strike,    // K2
        float upper_strike,     // K3
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Call Butterfly",
            "Buy low call, sell 2 mid calls, buy high call - profit from minimal movement",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Buy 1 call at lower strike
        float lower_premium = simd::blackScholesCall(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = lower_premium
        });

        // Sell 2 calls at middle strike
        float middle_premium = simd::blackScholesCall(
            underlying_price, middle_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = middle_strike,
            .quantity = 2.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = middle_premium
        });

        // Buy 1 call at upper strike
        float upper_premium = simd::blackScholesCall(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = upper_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call
                ? std::max(0.0f, underlying_price - leg.strike)
                : std::max(0.0f, leg.strike - underlying_price);

            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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
        float lower_strike = legs_[0].strike;
        float middle_strike = legs_[1].strike;
        float net_premium = -legs_[0].premium + 2.0f * legs_[1].premium - legs_[2].premium;
        return (middle_strike - lower_strike) - net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float net_premium = -legs_[0].premium + 2.0f * legs_[1].premium - legs_[2].premium;
        return net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float lower_strike = legs_[0].strike;
        float upper_strike = legs_[2].strike;
        float net_premium = -legs_[0].premium + 2.0f * legs_[1].premium - legs_[2].premium;
        return {lower_strike + net_premium, upper_strike - net_premium};
    }
};

/**
 * Short Call Butterfly
 *
 * Opposite of Long Call Butterfly
 * - Sell 1 call at lower strike
 * - Buy 2 calls at middle strike
 * - Sell 1 call at upper strike
 *
 * Market Outlook: Volatile (profit from large price movement)
 * Max Profit: Net premium received
 * Max Loss: (K2 - K1) - net premium
 */
class ShortCallButterflyStrategy final : public BaseOptionsStrategy<ShortCallButterflyStrategy> {
public:
    ShortCallButterflyStrategy(
        float underlying_price,
        float lower_strike,
        float middle_strike,
        float upper_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Call Butterfly",
            "Sell low call, buy 2 mid calls, sell high call - profit from large movements",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Sell 1 call at lower strike
        float lower_premium = simd::blackScholesCall(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = lower_premium
        });

        // Buy 2 calls at middle strike
        float middle_premium = simd::blackScholesCall(
            underlying_price, middle_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = middle_strike,
            .quantity = 2.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = middle_premium
        });

        // Sell 1 call at upper strike
        float upper_premium = simd::blackScholesCall(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = upper_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call
                ? std::max(0.0f, underlying_price - leg.strike)
                : std::max(0.0f, leg.strike - underlying_price);

            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float net_premium = legs_[0].premium - 2.0f * legs_[1].premium + legs_[2].premium;
        return net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float lower_strike = legs_[0].strike;
        float middle_strike = legs_[1].strike;
        float net_premium = legs_[0].premium - 2.0f * legs_[1].premium + legs_[2].premium;
        return (middle_strike - lower_strike) - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float lower_strike = legs_[0].strike;
        float upper_strike = legs_[2].strike;
        float net_premium = legs_[0].premium - 2.0f * legs_[1].premium + legs_[2].premium;
        return {lower_strike + net_premium, upper_strike - net_premium};
    }
};

/**
 * Long Put Butterfly
 *
 * Construction:
 * - Buy 1 put at higher strike (K3)
 * - Sell 2 puts at middle strike (K2)
 * - Buy 1 put at lower strike (K1)
 *
 * Market Outlook: Neutral
 * Similar to Long Call Butterfly but using puts
 */
class LongPutButterflyStrategy final : public BaseOptionsStrategy<LongPutButterflyStrategy> {
public:
    LongPutButterflyStrategy(
        float underlying_price,
        float lower_strike,
        float middle_strike,
        float upper_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Put Butterfly",
            "Buy high put, sell 2 mid puts, buy low put - profit from minimal movement",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Buy 1 put at upper strike
        float upper_premium = simd::blackScholesPut(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = upper_premium
        });

        // Sell 2 puts at middle strike
        float middle_premium = simd::blackScholesPut(
            underlying_price, middle_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = middle_strike,
            .quantity = 2.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = middle_premium
        });

        // Buy 1 put at lower strike
        float lower_premium = simd::blackScholesPut(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = lower_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - underlying_price);
            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float middle_strike = legs_[1].strike;
        float upper_strike = legs_[0].strike;
        float net_premium = -legs_[0].premium + 2.0f * legs_[1].premium - legs_[2].premium;
        return (upper_strike - middle_strike) - net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float net_premium = -legs_[0].premium + 2.0f * legs_[1].premium - legs_[2].premium;
        return net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float lower_strike = legs_[2].strike;
        float upper_strike = legs_[0].strike;
        float net_premium = -legs_[0].premium + 2.0f * legs_[1].premium - legs_[2].premium;
        return {lower_strike + net_premium, upper_strike - net_premium};
    }
};

/**
 * Short Put Butterfly
 *
 * Opposite of Long Put Butterfly
 * Market Outlook: Volatile
 */
class ShortPutButterflyStrategy final : public BaseOptionsStrategy<ShortPutButterflyStrategy> {
public:
    ShortPutButterflyStrategy(
        float underlying_price,
        float lower_strike,
        float middle_strike,
        float upper_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Put Butterfly",
            "Sell high put, buy 2 mid puts, sell low put - profit from large movements",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Sell 1 put at upper strike
        float upper_premium = simd::blackScholesPut(
            underlying_price, upper_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = upper_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = upper_premium
        });

        // Buy 2 puts at middle strike
        float middle_premium = simd::blackScholesPut(
            underlying_price, middle_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = middle_strike,
            .quantity = 2.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = middle_premium
        });

        // Sell 1 put at lower strike
        float lower_premium = simd::blackScholesPut(
            underlying_price, lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = lower_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - underlying_price);
            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float net_premium = legs_[0].premium - 2.0f * legs_[1].premium + legs_[2].premium;
        return net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float middle_strike = legs_[1].strike;
        float upper_strike = legs_[0].strike;
        float net_premium = legs_[0].premium - 2.0f * legs_[1].premium + legs_[2].premium;
        return (upper_strike - middle_strike) - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float lower_strike = legs_[2].strike;
        float upper_strike = legs_[0].strike;
        float net_premium = legs_[0].premium - 2.0f * legs_[1].premium + legs_[2].premium;
        return {lower_strike + net_premium, upper_strike - net_premium};
    }
};

// ============================================================================
// CONDOR SPREADS (4 strategies)
// ============================================================================

/**
 * Long Call Condor
 *
 * Construction:
 * - Buy 1 call at K1 (lowest strike)
 * - Sell 1 call at K2 (low-middle strike)
 * - Sell 1 call at K3 (high-middle strike)
 * - Buy 1 call at K4 (highest strike)
 *
 * Market Outlook: Neutral (profit from price staying in range)
 * Max Profit: (K2 - K1) - net premium (when K2 <= S <= K3)
 * Max Loss: Net premium paid
 *
 * Characteristics:
 * - Wider profit zone than butterfly
 * - Lower maximum profit than butterfly
 * - Four different strikes
 */
class LongCallCondorStrategy final : public BaseOptionsStrategy<LongCallCondorStrategy> {
public:
    LongCallCondorStrategy(
        float underlying_price,
        float strike1,  // Lowest
        float strike2,  // Low-middle
        float strike3,  // High-middle
        float strike4,  // Highest
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Call Condor",
            "Buy K1 call, sell K2 call, sell K3 call, buy K4 call - wider profit zone",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Buy 1 call at strike1
        float premium1 = simd::blackScholesCall(
            underlying_price, strike1, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = strike1,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium1
        });

        // Sell 1 call at strike2
        float premium2 = simd::blackScholesCall(
            underlying_price, strike2, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = strike2,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium2
        });

        // Sell 1 call at strike3
        float premium3 = simd::blackScholesCall(
            underlying_price, strike3, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = strike3,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium3
        });

        // Buy 1 call at strike4
        float premium4 = simd::blackScholesCall(
            underlying_price, strike4, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = strike4,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium4
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, underlying_price - leg.strike);
            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float strike1 = legs_[0].strike;
        float strike2 = legs_[1].strike;
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return (strike2 - strike1) - net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float strike1 = legs_[0].strike;
        float strike4 = legs_[3].strike;
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return {strike1 + net_premium, strike4 - net_premium};
    }
};

/**
 * Short Call Condor - Opposite of Long Call Condor
 */
class ShortCallCondorStrategy final : public BaseOptionsStrategy<ShortCallCondorStrategy> {
public:
    ShortCallCondorStrategy(
        float underlying_price,
        float strike1,
        float strike2,
        float strike3,
        float strike4,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Call Condor",
            "Sell K1 call, buy K2 call, buy K3 call, sell K4 call - profit from breakout",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Sell 1 call at strike1
        float premium1 = simd::blackScholesCall(
            underlying_price, strike1, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = strike1,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium1
        });

        // Buy 1 call at strike2
        float premium2 = simd::blackScholesCall(
            underlying_price, strike2, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = strike2,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium2
        });

        // Buy 1 call at strike3
        float premium3 = simd::blackScholesCall(
            underlying_price, strike3, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = strike3,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium3
        });

        // Sell 1 call at strike4
        float premium4 = simd::blackScholesCall(
            underlying_price, strike4, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = strike4,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium4
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, underlying_price - leg.strike);
            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaCallBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float strike1 = legs_[0].strike;
        float strike2 = legs_[1].strike;
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return (strike2 - strike1) - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float strike1 = legs_[0].strike;
        float strike4 = legs_[3].strike;
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return {strike1 + net_premium, strike4 - net_premium};
    }
};

/**
 * Long Put Condor - Using puts instead of calls
 */
class LongPutCondorStrategy final : public BaseOptionsStrategy<LongPutCondorStrategy> {
public:
    LongPutCondorStrategy(
        float underlying_price,
        float strike1,
        float strike2,
        float strike3,
        float strike4,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Put Condor",
            "Buy K4 put, sell K3 put, sell K2 put, buy K1 put - wider profit zone with puts",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Buy 1 put at strike4 (highest)
        float premium4 = simd::blackScholesPut(
            underlying_price, strike4, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = strike4,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium4
        });

        // Sell 1 put at strike3
        float premium3 = simd::blackScholesPut(
            underlying_price, strike3, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = strike3,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium3
        });

        // Sell 1 put at strike2
        float premium2 = simd::blackScholesPut(
            underlying_price, strike2, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = strike2,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium2
        });

        // Buy 1 put at strike1 (lowest)
        float premium1 = simd::blackScholesPut(
            underlying_price, strike1, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = strike1,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium1
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - underlying_price);
            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float strike3 = legs_[1].strike;
        float strike4 = legs_[0].strike;
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return (strike4 - strike3) - net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float strike1 = legs_[3].strike;
        float strike4 = legs_[0].strike;
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return {strike1 + net_premium, strike4 - net_premium};
    }
};

/**
 * Short Put Condor - Opposite of Long Put Condor
 */
class ShortPutCondorStrategy final : public BaseOptionsStrategy<ShortPutCondorStrategy> {
public:
    ShortPutCondorStrategy(
        float underlying_price,
        float strike1,
        float strike2,
        float strike3,
        float strike4,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Put Condor",
            "Sell K4 put, buy K3 put, buy K2 put, sell K1 put - profit from breakout with puts",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Sell 1 put at strike4
        float premium4 = simd::blackScholesPut(
            underlying_price, strike4, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = strike4,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium4
        });

        // Buy 1 put at strike3
        float premium3 = simd::blackScholesPut(
            underlying_price, strike3, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = strike3,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium3
        });

        // Buy 1 put at strike2
        float premium2 = simd::blackScholesPut(
            underlying_price, strike2, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = strike2,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium2
        });

        // Sell 1 put at strike1
        float premium1 = simd::blackScholesPut(
            underlying_price, strike1, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = strike1,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = premium1
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - underlying_price);
            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            float delta = simd::deltaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.delta += delta * sign * leg.quantity;

            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * leg.quantity;

            float theta = simd::thetaPutBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.theta += theta * sign * leg.quantity;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * leg.quantity;

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
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float strike3 = legs_[1].strike;
        float strike4 = legs_[0].strike;
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return (strike4 - strike3) - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float strike1 = legs_[3].strike;
        float strike4 = legs_[0].strike;
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return {strike1 + net_premium, strike4 - net_premium};
    }
};

// ============================================================================
// IRON BUTTERFLIES (2 strategies)
// ============================================================================

/**
 * Long Iron Butterfly
 *
 * Construction:
 * - Sell 1 ATM call
 * - Sell 1 ATM put
 * - Buy 1 OTM call (higher strike)
 * - Buy 1 OTM put (lower strike)
 *
 * Market Outlook: Neutral (short straddle + protective wings)
 * Max Profit: Net premium received (when S = ATM strike)
 * Max Loss: (OTM strike - ATM strike) - net premium
 *
 * Characteristics:
 * - Limited risk version of short straddle
 * - Credit spread (receive premium)
 * - Defined risk/reward
 */
class LongIronButterflyStrategy final : public BaseOptionsStrategy<LongIronButterflyStrategy> {
public:
    LongIronButterflyStrategy(
        float underlying_price,
        float atm_strike,
        float otm_call_strike,
        float otm_put_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Iron Butterfly",
            "Sell ATM straddle + buy OTM strangle - limited risk neutral play",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Sell ATM call
        float atm_call_premium = simd::blackScholesCall(
            underlying_price, atm_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = atm_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = atm_call_premium
        });

        // Sell ATM put
        float atm_put_premium = simd::blackScholesPut(
            underlying_price, atm_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = atm_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = atm_put_premium
        });

        // Buy OTM call
        float otm_call_premium = simd::blackScholesCall(
            underlying_price, otm_call_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = otm_call_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = otm_call_premium
        });

        // Buy OTM put
        float otm_put_premium = simd::blackScholesPut(
            underlying_price, otm_put_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = otm_put_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = otm_put_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (size_t i = 0; i < legs_.size(); ++i) {
            auto const& leg = legs_[i];
            float intrinsic = leg.is_call
                ? std::max(0.0f, underlying_price - leg.strike)
                : std::max(0.0f, leg.strike - underlying_price);

            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            // Gamma and Vega same for calls/puts
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
        float net_premium = legs_[0].premium + legs_[1].premium -
                           legs_[2].premium - legs_[3].premium;
        return net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float atm_strike = legs_[0].strike;
        float otm_call_strike = legs_[2].strike;
        float net_premium = legs_[0].premium + legs_[1].premium -
                           legs_[2].premium - legs_[3].premium;
        return (otm_call_strike - atm_strike) - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float atm_strike = legs_[0].strike;
        float net_premium = legs_[0].premium + legs_[1].premium -
                           legs_[2].premium - legs_[3].premium;
        return {atm_strike - net_premium, atm_strike + net_premium};
    }
};

/**
 * Short Iron Butterfly
 *
 * Opposite of Long Iron Butterfly
 * - Buy 1 ATM call
 * - Buy 1 ATM put
 * - Sell 1 OTM call
 * - Sell 1 OTM put
 *
 * Market Outlook: Volatile (profit from large moves)
 */
class ShortIronButterflyStrategy final : public BaseOptionsStrategy<ShortIronButterflyStrategy> {
public:
    ShortIronButterflyStrategy(
        float underlying_price,
        float atm_strike,
        float otm_call_strike,
        float otm_put_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Iron Butterfly",
            "Buy ATM straddle + sell OTM strangle - profit from breakouts",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Buy ATM call
        float atm_call_premium = simd::blackScholesCall(
            underlying_price, atm_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = atm_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = atm_call_premium
        });

        // Buy ATM put
        float atm_put_premium = simd::blackScholesPut(
            underlying_price, atm_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = atm_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = atm_put_premium
        });

        // Sell OTM call
        float otm_call_premium = simd::blackScholesCall(
            underlying_price, otm_call_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = otm_call_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = otm_call_premium
        });

        // Sell OTM put
        float otm_put_premium = simd::blackScholesPut(
            underlying_price, otm_put_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = otm_put_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = otm_put_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (size_t i = 0; i < legs_.size(); ++i) {
            auto const& leg = legs_[i];
            float intrinsic = leg.is_call
                ? std::max(0.0f, underlying_price - leg.strike)
                : std::max(0.0f, leg.strike - underlying_price);

            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            // Gamma and Vega same for calls/puts
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
        float atm_strike = legs_[0].strike;
        float otm_call_strike = legs_[2].strike;
        float net_premium = -legs_[0].premium - legs_[1].premium +
                           legs_[2].premium + legs_[3].premium;
        return (otm_call_strike - atm_strike) + net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float net_premium = -legs_[0].premium - legs_[1].premium +
                           legs_[2].premium + legs_[3].premium;
        return -net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float atm_strike = legs_[0].strike;
        float net_premium = -legs_[0].premium - legs_[1].premium +
                           legs_[2].premium + legs_[3].premium;
        return {atm_strike + net_premium, atm_strike - net_premium};
    }
};

// ============================================================================
// IRON CONDORS (2 strategies)
// ============================================================================

/**
 * Long Iron Condor
 *
 * Construction:
 * - Sell OTM put spread (sell higher put, buy lower put)
 * - Sell OTM call spread (sell lower call, buy higher call)
 *
 * Market Outlook: Neutral (profit from price staying in range)
 * Max Profit: Net premium received
 * Max Loss: Width of spread - net premium
 *
 * Characteristics:
 * - Most popular neutral strategy
 * - Credit spread
 * - Wide profit zone
 * - Defined risk
 */
class LongIronCondorStrategy final : public BaseOptionsStrategy<LongIronCondorStrategy> {
public:
    LongIronCondorStrategy(
        float underlying_price,
        float put_lower_strike,   // Lowest strike
        float put_higher_strike,  // Low-middle strike
        float call_lower_strike,  // High-middle strike
        float call_higher_strike, // Highest strike
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Long Iron Condor",
            "Sell OTM put spread + sell OTM call spread - wide profit zone",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Buy lower put
        float put_lower_premium = simd::blackScholesPut(
            underlying_price, put_lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = put_lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_lower_premium
        });

        // Sell higher put
        float put_higher_premium = simd::blackScholesPut(
            underlying_price, put_higher_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = put_higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_higher_premium
        });

        // Sell lower call
        float call_lower_premium = simd::blackScholesCall(
            underlying_price, call_lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = call_lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_lower_premium
        });

        // Buy higher call
        float call_higher_premium = simd::blackScholesCall(
            underlying_price, call_higher_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = call_higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_higher_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call
                ? std::max(0.0f, underlying_price - leg.strike)
                : std::max(0.0f, leg.strike - underlying_price);

            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            // Gamma and Vega same for calls/puts
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
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float put_spread_width = legs_[1].strike - legs_[0].strike;
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return put_spread_width - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float put_higher_strike = legs_[1].strike;
        float call_lower_strike = legs_[2].strike;
        float net_premium = -legs_[0].premium + legs_[1].premium +
                           legs_[2].premium - legs_[3].premium;
        return {put_higher_strike - net_premium, call_lower_strike + net_premium};
    }
};

/**
 * Short Iron Condor
 *
 * Opposite of Long Iron Condor
 * - Buy OTM put spread
 * - Buy OTM call spread
 *
 * Market Outlook: Volatile (profit from large moves)
 */
class ShortIronCondorStrategy final : public BaseOptionsStrategy<ShortIronCondorStrategy> {
public:
    ShortIronCondorStrategy(
        float underlying_price,
        float put_lower_strike,
        float put_higher_strike,
        float call_lower_strike,
        float call_higher_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f)
        : BaseOptionsStrategy(
            "Short Iron Condor",
            "Buy OTM put spread + buy OTM call spread - profit from breakouts",
            StrategyType::COMBINATION,
            MarketOutlook::VOLATILE,
            ComplexityLevel::ADVANCED)
    {
        float T = days_to_expiration / 365.0f;

        // Sell lower put
        float put_lower_premium = simd::blackScholesPut(
            underlying_price, put_lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,
            .strike = put_lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_lower_premium
        });

        // Buy higher put
        float put_higher_premium = simd::blackScholesPut(
            underlying_price, put_higher_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,
            .strike = put_higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_higher_premium
        });

        // Buy lower call
        float call_lower_premium = simd::blackScholesCall(
            underlying_price, call_lower_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = true,
            .strike = call_lower_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_lower_premium
        });

        // Sell higher call
        float call_higher_premium = simd::blackScholesCall(
            underlying_price, call_higher_strike, T, risk_free_rate, implied_volatility);
        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,
            .strike = call_higher_strike,
            .quantity = 1.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_higher_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {
        float pnl = 0.0f;

        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call
                ? std::max(0.0f, underlying_price - leg.strike)
                : std::max(0.0f, leg.strike - underlying_price);

            float leg_pnl = (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
            pnl += leg_pnl;
        }

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

            // Gamma and Vega same for calls/puts
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
        float put_spread_width = legs_[1].strike - legs_[0].strike;
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return put_spread_width + net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return -net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float put_higher_strike = legs_[1].strike;
        float call_lower_strike = legs_[2].strike;
        float net_premium = legs_[0].premium - legs_[1].premium -
                           legs_[2].premium + legs_[3].premium;
        return {put_higher_strike + net_premium, call_lower_strike - net_premium};
    }
};

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

// Butterfly factories
[[nodiscard]] inline auto createLongCallButterfly(
    float underlying_price, float lower_strike, float middle_strike, float upper_strike,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongCallButterflyStrategy>(
        underlying_price, lower_strike, middle_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortCallButterfly(
    float underlying_price, float lower_strike, float middle_strike, float upper_strike,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortCallButterflyStrategy>(
        underlying_price, lower_strike, middle_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createLongPutButterfly(
    float underlying_price, float lower_strike, float middle_strike, float upper_strike,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongPutButterflyStrategy>(
        underlying_price, lower_strike, middle_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortPutButterfly(
    float underlying_price, float lower_strike, float middle_strike, float upper_strike,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortPutButterflyStrategy>(
        underlying_price, lower_strike, middle_strike, upper_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

// Condor factories
[[nodiscard]] inline auto createLongCallCondor(
    float underlying_price, float strike1, float strike2, float strike3, float strike4,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongCallCondorStrategy>(
        underlying_price, strike1, strike2, strike3, strike4,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortCallCondor(
    float underlying_price, float strike1, float strike2, float strike3, float strike4,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortCallCondorStrategy>(
        underlying_price, strike1, strike2, strike3, strike4,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createLongPutCondor(
    float underlying_price, float strike1, float strike2, float strike3, float strike4,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongPutCondorStrategy>(
        underlying_price, strike1, strike2, strike3, strike4,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortPutCondor(
    float underlying_price, float strike1, float strike2, float strike3, float strike4,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortPutCondorStrategy>(
        underlying_price, strike1, strike2, strike3, strike4,
        days_to_expiration, implied_volatility, risk_free_rate);
}

// Iron Butterfly factories
[[nodiscard]] inline auto createLongIronButterfly(
    float underlying_price, float atm_strike, float otm_call_strike, float otm_put_strike,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongIronButterflyStrategy>(
        underlying_price, atm_strike, otm_call_strike, otm_put_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortIronButterfly(
    float underlying_price, float atm_strike, float otm_call_strike, float otm_put_strike,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortIronButterflyStrategy>(
        underlying_price, atm_strike, otm_call_strike, otm_put_strike,
        days_to_expiration, implied_volatility, risk_free_rate);
}

// Iron Condor factories
[[nodiscard]] inline auto createLongIronCondor(
    float underlying_price, float put_lower, float put_higher, float call_lower, float call_higher,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongIronCondorStrategy>(
        underlying_price, put_lower, put_higher, call_lower, call_higher,
        days_to_expiration, implied_volatility, risk_free_rate);
}

[[nodiscard]] inline auto createShortIronCondor(
    float underlying_price, float put_lower, float put_higher, float call_lower, float call_higher,
    float days_to_expiration, float implied_volatility, float risk_free_rate = 0.05f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortIronCondorStrategy>(
        underlying_price, put_lower, put_higher, call_lower, call_higher,
        days_to_expiration, implied_volatility, risk_free_rate);
}

}  // namespace bigbrother::options_strategies
