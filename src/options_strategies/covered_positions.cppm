/**
 * @file covered_positions.cppm
 * @brief Covered position options strategies (Tier 4)
 *
 * Implements 3 strategies combining stock positions with options:
 * - Covered Call: Long stock + short call (income generation)
 * - Covered Put: Short stock + short put (bearish income)
 * - Collar: Long stock + short call + long put (downside protection)
 *
 * These are among the most popular real-world strategies for income generation
 * and risk management.
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

export module bigbrother.options_strategies.covered_positions;

import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

/**
 * Covered Call Strategy
 *
 * Construction:
 * - Buy 100 shares of stock
 * - Sell 1 call option (typically OTM)
 *
 * Market Outlook: Neutral to slightly bullish
 * Max Profit: (Strike - Stock Price) + Premium received
 * Max Loss: Stock price - Premium (if stock goes to zero)
 * Breakeven: Stock purchase price - Premium received
 *
 * Characteristics:
 * - Most popular income-generating strategy
 * - Reduces cost basis of stock by premium
 * - Limits upside to strike price
 * - Provides downside protection equal to premium
 *
 * Use Cases:
 * - Generate income on existing stock holdings
 * - Willing to sell stock at strike price
 * - Expect low to moderate price appreciation
 */
class CoveredCallStrategy final : public BaseOptionsStrategy<CoveredCallStrategy> {
private:
    float stock_quantity_{100.0f};  // Standard lot
    float stock_entry_price_;

public:
    CoveredCallStrategy(
        float stock_price,
        float call_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f,
        float stock_quantity = 100.0f)
        : BaseOptionsStrategy(
            "Covered Call",
            "Buy stock + sell call - generate income on stock holdings",
            StrategyType::COMBINATION,
            MarketOutlook::BULLISH,
            ComplexityLevel::BEGINNER),
          stock_quantity_(stock_quantity),
          stock_entry_price_(stock_price)
    {
        float T = days_to_expiration / 365.0f;

        // Short call option
        float call_premium = simd::blackScholesCall(
            stock_price, call_strike, T, risk_free_rate, implied_volatility);

        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,  // Short call
            .strike = call_strike,
            .quantity = stock_quantity / 100.0f,  // 1 contract per 100 shares
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        // Stock P&L: (Current Price - Entry Price) * Quantity
        float stock_pnl = (underlying_price - stock_entry_price_) * stock_quantity_;

        // Option P&L
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        float call_pnl = (call_intrinsic - call_leg.premium) * call_leg.getSign() * call_leg.quantity * 100.0f;

        return stock_pnl + call_pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        // Stock contributes delta of 1.0 per share (long position)
        total.delta = stock_quantity_;

        // Option Greeks (short call)
        auto const& leg = legs_[0];
        float T = leg.days_to_expiration / 365.0f;
        float sign = leg.getSign();
        float contracts = leg.quantity;

        float delta = simd::deltaCallBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.delta += delta * sign * contracts * 100.0f;

        float gamma = simd::gammaBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.gamma = gamma * sign * contracts * 100.0f;

        float theta = simd::thetaCallBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.theta = theta * sign * contracts * 100.0f;

        float vega = simd::vegaBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.vega = vega * sign * contracts * 100.0f;

        float rho = simd::rhoCallBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.rho = rho * sign * contracts * 100.0f;

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float strike = legs_[0].strike;
        float premium = legs_[0].premium * legs_[0].quantity * 100.0f;
        return (strike - stock_entry_price_) * stock_quantity_ + premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float premium = legs_[0].premium * legs_[0].quantity * 100.0f;
        return stock_entry_price_ * stock_quantity_ - premium;  // Stock goes to zero
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float premium_per_share = legs_[0].premium;
        return {stock_entry_price_ - premium_per_share};
    }
};

/**
 * Covered Put Strategy
 *
 * Construction:
 * - Short 100 shares of stock
 * - Sell 1 put option (typically OTM)
 *
 * Market Outlook: Neutral to slightly bearish
 * Max Profit: (Stock Price - Strike) + Premium received
 * Max Loss: Unlimited (stock can rise indefinitely)
 * Breakeven: Stock short price + Premium received
 *
 * Characteristics:
 * - Bearish version of covered call
 * - Generate income while short stock
 * - Limits downside gains to strike price
 * - Provides upside protection equal to premium
 *
 * Use Cases:
 * - Generate income on short stock positions
 * - Willing to close short at strike price
 * - Expect low to moderate price decline
 */
class CoveredPutStrategy final : public BaseOptionsStrategy<CoveredPutStrategy> {
private:
    float stock_quantity_{-100.0f};  // Short 100 shares
    float stock_entry_price_;

public:
    CoveredPutStrategy(
        float stock_price,
        float put_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f,
        float stock_quantity = 100.0f)
        : BaseOptionsStrategy(
            "Covered Put",
            "Short stock + sell put - generate income on short stock positions",
            StrategyType::COMBINATION,
            MarketOutlook::BEARISH,
            ComplexityLevel::BEGINNER),
          stock_quantity_(-stock_quantity),  // Negative for short
          stock_entry_price_(stock_price)
    {
        float T = days_to_expiration / 365.0f;

        // Short put option
        float put_premium = simd::blackScholesPut(
            stock_price, put_strike, T, risk_free_rate, implied_volatility);

        addLeg(OptionLeg{
            .is_call = false,
            .is_long = false,  // Short put
            .strike = put_strike,
            .quantity = stock_quantity / 100.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        // Stock P&L: (Entry Price - Current Price) * Quantity (negative quantity for short)
        float stock_pnl = (stock_entry_price_ - underlying_price) * std::abs(stock_quantity_);

        // Option P&L
        auto const& put_leg = legs_[0];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        float put_pnl = (put_intrinsic - put_leg.premium) * put_leg.getSign() * put_leg.quantity * 100.0f;

        return stock_pnl + put_pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        // Stock contributes delta of -1.0 per share (short position)
        total.delta = stock_quantity_;  // Already negative

        // Option Greeks (short put)
        auto const& leg = legs_[0];
        float T = leg.days_to_expiration / 365.0f;
        float sign = leg.getSign();
        float contracts = leg.quantity;

        float delta = simd::deltaPutBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.delta += delta * sign * contracts * 100.0f;

        float gamma = simd::gammaBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.gamma = gamma * sign * contracts * 100.0f;

        float theta = simd::thetaPutBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.theta = theta * sign * contracts * 100.0f;

        float vega = simd::vegaBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.vega = vega * sign * contracts * 100.0f;

        float rho = simd::rhoPutBatch(
            _mm256_set1_ps(underlying_price),
            _mm256_set1_ps(leg.strike),
            _mm256_set1_ps(T),
            _mm256_set1_ps(risk_free_rate),
            _mm256_set1_ps(leg.implied_volatility)
        )[0];
        total.rho = rho * sign * contracts * 100.0f;

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float strike = legs_[0].strike;
        float premium = legs_[0].premium * legs_[0].quantity * 100.0f;
        return (stock_entry_price_ - strike) * std::abs(stock_quantity_) + premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited (stock can rise indefinitely)
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float premium_per_share = legs_[0].premium;
        return {stock_entry_price_ + premium_per_share};
    }
};

/**
 * Collar Strategy (also called "Hedge Wrapper")
 *
 * Construction:
 * - Buy 100 shares of stock
 * - Sell 1 call option (OTM)
 * - Buy 1 put option (OTM)
 *
 * Market Outlook: Neutral (protecting existing long stock position)
 * Max Profit: (Call Strike - Stock Price) + Net Premium
 * Max Loss: (Stock Price - Put Strike) - Net Premium
 * Breakeven: Stock Price + Net Debit (or - Net Credit)
 *
 * Characteristics:
 * - Protective strategy for stock holdings
 * - Often zero cost or small credit (sell call covers put cost)
 * - Limits both upside and downside
 * - Popular for earnings protection
 *
 * Use Cases:
 * - Protect stock gains during volatile periods
 * - Lock in profits without selling stock (tax deferral)
 * - Earnings protection strategy
 * - Cost-effective downside insurance
 */
class CollarStrategy final : public BaseOptionsStrategy<CollarStrategy> {
private:
    float stock_quantity_{100.0f};
    float stock_entry_price_;

public:
    CollarStrategy(
        float stock_price,
        float call_strike,
        float put_strike,
        float days_to_expiration,
        float implied_volatility,
        float risk_free_rate = 0.05f,
        float stock_quantity = 100.0f)
        : BaseOptionsStrategy(
            "Collar",
            "Buy stock + sell call + buy put - protect stock with defined risk",
            StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL,
            ComplexityLevel::INTERMEDIATE),
          stock_quantity_(stock_quantity),
          stock_entry_price_(stock_price)
    {
        float T = days_to_expiration / 365.0f;

        // Short call option (OTM)
        float call_premium = simd::blackScholesCall(
            stock_price, call_strike, T, risk_free_rate, implied_volatility);

        addLeg(OptionLeg{
            .is_call = true,
            .is_long = false,  // Short call
            .strike = call_strike,
            .quantity = stock_quantity / 100.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = call_premium
        });

        // Long put option (OTM)
        float put_premium = simd::blackScholesPut(
            stock_price, put_strike, T, risk_free_rate, implied_volatility);

        addLeg(OptionLeg{
            .is_call = false,
            .is_long = true,  // Long put
            .strike = put_strike,
            .quantity = stock_quantity / 100.0f,
            .days_to_expiration = days_to_expiration,
            .implied_volatility = implied_volatility,
            .premium = put_premium
        });
    }

    [[nodiscard]] auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float override {

        // Stock P&L
        float stock_pnl = (underlying_price - stock_entry_price_) * stock_quantity_;

        // Call P&L (short)
        auto const& call_leg = legs_[0];
        float call_intrinsic = std::max(0.0f, underlying_price - call_leg.strike);
        float call_pnl = (call_intrinsic - call_leg.premium) * call_leg.getSign() * call_leg.quantity * 100.0f;

        // Put P&L (long)
        auto const& put_leg = legs_[1];
        float put_intrinsic = std::max(0.0f, put_leg.strike - underlying_price);
        float put_pnl = (put_intrinsic - put_leg.premium) * put_leg.getSign() * put_leg.quantity * 100.0f;

        return stock_pnl + call_pnl + put_pnl;
    }

    [[nodiscard]] auto calculateExpirationPL(
        float underlying_price) const -> float override {
        return calculateProfitLoss(underlying_price, legs_[0].days_to_expiration);
    }

    [[nodiscard]] auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks override {

        Greeks total{};

        // Stock contributes delta of 1.0 per share
        total.delta = stock_quantity_;

        // Options Greeks
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f;
            float sign = leg.getSign();
            float contracts = leg.quantity;

            if (leg.is_call) {
                float delta = simd::deltaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * contracts * 100.0f;

                float theta = simd::thetaCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * contracts * 100.0f;

                float rho = simd::rhoCallBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * contracts * 100.0f;
            } else {
                float delta = simd::deltaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.delta += delta * sign * contracts * 100.0f;

                float theta = simd::thetaPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.theta += theta * sign * contracts * 100.0f;

                float rho = simd::rhoPutBatch(
                    _mm256_set1_ps(underlying_price),
                    _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T),
                    _mm256_set1_ps(risk_free_rate),
                    _mm256_set1_ps(leg.implied_volatility)
                )[0];
                total.rho += rho * sign * contracts * 100.0f;
            }

            // Gamma and Vega same for calls/puts
            float gamma = simd::gammaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.gamma += gamma * sign * contracts * 100.0f;

            float vega = simd::vegaBatch(
                _mm256_set1_ps(underlying_price),
                _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T),
                _mm256_set1_ps(risk_free_rate),
                _mm256_set1_ps(leg.implied_volatility)
            )[0];
            total.vega += vega * sign * contracts * 100.0f;
        }

        return total;
    }

    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float call_strike = legs_[0].strike;
        float call_premium = legs_[0].premium * legs_[0].quantity * 100.0f;
        float put_premium = legs_[1].premium * legs_[1].quantity * 100.0f;
        float net_premium = call_premium - put_premium;

        return (call_strike - stock_entry_price_) * stock_quantity_ + net_premium;
    }

    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float put_strike = legs_[1].strike;
        float call_premium = legs_[0].premium * legs_[0].quantity * 100.0f;
        float put_premium = legs_[1].premium * legs_[1].quantity * 100.0f;
        float net_premium = call_premium - put_premium;

        return (stock_entry_price_ - put_strike) * stock_quantity_ - net_premium;
    }

    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float call_premium = legs_[0].premium;
        float put_premium = legs_[1].premium;
        float net_cost_per_share = put_premium - call_premium;  // Could be credit or debit

        return {stock_entry_price_ + net_cost_per_share};
    }
};

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

[[nodiscard]] inline auto createCoveredCall(
    float stock_price, float call_strike, float days_to_expiration,
    float implied_volatility, float risk_free_rate = 0.05f, float stock_quantity = 100.0f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<CoveredCallStrategy>(
        stock_price, call_strike, days_to_expiration,
        implied_volatility, risk_free_rate, stock_quantity);
}

[[nodiscard]] inline auto createCoveredPut(
    float stock_price, float put_strike, float days_to_expiration,
    float implied_volatility, float risk_free_rate = 0.05f, float stock_quantity = 100.0f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<CoveredPutStrategy>(
        stock_price, put_strike, days_to_expiration,
        implied_volatility, risk_free_rate, stock_quantity);
}

[[nodiscard]] inline auto createCollar(
    float stock_price, float call_strike, float put_strike, float days_to_expiration,
    float implied_volatility, float risk_free_rate = 0.05f, float stock_quantity = 100.0f
) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<CollarStrategy>(
        stock_price, call_strike, put_strike, days_to_expiration,
        implied_volatility, risk_free_rate, stock_quantity);
}

}  // namespace bigbrother::options_strategies
