/**
 * @file ratio_spreads.cppm
 * @brief Ratio spread strategies with unequal leg quantities (Tier 6)
 * Implements 8 strategies with 1:2, 2:1, or other ratio configurations
 * Author: Olumuyiwa Oluwasanmi, Date: November 12, 2025
 */
module;
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

export module bigbrother.options_strategies.ratio_spreads;
import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// Helper macro to reduce code duplication
#define RATIO_GREEKS_CALC(leg_container) \
    Greeks total{}; \
    for (auto const& leg : leg_container) { \
        float T = leg.days_to_expiration / 365.0f, sign = leg.getSign(); \
        if (leg.is_call) { \
            total.delta += simd::deltaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
            total.theta += simd::thetaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
            total.rho += simd::rhoCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
        } else { \
            total.delta += simd::deltaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
            total.theta += simd::thetaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
            total.rho += simd::rhoPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
        } \
        total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
            _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
        total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike), \
            _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity; \
    } \
    return total;

// 1. Call Ratio Spread: Buy 1 call at lower strike, sell 2 calls at higher strike
class CallRatioSpreadStrategy final : public BaseOptionsStrategy<CallRatioSpreadStrategy> {
public:
    CallRatioSpreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Call Ratio Spread", "Buy 1 call + sell 2 calls - limited profit, unlimited risk",
            StrategyType::RATIO, MarketOutlook::BULLISH, ComplexityLevel::ADVANCED) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, true, lower_strike, 1.0f, days, iv,
            simd::blackScholesCall(price, lower_strike, T, r, iv)});
        addLeg(OptionLeg{true, false, higher_strike, 2.0f, days, iv,
            simd::blackScholesCall(price, higher_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, price - leg.strike);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float spread = legs_[1].strike - legs_[0].strike;
        float net = -legs_[0].premium + 2.0f * legs_[1].premium;
        return spread + net;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited upside risk
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float lower_be = legs_[0].strike + (-legs_[0].premium + 2.0f * legs_[1].premium);
        float upper_be = legs_[1].strike + (legs_[1].strike - legs_[0].strike + (-legs_[0].premium + 2.0f * legs_[1].premium));
        return {lower_be, upper_be};
    }
};

// 2. Put Ratio Spread: Buy 1 put at higher strike, sell 2 puts at lower strike
class PutRatioSpreadStrategy final : public BaseOptionsStrategy<PutRatioSpreadStrategy> {
public:
    PutRatioSpreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Put Ratio Spread", "Buy 1 put + sell 2 puts - limited profit, significant downside risk",
            StrategyType::RATIO, MarketOutlook::BEARISH, ComplexityLevel::ADVANCED) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, true, higher_strike, 1.0f, days, iv,
            simd::blackScholesPut(price, higher_strike, T, r, iv)});
        addLeg(OptionLeg{false, false, lower_strike, 2.0f, days, iv,
            simd::blackScholesPut(price, lower_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        float spread = legs_[0].strike - legs_[1].strike;
        float net = -legs_[0].premium + 2.0f * legs_[1].premium;
        return spread + net;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Large downside risk
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net = -legs_[0].premium + 2.0f * legs_[1].premium;
        float upper_be = legs_[0].strike - net;
        float lower_be = legs_[1].strike - (legs_[0].strike - legs_[1].strike + net);
        return {lower_be, upper_be};
    }
};

// 3. Call Ratio Backspread: Sell 1 call at lower strike, buy 2 calls at higher strike
class CallRatioBackspreadStrategy final : public BaseOptionsStrategy<CallRatioBackspreadStrategy> {
public:
    CallRatioBackspreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Call Ratio Backspread", "Sell 1 call + buy 2 calls - unlimited profit potential",
            StrategyType::RATIO, MarketOutlook::BULLISH_VOLATILE, ComplexityLevel::ADVANCED) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, false, lower_strike, 1.0f, days, iv,
            simd::blackScholesCall(price, lower_strike, T, r, iv)});
        addLeg(OptionLeg{true, true, higher_strike, 2.0f, days, iv,
            simd::blackScholesCall(price, higher_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, price - leg.strike);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited upside
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float spread = legs_[1].strike - legs_[0].strike;
        float net = legs_[0].premium - 2.0f * legs_[1].premium;
        return spread - net;
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net = legs_[0].premium - 2.0f * legs_[1].premium;
        float lower_be = legs_[0].strike + net;
        float upper_be = legs_[1].strike + (legs_[1].strike - legs_[0].strike - net);
        return {lower_be, upper_be};
    }
};

// 4. Put Ratio Backspread: Sell 1 put at higher strike, buy 2 puts at lower strike
class PutRatioBackspreadStrategy final : public BaseOptionsStrategy<PutRatioBackspreadStrategy> {
public:
    PutRatioBackspreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Put Ratio Backspread", "Sell 1 put + buy 2 puts - profit from large downside moves",
            StrategyType::RATIO, MarketOutlook::BEARISH_VOLATILE, ComplexityLevel::ADVANCED) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, false, higher_strike, 1.0f, days, iv,
            simd::blackScholesPut(price, higher_strike, T, r, iv)});
        addLeg(OptionLeg{false, true, lower_strike, 2.0f, days, iv,
            simd::blackScholesPut(price, lower_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        // Max profit when stock goes to zero
        float net = legs_[0].premium - 2.0f * legs_[1].premium;
        return 2.0f * legs_[1].strike - legs_[0].strike - net;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        float spread = legs_[0].strike - legs_[1].strike;
        float net = legs_[0].premium - 2.0f * legs_[1].premium;
        return spread - net;
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        float net = legs_[0].premium - 2.0f * legs_[1].premium;
        float upper_be = legs_[0].strike - net;
        float lower_be = legs_[1].strike - (legs_[0].strike - legs_[1].strike - net);
        return {lower_be, upper_be};
    }
};

// 5-8: Additional ratio variants (simplified implementations)
// 5. Bull Call Ratio Spread (variant of call ratio spread, slightly bullish bias)
class BullCallRatioSpreadStrategy final : public BaseOptionsStrategy<BullCallRatioSpreadStrategy> {
public:
    BullCallRatioSpreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Bull Call Ratio Spread", "Bullish ratio spread with 1:2 call ratio",
            StrategyType::RATIO, MarketOutlook::BULLISH, ComplexityLevel::ADVANCED) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, true, lower_strike, 1.0f, days, iv,
            simd::blackScholesCall(price, lower_strike, T, r, iv)});
        addLeg(OptionLeg{true, false, higher_strike, 2.0f, days, iv,
            simd::blackScholesCall(price, higher_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, price - leg.strike);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[1].strike - legs_[0].strike) + (-legs_[0].premium + 2.0f * legs_[1].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 6. Bear Put Ratio Spread
class BearPutRatioSpreadStrategy final : public BaseOptionsStrategy<BearPutRatioSpreadStrategy> {
public:
    BearPutRatioSpreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Bear Put Ratio Spread", "Bearish ratio spread with 1:2 put ratio",
            StrategyType::RATIO, MarketOutlook::BEARISH, ComplexityLevel::ADVANCED) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, true, higher_strike, 1.0f, days, iv,
            simd::blackScholesPut(price, higher_strike, T, r, iv)});
        addLeg(OptionLeg{false, false, lower_strike, 2.0f, days, iv,
            simd::blackScholesPut(price, lower_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[1].strike) + (-legs_[0].premium + 2.0f * legs_[1].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 7. Variable Ratio Call Spread (1:3 ratio for more aggressive)
class VariableRatioCallSpreadStrategy final : public BaseOptionsStrategy<VariableRatioCallSpreadStrategy> {
public:
    VariableRatioCallSpreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Variable Ratio Call Spread", "Buy 1 call + sell 3 calls - aggressive neutral strategy",
            StrategyType::RATIO, MarketOutlook::NEUTRAL, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, true, lower_strike, 1.0f, days, iv,
            simd::blackScholesCall(price, lower_strike, T, r, iv)});
        addLeg(OptionLeg{true, false, higher_strike, 3.0f, days, iv,
            simd::blackScholesCall(price, higher_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, price - leg.strike);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[1].strike - legs_[0].strike) + (-legs_[0].premium + 3.0f * legs_[1].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 8. Variable Ratio Put Spread (1:3 ratio)
class VariableRatioPutSpreadStrategy final : public BaseOptionsStrategy<VariableRatioPutSpreadStrategy> {
public:
    VariableRatioPutSpreadStrategy(float price, float lower_strike, float higher_strike,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Variable Ratio Put Spread", "Buy 1 put + sell 3 puts - aggressive bearish neutral",
            StrategyType::RATIO, MarketOutlook::NEUTRAL, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, true, higher_strike, 1.0f, days, iv,
            simd::blackScholesPut(price, higher_strike, T, r, iv)});
        addLeg(OptionLeg{false, false, lower_strike, 3.0f, days, iv,
            simd::blackScholesPut(price, lower_strike, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        RATIO_GREEKS_CALC(legs_);
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[1].strike) + (-legs_[0].premium + 3.0f * legs_[1].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Factory functions
[[nodiscard]] inline auto createCallRatioSpread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<CallRatioSpreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createPutRatioSpread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<PutRatioSpreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createCallRatioBackspread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<CallRatioBackspreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createPutRatioBackspread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<PutRatioBackspreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createBullCallRatioSpread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<BullCallRatioSpreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createBearPutRatioSpread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<BearPutRatioSpreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createVariableRatioCallSpread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<VariableRatioCallSpreadStrategy>(price, lower, higher, days, iv, r);
}
[[nodiscard]] inline auto createVariableRatioPutSpread(float price, float lower, float higher,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<VariableRatioPutSpreadStrategy>(price, lower, higher, days, iv, r);
}

#undef RATIO_GREEKS_CALC

}  // namespace bigbrother::options_strategies
