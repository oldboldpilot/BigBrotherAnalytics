/**
 * @file albatross_ladder.cppm
 * @brief Albatross and Ladder spread strategies (Tier 7 - Exotic)
 * Implements 7 advanced strategies - wide-bodied condor variants
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

export module bigbrother.options_strategies.albatross_ladder;
import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// Helper for Greeks calculation (same as ratio spreads)
#define EXOTIC_GREEKS(legs) \
    Greeks total{}; \
    for (auto const& leg : legs) { \
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

// 1. Long Call Albatross: Like condor but with wider body (4 legs, wider middle strikes)
// Buy K1, Sell K2, Sell K3, Buy K4 where K3-K2 > K2-K1
class LongCallAlbatrossStrategy final : public BaseOptionsStrategy<LongCallAlbatrossStrategy> {
public:
    LongCallAlbatrossStrategy(float price, float k1, float k2, float k3, float k4,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Long Call Albatross", "Wide-bodied condor with larger profit zone",
            StrategyType::COMBINATION, MarketOutlook::NEUTRAL, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, true, k1, 1.0f, days, iv, simd::blackScholesCall(price, k1, T, r, iv)});
        addLeg(OptionLeg{true, false, k2, 1.0f, days, iv, simd::blackScholesCall(price, k2, T, r, iv)});
        addLeg(OptionLeg{true, false, k3, 1.0f, days, iv, simd::blackScholesCall(price, k3, T, r, iv)});
        addLeg(OptionLeg{true, true, k4, 1.0f, days, iv, simd::blackScholesCall(price, k4, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            pnl += (std::max(0.0f, price - leg.strike) - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[1].strike - legs_[0].strike) - 
            (-legs_[0].premium + legs_[1].premium + legs_[2].premium - legs_[3].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return -legs_[0].premium + legs_[1].premium + legs_[2].premium - legs_[3].premium;
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 2. Short Call Albatross
class ShortCallAlbatrossStrategy final : public BaseOptionsStrategy<ShortCallAlbatrossStrategy> {
public:
    ShortCallAlbatrossStrategy(float price, float k1, float k2, float k3, float k4,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Short Call Albatross", "Opposite of long call albatross",
            StrategyType::COMBINATION, MarketOutlook::VOLATILE, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, false, k1, 1.0f, days, iv, simd::blackScholesCall(price, k1, T, r, iv)});
        addLeg(OptionLeg{true, true, k2, 1.0f, days, iv, simd::blackScholesCall(price, k2, T, r, iv)});
        addLeg(OptionLeg{true, true, k3, 1.0f, days, iv, simd::blackScholesCall(price, k3, T, r, iv)});
        addLeg(OptionLeg{true, false, k4, 1.0f, days, iv, simd::blackScholesCall(price, k4, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            pnl += (std::max(0.0f, price - leg.strike) - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[0].premium - legs_[1].premium - legs_[2].premium + legs_[3].premium;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return (legs_[1].strike - legs_[0].strike) - 
            (legs_[0].premium - legs_[1].premium - legs_[2].premium + legs_[3].premium);
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 3. Long Put Albatross
class LongPutAlbatrossStrategy final : public BaseOptionsStrategy<LongPutAlbatrossStrategy> {
public:
    LongPutAlbatrossStrategy(float price, float k1, float k2, float k3, float k4,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Long Put Albatross", "Wide-bodied put condor",
            StrategyType::COMBINATION, MarketOutlook::NEUTRAL, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, true, k4, 1.0f, days, iv, simd::blackScholesPut(price, k4, T, r, iv)});
        addLeg(OptionLeg{false, false, k3, 1.0f, days, iv, simd::blackScholesPut(price, k3, T, r, iv)});
        addLeg(OptionLeg{false, false, k2, 1.0f, days, iv, simd::blackScholesPut(price, k2, T, r, iv)});
        addLeg(OptionLeg{false, true, k1, 1.0f, days, iv, simd::blackScholesPut(price, k1, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            pnl += (std::max(0.0f, leg.strike - price) - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[1].strike) - 
            (-legs_[0].premium + legs_[1].premium + legs_[2].premium - legs_[3].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return -legs_[0].premium + legs_[1].premium + legs_[2].premium - legs_[3].premium;
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 4. Short Put Albatross
class ShortPutAlbatrossStrategy final : public BaseOptionsStrategy<ShortPutAlbatrossStrategy> {
public:
    ShortPutAlbatrossStrategy(float price, float k1, float k2, float k3, float k4,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Short Put Albatross", "Opposite of long put albatross",
            StrategyType::COMBINATION, MarketOutlook::VOLATILE, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, false, k4, 1.0f, days, iv, simd::blackScholesPut(price, k4, T, r, iv)});
        addLeg(OptionLeg{false, true, k3, 1.0f, days, iv, simd::blackScholesPut(price, k3, T, r, iv)});
        addLeg(OptionLeg{false, true, k2, 1.0f, days, iv, simd::blackScholesPut(price, k2, T, r, iv)});
        addLeg(OptionLeg{false, false, k1, 1.0f, days, iv, simd::blackScholesPut(price, k1, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            pnl += (std::max(0.0f, leg.strike - price) - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[0].premium - legs_[1].premium - legs_[2].premium + legs_[3].premium;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[1].strike) - 
            (legs_[0].premium - legs_[1].premium - legs_[2].premium + legs_[3].premium);
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 5. Call Ladder Spread: Buy 1 call, sell 1 call, sell 1 call (3 ascending strikes)
// Profits from moderate bullish move, unlimited risk above top strike
class CallLadderSpreadStrategy final : public BaseOptionsStrategy<CallLadderSpreadStrategy> {
public:
    CallLadderSpreadStrategy(float price, float k1, float k2, float k3,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Call Ladder Spread", "Buy 1 call + sell 2 calls at higher strikes",
            StrategyType::LADDER, MarketOutlook::BULLISH, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{true, true, k1, 1.0f, days, iv, simd::blackScholesCall(price, k1, T, r, iv)});
        addLeg(OptionLeg{true, false, k2, 1.0f, days, iv, simd::blackScholesCall(price, k2, T, r, iv)});
        addLeg(OptionLeg{true, false, k3, 1.0f, days, iv, simd::blackScholesCall(price, k3, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            pnl += (std::max(0.0f, price - leg.strike) - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[1].strike - legs_[0].strike) + 
            (-legs_[0].premium + legs_[1].premium + legs_[2].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Unlimited above K3
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 6. Put Ladder Spread: Buy 1 put, sell 1 put, sell 1 put (3 descending strikes)
class PutLadderSpreadStrategy final : public BaseOptionsStrategy<PutLadderSpreadStrategy> {
public:
    PutLadderSpreadStrategy(float price, float k1, float k2, float k3,
        float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Put Ladder Spread", "Buy 1 put + sell 2 puts at lower strikes",
            StrategyType::LADDER, MarketOutlook::BEARISH, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, true, k3, 1.0f, days, iv, simd::blackScholesPut(price, k3, T, r, iv)});
        addLeg(OptionLeg{false, false, k2, 1.0f, days, iv, simd::blackScholesPut(price, k2, T, r, iv)});
        addLeg(OptionLeg{false, false, k1, 1.0f, days, iv, simd::blackScholesPut(price, k1, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            pnl += (std::max(0.0f, leg.strike - price) - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[0].strike - legs_[1].strike) + 
            (-legs_[0].premium + legs_[1].premium + legs_[2].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;  // Large downside risk below K1
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// 7. Iron Albatross: Combination of wide iron butterfly (4 legs, calls + puts)
class IronAlbatrossStrategy final : public BaseOptionsStrategy<IronAlbatrossStrategy> {
public:
    IronAlbatrossStrategy(float price, float put_lower, float put_higher, 
        float call_lower, float call_higher, float days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Iron Albatross", "Wide iron condor variant with larger body",
            StrategyType::COMBINATION, MarketOutlook::NEUTRAL, ComplexityLevel::COMPLEX) {
        float T = days / 365.0f;
        addLeg(OptionLeg{false, true, put_lower, 1.0f, days, iv, simd::blackScholesPut(price, put_lower, T, r, iv)});
        addLeg(OptionLeg{false, false, put_higher, 1.0f, days, iv, simd::blackScholesPut(price, put_higher, T, r, iv)});
        addLeg(OptionLeg{true, false, call_lower, 1.0f, days, iv, simd::blackScholesCall(price, call_lower, T, r, iv)});
        addLeg(OptionLeg{true, true, call_higher, 1.0f, days, iv, simd::blackScholesCall(price, call_higher, T, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call ? std::max(0.0f, price - leg.strike) : std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override { return calculateProfitLoss(price); }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override { EXOTIC_GREEKS(legs_); }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return -legs_[0].premium + legs_[1].premium + legs_[2].premium - legs_[3].premium;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return (legs_[1].strike - legs_[0].strike) - 
            (-legs_[0].premium + legs_[1].premium + legs_[2].premium - legs_[3].premium);
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Factory functions
[[nodiscard]] inline auto createLongCallAlbatross(float price, float k1, float k2, float k3, float k4,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongCallAlbatrossStrategy>(price, k1, k2, k3, k4, days, iv, r);
}
[[nodiscard]] inline auto createShortCallAlbatross(float price, float k1, float k2, float k3, float k4,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortCallAlbatrossStrategy>(price, k1, k2, k3, k4, days, iv, r);
}
[[nodiscard]] inline auto createLongPutAlbatross(float price, float k1, float k2, float k3, float k4,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongPutAlbatrossStrategy>(price, k1, k2, k3, k4, days, iv, r);
}
[[nodiscard]] inline auto createShortPutAlbatross(float price, float k1, float k2, float k3, float k4,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortPutAlbatrossStrategy>(price, k1, k2, k3, k4, days, iv, r);
}
[[nodiscard]] inline auto createCallLadderSpread(float price, float k1, float k2, float k3,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<CallLadderSpreadStrategy>(price, k1, k2, k3, days, iv, r);
}
[[nodiscard]] inline auto createPutLadderSpread(float price, float k1, float k2, float k3,
    float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<PutLadderSpreadStrategy>(price, k1, k2, k3, days, iv, r);
}
[[nodiscard]] inline auto createIronAlbatross(float price, float put_lower, float put_higher,
    float call_lower, float call_higher, float days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<IronAlbatrossStrategy>(price, put_lower, put_higher, call_lower, call_higher, days, iv, r);
}

#undef EXOTIC_GREEKS

}  // namespace bigbrother::options_strategies
