/**
 * @file calendar_spreads.cppm  
 * @brief Calendar spread options strategies using different expirations (Tier 5)
 * Implements 6 strategies exploiting time decay differences
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

export module bigbrother.options_strategies.calendar_spreads;
import bigbrother.options_strategies.base;
import bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies {

// Long Call Calendar: Sell near-term call, buy far-term call (same strike)
// Profit from near-term decay exceeding far-term decay
class LongCallCalendarStrategy final : public BaseOptionsStrategy<LongCallCalendarStrategy> {
public:
    LongCallCalendarStrategy(float underlying_price, float strike, 
        float near_days, float far_days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Long Call Calendar",
            "Sell near call + buy far call - profit from time decay differential",
            StrategyType::COMBINATION, MarketOutlook::NEUTRAL, ComplexityLevel::INTERMEDIATE) {
        float T_near = near_days / 365.0f, T_far = far_days / 365.0f;
        addLeg(OptionLeg{true, false, strike, 1.0f, near_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_near, r, iv)});
        addLeg(OptionLeg{true, true, strike, 1.0f, far_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_far, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float days = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, price - leg.strike);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price, legs_[0].days_to_expiration);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        Greeks total{};
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f, sign = leg.getSign();
            total.delta += simd::deltaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.theta += simd::thetaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.rho += simd::rhoCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
        }
        return total;
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return std::nullopt;  // Depends on time and volatility
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return legs_[1].premium - legs_[0].premium;  // Net debit
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override {
        return {};  // Complex, depends on time to near expiration
    }
};

// Short Call Calendar: opposite
class ShortCallCalendarStrategy final : public BaseOptionsStrategy<ShortCallCalendarStrategy> {
public:
    ShortCallCalendarStrategy(float underlying_price, float strike,
        float near_days, float far_days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Short Call Calendar",
            "Buy near call + sell far call", StrategyType::COMBINATION, 
            MarketOutlook::VOLATILE, ComplexityLevel::INTERMEDIATE) {
        float T_near = near_days / 365.0f, T_far = far_days / 365.0f;
        addLeg(OptionLeg{true, true, strike, 1.0f, near_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_near, r, iv)});
        addLeg(OptionLeg{true, false, strike, 1.0f, far_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_far, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float days = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, price - leg.strike);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price, legs_[0].days_to_expiration);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        Greeks total{};
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f, sign = leg.getSign();
            total.delta += simd::deltaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.theta += simd::thetaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.rho += simd::rhoCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
        }
        return total;
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[1].premium - legs_[0].premium;  // Net credit
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return std::nullopt;
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Long Put Calendar
class LongPutCalendarStrategy final : public BaseOptionsStrategy<LongPutCalendarStrategy> {
public:
    LongPutCalendarStrategy(float underlying_price, float strike,
        float near_days, float far_days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Long Put Calendar",
            "Sell near put + buy far put", StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL, ComplexityLevel::INTERMEDIATE) {
        float T_near = near_days / 365.0f, T_far = far_days / 365.0f;
        addLeg(OptionLeg{false, false, strike, 1.0f, near_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_near, r, iv)});
        addLeg(OptionLeg{false, true, strike, 1.0f, far_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_far, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float days = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price, legs_[0].days_to_expiration);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        Greeks total{};
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f, sign = leg.getSign();
            total.delta += simd::deltaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.theta += simd::thetaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.rho += simd::rhoPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
        }
        return total;
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return legs_[1].premium - legs_[0].premium;
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Short Put Calendar
class ShortPutCalendarStrategy final : public BaseOptionsStrategy<ShortPutCalendarStrategy> {
public:
    ShortPutCalendarStrategy(float underlying_price, float strike,
        float near_days, float far_days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Short Put Calendar",
            "Buy near put + sell far put", StrategyType::COMBINATION,
            MarketOutlook::VOLATILE, ComplexityLevel::INTERMEDIATE) {
        float T_near = near_days / 365.0f, T_far = far_days / 365.0f;
        addLeg(OptionLeg{false, true, strike, 1.0f, near_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_near, r, iv)});
        addLeg(OptionLeg{false, false, strike, 1.0f, far_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_far, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float days = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price, legs_[0].days_to_expiration);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        Greeks total{};
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f, sign = leg.getSign();
            total.delta += simd::deltaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.theta += simd::thetaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.rho += simd::rhoPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
        }
        return total;
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return legs_[1].premium - legs_[0].premium;
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Long Calendar Straddle: Sell near straddle + buy far straddle
class LongCalendarStraddleStrategy final : public BaseOptionsStrategy<LongCalendarStraddleStrategy> {
public:
    LongCalendarStraddleStrategy(float underlying_price, float strike,
        float near_days, float far_days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Long Calendar Straddle",
            "Sell near straddle + buy far straddle", StrategyType::COMBINATION,
            MarketOutlook::NEUTRAL, ComplexityLevel::ADVANCED) {
        float T_near = near_days / 365.0f, T_far = far_days / 365.0f;
        // Near-term short call
        addLeg(OptionLeg{true, false, strike, 1.0f, near_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_near, r, iv)});
        // Near-term short put
        addLeg(OptionLeg{false, false, strike, 1.0f, near_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_near, r, iv)});
        // Far-term long call
        addLeg(OptionLeg{true, true, strike, 1.0f, far_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_far, r, iv)});
        // Far-term long put
        addLeg(OptionLeg{false, true, strike, 1.0f, far_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_far, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float days = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call ? 
                std::max(0.0f, price - leg.strike) : std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price, legs_[0].days_to_expiration);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        Greeks total{};
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f, sign = leg.getSign();
            if (leg.is_call) {
                total.delta += simd::deltaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.theta += simd::thetaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.rho += simd::rhoCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            } else {
                total.delta += simd::deltaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.theta += simd::thetaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.rho += simd::rhoPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            }
            total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
        }
        return total;
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override {
        return (legs_[2].premium + legs_[3].premium) - (legs_[0].premium + legs_[1].premium);
    }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Short Calendar Straddle
class ShortCalendarStraddleStrategy final : public BaseOptionsStrategy<ShortCalendarStraddleStrategy> {
public:
    ShortCalendarStraddleStrategy(float underlying_price, float strike,
        float near_days, float far_days, float iv, float r = 0.05f)
        : BaseOptionsStrategy("Short Calendar Straddle",
            "Buy near straddle + sell far straddle", StrategyType::COMBINATION,
            MarketOutlook::VOLATILE, ComplexityLevel::ADVANCED) {
        float T_near = near_days / 365.0f, T_far = far_days / 365.0f;
        addLeg(OptionLeg{true, true, strike, 1.0f, near_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_near, r, iv)});
        addLeg(OptionLeg{false, true, strike, 1.0f, near_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_near, r, iv)});
        addLeg(OptionLeg{true, false, strike, 1.0f, far_days, iv,
            simd::blackScholesCall(underlying_price, strike, T_far, r, iv)});
        addLeg(OptionLeg{false, false, strike, 1.0f, far_days, iv,
            simd::blackScholesPut(underlying_price, strike, T_far, r, iv)});
    }
    [[nodiscard]] auto calculateProfitLoss(float price, float days = 0.0f) const -> float override {
        float pnl = 0.0f;
        for (auto const& leg : legs_) {
            float intrinsic = leg.is_call ?
                std::max(0.0f, price - leg.strike) : std::max(0.0f, leg.strike - price);
            pnl += (intrinsic - leg.premium) * leg.getSign() * leg.quantity;
        }
        return pnl;
    }
    [[nodiscard]] auto calculateExpirationPL(float price) const -> float override {
        return calculateProfitLoss(price, legs_[0].days_to_expiration);
    }
    [[nodiscard]] auto calculateGreeks(float price, float r) const -> Greeks override {
        Greeks total{};
        for (auto const& leg : legs_) {
            float T = leg.days_to_expiration / 365.0f, sign = leg.getSign();
            if (leg.is_call) {
                total.delta += simd::deltaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.theta += simd::thetaCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.rho += simd::rhoCallBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            } else {
                total.delta += simd::deltaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.theta += simd::thetaPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
                total.rho += simd::rhoPutBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                    _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            }
            total.gamma += simd::gammaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
            total.vega += simd::vegaBatch(_mm256_set1_ps(price), _mm256_set1_ps(leg.strike),
                _mm256_set1_ps(T), _mm256_set1_ps(r), _mm256_set1_ps(leg.implied_volatility))[0] * sign * leg.quantity;
        }
        return total;
    }
    [[nodiscard]] auto getMaxProfit() const -> std::optional<float> override {
        return (legs_[2].premium + legs_[3].premium) - (legs_[0].premium + legs_[1].premium);
    }
    [[nodiscard]] auto getMaxLoss() const -> std::optional<float> override { return std::nullopt; }
    [[nodiscard]] auto getBreakevens() const -> std::vector<float> override { return {}; }
};

// Factory functions
[[nodiscard]] inline auto createLongCallCalendar(float price, float strike,
    float near_days, float far_days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongCallCalendarStrategy>(price, strike, near_days, far_days, iv, r);
}
[[nodiscard]] inline auto createShortCallCalendar(float price, float strike,
    float near_days, float far_days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortCallCalendarStrategy>(price, strike, near_days, far_days, iv, r);
}
[[nodiscard]] inline auto createLongPutCalendar(float price, float strike,
    float near_days, float far_days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongPutCalendarStrategy>(price, strike, near_days, far_days, iv, r);
}
[[nodiscard]] inline auto createShortPutCalendar(float price, float strike,
    float near_days, float far_days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortPutCalendarStrategy>(price, strike, near_days, far_days, iv, r);
}
[[nodiscard]] inline auto createLongCalendarStraddle(float price, float strike,
    float near_days, float far_days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<LongCalendarStraddleStrategy>(price, strike, near_days, far_days, iv, r);
}
[[nodiscard]] inline auto createShortCalendarStraddle(float price, float strike,
    float near_days, float far_days, float iv, float r = 0.05f) -> std::unique_ptr<IOptionsStrategy> {
    return std::make_unique<ShortCalendarStraddleStrategy>(price, strike, near_days, far_days, iv, r);
}

}  // namespace bigbrother::options_strategies
