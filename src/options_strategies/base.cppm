/**
 * BigBrotherAnalytics - Options Strategies Base Module (C++23)
 *
 * Foundation for all 52+ options strategies with SIMD optimization.
 * Implements base classes, interfaces, and common functionality.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 * Phase: Options Strategies Implementation - Complete A-Z List
 *
 * Performance Targets:
 * - Single option pricing: <1μs (AVX2)
 * - Greeks calculation: <5μs (AVX2)
 * - Batch pricing (100): <100μs (AVX2)
 */

module;

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

export module bigbrother.options_strategies.base;

export namespace bigbrother::options_strategies {

// ============================================================================
// Strategy Taxonomy
// ============================================================================

enum class StrategyType {
    SINGLE_LEG,         // Long/Short Call/Put
    VERTICAL_SPREAD,    // Bull/Bear Call/Put Spread
    HORIZONTAL_SPREAD,  // Calendar spreads
    DIAGONAL_SPREAD,    // Diagonal time/strike spreads
    BUTTERFLY,          // Butterfly variants
    CONDOR,             // Condor variants
    RATIO,              // Ratio spreads, backspreads
    COMBINATION,        // Straddles, strangles, covered positions
    LADDER,             // Ladder spreads
    ALBATROSS           // Albatross variants
};

enum class MarketOutlook {
    BULLISH,            // Expect price to rise
    BEARISH,            // Expect price to fall
    NEUTRAL,            // Expect price to stay flat
    VOLATILE,           // Expect big move (either direction)
    BULLISH_VOLATILE,   // Expect move up with volatility
    BEARISH_VOLATILE    // Expect move down with volatility
};

enum class ComplexityLevel {
    BEGINNER,       // Simple, 1-2 legs
    INTERMEDIATE,   // 2-3 legs, moderate risk
    ADVANCED,       // 3-4 legs, complex P&L
    COMPLEX         // 4+ legs, very complex risk profile
};

// ============================================================================
// Greeks Structure (SIMD-aligned)
// ============================================================================

/**
 * Greeks aligned to 32 bytes for AVX2 operations
 *
 * Delta: Rate of change of option price with respect to underlying
 * Gamma: Rate of change of delta with respect to underlying
 * Theta: Rate of option decay per day
 * Vega: Sensitivity to volatility changes
 * Rho: Sensitivity to interest rate changes
 */
struct alignas(32) Greeks {
    float delta{0.0f};      // $delta per $1 underlying move
    float gamma{0.0f};      // delta change per $1 underlying move
    float theta{0.0f};      // $ per day time decay
    float vega{0.0f};       // $delta per 1% volatility change
    float rho{0.0f};        // $delta per 1% rate change
    float _padding[3];      // Align to 32 bytes for AVX2

    // Helper to check if Greeks are neutral
    [[nodiscard]] auto isNeutral(float tolerance = 0.01f) const -> bool {
        return std::abs(delta) < tolerance;
    }

    // Scale all Greeks by multiplier (for position sizing)
    auto scale(float multiplier) -> void {
        delta *= multiplier;
        gamma *= multiplier;
        theta *= multiplier;
        vega *= multiplier;
        rho *= multiplier;
    }
};

// ============================================================================
// Option Leg Structure
// ============================================================================

struct OptionLeg {
    bool is_call{true};                 // Call (true) or Put (false)
    bool is_long{true};                 // Long (true) or Short (false)
    float strike{100.0f};               // Strike price
    float quantity{1.0f};               // Number of contracts
    float days_to_expiration{30.0f};    // Time to expiration (days)
    float implied_volatility{0.20f};    // IV (20% = 0.20)
    float premium{5.0f};                // Premium paid/received per contract

    // Helper to get position sign (+1 long, -1 short)
    [[nodiscard]] auto getSign() const -> float {
        return is_long ? 1.0f : -1.0f;
    }

    // Calculate intrinsic value at expiration
    [[nodiscard]] auto intrinsicValue(float underlying_price) const -> float {
        float payoff = is_call ?
            std::max(0.0f, underlying_price - strike) :
            std::max(0.0f, strike - underlying_price);
        return payoff * getSign() * quantity;
    }
};

// ============================================================================
// Base Strategy Interface
// ============================================================================

/**
 * Abstract base class for all options strategies
 *
 * Implements common functionality and defines interface for:
 * - Strategy metadata (name, type, outlook, complexity)
 * - Pricing calculations (P&L, Greeks)
 * - Batch operations (SIMD-optimized)
 * - Position management (legs, breakevens, max profit/loss)
 */
class IOptionsStrategy {
public:
    virtual ~IOptionsStrategy() = default;

    // ------------------------------------------------------------------------
    // Strategy Metadata
    // ------------------------------------------------------------------------

    [[nodiscard]] virtual auto getName() const -> std::string_view = 0;
    [[nodiscard]] virtual auto getType() const -> StrategyType = 0;
    [[nodiscard]] virtual auto getOutlook() const -> MarketOutlook = 0;
    [[nodiscard]] virtual auto getComplexity() const -> ComplexityLevel = 0;
    [[nodiscard]] virtual auto getDescription() const -> std::string_view = 0;

    // ------------------------------------------------------------------------
    // Pricing & P&L Calculations
    // ------------------------------------------------------------------------

    /**
     * Calculate profit/loss at a given underlying price
     *
     * @param underlying_price Current price of underlying asset
     * @param days_elapsed Days elapsed since position opened
     * @return P&L in dollars (positive = profit, negative = loss)
     */
    [[nodiscard]] virtual auto calculateProfitLoss(
        float underlying_price,
        float days_elapsed = 0.0f) const -> float = 0;

    /**
     * Calculate profit/loss at expiration (ignoring time decay)
     *
     * @param underlying_price Price at expiration
     * @return P&L at expiration
     */
    [[nodiscard]] virtual auto calculateExpirationPL(
        float underlying_price) const -> float = 0;

    // ------------------------------------------------------------------------
    // Greeks Calculations
    // ------------------------------------------------------------------------

    /**
     * Calculate all Greeks for current position
     *
     * @param underlying_price Current underlying price
     * @param risk_free_rate Current risk-free rate (e.g., 0.05 = 5%)
     * @return Greeks structure with all 5 Greeks
     */
    [[nodiscard]] virtual auto calculateGreeks(
        float underlying_price,
        float risk_free_rate) const -> Greeks = 0;

    // ------------------------------------------------------------------------
    // Batch Calculations (SIMD-optimized)
    // ------------------------------------------------------------------------

    /**
     * Calculate P&L for multiple underlying prices (AVX2: 8 at once)
     *
     * @param underlying_prices Array of prices to evaluate
     * @param days_elapsed Days elapsed since position opened
     * @return Vector of P&L values (same order as input)
     */
    [[nodiscard]] virtual auto calculateProfitLossBatch(
        std::span<float const> underlying_prices,
        float days_elapsed = 0.0f) const -> std::vector<float> = 0;

    // ------------------------------------------------------------------------
    // Position Management
    // ------------------------------------------------------------------------

    /**
     * Get all legs of the strategy
     */
    [[nodiscard]] virtual auto getLegs() const -> std::span<OptionLeg const> = 0;

    /**
     * Get maximum profit potential
     * Returns nullopt if unlimited profit potential
     */
    [[nodiscard]] virtual auto getMaxProfit() const -> std::optional<float> = 0;

    /**
     * Get maximum loss potential
     * Returns nullopt if unlimited loss potential
     */
    [[nodiscard]] virtual auto getMaxLoss() const -> std::optional<float> = 0;

    /**
     * Get breakeven points (prices where P&L = 0)
     */
    [[nodiscard]] virtual auto getBreakevens() const -> std::vector<float> = 0;

    /**
     * Get net debit/credit for opening the position
     * Positive = net debit (money paid)
     * Negative = net credit (money received)
     */
    [[nodiscard]] virtual auto getNetDebit() const -> float = 0;

    /**
     * Check if strategy is currently profitable
     */
    [[nodiscard]] virtual auto isProfitable(
        float underlying_price,
        float days_elapsed = 0.0f) const -> bool {
        return calculateProfitLoss(underlying_price, days_elapsed) > 0.0f;
    }
};

// ============================================================================
// Base Strategy Implementation (CRTP Pattern)
// ============================================================================

/**
 * CRTP base class providing common implementations
 * Reduces code duplication across 52+ strategies
 */
template<typename Derived>
class BaseOptionsStrategy : public IOptionsStrategy {
protected:
    std::string name_;
    std::string description_;
    StrategyType type_;
    MarketOutlook outlook_;
    ComplexityLevel complexity_;
    std::vector<OptionLeg> legs_;

public:
    // Constructor
    BaseOptionsStrategy(
        std::string name,
        std::string description,
        StrategyType type,
        MarketOutlook outlook,
        ComplexityLevel complexity)
        : name_(std::move(name))
        , description_(std::move(description))
        , type_(type)
        , outlook_(outlook)
        , complexity_(complexity) {}

    // Metadata (final implementations)
    [[nodiscard]] auto getName() const -> std::string_view final {
        return name_;
    }

    [[nodiscard]] auto getDescription() const -> std::string_view final {
        return description_;
    }

    [[nodiscard]] auto getType() const -> StrategyType final {
        return type_;
    }

    [[nodiscard]] auto getOutlook() const -> MarketOutlook final {
        return outlook_;
    }

    [[nodiscard]] auto getComplexity() const -> ComplexityLevel final {
        return complexity_;
    }

    // Legs
    [[nodiscard]] auto getLegs() const -> std::span<OptionLeg const> final {
        return legs_;
    }

    // Net debit calculation
    [[nodiscard]] auto getNetDebit() const -> float final {
        float net_debit = 0.0f;
        for (auto const& leg : legs_) {
            net_debit += leg.premium * leg.getSign() * leg.quantity;
        }
        return net_debit;
    }

    // Default batch implementation (can be overridden for SIMD)
    [[nodiscard]] auto calculateProfitLossBatch(
        std::span<float const> underlying_prices,
        float days_elapsed = 0.0f) const -> std::vector<float> override {

        std::vector<float> results;
        results.reserve(underlying_prices.size());

        for (float price : underlying_prices) {
            results.push_back(calculateProfitLoss(price, days_elapsed));
        }

        return results;
    }

protected:
    // Helper: Add a leg to the strategy
    auto addLeg(OptionLeg leg) -> void {
        legs_.push_back(std::move(leg));
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create strategy by name (factory pattern)
 * Returns nullptr if strategy name not recognized
 */
[[nodiscard]] auto createStrategy(std::string_view name)
    -> std::unique_ptr<IOptionsStrategy>;

/**
 * Get list of all available strategy names
 */
[[nodiscard]] auto getAllStrategyNames() -> std::vector<std::string_view>;

} // namespace bigbrother::options_strategies
