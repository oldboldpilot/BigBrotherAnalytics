#pragma once

#include "strategy.hpp"

namespace bigbrother::strategy {

/**
 * Delta-Neutral Straddle Strategy
 *
 * Options strategy that profits from volatility regardless of direction.
 *
 * Strategy:
 * - Buy 1 ATM call + 1 ATM put with same strike and expiration
 * - Delta-neutral: Δ_call + Δ_put ≈ 0
 * - Profits when underlying moves significantly in either direction
 * - Maximum loss: Total premium paid (if stock doesn't move)
 *
 * Ideal Conditions:
 * - Expecting significant price movement (earnings, Fed announcement)
 * - Low current IV (cheap options)
 * - High realized volatility expected
 * - Time to expiration: 7-45 days optimal
 *
 * Entry Criteria:
 * - IV percentile < 50% (options are cheap)
 * - Upcoming catalyst (earnings, economic data)
 * - Expected move > breakeven move
 * - Positive expected value from Monte Carlo
 *
 * Exit Criteria:
 * - Profit target reached (50-100% of premium)
 * - Stop loss (50% of premium)
 * - Time decay threshold (70% of maximum theta)
 * - Volatility spike achieved (sell into high IV)
 *
 * Risk Management:
 * - Maximum position: $1,500 (5% of $30k account)
 * - Hard stop loss at 50% of premium
 * - Close before expiration week
 * - Maximum 3 straddles simultaneously
 */
class DeltaNeutralStraddleStrategy : public BaseStrategy {
public:
    struct Parameters {
        double max_position_size{1500.0};      // $1,500 max per PRD
        double min_expected_return{50.0};      // $50 minimum EV
        double min_win_probability{0.60};      // 60% minimum win rate
        int min_days_to_expiration{7};         // Minimum DTE
        int max_days_to_expiration{45};        // Maximum DTE
        double max_iv_percentile{50.0};        // Buy when IV < 50th percentile
        double profit_target_percent{75.0};    // Take profit at 75% gain
        double stop_loss_percent{50.0};        // Stop loss at 50% loss
        int max_concurrent_positions{3};       // Max 3 straddles

        [[nodiscard]] auto validate() const noexcept -> Result<void>;
    };

    DeltaNeutralStraddleStrategy();  // Use default parameters
    explicit DeltaNeutralStraddleStrategy(Parameters params);

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> override;

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override;

private:
    /**
     * Find best straddle opportunity for a symbol
     */
    [[nodiscard]] auto findBestStraddle(
        std::string const& symbol,
        StrategyContext const& context
    ) const -> std::optional<TradingSignal>;

    /**
     * Calculate straddle breakeven points
     */
    [[nodiscard]] auto calculateBreakevens(
        Price strike,
        Price call_premium,
        Price put_premium
    ) const noexcept -> std::pair<Price, Price>;

    /**
     * Calculate expected move based on IV
     */
    [[nodiscard]] auto calculateExpectedMove(
        Price spot,
        double implied_volatility,
        double days_to_expiration
    ) const noexcept -> Price;

    /**
     * Check if delta is near-neutral
     */
    [[nodiscard]] auto isDeltaNeutral(
        options::Greeks const& call_greeks,
        options::Greeks const& put_greeks,
        double tolerance = 0.1
    ) const noexcept -> bool;

    Parameters params_;
};

/**
 * Delta-Neutral Strangle Strategy
 *
 * Similar to straddle but with different strikes (OTM call + OTM put).
 *
 * Strategy:
 * - Buy 1 OTM call + 1 OTM put (different strikes)
 * - Cheaper than straddle (lower premium)
 * - Requires larger move to profit
 * - Better risk/reward ratio
 *
 * Differences from Straddle:
 * - Lower cost (OTM options cheaper)
 * - Wider breakeven points
 * - Still delta-neutral
 * - Lower theta decay
 */
class DeltaNeutralStrangleStrategy : public BaseStrategy {
public:
    struct Parameters {
        double max_position_size{1500.0};
        double min_expected_return{50.0};
        double min_win_probability{0.55};      // Lower than straddle
        int min_days_to_expiration{14};
        int max_days_to_expiration{60};
        double call_delta_target{0.30};        // Buy 30-delta calls
        double put_delta_target{-0.30};        // Buy -30-delta puts
        double profit_target_percent{100.0};   // 100% profit target
        double stop_loss_percent{50.0};
        int max_concurrent_positions{3};

        [[nodiscard]] auto validate() const noexcept -> Result<void>;
    };

    DeltaNeutralStrangleStrategy();  // Use default parameters
    explicit DeltaNeutralStrangleStrategy(Parameters params);

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> override;

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override;

private:
    [[nodiscard]] auto findBestStrangle(
        std::string const& symbol,
        StrategyContext const& context
    ) const -> std::optional<TradingSignal>;

    /**
     * Find strike with target delta
     */
    [[nodiscard]] auto findStrikeByDelta(
        schwab::OptionsChainData const& chain,
        options::OptionType type,
        double target_delta,
        Price spot_price
    ) const -> std::optional<schwab::OptionsChainData::OptionQuote>;

    Parameters params_;
};

} // namespace bigbrother::strategy
