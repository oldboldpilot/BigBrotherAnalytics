#pragma once

#include "strategy.hpp"

namespace bigbrother::strategy {

/**
 * Volatility Arbitrage Strategy
 *
 * Profits from mispricing between implied volatility (IV) and realized volatility (RV).
 *
 * Strategy:
 * - Buy options when IV < RV (options are cheap)
 * - Sell options when IV > RV (options are expensive)
 * - Delta-hedge to isolate volatility exposure
 * - Rebalance hedge periodically
 *
 * Key Concepts:
 * - Implied Volatility (IV): Market's expectation of future volatility (from option prices)
 * - Realized Volatility (RV): Actual historical volatility (from price movements)
 * - Volatility Risk Premium: IV typically > RV (options trade at premium)
 *
 * Entry Criteria:
 * - IV < RV: Buy options (expect IV to increase or RV to validate IV)
 * - IV > RV * 1.5: Sell options (IV overpriced)
 * - Significant IV percentile deviation (< 30th or > 70th percentile)
 * - Positive expected value from mispricing
 *
 * Delta Hedging:
 * - If we buy call with Δ=0.5, short 50 shares of stock
 * - Rebalance when delta changes significantly
 * - Goal: Profit purely from volatility, not direction
 *
 * Risk Management:
 * - Gamma risk: Large price moves can blow through delta hedge
 * - Vega risk: IV can continue to move against us
 * - Theta decay: Time works against long options
 * - Maximum position: $1,500 per trade
 */
class VolatilityArbitrageStrategy : public BaseStrategy {
public:
    struct Parameters {
        double max_position_size{1500.0};
        double min_iv_rv_spread{0.05};         // 5% spread minimum
        double min_expected_return{75.0};      // Higher than straddle (more complex)
        double min_win_probability{0.65};
        int lookback_period{20};               // Days for RV calculation
        double iv_percentile_low{30.0};        // Buy when IV < 30th percentile
        double iv_percentile_high{70.0};       // Sell when IV > 70th percentile
        double delta_rebalance_threshold{0.15}; // Rebalance when |Δ| > 0.15
        double gamma_risk_limit{0.10};         // Max gamma exposure
        int min_days_to_expiration{14};
        int max_days_to_expiration{60};

        [[nodiscard]] auto validate() const noexcept -> Result<void>;
    };

    VolatilityArbitrageStrategy();  // Use default parameters
    explicit VolatilityArbitrageStrategy(Parameters params);

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> override;

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override;

private:
    /**
     * Calculate realized volatility from historical data
     */
    [[nodiscard]] auto calculateRealizedVolatility(
        std::vector<Bar> const& bars,
        int lookback_period
    ) const -> double;

    /**
     * Calculate IV percentile (current IV vs historical IV)
     */
    [[nodiscard]] auto calculateIVPercentile(
        double current_iv,
        std::vector<double> const& historical_iv
    ) const -> double;

    /**
     * Find best volatility arbitrage opportunity
     */
    [[nodiscard]] auto findBestVolArb(
        std::string const& symbol,
        StrategyContext const& context
    ) const -> std::optional<TradingSignal>;

    /**
     * Calculate delta hedge size
     */
    [[nodiscard]] auto calculateDeltaHedge(
        int num_contracts,
        double delta
    ) const noexcept -> int;

    Parameters params_;
};

/**
 * Mean Reversion Strategy
 *
 * Trade based on correlation breakdown and mean reversion.
 *
 * Strategy:
 * - Identify pairs with historical strong correlation
 * - Detect when correlation breaks down temporarily
 * - Trade expecting correlation to restore (mean reversion)
 *
 * Example:
 * - AAPL and MSFT historically correlated 0.85
 * - Recent divergence: AAPL up 5%, MSFT flat
 * - Trade: Short AAPL, Long MSFT (expect convergence)
 */
class MeanReversionStrategy : public BaseStrategy {
public:
    struct Parameters {
        double min_historical_correlation{0.70};
        double min_correlation_breakdown{0.30};  // Require 0.30 drop in correlation
        double max_position_size{1500.0};
        double min_expected_return{50.0};
        double min_win_probability{0.60};
        int correlation_lookback{252};  // 1 year
        int recent_period{20};           // Last 20 days

        [[nodiscard]] auto validate() const noexcept -> Result<void>;
    };

    MeanReversionStrategy();  // Use default parameters
    explicit MeanReversionStrategy(Parameters params);

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> override;

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override;

private:
    Parameters params_;
};

} // namespace bigbrother::strategy
