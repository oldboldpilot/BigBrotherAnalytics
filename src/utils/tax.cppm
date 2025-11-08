/**
 * BigBrotherAnalytics - Tax Calculation Module (C++23)
 *
 * Comprehensive tax calculation for algorithmic trading.
 * CRITICAL: True profitability requires after-tax returns.
 *
 * US Tax Rules for Trading (2025):
 * - Short-term capital gains (<= 1 year): Taxed as ordinary income (up to 37%)
 * - Long-term capital gains (> 1 year): Preferential rates (0%, 15%, 20%)
 * - Wash sale rule: Cannot claim loss if repurchase within 30 days
 * - Day trading: Pattern day trader rules (25k minimum)
 * - Options: Generally short-term treatment
 * - Section 1256 contracts: 60/40 rule (60% long-term, 40% short-term)
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - Fluent API for tax calculation
 * - std::expected for error handling
 * - Modern C++23 features
 *
 * References:
 * - IRS Publication 550 (Investment Income and Expenses)
 * - IRS Publication 17 (Federal Income Tax)
 */

// Global module fragment
module;

#include <vector>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <string>
#include <chrono>
#include <algorithm>
#include <optional>
#include <cmath>

// Module declaration
export module bigbrother.utils.tax;

// Import dependencies
import bigbrother.utils.types;

export namespace bigbrother::utils::tax {

using namespace bigbrother::types;

// ============================================================================
// Tax Configuration
// ============================================================================

/**
 * Tax Configuration for US Taxpayers
 * C.1: Struct for passive data
 */
struct TaxConfig {
    // Federal tax rates (2025 brackets for single filer)
    double short_term_rate{0.24};         // 24% (assume $90k-$190k bracket)
    double long_term_rate{0.15};          // 15% (assume $44k-$492k bracket)
    double medicare_surtax{0.038};        // 3.8% Net Investment Income Tax (NIIT)

    // State tax (varies by state)
    double state_tax_rate{0.05};          // 5% (conservative estimate)

    // Trading status
    bool is_pattern_day_trader{true};     // Day trading = short-term treatment
    bool is_section_1256_trader{false};   // Futures/options on indices (60/40 rule)

    // Wash sale tracking
    bool track_wash_sales{true};          // Enable wash sale rule enforcement
    int wash_sale_window_days{30};        // IRS: 30 days before/after

    /**
     * Calculate effective tax rate for short-term gains
     */
    [[nodiscard]] constexpr auto effectiveShortTermRate() const noexcept -> double {
        return short_term_rate + medicare_surtax + state_tax_rate;
    }

    /**
     * Calculate effective tax rate for long-term gains
     */
    [[nodiscard]] constexpr auto effectiveLongTermRate() const noexcept -> double {
        return long_term_rate + medicare_surtax + state_tax_rate;
    }

    /**
     * Validate configuration
     */
    [[nodiscard]] auto validate() const noexcept -> Result<void> {
        if (short_term_rate < 0.0 || short_term_rate > 1.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid short-term rate");
        }
        if (long_term_rate < 0.0 || long_term_rate > 1.0) {
            return makeError<void>(ErrorCode::InvalidParameter, "Invalid long-term rate");
        }
        return {};
    }
};

/**
 * Trade for Tax Purposes
 */
struct TaxTrade {
    std::string trade_id;
    std::string symbol;
    Timestamp entry_time{0};
    Timestamp exit_time{0};
    double cost_basis{0.0};          // Entry cost
    double proceeds{0.0};            // Exit proceeds
    double gross_pnl{0.0};           // Gross profit/loss
    bool is_options{false};          // Options vs stocks
    bool is_index_option{false};     // Section 1256 eligible

    // Wash sale tracking
    bool wash_sale_disallowed{false};
    double wash_sale_amount{0.0};

    /**
     * Calculate holding period in days
     */
    [[nodiscard]] constexpr auto holdingPeriodDays() const noexcept -> int {
        return static_cast<int>((exit_time - entry_time) / (1'000'000LL * 86400LL));
    }

    /**
     * Check if long-term (> 365 days)
     */
    [[nodiscard]] constexpr auto isLongTerm() const noexcept -> bool {
        return holdingPeriodDays() > 365;
    }

    /**
     * Check if short-term (<= 365 days)
     */
    [[nodiscard]] constexpr auto isShortTerm() const noexcept -> bool {
        return !isLongTerm();
    }

    /**
     * Check if gain or loss
     */
    [[nodiscard]] constexpr auto isGain() const noexcept -> bool {
        return gross_pnl > 0.0;
    }

    [[nodiscard]] constexpr auto isLoss() const noexcept -> bool {
        return gross_pnl < 0.0;
    }
};

/**
 * Tax Calculation Result
 */
struct TaxResult {
    // Gross P&L breakdown
    double total_gross_pnl{0.0};
    double short_term_gains{0.0};
    double long_term_gains{0.0};
    double short_term_losses{0.0};
    double long_term_losses{0.0};

    // Tax calculations
    double taxable_short_term{0.0};    // After offsetting losses
    double taxable_long_term{0.0};     // After offsetting losses
    double total_tax_owed{0.0};

    // After-tax results
    double net_pnl_after_tax{0.0};
    double effective_tax_rate{0.0};    // Total tax / Total gains

    // Wash sale impact
    int wash_sales_disallowed{0};
    double wash_sale_loss_disallowed{0.0};

    // Carryforward
    double capital_loss_carryforward{0.0};  // Losses > $3,000 carry forward

    /**
     * Calculate tax efficiency
     * Higher is better (less tax drag)
     */
    [[nodiscard]] constexpr auto taxEfficiency() const noexcept -> double {
        if (total_gross_pnl <= 0.0) return 1.0;
        return net_pnl_after_tax / total_gross_pnl;
    }

    /**
     * Check if profitable after tax
     */
    [[nodiscard]] constexpr auto isProfitableAfterTax() const noexcept -> bool {
        return net_pnl_after_tax > 0.0;
    }
};

// ============================================================================
// Tax Calculator - Core Tax Engine
// ============================================================================

/**
 * Tax Calculator
 *
 * Calculates taxes on trading activity following IRS rules.
 * All functions use trailing return syntax and modern C++23 features.
 */
class TaxCalculator {
public:
    explicit TaxCalculator(TaxConfig config = TaxConfig{})
        : config_{std::move(config)} {}

    /**
     * Calculate taxes on all trades
     *
     * @param trades Vector of completed trades
     * @return Complete tax calculation with breakdown
     */
    [[nodiscard]] auto calculateTaxes(std::vector<TaxTrade> const& trades) -> Result<TaxResult> {

        TaxResult result{};

        // Apply wash sale rules first
        auto trades_after_wash = applyWashSaleRules(trades);

        // Categorize gains and losses
        for (auto const& trade : trades_after_wash) {
            if (trade.is_index_option && config_.is_section_1256_trader) {
                // Section 1256: 60% long-term, 40% short-term regardless of holding period
                if (trade.isGain()) {
                    result.long_term_gains += trade.gross_pnl * 0.60;
                    result.short_term_gains += trade.gross_pnl * 0.40;
                } else {
                    result.long_term_losses += std::abs(trade.gross_pnl) * 0.60;
                    result.short_term_losses += std::abs(trade.gross_pnl) * 0.40;
                }
            } else if (trade.isLongTerm()) {
                // Long-term treatment
                if (trade.isGain()) {
                    result.long_term_gains += trade.gross_pnl;
                } else {
                    result.long_term_losses += std::abs(trade.gross_pnl);
                }
            } else {
                // Short-term treatment (most day trading)
                if (trade.isGain()) {
                    result.short_term_gains += trade.gross_pnl;
                } else if (!trade.wash_sale_disallowed) {
                    result.short_term_losses += std::abs(trade.gross_pnl);
                }
            }

            result.total_gross_pnl += trade.gross_pnl;

            if (trade.wash_sale_disallowed) {
                result.wash_sales_disallowed++;
                result.wash_sale_loss_disallowed += std::abs(trade.wash_sale_amount);
            }
        }

        // Net short-term and long-term (offset gains with losses)
        result.taxable_short_term = std::max(0.0,
            result.short_term_gains - result.short_term_losses);

        result.taxable_long_term = std::max(0.0,
            result.long_term_gains - result.long_term_losses);

        // Calculate taxes owed
        double const st_tax = result.taxable_short_term * config_.effectiveShortTermRate();
        double const lt_tax = result.taxable_long_term * config_.effectiveLongTermRate();

        result.total_tax_owed = st_tax + lt_tax;

        // After-tax P&L
        result.net_pnl_after_tax = result.total_gross_pnl - result.total_tax_owed;

        // Effective tax rate
        if (result.total_gross_pnl > 0.0) {
            result.effective_tax_rate = result.total_tax_owed / result.total_gross_pnl;
        }

        // Capital loss carryforward (IRS: max $3,000 deduction per year)
        double const total_losses = result.short_term_losses + result.long_term_losses;
        double const total_gains = result.short_term_gains + result.long_term_gains;
        double const net_loss = total_losses - total_gains;

        if (net_loss > 3'000.0) {
            result.capital_loss_carryforward = net_loss - 3'000.0;
        }

        return result;
    }

    /**
     * Calculate estimated quarterly tax payment
     * (Required for traders to avoid penalties)
     */
    [[nodiscard]] auto calculateQuarterlyTax(
        double quarterly_profit
    ) const noexcept -> double {
        // Most day trading is short-term
        return quarterly_profit * config_.effectiveShortTermRate();
    }

    /**
     * Apply wash sale rules
     *
     * IRS Wash Sale Rule: If you sell a security at a loss and buy
     * substantially identical security within 30 days before or after,
     * you cannot deduct the loss.
     */
    [[nodiscard]] auto applyWashSaleRules(std::vector<TaxTrade> trades) const
        -> std::vector<TaxTrade> {

        if (!config_.track_wash_sales) {
            return trades;
        }

        auto result = trades;

        // Sort by exit time
        std::ranges::sort(result, [](auto const& a, auto const& b) {
            return a.exit_time < b.exit_time;
        });

        // Check each loss for wash sales
        for (size_t i = 0; i < result.size(); ++i) {
            if (!result[i].isLoss()) {
                continue;  // Wash sale only applies to losses
            }

            // Check 30-day window before and after
            for (size_t j = 0; j < result.size(); ++j) {
                if (i == j) continue;

                // Same or substantially identical security
                if (result[i].symbol != result[j].symbol) {
                    continue;
                }

                // Check if within wash sale window
                auto const days_diff = std::abs(
                    static_cast<int>((result[j].entry_time - result[i].exit_time) /
                    (1'000'000LL * 86400LL))
                );

                if (days_diff <= config_.wash_sale_window_days) {
                    // Wash sale! Disallow the loss
                    result[i].wash_sale_disallowed = true;
                    result[i].wash_sale_amount = std::abs(result[i].gross_pnl);

                    // Add disallowed loss to cost basis of replacement
                    // (IRS rule: loss is deferred, not lost forever)
                    result[j].cost_basis += std::abs(result[i].gross_pnl);
                    break;
                }
            }
        }

        return result;
    }

private:
    TaxConfig config_;
};

// ============================================================================
// Fluent API for Tax Calculation
// ============================================================================

/**
 * Tax Calculator Builder - Fluent API
 *
 * Example Usage:
 *
 *   // Calculate taxes on trades
 *   auto tax_result = TaxCalculatorBuilder()
 *       .federalRate(0.24)
 *       .stateRate(0.05)
 *       .withMedicareSurtax()
 *       .patternDayTrader()
 *       .trackWashSales()
 *       .addTrades(all_trades)
 *       .calculate();
 *
 *   std::println("Gross P&L: ${}", tax_result->total_gross_pnl);
 *   std::println("Tax Owed: ${}", tax_result->total_tax_owed);
 *   std::println("Net After Tax: ${}", tax_result->net_pnl_after_tax);
 *   std::println("Effective Tax Rate: {:.1f}%", tax_result->effective_tax_rate * 100);
 *
 *   // Quick after-tax return calculation
 *   auto after_tax_return = TaxCalculatorBuilder()
 *       .federalRate(0.24)
 *       .grossReturn(10000.0)
 *       .shortTermGains()
 *       .calculateAfterTax();
 */
class TaxCalculatorBuilder {
public:
    TaxCalculatorBuilder()
        : config_{} {}

    /**
     * Set federal tax rate
     */
    [[nodiscard]] auto federalRate(double rate) noexcept -> TaxCalculatorBuilder& {
        config_.short_term_rate = rate;
        return *this;
    }

    /**
     * Set long-term capital gains rate
     */
    [[nodiscard]] auto longTermRate(double rate) noexcept -> TaxCalculatorBuilder& {
        config_.long_term_rate = rate;
        return *this;
    }

    /**
     * Set state tax rate
     */
    [[nodiscard]] auto stateRate(double rate) noexcept -> TaxCalculatorBuilder& {
        config_.state_tax_rate = rate;
        return *this;
    }

    /**
     * Include Medicare surtax (3.8% NIIT)
     */
    [[nodiscard]] auto withMedicareSurtax() noexcept -> TaxCalculatorBuilder& {
        config_.medicare_surtax = 0.038;
        return *this;
    }

    /**
     * Mark as pattern day trader (all short-term)
     */
    [[nodiscard]] auto patternDayTrader() noexcept -> TaxCalculatorBuilder& {
        config_.is_pattern_day_trader = true;
        return *this;
    }

    /**
     * Enable Section 1256 treatment (60/40 rule)
     */
    [[nodiscard]] auto section1256Trader() noexcept -> TaxCalculatorBuilder& {
        config_.is_section_1256_trader = true;
        return *this;
    }

    /**
     * Enable wash sale tracking
     */
    [[nodiscard]] auto trackWashSales() noexcept -> TaxCalculatorBuilder& {
        config_.track_wash_sales = true;
        return *this;
    }

    /**
     * Add trades for calculation
     */
    [[nodiscard]] auto addTrades(std::vector<TaxTrade> trades) -> TaxCalculatorBuilder& {
        trades_ = std::move(trades);
        return *this;
    }

    [[nodiscard]] auto addTrade(TaxTrade trade) -> TaxCalculatorBuilder& {
        trades_.push_back(std::move(trade));
        return *this;
    }

    /**
     * Calculate full tax result (terminal operation)
     */
    [[nodiscard]] auto calculate() -> Result<TaxResult> {
        TaxCalculator calc{config_};
        return calc.calculateTaxes(trades_);
    }

    /**
     * Quick after-tax return calculation (terminal operation)
     */
    [[nodiscard]] auto calculateAfterTax(double gross_return) const noexcept -> double {
        // Assume short-term for day trading
        double const tax = gross_return * config_.effectiveShortTermRate();
        return gross_return - tax;
    }

private:
    TaxConfig config_;
    std::vector<TaxTrade> trades_;
};

// ============================================================================
// Tax-Adjusted Performance Metrics
// ============================================================================

/**
 * Tax-Adjusted Metrics
 *
 * Performance metrics that account for tax impact
 */
struct TaxAdjustedMetrics {
    // Pre-tax metrics
    double gross_return{0.0};
    double gross_return_percent{0.0};

    // Tax impact
    double total_tax_paid{0.0};
    double effective_tax_rate{0.0};

    // After-tax metrics
    double net_return_after_tax{0.0};
    double net_return_percent_after_tax{0.0};
    double after_tax_sharpe_ratio{0.0};

    // Tax efficiency
    double tax_efficiency{0.0};        // Net / Gross
    double tax_drag{0.0};              // Tax / Gross

    /**
     * Compare to pre-tax metrics
     */
    [[nodiscard]] constexpr auto taxDragPercent() const noexcept -> double {
        if (gross_return <= 0.0) return 0.0;
        return (total_tax_paid / gross_return) * 100.0;
    }
};

/**
 * Calculate after-tax Sharpe ratio
 *
 * Critical: Sharpe ratio must use after-tax returns for accuracy
 */
[[nodiscard]] inline auto calculateAfterTaxSharpe(
    std::vector<double> const& daily_returns,
    TaxConfig const& tax_config,
    double risk_free_rate = 0.0
) -> double {

    if (daily_returns.empty()) {
        return 0.0;
    }

    // Apply tax drag to returns
    std::vector<double> after_tax_returns;
    after_tax_returns.reserve(daily_returns.size());

    double const tax_rate = tax_config.effectiveShortTermRate();

    for (auto const& ret : daily_returns) {
        // Only tax positive returns
        double const after_tax = ret > 0.0 ? ret * (1.0 - tax_rate) : ret;
        after_tax_returns.push_back(after_tax);
    }

    // Calculate Sharpe on after-tax returns
    double const mean_return = std::accumulate(
        after_tax_returns.begin(), after_tax_returns.end(), 0.0
    ) / static_cast<double>(after_tax_returns.size());

    double const variance = std::accumulate(
        after_tax_returns.begin(), after_tax_returns.end(), 0.0,
        [mean_return](double acc, double ret) {
            double const diff = ret - mean_return;
            return acc + diff * diff;
        }
    ) / static_cast<double>(after_tax_returns.size());

    double const std_dev = std::sqrt(variance);

    if (std_dev == 0.0) {
        return 0.0;
    }

    return (mean_return - risk_free_rate) / std_dev * std::sqrt(252.0);
}

} // export namespace bigbrother::utils::tax
