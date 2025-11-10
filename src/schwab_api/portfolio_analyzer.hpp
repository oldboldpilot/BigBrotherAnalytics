/**
 * BigBrotherAnalytics - Portfolio Analyzer (C++23)
 *
 * Real-time portfolio analytics:
 * - Total equity and P/L tracking
 * - Sector exposure analysis
 * - Risk metrics (portfolio heat, correlation, VaR)
 * - Position concentration monitoring
 * - Performance attribution
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#pragma once

#include "account_types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace bigbrother::schwab {

/**
 * Sector exposure breakdown
 */
struct SectorExposure {
    std::string sector_name;
    double market_value{0.0};
    double percent_of_portfolio{0.0};
    int position_count{0};
    double total_pnl{0.0};
    double avg_pnl_percent{0.0};
};

/**
 * Risk metrics
 */
struct RiskMetrics {
    double portfolio_heat{0.0};       // Total risk as % of capital
    double value_at_risk_95{0.0};     // VaR at 95% confidence
    double expected_shortfall{0.0};   // Conditional VaR
    double portfolio_beta{0.0};       // Market beta
    double portfolio_volatility{0.0}; // Portfolio volatility
    double sharpe_ratio{0.0};         // Risk-adjusted return
    double sortino_ratio{0.0};        // Downside risk-adjusted return
    double max_drawdown{0.0};         // Maximum drawdown
    double concentration_risk{0.0};   // Herfindahl index
    int positions_at_risk{0};         // Positions with stop-loss approaching
};

/**
 * Performance metrics
 */
struct PerformanceMetrics {
    double total_return{0.0};
    double total_return_percent{0.0};
    double day_pnl{0.0};
    double day_pnl_percent{0.0};
    double week_pnl{0.0};
    double month_pnl{0.0};
    double ytd_pnl{0.0};
    double annualized_return{0.0};
    double win_rate{0.0};      // % of winning trades
    double profit_factor{0.0}; // Gross profit / gross loss
    double avg_win{0.0};
    double avg_loss{0.0};
    double largest_win{0.0};
    double largest_loss{0.0};
};

/**
 * Portfolio Analyzer - Advanced portfolio analytics
 */
class PortfolioAnalyzer {
  public:
    PortfolioAnalyzer() = default;

    // ========================================================================
    // Core Analytics
    // ========================================================================

    /**
     * Calculate comprehensive portfolio summary
     */
    [[nodiscard]] auto analyzePortfolio(std::vector<Position> const& positions,
                                        Balance const& balance) const -> PortfolioSummary {

        PortfolioSummary summary;
        summary.total_equity = balance.total_equity;
        summary.total_cash = balance.cash;
        summary.total_market_value = 0.0;
        summary.total_cost_basis = 0.0;
        summary.total_unrealized_pnl = 0.0;
        summary.total_day_pnl = 0.0;
        summary.position_count = 0;
        summary.long_position_count = 0;
        summary.short_position_count = 0;
        summary.largest_position_percent = 0.0;
        summary.portfolio_concentration = 0.0;

        for (auto const& pos : positions) {
            if (std::abs(pos.quantity) <= quantity_epsilon)
                continue;

            summary.position_count++;
            summary.total_market_value += pos.market_value;
            summary.total_cost_basis += pos.cost_basis;
            summary.total_unrealized_pnl += pos.unrealized_pnl;
            summary.total_day_pnl += pos.day_pnl;

            if (pos.isLong()) {
                summary.long_position_count++;
            } else if (pos.isShort()) {
                summary.short_position_count++;
            }

            // Calculate position weight
            if (balance.total_equity > 0.0) {
                double weight = (std::abs(pos.market_value) / balance.total_equity) * 100.0;
                summary.largest_position_percent =
                    std::max(summary.largest_position_percent, weight);

                // Herfindahl index
                double weight_fraction = weight / 100.0;
                summary.portfolio_concentration += weight_fraction * weight_fraction;
            }
        }

        // Calculate percentages
        if (summary.total_cost_basis > 0.0) {
            summary.total_unrealized_pnl_percent =
                (summary.total_unrealized_pnl / summary.total_cost_basis) * 100.0;
        }

        if (balance.total_equity > 0.0) {
            summary.total_day_pnl_percent = (summary.total_day_pnl / balance.total_equity) * 100.0;
        }

        summary.updated_at = getCurrentTimestamp();

        return summary;
    }

    /**
     * Calculate sector exposure
     * Requires sector mapping (symbol -> sector)
     */
    [[nodiscard]] auto
    calculateSectorExposure(std::vector<Position> const& positions,
                            std::unordered_map<std::string, std::string> const& sector_map,
                            double total_equity) const -> std::vector<SectorExposure> {

        std::unordered_map<std::string, SectorExposure> sector_data;

        for (auto const& pos : positions) {
            if (std::abs(pos.quantity) <= quantity_epsilon)
                continue;

            // Get sector for symbol
            std::string sector = "Unknown";
            if (sector_map.find(pos.symbol) != sector_map.end()) {
                sector = sector_map.at(pos.symbol);
            }

            // Accumulate sector metrics
            auto& exposure = sector_data[sector];
            exposure.sector_name = sector;
            exposure.market_value += pos.market_value;
            exposure.position_count++;
            exposure.total_pnl += pos.unrealized_pnl;
        }

        // Calculate percentages
        std::vector<SectorExposure> exposures;
        exposures.reserve(sector_data.size());

        for (auto& [sector, exposure] : sector_data) {
            if (total_equity > 0.0) {
                exposure.percent_of_portfolio = (exposure.market_value / total_equity) * 100.0;
            }

            if (exposure.market_value > 0.0) {
                exposure.avg_pnl_percent = (exposure.total_pnl / exposure.market_value) * 100.0;
            }

            exposures.push_back(exposure);
        }

        // Sort by market value descending
        std::sort(exposures.begin(), exposures.end(), [](auto const& a, auto const& b) -> bool {
            return a.market_value > b.market_value;
        });

        return exposures;
    }

    /**
     * Calculate risk metrics
     */
    [[nodiscard]] auto calculateRiskMetrics(std::vector<Position> const& positions,
                                            Balance const& balance) const -> RiskMetrics {

        RiskMetrics metrics;

        if (balance.total_equity <= 0.0) {
            return metrics;
        }

        // Portfolio heat (total position size as % of capital)
        double total_position_value = 0.0;
        for (auto const& pos : positions) {
            total_position_value += std::abs(pos.market_value);
        }
        metrics.portfolio_heat = (total_position_value / balance.total_equity) * 100.0;

        // Concentration risk (Herfindahl index)
        metrics.concentration_risk = 0.0;
        double max_position_percent = 0.0;

        for (auto const& pos : positions) {
            if (std::abs(pos.quantity) <= quantity_epsilon)
                continue;

            double weight = std::abs(pos.market_value) / balance.total_equity;
            metrics.concentration_risk += weight * weight;
            max_position_percent = std::max(max_position_percent, weight);
        }

        // Simple VaR estimation (95% confidence, assuming normal distribution)
        // VaR = 1.65 * sigma * portfolio_value
        // Using average position volatility as proxy
        std::vector<double> returns;
        for (auto const& pos : positions) {
            if (pos.cost_basis > 0.0) {
                returns.push_back(pos.unrealized_pnl_percent);
            }
        }

        if (!returns.empty()) {
            double avg_return =
                std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

            double variance = 0.0;
            for (auto const& ret : returns) {
                variance += (ret - avg_return) * (ret - avg_return);
            }
            variance /= returns.size();
            double volatility = std::sqrt(variance);

            metrics.portfolio_volatility = volatility;
            metrics.value_at_risk_95 = 1.65 * (volatility / 100.0) * balance.total_equity;
            metrics.expected_shortfall = 2.06 * (volatility / 100.0) * balance.total_equity;

            // Sharpe ratio (assuming risk-free rate of 4%)
            constexpr double risk_free_rate = 4.0;
            if (volatility > 0.0) {
                metrics.sharpe_ratio = (avg_return - risk_free_rate) / volatility;
            }
        }

        // Count positions at risk (losing > 5%)
        metrics.positions_at_risk =
            std::count_if(positions.begin(), positions.end(), [](auto const& pos) -> bool {
                return pos.unrealized_pnl_percent < -5.0;
            });

        // Portfolio beta (placeholder - would need market data)
        metrics.portfolio_beta = 1.0;

        return metrics;
    }

    /**
     * Calculate performance metrics
     */
    [[nodiscard]] auto calculatePerformanceMetrics(std::vector<Position> const& positions,
                                                   std::vector<Transaction> const& transactions,
                                                   Balance const& balance) const
        -> PerformanceMetrics {

        PerformanceMetrics metrics;

        // Total return from positions
        metrics.total_return = 0.0;
        metrics.day_pnl = 0.0;

        for (auto const& pos : positions) {
            metrics.total_return += pos.unrealized_pnl;
            metrics.day_pnl += pos.day_pnl;
        }

        if (balance.total_equity > 0.0) {
            metrics.total_return_percent = (metrics.total_return / balance.total_equity) * 100.0;
            metrics.day_pnl_percent = (metrics.day_pnl / balance.total_equity) * 100.0;
        }

        // Analyze completed trades from transactions
        std::vector<double> wins;
        std::vector<double> losses;

        for (auto const& txn : transactions) {
            if (!txn.isTradeTransaction())
                continue;

            // Simplified P/L calculation
            double pnl = txn.net_amount;

            if (pnl > 0) {
                wins.push_back(pnl);
            } else if (pnl < 0) {
                losses.push_back(std::abs(pnl));
            }
        }

        int total_trades = wins.size() + losses.size();
        if (total_trades > 0) {
            metrics.win_rate = (static_cast<double>(wins.size()) / total_trades) * 100.0;
        }

        if (!wins.empty()) {
            metrics.avg_win = std::accumulate(wins.begin(), wins.end(), 0.0) / wins.size();
            metrics.largest_win = *std::max_element(wins.begin(), wins.end());
        }

        if (!losses.empty()) {
            metrics.avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
            metrics.largest_loss = *std::max_element(losses.begin(), losses.end());
        }

        // Profit factor
        double gross_profit = std::accumulate(wins.begin(), wins.end(), 0.0);
        double gross_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
        if (gross_loss > 0.0) {
            metrics.profit_factor = gross_profit / gross_loss;
        }

        return metrics;
    }

    // ========================================================================
    // Position Analysis
    // ========================================================================

    /**
     * Find largest positions
     */
    [[nodiscard]] auto getLargestPositions(std::vector<Position> const& positions,
                                           int limit = 10) const -> std::vector<Position> {

        auto sorted = positions;

        std::sort(sorted.begin(), sorted.end(), [](auto const& a, auto const& b) -> bool {
            return std::abs(a.market_value) > std::abs(b.market_value);
        });

        if (sorted.size() > static_cast<size_t>(limit)) {
            sorted.resize(limit);
        }

        return sorted;
    }

    /**
     * Find top performers (by P/L)
     */
    [[nodiscard]] auto getTopPerformers(std::vector<Position> const& positions,
                                        int limit = 10) const -> std::vector<Position> {

        auto sorted = positions;

        std::sort(sorted.begin(), sorted.end(), [](auto const& a, auto const& b) -> bool {
            return a.unrealized_pnl > b.unrealized_pnl;
        });

        if (sorted.size() > static_cast<size_t>(limit)) {
            sorted.resize(limit);
        }

        return sorted;
    }

    /**
     * Find worst performers (by P/L)
     */
    [[nodiscard]] auto getWorstPerformers(std::vector<Position> const& positions,
                                          int limit = 10) const -> std::vector<Position> {

        auto sorted = positions;

        std::sort(sorted.begin(), sorted.end(), [](auto const& a, auto const& b) -> bool {
            return a.unrealized_pnl < b.unrealized_pnl;
        });

        if (sorted.size() > static_cast<size_t>(limit)) {
            sorted.resize(limit);
        }

        return sorted;
    }

    /**
     * Check position concentration risk
     */
    [[nodiscard]] auto hasConcentrationRisk(std::vector<Position> const& positions,
                                            double total_equity,
                                            double threshold_percent = 20.0) const -> bool {

        for (auto const& pos : positions) {
            if (total_equity <= 0.0)
                continue;

            double weight = (std::abs(pos.market_value) / total_equity) * 100.0;
            if (weight > threshold_percent) {
                return true;
            }
        }

        return false;
    }

  private:
    [[nodiscard]] auto getCurrentTimestamp() const noexcept -> Timestamp {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
};

} // namespace bigbrother::schwab
