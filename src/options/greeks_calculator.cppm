/**
 * @file greeks_calculator.cppm
 * @brief Real-Time Greeks Calculator for Options Portfolios
 *
 * Provides comprehensive Greeks calculation and portfolio risk analysis:
 * - Individual option Greeks (delta, gamma, theta, vega, rho)
 * - Portfolio aggregate Greeks
 * - Position Greeks tracking
 * - Dynamic hedging recommendations
 *
 * Greeks Definitions:
 * - Delta: Rate of change of option value w.r.t. underlying price (∂V/∂S)
 * - Gamma: Rate of change of delta w.r.t. underlying price (∂²V/∂S²)
 * - Theta: Time decay per day (∂V/∂t)
 * - Vega: Sensitivity to volatility changes (∂V/∂σ)
 * - Rho: Sensitivity to interest rate changes (∂V/∂r)
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 */

module;

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

export module options:greeks_calculator;

import options:trinomial_pricer;
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace options {

using namespace bigbrother::types;
using namespace bigbrother::utils;

// ============================================================================
// Portfolio Greeks Structure
// ============================================================================

struct PortfolioGreeks {
    double total_delta{0.0};          // Net delta exposure
    double total_gamma{0.0};          // Net gamma exposure
    double total_theta{0.0};          // Net theta decay (per day)
    double total_vega{0.0};           // Net vega exposure
    double total_rho{0.0};            // Net rho exposure

    double dollar_delta{0.0};         // Delta in dollar terms
    double dollar_gamma{0.0};         // Gamma in dollar terms
    double dollar_theta{0.0};         // Theta in dollar terms
    double dollar_vega{0.0};          // Vega in dollar terms

    int total_long_contracts{0};      // Total long option contracts
    int total_short_contracts{0};     // Total short option contracts
    double total_notional_value{0.0}; // Total notional value of options

    // Risk metrics
    double delta_neutrality_score{0.0};   // How close to delta-neutral (0-1)
    double gamma_risk_score{0.0};         // Gamma risk level (0-1)
    double theta_per_day_pct{0.0};        // Daily theta decay as % of portfolio
    double vega_risk_pct{0.0};            // Vega risk as % of portfolio

    [[nodiscard]] auto isDeltaNeutral(double tolerance = 0.10) const noexcept -> bool {
        return std::abs(total_delta) <= tolerance;
    }

    [[nodiscard]] auto getDeltaAdjustment() const noexcept -> double {
        // Shares needed to hedge delta
        return -total_delta;
    }
};

// ============================================================================
// Position Greeks Tracker
// ============================================================================

struct PositionGreeks {
    std::string symbol;
    std::string position_id;
    StrategyType strategy_type;

    // Individual Greeks
    double delta{0.0};
    double gamma{0.0};
    double theta{0.0};
    double vega{0.0};
    double rho{0.0};

    // Position details
    int quantity{0};                  // Number of contracts
    double spot_price{0.0};           // Current underlying price
    double strike_price{0.0};
    double time_to_expiry{0.0};       // Years
    double implied_volatility{0.0};
    double risk_free_rate{0.05};

    // Updated values
    double current_value{0.0};
    double unrealized_pnl{0.0};
    Timestamp last_update{0};

    [[nodiscard]] auto getDollarDelta() const noexcept -> double {
        return delta * quantity * 100.0; // 100 shares per contract
    }

    [[nodiscard]] auto getDollarGamma() const noexcept -> double {
        return gamma * quantity * 100.0 * spot_price / 100.0;
    }

    [[nodiscard]] auto getDollarTheta() const noexcept -> double {
        return theta * quantity;
    }

    [[nodiscard]] auto getDollarVega() const noexcept -> double {
        return vega * quantity;
    }
};

// ============================================================================
// Greeks Calculator Engine
// ============================================================================

class GreeksCalculator {
public:
    GreeksCalculator() : pricer_(100) {}

    /**
     * Calculate Greeks for a single option position
     */
    [[nodiscard]] auto calculatePositionGreeks(
        std::string const& symbol,
        double spot_price,
        double strike_price,
        double time_to_expiry,
        double volatility,
        double risk_free_rate,
        OptionType option_type,
        int quantity
    ) -> PositionGreeks {

        PositionGreeks pos_greeks;
        pos_greeks.symbol = symbol;
        pos_greeks.quantity = quantity;
        pos_greeks.spot_price = spot_price;
        pos_greeks.strike_price = strike_price;
        pos_greeks.time_to_expiry = time_to_expiry;
        pos_greeks.implied_volatility = volatility;
        pos_greeks.risk_free_rate = risk_free_rate;

        // Calculate option price and Greeks
        auto result = pricer_.price(
            spot_price, strike_price, time_to_expiry, volatility, risk_free_rate,
            option_type, OptionStyle::AMERICAN
        );

        pos_greeks.current_value = result.price * 100.0 * std::abs(quantity);

        // Greeks (adjust sign for short positions)
        double sign = quantity > 0 ? 1.0 : -1.0;
        pos_greeks.delta = result.greeks.delta * sign;
        pos_greeks.gamma = result.greeks.gamma * sign;
        pos_greeks.theta = result.greeks.theta * sign;
        pos_greeks.vega = result.greeks.vega * sign;
        pos_greeks.rho = result.greeks.rho * sign;

        pos_greeks.last_update = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        Logger::getInstance().debug(
            "Position Greeks for {} {}: Delta={:.4f}, Gamma={:.4f}, Theta={:.4f}, Vega={:.4f}",
            symbol, quantity > 0 ? "Long" : "Short",
            pos_greeks.delta, pos_greeks.gamma, pos_greeks.theta, pos_greeks.vega
        );

        return pos_greeks;
    }

    /**
     * Calculate aggregate portfolio Greeks
     */
    [[nodiscard]] auto calculatePortfolioGreeks(
        std::vector<PositionGreeks> const& positions,
        double portfolio_value
    ) -> PortfolioGreeks {

        PortfolioGreeks portfolio;

        for (auto const& pos : positions) {
            portfolio.total_delta += pos.delta * std::abs(pos.quantity);
            portfolio.total_gamma += pos.gamma * std::abs(pos.quantity);
            portfolio.total_theta += pos.theta * std::abs(pos.quantity);
            portfolio.total_vega += pos.vega * std::abs(pos.quantity);
            portfolio.total_rho += pos.rho * std::abs(pos.quantity);

            portfolio.dollar_delta += pos.getDollarDelta();
            portfolio.dollar_gamma += pos.getDollarGamma();
            portfolio.dollar_theta += pos.getDollarTheta();
            portfolio.dollar_vega += pos.getDollarVega();

            portfolio.total_notional_value += pos.spot_price * std::abs(pos.quantity) * 100.0;

            if (pos.quantity > 0) {
                portfolio.total_long_contracts += pos.quantity;
            } else {
                portfolio.total_short_contracts += std::abs(pos.quantity);
            }
        }

        // Calculate risk metrics
        portfolio.delta_neutrality_score = 1.0 - std::min(1.0, std::abs(portfolio.total_delta) / 10.0);
        portfolio.gamma_risk_score = std::min(1.0, std::abs(portfolio.total_gamma) / 0.10);

        if (portfolio_value > 0.0) {
            portfolio.theta_per_day_pct = (portfolio.dollar_theta / portfolio_value) * 100.0;
            portfolio.vega_risk_pct = (std::abs(portfolio.dollar_vega) / portfolio_value) * 100.0;
        }

        Logger::getInstance().info(
            "Portfolio Greeks: Delta={:.2f}, Gamma={:.4f}, Theta={:.2f}/day, Vega={:.2f}",
            portfolio.total_delta, portfolio.total_gamma, portfolio.total_theta, portfolio.total_vega
        );

        Logger::getInstance().info(
            "Portfolio Risk: Delta neutrality={:.1f}%, Gamma risk={:.1f}%, "
            "Theta decay={:.3f}%/day, Vega risk={:.2f}%",
            portfolio.delta_neutrality_score * 100.0,
            portfolio.gamma_risk_score * 100.0,
            portfolio.theta_per_day_pct,
            portfolio.vega_risk_pct
        );

        return portfolio;
    }

    /**
     * Calculate delta hedge recommendation
     *
     * Returns number of shares to buy/sell to achieve delta neutrality
     */
    [[nodiscard]] auto calculateDeltaHedge(
        PortfolioGreeks const& portfolio,
        double current_stock_price
    ) -> struct HedgeRecommendation {
        struct HedgeRecommendation {
            double shares_to_trade{0.0};
            std::string action;           // "BUY" or "SELL"
            double cost_estimate{0.0};
            double resulting_delta{0.0};
            std::string rationale;
        };

        HedgeRecommendation hedge;

        // Shares needed to neutralize delta
        double shares_needed = -portfolio.total_delta;

        hedge.shares_to_trade = std::abs(shares_needed);
        hedge.action = shares_needed > 0 ? "BUY" : "SELL";
        hedge.cost_estimate = hedge.shares_to_trade * current_stock_price;
        hedge.resulting_delta = portfolio.total_delta + shares_needed;

        hedge.rationale = "Current portfolio delta: " + std::to_string(portfolio.total_delta) +
                         ". " + hedge.action + " " + std::to_string(static_cast<int>(hedge.shares_to_trade)) +
                         " shares to achieve delta neutrality. Cost: $" + std::to_string(hedge.cost_estimate);

        return hedge;
    };

    /**
     * Calculate gamma scalping opportunity
     *
     * When gamma is high, small price moves create rehedging opportunities
     */
    [[nodiscard]] auto evaluateGammaScalping(
        PortfolioGreeks const& portfolio,
        double current_price,
        double price_change
    ) -> std::optional<struct ScalpingOpportunity> {

        struct ScalpingOpportunity {
            double price_move_threshold{0.0};
            double shares_to_adjust{0.0};
            double expected_profit{0.0};
            std::string recommendation;
        };

        // Only worthwhile if gamma is significant
        if (std::abs(portfolio.total_gamma) < 0.01) {
            return std::nullopt;
        }

        ScalpingOpportunity opp;

        // Delta change from price move
        double delta_change = portfolio.total_gamma * price_change;

        opp.price_move_threshold = current_price * 0.01; // 1% move
        opp.shares_to_adjust = std::abs(delta_change * 100.0);
        opp.expected_profit = 0.5 * portfolio.total_gamma * price_change * price_change * 100.0;

        opp.recommendation = "Price moved $" + std::to_string(price_change) +
                           ". Gamma=" + std::to_string(portfolio.total_gamma) +
                           ". Rehedge by trading " + std::to_string(static_cast<int>(opp.shares_to_adjust)) +
                           " shares. Expected P/L: $" + std::to_string(opp.expected_profit);

        return opp;
    }

    /**
     * Estimate P/L from theta decay
     *
     * Calculate expected profit/loss from time decay over given period
     */
    [[nodiscard]] auto estimateThetaDecay(
        PortfolioGreeks const& portfolio,
        int days
    ) -> double {
        return portfolio.dollar_theta * days;
    }

    /**
     * Estimate P/L from volatility change
     */
    [[nodiscard]] auto estimateVolatilityPnL(
        PortfolioGreeks const& portfolio,
        double vol_change_pct
    ) -> double {
        // Vega is per 1% IV change
        return portfolio.dollar_vega * vol_change_pct;
    }

    /**
     * Get Greeks sensitivity analysis
     *
     * Show how portfolio value changes with various scenarios
     */
    [[nodiscard]] auto getSensitivityAnalysis(
        std::vector<PositionGreeks> const& positions,
        double current_price
    ) -> std::string {

        std::ostringstream oss;
        oss << "Options Portfolio Sensitivity Analysis\n";
        oss << "========================================\n\n";

        auto portfolio = calculatePortfolioGreeks(positions, 0.0);

        oss << "Current Portfolio Greeks:\n";
        oss << "  Delta: " << portfolio.total_delta << " (Dollar Delta: $" << portfolio.dollar_delta << ")\n";
        oss << "  Gamma: " << portfolio.total_gamma << "\n";
        oss << "  Theta: " << portfolio.total_theta << " (Dollar Theta: $" << portfolio.dollar_theta << "/day)\n";
        oss << "  Vega:  " << portfolio.total_vega << " (Dollar Vega: $" << portfolio.dollar_vega << ")\n\n";

        oss << "Price Scenarios (P/L impact):\n";
        for (double price_change : {-5.0, -2.0, -1.0, 1.0, 2.0, 5.0}) {
            double pnl = portfolio.total_delta * price_change +
                        0.5 * portfolio.total_gamma * price_change * price_change;
            oss << "  Price change $" << price_change << ": $" << (pnl * 100.0) << "\n";
        }
        oss << "\n";

        oss << "Time Decay Scenarios (P/L from theta):\n";
        for (int days : {1, 7, 14, 30}) {
            double theta_pnl = estimateThetaDecay(portfolio, days);
            oss << "  " << days << " days: $" << theta_pnl << "\n";
        }
        oss << "\n";

        oss << "Volatility Scenarios (P/L from vega):\n";
        for (double vol_change : {-5.0, -2.0, 2.0, 5.0}) {
            double vega_pnl = estimateVolatilityPnL(portfolio, vol_change);
            oss << "  IV change " << vol_change << "%: $" << vega_pnl << "\n";
        }
        oss << "\n";

        return oss.str();
    }

    /**
     * Monitor Greeks limits and generate alerts
     */
    [[nodiscard]] auto checkGreeksLimits(
        PortfolioGreeks const& portfolio,
        double max_delta,
        double max_gamma,
        double max_theta_per_day,
        double max_vega
    ) -> std::vector<std::string> {

        std::vector<std::string> alerts;

        if (std::abs(portfolio.total_delta) > max_delta) {
            alerts.push_back("WARNING: Portfolio delta " + std::to_string(portfolio.total_delta) +
                           " exceeds limit " + std::to_string(max_delta));
        }

        if (std::abs(portfolio.total_gamma) > max_gamma) {
            alerts.push_back("WARNING: Portfolio gamma " + std::to_string(portfolio.total_gamma) +
                           " exceeds limit " + std::to_string(max_gamma));
        }

        if (std::abs(portfolio.total_theta) > max_theta_per_day) {
            alerts.push_back("WARNING: Portfolio theta " + std::to_string(portfolio.total_theta) +
                           " exceeds limit " + std::to_string(max_theta_per_day));
        }

        if (std::abs(portfolio.total_vega) > max_vega) {
            alerts.push_back("WARNING: Portfolio vega " + std::to_string(portfolio.total_vega) +
                           " exceeds limit " + std::to_string(max_vega));
        }

        return alerts;
    }

private:
    TrinomialPricer pricer_;
};

// ============================================================================
// Real-Time Greeks Monitor
// ============================================================================

class GreeksMonitor {
public:
    GreeksMonitor() = default;

    /**
     * Add position to monitor
     */
    auto addPosition(PositionGreeks const& position) -> void {
        positions_[position.position_id] = position;
        Logger::getInstance().info("Added position {} to Greeks monitor", position.position_id);
    }

    /**
     * Update position Greeks
     */
    auto updatePosition(
        std::string const& position_id,
        double new_spot_price,
        double new_volatility
    ) -> void {

        auto it = positions_.find(position_id);
        if (it == positions_.end()) {
            Logger::getInstance().warn("Position {} not found in monitor", position_id);
            return;
        }

        auto& pos = it->second;
        pos.spot_price = new_spot_price;
        pos.implied_volatility = new_volatility;

        // Recalculate Greeks
        GreeksCalculator calc;
        OptionType type = OptionType::CALL; // Would need to track this
        auto updated = calc.calculatePositionGreeks(
            pos.symbol, new_spot_price, pos.strike_price, pos.time_to_expiry,
            new_volatility, pos.risk_free_rate, type, pos.quantity
        );

        pos.delta = updated.delta;
        pos.gamma = updated.gamma;
        pos.theta = updated.theta;
        pos.vega = updated.vega;
        pos.rho = updated.rho;
        pos.current_value = updated.current_value;
        pos.last_update = updated.last_update;

        Logger::getInstance().debug("Updated Greeks for position {}", position_id);
    }

    /**
     * Get current portfolio Greeks
     */
    [[nodiscard]] auto getPortfolioGreeks(double portfolio_value) const -> PortfolioGreeks {
        std::vector<PositionGreeks> pos_list;
        for (auto const& [id, pos] : positions_) {
            pos_list.push_back(pos);
        }

        GreeksCalculator calc;
        return calc.calculatePortfolioGreeks(pos_list, portfolio_value);
    }

    /**
     * Get position by ID
     */
    [[nodiscard]] auto getPosition(std::string const& position_id) const
        -> std::optional<PositionGreeks> {

        auto it = positions_.find(position_id);
        if (it != positions_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * Remove position
     */
    auto removePosition(std::string const& position_id) -> void {
        positions_.erase(position_id);
        Logger::getInstance().info("Removed position {} from Greeks monitor", position_id);
    }

    /**
     * Get all positions
     */
    [[nodiscard]] auto getAllPositions() const -> std::vector<PositionGreeks> {
        std::vector<PositionGreeks> result;
        for (auto const& [id, pos] : positions_) {
            result.push_back(pos);
        }
        return result;
    }

private:
    std::unordered_map<std::string, PositionGreeks> positions_;
};

} // namespace options
