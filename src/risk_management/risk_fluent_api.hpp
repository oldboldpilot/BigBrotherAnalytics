#pragma once

#include "risk_manager.hpp"
#include <optional>

namespace bigbrother::risk {

/**
 * Fluent API for Risk Management
 *
 * Provides intuitive, chainable interface for risk assessment and management.
 *
 * Example Usage:
 *
 *   // Assess a stock trade
 *   auto risk = RiskAssessor()
 *       .forStock("AAPL")
 *       .entryPrice(150.0)
 *       .targetPrice(160.0)
 *       .stopPrice(145.0)
 *       .positionSize(1000.0)
 *       .winProbability(0.65)
 *       .assess();
 *
 *   if (risk->isApproved()) {
 *       // Execute trade
 *   }
 *
 *   // Assess an options trade
 *   auto option_risk = RiskAssessor()
 *       .forOption("SPY")
 *       .call()
 *       .strike(450.0)
 *       .daysToExpiration(30)
 *       .volatility(0.20)
 *       .positionSize(2000.0)
 *       .assess();
 *
 *   // Position sizing
 *   auto size = PositionSizeCalculator()
 *       .accountValue(30000.0)
 *       .winProbability(0.70)
 *       .expectedWin(500.0)
 *       .expectedLoss(200.0)
 *       .useKellyHalf()
 *       .calculate();
 *
 *   // Monte Carlo simulation
 *   auto simulation = MonteCarloSimulation()
 *       .forOption(params)
 *       .positionSize(1500.0)
 *       .simulations(10000)
 *       .run();
 */

class RiskAssessor {
public:
    RiskAssessor() = default;

    // Symbol selection
    [[nodiscard]] auto forStock(std::string symbol) noexcept -> RiskAssessor& {
        symbol_ = std::move(symbol);
        is_option_ = false;
        return *this;
    }

    [[nodiscard]] auto forOption(std::string symbol) noexcept -> RiskAssessor& {
        symbol_ = std::move(symbol);
        is_option_ = true;
        return *this;
    }

    // Stock trade parameters
    [[nodiscard]] auto entryPrice(Price price) noexcept -> RiskAssessor& {
        entry_price_ = price;
        return *this;
    }

    [[nodiscard]] auto targetPrice(Price price) noexcept -> RiskAssessor& {
        target_price_ = price;
        return *this;
    }

    [[nodiscard]] auto stopPrice(Price price) noexcept -> RiskAssessor& {
        stop_price_ = price;
        return *this;
    }

    // Option parameters (chainable with options pricing API)
    [[nodiscard]] auto call() noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->option_type = options::OptionType::Call;
        return *this;
    }

    [[nodiscard]] auto put() noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->option_type = options::OptionType::Put;
        return *this;
    }

    [[nodiscard]] auto american() noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->option_style = options::OptionStyle::American;
        return *this;
    }

    [[nodiscard]] auto spot(Price price) noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->spot_price = price;
        entry_price_ = price;  // For risk calculations
        return *this;
    }

    [[nodiscard]] auto strike(Price price) noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->strike_price = price;
        return *this;
    }

    [[nodiscard]] auto daysToExpiration(int days) noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->time_to_expiration = static_cast<double>(days) / 365.0;
        return *this;
    }

    [[nodiscard]] auto volatility(double vol) noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->volatility = vol;
        volatility_ = vol;
        return *this;
    }

    [[nodiscard]] auto riskFreeRate(double rate) noexcept -> RiskAssessor& {
        if (!option_params_) {
            option_params_ = options::PricingParams{};
        }
        option_params_->risk_free_rate = rate;
        return *this;
    }

    // Common parameters
    [[nodiscard]] auto positionSize(double size) noexcept -> RiskAssessor& {
        position_size_ = size;
        return *this;
    }

    [[nodiscard]] auto winProbability(double prob) noexcept -> RiskAssessor& {
        win_probability_ = prob;
        return *this;
    }

    [[nodiscard]] auto targetProfit(double profit) noexcept -> RiskAssessor& {
        target_profit_ = profit;
        return *this;
    }

    // Risk manager to use
    [[nodiscard]] auto withRiskManager(RiskManager& manager) noexcept -> RiskAssessor& {
        risk_manager_ = &manager;
        return *this;
    }

    // Terminal operation: assess the trade
    [[nodiscard]] auto assess() const noexcept -> Result<TradeRisk> {
        if (!risk_manager_) {
            return makeError<TradeRisk>(
                ErrorCode::InvalidParameter,
                "Risk manager not specified. Use withRiskManager()"
            );
        }

        if (is_option_ && option_params_) {
            // Options trade assessment
            return risk_manager_->assessOptionsTrade(
                symbol_,
                *option_params_,
                position_size_.value_or(0.0),
                win_probability_.value_or(0.5),
                target_profit_.value_or(0.0)
            );
        } else {
            // Stock trade assessment
            return risk_manager_->assessTrade(
                symbol_,
                position_size_.value_or(0.0),
                entry_price_.value_or(0.0),
                stop_price_.value_or(0.0),
                target_price_.value_or(0.0),
                win_probability_.value_or(0.5)
            );
        }
    }

    // Terminal operation: run Monte Carlo simulation
    [[nodiscard]] auto simulate(int num_simulations = 10'000) const noexcept
        -> Result<MonteCarloSimulator::SimulationResult> {

        if (is_option_ && option_params_) {
            return MonteCarloSimulator::simulateOptionTrade(
                *option_params_,
                position_size_.value_or(1000.0),
                num_simulations
            );
        } else {
            return MonteCarloSimulator::simulateStockTrade(
                entry_price_.value_or(0.0),
                target_price_.value_or(0.0),
                stop_price_.value_or(0.0),
                volatility_.value_or(0.25),
                num_simulations
            );
        }
    }

private:
    std::string symbol_;
    bool is_option_{false};

    // Stock parameters
    std::optional<Price> entry_price_;
    std::optional<Price> target_price_;
    std::optional<Price> stop_price_;
    std::optional<double> volatility_;

    // Options parameters
    std::optional<options::PricingParams> option_params_;

    // Common parameters
    std::optional<double> position_size_;
    std::optional<double> win_probability_;
    std::optional<double> target_profit_;

    // Risk manager
    RiskManager* risk_manager_{nullptr};
};

/**
 * Position Size Calculator Fluent API
 */
class PositionSizeCalculator {
public:
    PositionSizeCalculator() = default;

    [[nodiscard]] auto accountValue(double value) noexcept -> PositionSizeCalculator& {
        account_value_ = value;
        return *this;
    }

    [[nodiscard]] auto winProbability(double prob) noexcept -> PositionSizeCalculator& {
        win_prob_ = prob;
        return *this;
    }

    [[nodiscard]] auto expectedWin(double amount) noexcept -> PositionSizeCalculator& {
        win_amount_ = amount;
        return *this;
    }

    [[nodiscard]] auto expectedLoss(double amount) noexcept -> PositionSizeCalculator& {
        loss_amount_ = amount;
        return *this;
    }

    [[nodiscard]] auto maxPosition(double max) noexcept -> PositionSizeCalculator& {
        max_position_ = max;
        return *this;
    }

    // Sizing methods
    [[nodiscard]] auto useFixedDollar() noexcept -> PositionSizeCalculator& {
        method_ = PositionSizer::Method::FixedDollar;
        return *this;
    }

    [[nodiscard]] auto useFixedPercent() noexcept -> PositionSizeCalculator& {
        method_ = PositionSizer::Method::FixedPercent;
        return *this;
    }

    [[nodiscard]] auto useKelly() noexcept -> PositionSizeCalculator& {
        method_ = PositionSizer::Method::KellyCriterion;
        return *this;
    }

    [[nodiscard]] auto useKellyHalf() noexcept -> PositionSizeCalculator& {
        method_ = PositionSizer::Method::KellyHalf;
        return *this;
    }

    [[nodiscard]] auto useVolatilityAdjusted() noexcept -> PositionSizeCalculator& {
        method_ = PositionSizer::Method::VolatilityAdjusted;
        return *this;
    }

    // Terminal operation
    [[nodiscard]] auto calculate() const noexcept -> Result<double> {
        return PositionSizer::calculateSize(
            method_,
            account_value_,
            win_prob_,
            win_amount_,
            loss_amount_,
            max_position_
        );
    }

private:
    double account_value_{30'000.0};
    double win_prob_{0.6};
    double win_amount_{500.0};
    double loss_amount_{200.0};
    double max_position_{std::numeric_limits<double>::infinity()};
    PositionSizer::Method method_{PositionSizer::Method::KellyHalf};
};

/**
 * Monte Carlo Simulation Fluent API
 */
class MonteCarloSimulation {
public:
    MonteCarloSimulation() = default;

    [[nodiscard]] auto forStock(
        Price entry,
        Price target,
        Price stop,
        double vol
    ) noexcept -> MonteCarloSimulation& {
        is_option_ = false;
        entry_price_ = entry;
        target_price_ = target;
        stop_price_ = stop;
        volatility_ = vol;
        return *this;
    }

    [[nodiscard]] auto forOption(
        options::PricingParams params
    ) noexcept -> MonteCarloSimulation& {
        is_option_ = true;
        option_params_ = std::move(params);
        return *this;
    }

    [[nodiscard]] auto positionSize(double size) noexcept -> MonteCarloSimulation& {
        position_size_ = size;
        return *this;
    }

    [[nodiscard]] auto simulations(int num) noexcept -> MonteCarloSimulation& {
        num_simulations_ = num;
        return *this;
    }

    [[nodiscard]] auto steps(int num) noexcept -> MonteCarloSimulation& {
        num_steps_ = num;
        return *this;
    }

    // Terminal operation
    [[nodiscard]] auto run() const noexcept
        -> Result<MonteCarloSimulator::SimulationResult> {

        if (is_option_) {
            return MonteCarloSimulator::simulateOptionTrade(
                option_params_,
                position_size_,
                num_simulations_,
                num_steps_
            );
        } else {
            return MonteCarloSimulator::simulateStockTrade(
                entry_price_,
                target_price_,
                stop_price_,
                volatility_,
                num_simulations_
            );
        }
    }

private:
    bool is_option_{false};

    // Stock parameters
    Price entry_price_{0.0};
    Price target_price_{0.0};
    Price stop_price_{0.0};
    double volatility_{0.25};

    // Option parameters
    options::PricingParams option_params_{};

    // Simulation parameters
    double position_size_{1000.0};
    int num_simulations_{10'000};
    int num_steps_{100};
};

/**
 * Convenience functions for quick risk assessment
 */

// Quick position size calculation with Kelly Half
[[nodiscard]] inline auto calculateKellyHalfSize(
    double account_value,
    double win_probability,
    double expected_win,
    double expected_loss,
    double max_position = 1500.0
) noexcept -> Result<double> {

    return PositionSizeCalculator()
        .accountValue(account_value)
        .winProbability(win_probability)
        .expectedWin(expected_win)
        .expectedLoss(expected_loss)
        .maxPosition(max_position)
        .useKellyHalf()
        .calculate();
}

// Quick Monte Carlo for options
[[nodiscard]] inline auto simulateOption(
    options::PricingParams const& params,
    double position_size,
    int simulations = 10'000
) noexcept -> Result<MonteCarloSimulator::SimulationResult> {

    return MonteCarloSimulation()
        .forOption(params)
        .positionSize(position_size)
        .simulations(simulations)
        .run();
}

} // namespace bigbrother::risk
