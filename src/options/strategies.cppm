/**
 * @file strategies.cppm
 * @brief Comprehensive Options Trading Strategies Module
 *
 * Implements professional-grade options trading strategies with ML integration:
 * - Covered Calls: Sell calls against existing stock positions
 * - Cash-Secured Puts: Sell puts with cash reserve
 * - Vertical Spreads: Bull/bear call/put spreads
 * - Iron Condors: Combined credit spreads for range-bound markets
 * - Protective Puts: Portfolio hedging
 *
 * ============================================================================
 * CRITICAL: CORRECT OPTIONS PRICING METHODOLOGY
 * ============================================================================
 *
 * ML PricePredictor predicts UNDERLYING STOCK PRICE, NOT option prices directly.
 *
 * Correct Flow:
 * 1. ML PricePredictor → Predict underlying stock price (e.g., SPY: $450 → $462)
 * 2. Get Implied Volatility (IV) from market data or estimate
 * 3. Trinomial Tree + Risk-Free Rate + IV + Predicted Stock Price → Calculate Option Fair Value
 * 4. Calculate Greeks from trinomial model (delta, gamma, theta, vega, rho)
 * 5. Trading Decision: Compare fair value to market price, use Greeks for risk assessment
 *
 * ML Integration (CORRECT):
 * - Uses PricePredictor for STOCK directional bias (predict future stock price)
 * - Uses predicted stock price to inform strike selection
 * - Options are priced using trinomial tree with IV from market
 * - Greeks calculated from trinomial model, NOT from ML
 * - Strike price optimization based on predicted stock movement
 * - DTE (Days To Expiration) optimization
 *
 * Risk Management:
 * - Position sizing based on account value
 * - Greeks limits (delta, theta, vega exposure)
 * - Collateral requirements
 * - Stop-loss triggers
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-15 (Updated with correct methodology)
 */

module;

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

export module options:strategies;

import options:trinomial_pricer;
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.market_intelligence.price_predictor;

export namespace options {

using namespace bigbrother::types;
using namespace bigbrother::utils;

// ============================================================================
// Options Strategy Enumerations
// ============================================================================

enum class StrategyType {
    CoveredCall,
    CashSecuredPut,
    BullCallSpread,
    BearCallSpread,
    BullPutSpread,
    BearPutSpread,
    IronCondor,
    ProtectivePut
};

enum class MarketOutlook {
    Bullish,
    Bearish,
    Neutral,
    HighVolatility,
    LowVolatility
};

// ============================================================================
// Options Position Structure
// ============================================================================

struct OptionsPosition {
    std::string symbol;
    StrategyType strategy_type;
    std::vector<OptionLeg> legs;
    double max_profit{0.0};
    double max_loss{0.0};
    double breakeven_price{0.0};
    double required_capital{0.0};
    double collateral_required{0.0};
    Greeks aggregate_greeks;
    int days_to_expiration{0};
    double confidence{0.0};
    std::string rationale;
};

struct OptionLeg {
    OptionType type{OptionType::CALL};
    double strike{0.0};
    double premium{0.0};
    int quantity{0};      // Positive = long, negative = short
    int days_to_expiry{0};
    Greeks greeks;

    [[nodiscard]] auto isLong() const noexcept -> bool { return quantity > 0; }
    [[nodiscard]] auto isShort() const noexcept -> bool { return quantity < 0; }
};

// ============================================================================
// Strategy Configuration
// ============================================================================

struct StrategyConfig {
    // Position sizing
    double max_position_size_pct{0.05};        // 5% max per position
    double max_portfolio_allocation_pct{0.20}; // 20% max total options exposure

    // Greeks limits
    double max_delta_per_position{0.50};       // Max delta per position
    double max_portfolio_delta{2.00};          // Max total delta exposure
    double max_theta_per_position{-50.0};      // Max theta decay per position
    double max_vega_per_position{100.0};       // Max vega per position

    // Strategy preferences
    int min_dte{7};                            // Minimum days to expiration
    int max_dte{45};                           // Maximum days to expiration
    int preferred_dte{30};                     // Preferred DTE for new positions

    // IV targets
    double min_iv_rank{0.30};                  // Minimum IV rank to sell premium
    double max_iv_rank{0.90};                  // Maximum IV rank (avoid selling)

    // Spread width for vertical spreads
    double spread_width_pct{0.05};             // 5% spread width

    // Probability targets
    double min_win_probability{0.60};          // 60% minimum win rate

    // Stop loss
    double stop_loss_pct{0.50};                // Stop at 50% of max loss
};

// ============================================================================
// Options Strategy Engine
// ============================================================================

class OptionsStrategyEngine {
public:
    explicit OptionsStrategyEngine(StrategyConfig config = StrategyConfig{})
        : config_{std::move(config)}, pricer_{100} {}

    /**
     * Generate covered call strategy
     * Sell call against existing stock position
     *
     * CORRECT METHODOLOGY:
     * 1. Use ML to predict future STOCK price (not option price)
     * 2. Use predicted stock price to select optimal strike
     * 3. Price option using trinomial tree with market IV
     * 4. Calculate Greeks from trinomial model
     *
     * @param symbol Stock symbol
     * @param current_price Current stock price
     * @param shares_owned Number of shares owned
     * @param account_value Total account value
     * @return Covered call position or nullopt
     */
    [[nodiscard]] auto generateCoveredCall(
        std::string const& symbol,
        double current_price,
        int shares_owned,
        double account_value
    ) -> std::optional<OptionsPosition> {

        if (shares_owned < 100) {
            Logger::getInstance().warn("Covered call requires at least 100 shares");
            return std::nullopt;
        }

        // TODO: INTEGRATE ML FOR STOCK PRICE PREDICTION
        // Correct flow:
        // 1. Extract 85 features from market data for symbol
        // 2. Call predictor.predictStockPrice(symbol, current_price, features, 20)
        // 3. Use predicted stock price to determine strike selection
        //
        // For now, use simple heuristic for strike selection
        // This is a TEMPORARY placeholder until feature extraction is integrated

        double strike_multiplier = 1.03; // Default: 3% OTM

        // PLACEHOLDER: In production, this would be:
        // auto features = featureExtractor.extract(symbol, current_price);
        // auto predicted_price = predictor.predictStockPrice(symbol, current_price, features, 20);
        // if (predicted_price) {
        //     // If bullish prediction, sell higher OTM call
        //     if (*predicted_price > current_price * 1.02) {
        //         strike_multiplier = 1.05;  // 5% OTM if bullish
        //     } else {
        //         strike_multiplier = 1.02;  // 2% OTM if neutral
        //     }
        // }

        double strike = std::round(current_price * strike_multiplier);

        // Calculate premium and Greeks
        auto pricing = pricer_.price(
            current_price, strike, 30.0/365.0, 0.25, 0.05,
            OptionType::CALL, OptionStyle::AMERICAN
        );

        // Sell 1 call per 100 shares
        int contracts = shares_owned / 100;

        OptionsPosition position;
        position.symbol = symbol;
        position.strategy_type = StrategyType::CoveredCall;
        position.days_to_expiration = 30;

        // Short call leg
        OptionLeg call_leg;
        call_leg.type = OptionType::CALL;
        call_leg.strike = strike;
        call_leg.premium = pricing.price * 100.0; // Premium per contract
        call_leg.quantity = -contracts; // Short
        call_leg.days_to_expiry = 30;
        call_leg.greeks = pricing.greeks;

        position.legs.push_back(call_leg);

        // Calculate P/L
        position.max_profit = call_leg.premium * std::abs(contracts) +
                             (strike - current_price) * shares_owned;
        position.max_loss = current_price * shares_owned - call_leg.premium * std::abs(contracts);
        position.breakeven_price = current_price - (call_leg.premium / 100.0);
        position.required_capital = 0.0; // Covered by shares
        position.collateral_required = current_price * shares_owned;

        // Aggregate Greeks (negative because we're short)
        position.aggregate_greeks = pricing.greeks;
        position.aggregate_greeks.delta *= -contracts;
        position.aggregate_greeks.theta *= -contracts;
        position.aggregate_greeks.vega *= -contracts;

        // TODO: Update confidence from ML prediction when integrated
        position.confidence = 0.70; // Default confidence until ML integration
        position.rationale = "Covered call: Generate income from " + symbol +
                           " shares. Strike: $" + std::to_string(strike) +
                           ", Premium: $" + std::to_string(call_leg.premium);

        return position;
    }

    /**
     * Generate cash-secured put strategy
     * Sell put with cash reserve to buy stock at lower price
     *
     * CORRECT METHODOLOGY:
     * 1. Use ML to predict future STOCK price
     * 2. Only sell puts if predicted stock price > current (bullish/neutral)
     * 3. Use predicted price to optimize strike selection
     * 4. Price option using trinomial tree with market IV
     * 5. Calculate Greeks from trinomial model
     */
    [[nodiscard]] auto generateCashSecuredPut(
        std::string const& symbol,
        double current_price,
        double available_cash,
        double account_value
    ) -> std::optional<OptionsPosition> {

        // TODO: INTEGRATE ML FOR STOCK PRICE PREDICTION
        // Correct flow:
        // 1. Extract features from market data
        // 2. Predict future stock price using ML
        // 3. Only sell puts if bullish/neutral prediction
        //
        // PLACEHOLDER: Using default strike selection for now

        double strike_multiplier = 0.97; // 3% OTM

        // PLACEHOLDER: In production, this would be:
        // auto features = featureExtractor.extract(symbol, current_price);
        // auto predicted_price = predictor.predictStockPrice(symbol, current_price, features, 20);
        // if (predicted_price) {
        //     if (*predicted_price < current_price * 0.97) {
        //         Logger::getInstance().info("Bearish prediction, skipping cash-secured put");
        //         return std::nullopt;
        //     }
        //     // Adjust strike based on how bullish the prediction is
        //     if (*predicted_price > current_price * 1.03) {
        //         strike_multiplier = 0.95;  // More aggressive if very bullish
        //     }
        // }

        double strike = std::round(current_price * strike_multiplier);

        // Calculate collateral required
        double collateral_per_contract = strike * 100.0;
        int max_contracts = static_cast<int>(available_cash / collateral_per_contract);

        if (max_contracts < 1) {
            Logger::getInstance().warn("Insufficient cash for cash-secured put");
            return std::nullopt;
        }

        // Limit position size
        double max_position_value = account_value * config_.max_position_size_pct;
        int position_limit = static_cast<int>(max_position_value / collateral_per_contract);
        int contracts = std::min(max_contracts, position_limit);

        // Price the put
        auto pricing = pricer_.price(
            current_price, strike, 30.0/365.0, 0.25, 0.05,
            OptionType::PUT, OptionStyle::AMERICAN
        );

        OptionsPosition position;
        position.symbol = symbol;
        position.strategy_type = StrategyType::CashSecuredPut;
        position.days_to_expiration = 30;

        OptionLeg put_leg;
        put_leg.type = OptionType::PUT;
        put_leg.strike = strike;
        put_leg.premium = pricing.price * 100.0;
        put_leg.quantity = -contracts; // Short
        put_leg.days_to_expiry = 30;
        put_leg.greeks = pricing.greeks;

        position.legs.push_back(put_leg);

        position.max_profit = put_leg.premium * contracts;
        position.max_loss = (strike * 100.0 * contracts) - position.max_profit;
        position.breakeven_price = strike - (put_leg.premium / 100.0);
        position.required_capital = strike * 100.0 * contracts;
        position.collateral_required = position.required_capital;

        position.aggregate_greeks = pricing.greeks;
        position.aggregate_greeks.delta *= -contracts;
        position.aggregate_greeks.theta *= -contracts;
        position.aggregate_greeks.vega *= -contracts;

        // TODO: Update confidence from ML prediction when integrated
        position.confidence = 0.70; // Default confidence until ML integration
        position.rationale = "Cash-secured put: Willing to buy " + symbol +
                           " at $" + std::to_string(strike) +
                           ". Premium: $" + std::to_string(put_leg.premium * contracts);

        return position;
    }

    /**
     * Generate bull call spread
     * Buy lower strike call, sell higher strike call
     *
     * CORRECT METHODOLOGY:
     * 1. Use ML to predict future STOCK price
     * 2. Only trade if bullish prediction (predicted price > current)
     * 3. Use predicted price to optimize strike selection
     * 4. Price both legs using trinomial tree with market IV
     * 5. Calculate net Greeks from both legs
     */
    [[nodiscard]] auto generateBullCallSpread(
        std::string const& symbol,
        double current_price,
        double available_cash,
        double account_value
    ) -> std::optional<OptionsPosition> {

        // TODO: INTEGRATE ML FOR STOCK PRICE PREDICTION
        // Correct flow:
        // 1. Extract features and predict future stock price
        // 2. Only trade if sufficiently bullish (predicted_price > current_price * 1.02)
        // 3. Optimize strike selection based on predicted price
        //
        // PLACEHOLDER: Using default strikes for now

        // PLACEHOLDER: In production, check ML prediction:
        // auto features = featureExtractor.extract(symbol, current_price);
        // auto predicted_price = predictor.predictStockPrice(symbol, current_price, features, 20);
        // if (!predicted_price || *predicted_price < current_price * 1.02) {
        //     Logger::getInstance().info("Not bullish enough for bull call spread");
        //     return std::nullopt;
        // }

        // Long call strike: ATM or slightly OTM
        double long_strike = std::round(current_price * 1.00);

        // Short call strike: based on spread width
        double short_strike = std::round(current_price * (1.0 + config_.spread_width_pct));

        // Price both legs
        auto long_call_pricing = pricer_.price(
            current_price, long_strike, 30.0/365.0, 0.25, 0.05,
            OptionType::CALL, OptionStyle::AMERICAN
        );

        auto short_call_pricing = pricer_.price(
            current_price, short_strike, 30.0/365.0, 0.25, 0.05,
            OptionType::CALL, OptionStyle::AMERICAN
        );

        // Net debit
        double net_debit = (long_call_pricing.price - short_call_pricing.price) * 100.0;

        // Calculate position size
        int max_contracts = static_cast<int>(available_cash / net_debit);
        double max_position_value = account_value * config_.max_position_size_pct;
        int position_limit = static_cast<int>(max_position_value / net_debit);
        int contracts = std::min(max_contracts, std::max(1, position_limit));

        OptionsPosition position;
        position.symbol = symbol;
        position.strategy_type = StrategyType::BullCallSpread;
        position.days_to_expiration = 30;

        // Long call leg
        OptionLeg long_call;
        long_call.type = OptionType::CALL;
        long_call.strike = long_strike;
        long_call.premium = long_call_pricing.price * 100.0;
        long_call.quantity = contracts; // Long
        long_call.days_to_expiry = 30;
        long_call.greeks = long_call_pricing.greeks;
        position.legs.push_back(long_call);

        // Short call leg
        OptionLeg short_call;
        short_call.type = OptionType::CALL;
        short_call.strike = short_strike;
        short_call.premium = short_call_pricing.price * 100.0;
        short_call.quantity = -contracts; // Short
        short_call.days_to_expiry = 30;
        short_call.greeks = short_call_pricing.greeks;
        position.legs.push_back(short_call);

        position.max_profit = ((short_strike - long_strike) * 100.0 - net_debit) * contracts;
        position.max_loss = net_debit * contracts;
        position.breakeven_price = long_strike + (net_debit / 100.0);
        position.required_capital = net_debit * contracts;
        position.collateral_required = position.required_capital;

        // Aggregate Greeks (long - short)
        position.aggregate_greeks.delta = (long_call_pricing.greeks.delta - short_call_pricing.greeks.delta) * contracts;
        position.aggregate_greeks.theta = (long_call_pricing.greeks.theta - short_call_pricing.greeks.theta) * contracts;
        position.aggregate_greeks.vega = (long_call_pricing.greeks.vega - short_call_pricing.greeks.vega) * contracts;
        position.aggregate_greeks.gamma = (long_call_pricing.greeks.gamma - short_call_pricing.greeks.gamma) * contracts;

        // TODO: Update confidence from ML prediction when integrated
        position.confidence = 0.70; // Default confidence until ML integration
        position.rationale = "Bull call spread: Bullish on " + symbol +
                           ". Max profit: $" + std::to_string(position.max_profit) +
                           ", Max loss: $" + std::to_string(position.max_loss);

        return position;
    }

    /**
     * Generate iron condor
     * Sell OTM put spread + sell OTM call spread
     * Profits in range-bound market
     *
     * CORRECT METHODOLOGY:
     * 1. Use ML to predict future STOCK price
     * 2. Only trade if neutral prediction (predicted price near current)
     * 3. Center strikes around predicted price
     * 4. Price all 4 legs using trinomial tree with market IV
     * 5. Calculate net Greeks from all legs
     */
    [[nodiscard]] auto generateIronCondor(
        std::string const& symbol,
        double current_price,
        double available_cash,
        double account_value
    ) -> std::optional<OptionsPosition> {

        // TODO: INTEGRATE ML FOR STOCK PRICE PREDICTION
        // Correct flow:
        // 1. Extract features and predict future stock price
        // 2. Only trade if neutral prediction (predicted price within ±3% of current)
        // 3. Center iron condor strikes around predicted price
        //
        // PLACEHOLDER: Using default strikes centered around current price

        // PLACEHOLDER: In production, check ML prediction:
        // auto features = featureExtractor.extract(symbol, current_price);
        // auto predicted_price = predictor.predictStockPrice(symbol, current_price, features, 20);
        // if (!predicted_price) {
        //     return std::nullopt;
        // }
        // double predicted_change_pct = (*predicted_price - current_price) / current_price * 100.0;
        // if (std::abs(predicted_change_pct) > 5.0) {
        //     Logger::getInstance().info("High predicted movement ({:.2f}%), skipping iron condor", predicted_change_pct);
        //     return std::nullopt;
        // }

        // Strike selection for iron condor
        // Lower put spread: 7-10% OTM
        double lower_put_strike = std::round(current_price * 0.93);
        double upper_put_strike = std::round(current_price * 0.95);

        // Upper call spread: 7-10% OTM
        double lower_call_strike = std::round(current_price * 1.05);
        double upper_call_strike = std::round(current_price * 1.07);

        // Price all four legs
        auto lower_put_pricing = pricer_.price(current_price, lower_put_strike, 30.0/365.0, 0.20, 0.05, OptionType::PUT);
        auto upper_put_pricing = pricer_.price(current_price, upper_put_strike, 30.0/365.0, 0.20, 0.05, OptionType::PUT);
        auto lower_call_pricing = pricer_.price(current_price, lower_call_strike, 30.0/365.0, 0.20, 0.05, OptionType::CALL);
        auto upper_call_pricing = pricer_.price(current_price, upper_call_strike, 30.0/365.0, 0.20, 0.05, OptionType::CALL);

        // Net credit = premium collected from short options - premium paid for long options
        double net_credit = (upper_put_pricing.price + lower_call_pricing.price -
                           lower_put_pricing.price - upper_call_pricing.price) * 100.0;

        if (net_credit <= 0.0) {
            Logger::getInstance().warn("Iron condor would result in net debit, skipping");
            return std::nullopt;
        }

        // Calculate max risk (width of spread - net credit)
        double put_spread_width = (upper_put_strike - lower_put_strike) * 100.0;
        double max_risk = put_spread_width - net_credit;

        int max_contracts = static_cast<int>(available_cash / max_risk);
        double max_position_value = account_value * config_.max_position_size_pct;
        int position_limit = static_cast<int>(max_position_value / max_risk);
        int contracts = std::min(max_contracts, std::max(1, position_limit));

        OptionsPosition position;
        position.symbol = symbol;
        position.strategy_type = StrategyType::IronCondor;
        position.days_to_expiration = 30;

        // Long lower put
        OptionLeg long_lower_put;
        long_lower_put.type = OptionType::PUT;
        long_lower_put.strike = lower_put_strike;
        long_lower_put.premium = lower_put_pricing.price * 100.0;
        long_lower_put.quantity = contracts;
        long_lower_put.days_to_expiry = 30;
        long_lower_put.greeks = lower_put_pricing.greeks;
        position.legs.push_back(long_lower_put);

        // Short upper put
        OptionLeg short_upper_put;
        short_upper_put.type = OptionType::PUT;
        short_upper_put.strike = upper_put_strike;
        short_upper_put.premium = upper_put_pricing.price * 100.0;
        short_upper_put.quantity = -contracts;
        short_upper_put.days_to_expiry = 30;
        short_upper_put.greeks = upper_put_pricing.greeks;
        position.legs.push_back(short_upper_put);

        // Short lower call
        OptionLeg short_lower_call;
        short_lower_call.type = OptionType::CALL;
        short_lower_call.strike = lower_call_strike;
        short_lower_call.premium = lower_call_pricing.price * 100.0;
        short_lower_call.quantity = -contracts;
        short_lower_call.days_to_expiry = 30;
        short_lower_call.greeks = lower_call_pricing.greeks;
        position.legs.push_back(short_lower_call);

        // Long upper call
        OptionLeg long_upper_call;
        long_upper_call.type = OptionType::CALL;
        long_upper_call.strike = upper_call_strike;
        long_upper_call.premium = upper_call_pricing.price * 100.0;
        long_upper_call.quantity = contracts;
        long_upper_call.days_to_expiry = 30;
        long_upper_call.greeks = upper_call_pricing.greeks;
        position.legs.push_back(long_upper_call);

        position.max_profit = net_credit * contracts;
        position.max_loss = (put_spread_width - net_credit) * contracts;
        position.breakeven_price = current_price; // Simplified
        position.required_capital = position.max_loss;
        position.collateral_required = position.max_loss;

        // Aggregate Greeks (should be near-neutral)
        position.aggregate_greeks.delta = 0.0; // Approximately delta-neutral
        position.aggregate_greeks.theta = (lower_put_pricing.greeks.theta * contracts -
                                          upper_put_pricing.greeks.theta * contracts -
                                          lower_call_pricing.greeks.theta * contracts +
                                          upper_call_pricing.greeks.theta * contracts);

        // TODO: Update confidence from ML prediction when integrated
        position.confidence = 0.70; // Default confidence until ML integration
        position.rationale = "Iron condor: Range-bound profit on " + symbol +
                           ". Max profit: $" + std::to_string(position.max_profit) +
                           ", Max risk: $" + std::to_string(position.max_loss);

        return position;
    }

    /**
     * Generate protective put
     * Buy put to hedge existing stock position
     *
     * CORRECT METHODOLOGY:
     * 1. Use ML to predict future STOCK price
     * 2. Only hedge if bearish/uncertain prediction
     * 3. Select strike based on predicted downside
     * 4. Price put using trinomial tree with market IV
     * 5. Calculate Greeks for hedge ratio optimization
     */
    [[nodiscard]] auto generateProtectivePut(
        std::string const& symbol,
        double current_price,
        int shares_owned,
        double available_cash
    ) -> std::optional<OptionsPosition> {

        if (shares_owned < 100) {
            Logger::getInstance().warn("Protective put requires at least 100 shares");
            return std::nullopt;
        }

        // TODO: INTEGRATE ML FOR STOCK PRICE PREDICTION
        // Correct flow:
        // 1. Extract features and predict future stock price
        // 2. Only hedge if bearish prediction or high uncertainty
        // 3. Select strike based on predicted downside risk
        //
        // PLACEHOLDER: Using default hedge strategy for now

        // PLACEHOLDER: In production, assess downside risk:
        // auto features = featureExtractor.extract(symbol, current_price);
        // auto predicted_price = predictor.predictStockPrice(symbol, current_price, features, 20);
        // if (predicted_price && *predicted_price > current_price * 1.05) {
        //     Logger::getInstance().info("Bullish prediction, protective put not needed");
        //     return std::nullopt;
        // }
        // // Adjust strike based on predicted downside
        // if (predicted_price) {
        //     double predicted_drop = (current_price - *predicted_price) / current_price;
        //     strike_multiplier = 1.0 - (predicted_drop * 0.7);  // Protect 70% of predicted drop
        // }

        // Select strike 5% below current price (5% downside protection)
        double strike = std::round(current_price * 0.95);

        int contracts = shares_owned / 100;

        // Price the put
        auto pricing = pricer_.price(
            current_price, strike, 30.0/365.0, 0.30, 0.05,
            OptionType::PUT, OptionStyle::AMERICAN
        );

        double total_cost = pricing.price * 100.0 * contracts;

        if (total_cost > available_cash) {
            Logger::getInstance().warn("Insufficient cash for protective put");
            return std::nullopt;
        }

        OptionsPosition position;
        position.symbol = symbol;
        position.strategy_type = StrategyType::ProtectivePut;
        position.days_to_expiration = 30;

        OptionLeg put_leg;
        put_leg.type = OptionType::PUT;
        put_leg.strike = strike;
        put_leg.premium = pricing.price * 100.0;
        put_leg.quantity = contracts; // Long
        put_leg.days_to_expiry = 30;
        put_leg.greeks = pricing.greeks;

        position.legs.push_back(put_leg);

        // Max loss is limited to distance to strike + premium paid
        position.max_loss = ((current_price - strike) * shares_owned) + total_cost;
        position.max_profit = std::numeric_limits<double>::max(); // Unlimited upside
        position.breakeven_price = current_price + (pricing.price);
        position.required_capital = total_cost;
        position.collateral_required = 0.0; // No collateral for long options

        position.aggregate_greeks = pricing.greeks;
        position.aggregate_greeks.delta *= contracts;
        position.aggregate_greeks.theta *= contracts;
        position.aggregate_greeks.vega *= contracts;

        // TODO: Update confidence from ML prediction when integrated
        position.confidence = 0.70; // Default confidence until ML integration
        position.rationale = "Protective put: Hedge " + symbol + " against downside. " +
                           "Protection below $" + std::to_string(strike) +
                           ". Cost: $" + std::to_string(total_cost);

        return position;
    }

    /**
     * Validate position against risk limits
     */
    [[nodiscard]] auto validatePosition(
        OptionsPosition const& position,
        double current_portfolio_delta,
        double current_portfolio_theta,
        double account_value
    ) const -> bool {

        // Check position size
        double position_pct = position.required_capital / account_value;
        if (position_pct > config_.max_position_size_pct) {
            Logger::getInstance().warn("Position size {} exceeds limit {}",
                                      position_pct, config_.max_position_size_pct);
            return false;
        }

        // Check delta limits
        if (std::abs(position.aggregate_greeks.delta) > config_.max_delta_per_position) {
            Logger::getInstance().warn("Position delta {} exceeds limit {}",
                                      position.aggregate_greeks.delta, config_.max_delta_per_position);
            return false;
        }

        // Check portfolio delta
        double new_portfolio_delta = current_portfolio_delta + position.aggregate_greeks.delta;
        if (std::abs(new_portfolio_delta) > config_.max_portfolio_delta) {
            Logger::getInstance().warn("Portfolio delta {} would exceed limit {}",
                                      new_portfolio_delta, config_.max_portfolio_delta);
            return false;
        }

        // Check theta limits
        if (position.aggregate_greeks.theta < config_.max_theta_per_position) {
            Logger::getInstance().warn("Position theta {} exceeds limit {}",
                                      position.aggregate_greeks.theta, config_.max_theta_per_position);
            return false;
        }

        // Check vega limits
        if (std::abs(position.aggregate_greeks.vega) > config_.max_vega_per_position) {
            Logger::getInstance().warn("Position vega {} exceeds limit {}",
                                      position.aggregate_greeks.vega, config_.max_vega_per_position);
            return false;
        }

        return true;
    }

    /**
     * Get recommended strategy based on market outlook
     */
    [[nodiscard]] auto recommendStrategy(
        MarketOutlook outlook,
        bool owns_stock,
        double cash_available,
        double stock_price
    ) const -> std::vector<StrategyType> {

        std::vector<StrategyType> recommendations;

        switch (outlook) {
            case MarketOutlook::Bullish:
                recommendations.push_back(StrategyType::BullCallSpread);
                recommendations.push_back(StrategyType::BullPutSpread);
                if (!owns_stock && cash_available >= stock_price * 100.0) {
                    recommendations.push_back(StrategyType::CashSecuredPut);
                }
                break;

            case MarketOutlook::Bearish:
                recommendations.push_back(StrategyType::BearCallSpread);
                recommendations.push_back(StrategyType::BearPutSpread);
                if (owns_stock) {
                    recommendations.push_back(StrategyType::ProtectivePut);
                }
                break;

            case MarketOutlook::Neutral:
                recommendations.push_back(StrategyType::IronCondor);
                if (owns_stock) {
                    recommendations.push_back(StrategyType::CoveredCall);
                }
                break;

            case MarketOutlook::HighVolatility:
                // High IV: sell premium
                if (owns_stock) {
                    recommendations.push_back(StrategyType::CoveredCall);
                }
                recommendations.push_back(StrategyType::IronCondor);
                break;

            case MarketOutlook::LowVolatility:
                // Low IV: buy options
                recommendations.push_back(StrategyType::BullCallSpread);
                break;
        }

        return recommendations;
    }

private:
    StrategyConfig config_;
    TrinomialPricer pricer_;
};

// ============================================================================
// IMPLEMENTATION NOTES: ML Integration Requirements
// ============================================================================
//
// The current implementation uses PLACEHOLDER logic for ML-based stock price
// prediction. To enable full ML integration, the following components are needed:
//
// 1. FEATURE EXTRACTION:
//    - Implement FeatureExtractor to build 85-feature vectors from market data
//    - Features include: price data, volume, technical indicators, lags, diffs
//    - Reference: src/market_intelligence/feature_extractor.cppm
//
// 2. MARKET DATA INTEGRATION:
//    - Connect to real-time market data feed (Schwab API, etc.)
//    - Fetch historical data for feature calculation
//    - Get implied volatility from options chain data
//
// 3. ML PREDICTOR INTEGRATION:
//    - Call PricePredictor::predictStockPrice() with extracted features
//    - Use predicted STOCK price (not option price) for strike selection
//    - Use prediction confidence for position sizing
//
// 4. OPTION PRICING:
//    - Use trinomial pricer with market-derived IV (NOT ML predicted)
//    - Feed predicted stock price as spot price for future scenarios
//    - Calculate Greeks from trinomial model
//
// 5. RISK-FREE RATE:
//    - Fetch current risk-free rate from FRED API
//    - Update pricer calls with actual rate (currently hardcoded to 0.05)
//
// Example of correct integration flow:
//
//   // 1. Extract features
//   auto features = featureExtractor.extract(symbol, current_price);
//
//   // 2. Predict future stock price using ML
//   auto predicted_price = predictor.predictStockPrice(
//       symbol, current_price, features, 20  // 20-day horizon
//   );
//
//   // 3. Get market IV (from options chain or estimate)
//   double market_iv = getImpliedVolatility(symbol, strike, expiration);
//
//   // 4. Get risk-free rate
//   double risk_free_rate = fredAPI.getTreasuryRate();
//
//   // 5. Price option using trinomial tree
//   auto pricing = pricer_.price(
//       current_price,           // Current stock price
//       strike,                  // Strike price (optimized using predicted_price)
//       30.0/365.0,             // Time to expiration
//       market_iv,              // Market implied volatility
//       risk_free_rate,         // FRED risk-free rate
//       OptionType::CALL,
//       OptionStyle::AMERICAN
//   );
//
//   // 6. Greeks come from trinomial model, NOT from ML
//   Greeks greeks = pricing.greeks;
//
// ============================================================================

} // namespace options
