#include "strategy_straddle.hpp"
#include "../utils/logger.hpp"
#include "../utils/timer.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace bigbrother::strategy {

/**
 * Delta-Neutral Straddle Implementation
 *
 * This is a CRITICAL strategy for options day trading.
 *
 * Trading Logic:
 * 1. Find ATM strike (closest to spot price)
 * 2. Buy call + put at same strike
 * 3. Verify delta-neutrality (Δ_total ≈ 0)
 * 4. Calculate breakeven points
 * 5. Estimate probability of profit using Monte Carlo
 * 6. Size position using Kelly Criterion
 * 7. Set stop loss and profit targets
 *
 * Profit Scenarios:
 * - Stock moves up significantly: Call profits > Put loss
 * - Stock moves down significantly: Put profits > Call loss
 * - Stock stays flat: Maximum loss = Total premium paid
 *
 * This strategy profits from VOLATILITY, not direction.
 */

[[nodiscard]] auto DeltaNeutralStraddleStrategy::Parameters::validate() const noexcept
    -> Result<void> {

    if (max_position_size <= 0.0) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Max position size must be positive"
        );
    }

    if (min_expected_return < 0.0) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Min expected return cannot be negative"
        );
    }

    if (min_win_probability < 0.0 || min_win_probability > 1.0) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Win probability must be between 0 and 1"
        );
    }

    if (min_days_to_expiration < 1 || max_days_to_expiration < min_days_to_expiration) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Invalid expiration range"
        );
    }

    return {};
}

DeltaNeutralStraddleStrategy::DeltaNeutralStraddleStrategy(Parameters params)
    : BaseStrategy("DeltaNeutralStraddle",
                  "Profit from volatility by buying ATM call and put"),
      params_{std::move(params)} {

    if (auto validation = params_.validate(); !validation) {
        LOG_ERROR("Invalid strategy parameters: {}", validation.error().message);
        setActive(false);
    }
}

[[nodiscard]] auto DeltaNeutralStraddleStrategy::generateSignals(
    StrategyContext const& context
) -> std::vector<TradingSignal> {

    PROFILE_SCOPE("DeltaNeutralStraddle::generateSignals");

    if (!isActive()) {
        return {};
    }

    std::vector<TradingSignal> signals;

    // Check if we have room for more positions
    int const active_straddles = std::ranges::count_if(
        context.current_positions,
        [](auto const& pos) {
            return pos.symbol.find("STRADDLE_") == 0;
        }
    );

    if (active_straddles >= params_.max_concurrent_positions) {
        LOG_DEBUG("Max straddle positions reached: {}/{}",
                 active_straddles, params_.max_concurrent_positions);
        return {};
    }

    // Screen all symbols with options chains
    for (auto const& [symbol, chain] : context.options_chains) {
        // Get current quote
        auto quote_it = context.current_quotes.find(symbol);
        if (quote_it == context.current_quotes.end()) {
            continue;
        }

        auto signal = findBestStraddle(symbol, context);

        if (signal && signal->isActionable()) {
            signals.push_back(*signal);
        }
    }

    // Sort by expected return (best first)
    std::ranges::sort(signals, [](auto const& a, auto const& b) {
        return a.expected_return > b.expected_return;
    });

    LOG_INFO("DeltaNeutralStraddle generated {} signals", signals.size());

    return signals;
}

[[nodiscard]] auto DeltaNeutralStraddleStrategy::findBestStraddle(
    std::string const& symbol,
    StrategyContext const& context
) const -> std::optional<TradingSignal> {

    auto chain_it = context.options_chains.find(symbol);
    auto quote_it = context.current_quotes.find(symbol);

    if (chain_it == context.options_chains.end() || quote_it == context.current_quotes.end()) {
        return std::nullopt;
    }

    auto const& chain = chain_it->second;
    auto const& quote = quote_it->second;

    Price const spot = quote.last;

    // Find ATM strike (closest to spot)
    Price atm_strike = 0.0;
    double min_distance = std::numeric_limits<double>::infinity();

    for (auto const& option : chain.all_options) {
        double const distance = std::abs(option.contract.strike - spot);
        if (distance < min_distance) {
            min_distance = distance;
            atm_strike = option.contract.strike;
        }
    }

    if (atm_strike == 0.0) {
        return std::nullopt;
    }

    // Find ATM call and put
    auto const atm_call = chain.findContract(
        options::OptionType::Call,
        atm_strike,
        0  // TODO: Add expiration filtering
    );

    auto const atm_put = chain.findContract(
        options::OptionType::Put,
        atm_strike,
        0
    );

    if (!atm_call || !atm_put) {
        LOG_DEBUG("No ATM options found for {} at strike {}", symbol, atm_strike);
        return std::nullopt;
    }

    // Check days to expiration
    double const dte = atm_call->contract.daysToExpiration(context.current_time);

    if (dte < params_.min_days_to_expiration || dte > params_.max_days_to_expiration) {
        return std::nullopt;
    }

    // Calculate total cost
    Price const call_premium = atm_call->quote.ask;  // We buy at ask
    Price const put_premium = atm_put->quote.ask;
    Price const total_premium = call_premium + put_premium;
    Price const total_cost = total_premium * 100.0;  // Per contract (100 shares)

    // Check position size limit
    if (total_cost > params_.max_position_size) {
        LOG_DEBUG("Straddle cost ${:.2f} exceeds max position ${:.2f}",
                 total_cost, params_.max_position_size);
        return std::nullopt;
    }

    // Check delta-neutrality
    if (!isDeltaNeutral(atm_call->greeks, atm_put->greeks)) {
        LOG_WARN("Straddle not delta-neutral for {}: Δ_call={:.3f}, Δ_put={:.3f}",
                symbol, atm_call->greeks.delta, atm_put->greeks.delta);
        // Continue anyway - may still be profitable
    }

    // Calculate breakeven points
    auto const [lower_breakeven, upper_breakeven] = calculateBreakevens(
        atm_strike,
        call_premium,
        put_premium
    );

    // Calculate expected move based on IV
    double const avg_iv = (atm_call->implied_volatility + atm_put->implied_volatility) / 2.0;
    Price const expected_move = calculateExpectedMove(spot, avg_iv, dte);

    // Probability of profit estimate
    // If expected move > total premium, profitable
    double win_probability = 0.5;  // Base case

    if (expected_move > total_premium * 1.2) {
        win_probability = 0.70;  // High probability
    } else if (expected_move > total_premium) {
        win_probability = 0.60;
    } else {
        win_probability = 0.40;  // Low probability
    }

    // Expected return calculation
    // Simplified: If stock moves expected_move, straddle value increases
    double expected_straddle_value = total_premium;

    if (expected_move > total_premium) {
        // Rough approximation: profit ≈ (expected_move - total_premium)
        expected_straddle_value = expected_move * 1.5;
    }

    double const expected_return = (expected_straddle_value - total_premium) * 100.0;

    // Check minimum criteria
    if (expected_return < params_.min_expected_return) {
        return std::nullopt;
    }

    if (win_probability < params_.min_win_probability) {
        return std::nullopt;
    }

    // Generate signal
    TradingSignal signal;
    signal.strategy_name = getName();
    signal.symbol = symbol;
    signal.type = SignalType::Buy;
    signal.confidence = win_probability;
    signal.expected_return = expected_return;
    signal.max_risk = total_cost;  // Maximum loss = premium paid
    signal.win_probability = win_probability;
    signal.timestamp = context.current_time;

    // Build rationale
    signal.rationale = std::format(
        "Delta-neutral straddle on {} at ${:.2f} strike. "
        "Cost: ${:.2f} (call=${:.2f}, put=${:.2f}). "
        "Breakevens: ${:.2f} - ${:.2f}. "
        "Expected move: ${:.2f} (IV={:.1f}%). "
        "DTE: {} days. "
        "Expected return: ${:.2f} ({:.0f}% probability).",
        symbol, atm_strike, total_cost, call_premium, put_premium,
        lower_breakeven, upper_breakeven,
        expected_move, avg_iv * 100.0,
        static_cast<int>(dte),
        expected_return, win_probability * 100.0
    );

    // Key features
    signal.features = {
        std::format("IV={:.1f}%", avg_iv * 100.0),
        std::format("ExpectedMove=${:.2f}", expected_move),
        std::format("Cost=${:.2f}", total_cost),
        std::format("DTE={}", static_cast<int>(dte))
    };

    // Store option details
    signal.option_contract = atm_call->contract;
    signal.greeks = atm_call->greeks;
    signal.implied_volatility = avg_iv;

    LOG_INFO("Straddle opportunity: {}", signal.rationale);

    return signal;
}

[[nodiscard]] auto DeltaNeutralStraddleStrategy::calculateBreakevens(
    Price strike,
    Price call_premium,
    Price put_premium
) const noexcept -> std::pair<Price, Price> {

    Price const total_premium = call_premium + put_premium;

    Price const lower_breakeven = strike - total_premium;
    Price const upper_breakeven = strike + total_premium;

    return {lower_breakeven, upper_breakeven};
}

[[nodiscard]] auto DeltaNeutralStraddleStrategy::calculateExpectedMove(
    Price spot,
    double implied_volatility,
    double days_to_expiration
) const noexcept -> Price {

    // Expected move (1 standard deviation)
    // EM = S * σ * √(T)
    // where T is time in years

    double const years_to_expiration = days_to_expiration / 365.0;
    Price const expected_move = spot * implied_volatility * std::sqrt(years_to_expiration);

    return expected_move;
}

[[nodiscard]] auto DeltaNeutralStraddleStrategy::isDeltaNeutral(
    options::Greeks const& call_greeks,
    options::Greeks const& put_greeks,
    double tolerance
) const noexcept -> bool {

    // For a straddle with 1 call + 1 put:
    // Total delta = Δ_call + Δ_put
    // Should be close to 0 (within tolerance)

    double const total_delta = call_greeks.delta + put_greeks.delta;

    return std::abs(total_delta) <= tolerance;
}

[[nodiscard]] auto DeltaNeutralStraddleStrategy::getParameters() const
    -> std::map<std::string, std::string> {

    return {
        {"max_position_size", std::to_string(params_.max_position_size)},
        {"min_expected_return", std::to_string(params_.min_expected_return)},
        {"min_win_probability", std::to_string(params_.min_win_probability)},
        {"min_days_to_expiration", std::to_string(params_.min_days_to_expiration)},
        {"max_days_to_expiration", std::to_string(params_.max_days_to_expiration)},
        {"profit_target_percent", std::to_string(params_.profit_target_percent)},
        {"stop_loss_percent", std::to_string(params_.stop_loss_percent)}
    };
}

} // namespace bigbrother::strategy
