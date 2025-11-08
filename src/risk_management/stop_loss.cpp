/**
 * Stop Loss Implementation
 * C++23 module implementation file
 */

import bigbrother.risk_management;
import bigbrother.utils.logger;
import bigbrother.utils.timer;

#include <algorithm>

namespace bigbrother::risk {

/**
 * Stop Loss Implementation
 *
 * Manages various types of stop losses to protect capital.
 */

[[nodiscard]] auto StopLossManager::Stop::isTriggered(Price current_price)
    const noexcept -> bool {

    if (triggered) {
        return true;  // Already triggered
    }

    switch (type) {
        case StopType::Hard: {
            // Simple price-based stop
            // Trigger if price crosses threshold
            if (initial_price > trigger_price) {
                // Long position - stop if price falls below
                return current_price <= trigger_price;
            } else {
                // Short position - stop if price rises above
                return current_price >= trigger_price;
            }
        }

        case StopType::Trailing: {
            // Trailing stop follows price
            // For long: stop trails below price by trail_amount
            // For short: stop trails above price by trail_amount

            if (initial_price > trigger_price) {
                // Long position
                double const trailing_stop = current_price - trail_amount;
                return current_price <= trailing_stop;
            } else {
                // Short position
                double const trailing_stop = current_price + trail_amount;
                return current_price >= trailing_stop;
            }
        }

        case StopType::TimeStop: {
            // Exit after time limit
            auto const now = utils::Timer::now();
            return now >= expiration;
        }

        case StopType::VolatilityStop: {
            // Stop based on volatility expansion
            // trail_amount represents volatility threshold
            double const price_change_pct = std::abs(
                (current_price - initial_price) / initial_price
            );
            return price_change_pct > trail_amount;
        }

        case StopType::Greeks: {
            // Stop based on Greeks deterioration
            // (Implementation would need Greeks data)
            // For now, treat like hard stop
            return current_price <= trigger_price;
        }

        default:
            return false;
    }
}

auto StopLossManager::addStop(
    std::string position_id,
    StopType type,
    Price trigger_price,
    Price initial_price,
    double trail_amount
) -> void {

    std::lock_guard lock{mutex_};

    // Remove existing stop for this position
    stops_.erase(
        std::remove_if(stops_.begin(), stops_.end(),
            [&](Stop const& s) { return s.position_id == position_id; }),
        stops_.end()
    );

    // Add new stop
    Stop stop{
        .position_id = std::move(position_id),
        .type = type,
        .trigger_price = trigger_price,
        .initial_price = initial_price,
        .trail_amount = trail_amount,
        .expiration = 0,
        .triggered = false
    };

    stops_.push_back(std::move(stop));

    LOG_INFO("Added {} stop for {}: trigger=${:.2f}",
             [type]() {
                 switch (type) {
                     case StopType::Hard: return "hard";
                     case StopType::Trailing: return "trailing";
                     case StopType::TimeStop: return "time";
                     case StopType::VolatilityStop: return "volatility";
                     case StopType::Greeks: return "greeks";
                     default: return "unknown";
                 }
             }(),
             stop.position_id,
             trigger_price);
}

[[nodiscard]] auto StopLossManager::update(
    std::unordered_map<std::string, Price> const& current_prices
) -> std::vector<std::string> {

    PROFILE_SCOPE("StopLossManager::update");

    std::lock_guard lock{mutex_};

    std::vector<std::string> triggered_positions;

    for (auto& stop : stops_) {
        if (stop.triggered) {
            continue;  // Already triggered
        }

        // Find current price for this position
        auto it = current_prices.find(stop.position_id);
        if (it == current_prices.end()) {
            continue;  // No price update for this position
        }

        Price const current_price = it->second;

        // Check if stop is triggered
        if (stop.isTriggered(current_price)) {
            stop.triggered = true;
            triggered_positions.push_back(stop.position_id);

            LOG_WARN("STOP LOSS TRIGGERED: {} at ${:.2f} (stop: ${:.2f})",
                     stop.position_id,
                     current_price,
                     stop.trigger_price);
        }

        // Update trailing stops
        if (stop.type == StopType::Trailing && !stop.triggered) {
            // Adjust trigger price if beneficial
            if (stop.initial_price > stop.trigger_price) {
                // Long position - raise stop if price increases
                double const new_stop = current_price - stop.trail_amount;
                if (new_stop > stop.trigger_price) {
                    LOG_DEBUG("Trailing stop updated for {}: ${:.2f} -> ${:.2f}",
                             stop.position_id,
                             stop.trigger_price,
                             new_stop);
                    stop.trigger_price = new_stop;
                }
            } else {
                // Short position - lower stop if price decreases
                double const new_stop = current_price + stop.trail_amount;
                if (new_stop < stop.trigger_price) {
                    LOG_DEBUG("Trailing stop updated for {}: ${:.2f} -> ${:.2f}",
                             stop.position_id,
                             stop.trigger_price,
                             new_stop);
                    stop.trigger_price = new_stop;
                }
            }
        }
    }

    return triggered_positions;
}

auto StopLossManager::removeStop(std::string const& position_id) -> void {
    std::lock_guard lock{mutex_};

    stops_.erase(
        std::remove_if(stops_.begin(), stops_.end(),
            [&](Stop const& s) { return s.position_id == position_id; }),
        stops_.end()
    );

    LOG_DEBUG("Removed stop for {}", position_id);
}

[[nodiscard]] auto StopLossManager::getActiveStops() const noexcept
    -> std::vector<Stop> const& {
    return stops_;
}

} // namespace bigbrother::risk
